import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from model.blocks import SpatialAdaLN
from model.padding import GeoCyclicPadding

class DRAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        mesh_size: tuple,
        num_heads: int = 4,
        window_radius: int = 5,
        zero_init_output: bool = False,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.H, self.W = mesh_size
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.window_radius = window_radius

        self.in_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)

        self.local_mixer = nn.Sequential(
            GeoCyclicPadding(2),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, groups=hidden_dim, bias=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=True),
        )

        self.norm = nn.GroupNorm(1, hidden_dim, affine=False)

        self.qkv_proj = nn.Linear(hidden_dim + 3, 3 * hidden_dim)
        self.attn_out = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.full((hidden_dim,), -4.0))

        x_coord = torch.cos(lat_grid) * torch.cos(lon_grid)
        y_coord = torch.cos(lat_grid) * torch.sin(lon_grid)
        z_coord = torch.sin(lat_grid)

        coords_3d = torch.stack([x_coord, y_coord, z_coord], dim=0).unsqueeze(0)
        self.register_buffer("coords_3d", coords_3d)

        self.block_mask = None

        self._init_weights(zero_init_output=zero_init_output)


    def _init_weights(self, zero_init_output: bool):
        nn.init.kaiming_normal_(self.in_proj.weight, mode="fan_in", nonlinearity="relu")
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.xavier_normal_(self.qkv_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)

        # Small attention branch, not exactly dead.
        nn.init.normal_(self.attn_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.attn_out.bias)

        if zero_init_output:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.out_proj.bias)
        else:
            nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_in", nonlinearity="relu")
            if self.out_proj.bias is not None:
                nn.init.zeros_(self.out_proj.bias)


    def to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)

    def to_grid(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        return x.permute(0, 2, 1).reshape(B, C, self.H, self.W)

    def enforce_pole_continuity(self, x: torch.Tensor) -> torch.Tensor:
        south_mean = x[:, :, 0:1, :].mean(dim=3, keepdim=True)
        north_mean = x[:, :, -1:, :].mean(dim=3, keepdim=True)

        x = x.clone()
        x[:, :, 0, :] = south_mean.squeeze(-1)
        x[:, :, -1, :] = north_mean.squeeze(-1)
        return x

    def setup_mask(self, device: torch.device):
        if self.block_mask is not None:
            return

        H_val = self.H
        W_val = self.W
        rad = self.window_radius

        L = H_val * W_val
        pad_len = (128 - (L % 128)) % 128
        L_padded = L + pad_len

        def geocyclic_mask(b, h, query_idx, key_idx):
            valid = (query_idx < L) & (key_idx < L)

            q_lat = query_idx // W_val
            q_lon = query_idx % W_val
            k_lat = key_idx // W_val
            k_lon = key_idx % W_val

            lat_dist = abs(q_lat - k_lat)

            direct_lon_dist = abs(q_lon - k_lon)
            wrap_lon_dist = W_val - direct_lon_dist
            lon_dist = torch.minimum(direct_lon_dist, wrap_lon_dist)

            in_direct = (lat_dist <= rad) & (lon_dist <= rad)

            cross_k_lon = (k_lon + W_val // 2) % W_val
            direct_cross_lon_dist = abs(q_lon - cross_k_lon)
            wrap_cross_lon_dist = W_val - direct_cross_lon_dist
            lon_dist_cross = torch.minimum(direct_cross_lon_dist, wrap_cross_lon_dist)

            near_row0 = (q_lat + k_lat <= rad) & (lon_dist_cross <= rad)
            near_rowN = ((H_val - 1 - q_lat) + (H_val - 1 - k_lat) <= rad) & (
                lon_dist_cross <= rad
            )

            return valid & (in_direct | near_row0 | near_rowN)

        self.block_mask = create_block_mask(
            geocyclic_mask,
            B=1,
            H=1,
            Q_LEN=L_padded,
            KV_LEN=L_padded,
            device=device,
            _compile=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_mask is None:
            self.setup_mask(x.device)

        B, _, H, W = x.shape
        assert H == self.H and W == self.W

        L = H * W
        pad_len = (128 - (L % 128)) % 128
        L_padded = L + pad_len

        # Base projection path.
        base = self.in_proj(x)
        local = self.local_mixer(base)

        coords = self.coords_3d.to(dtype=base.dtype, device=base.device)
        coords = coords.expand(B, -1, -1, -1)

        z = self.norm(base)
        z = torch.cat([z, coords], dim=1)
        z_tokens = self.to_tokens(z)

        if pad_len > 0:
            padding = torch.zeros(
                B,
                pad_len,
                z_tokens.shape[-1],
                device=z_tokens.device,
                dtype=z_tokens.dtype,
            )
            z_tokens = torch.cat([z_tokens, padding], dim=1)

        q, k, v = self.qkv_proj(z_tokens).chunk(3, dim=-1)

        q = q.reshape(B, L_padded, self.num_heads, self.head_dim)
        k = k.reshape(B, L_padded, self.num_heads, self.head_dim)
        v = v.reshape(B, L_padded, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        attn = flex_attention(q, k, v, block_mask=self.block_mask)

        attn = attn.permute(0, 2, 1, 3).contiguous()
        attn = attn.reshape(B, L_padded, self.hidden_dim)

        if pad_len > 0:
            attn = attn[:, :L]

        attn = self.attn_out(attn)
        attn = self.to_grid(attn)

        gate = torch.sigmoid(self.alpha).view(1, -1, 1, 1)

        y = base + local + gate * attn
        y = self.enforce_pole_continuity(y)

        return self.out_proj(y)




class CrossScaleGeoAttention(nn.Module):
    """
    Cross-resolution local attention on a lat/lon grid.

    Q lives on q_mesh.
    K,V live on kv_mesh.

    Use cases:
        restriction: Q = coarse, K,V = fine
        prolongation: Q = fine, K,V = coarse
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        output_dim: int,
        hidden_dim: int,
        q_lat_grid: torch.Tensor,
        q_lon_grid: torch.Tensor,
        kv_lat_grid: torch.Tensor,
        kv_lon_grid: torch.Tensor,
        q_mesh: tuple[int, int],
        kv_mesh: tuple[int, int],
        num_heads: int = 4,
        window_radius: int = 4,
        zero_init_output: bool = True,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.Hq, self.Wq = q_mesh
        self.Hk, self.Wk = kv_mesh
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        self.window_radius = window_radius

        self.q_proj_in = nn.Conv2d(q_dim, hidden_dim, kernel_size=1)
        self.kv_proj_in = nn.Conv2d(kv_dim, hidden_dim, kernel_size=1)

        self.q_norm = nn.GroupNorm(1, hidden_dim, affine=False)
        self.kv_norm = nn.GroupNorm(1, hidden_dim, affine=False)

        self.q_proj = nn.Linear(hidden_dim + 3, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim + 3, 2 * hidden_dim)

        self.attn_out = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

        self.alpha = nn.Parameter(torch.full((hidden_dim,), -4.0))

        self.register_buffer(
            "q_coords_3d",
            self._make_coords(q_lat_grid, q_lon_grid),
            persistent=False,
        )
        self.register_buffer(
            "kv_coords_3d",
            self._make_coords(kv_lat_grid, kv_lon_grid),
            persistent=False,
        )

        self.block_mask = None
        self._init_weights(zero_init_output)

    def _make_coords(self, lat_grid: torch.Tensor, lon_grid: torch.Tensor):
        x = torch.cos(lat_grid) * torch.cos(lon_grid)
        y = torch.cos(lat_grid) * torch.sin(lon_grid)
        z = torch.sin(lat_grid)
        return torch.stack([x, y, z], dim=0).unsqueeze(0)

    def _init_weights(self, zero_init_output: bool):
        nn.init.kaiming_normal_(self.q_proj_in.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.kv_proj_in.weight, mode="fan_in", nonlinearity="relu")

        if self.q_proj_in.bias is not None:
            nn.init.zeros_(self.q_proj_in.bias)
        if self.kv_proj_in.bias is not None:
            nn.init.zeros_(self.kv_proj_in.bias)

        nn.init.xavier_normal_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)

        nn.init.xavier_normal_(self.kv_proj.weight)
        nn.init.zeros_(self.kv_proj.bias)

        nn.init.normal_(self.attn_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.attn_out.bias)

        if zero_init_output:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-3)
        else:
            nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_in", nonlinearity="relu")

        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)

    def _to_q_grid(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        return x.permute(0, 2, 1).reshape(B, C, self.Hq, self.Wq)

    def setup_mask(self, device: torch.device):
        if self.block_mask is not None:
            return

        Hq, Wq = self.Hq, self.Wq
        Hk, Wk = self.Hk, self.Wk
        rad = self.window_radius

        Q_LEN = Hq * Wq
        KV_LEN = Hk * Wk

        q_pad = (128 - (Q_LEN % 128)) % 128
        kv_pad = (128 - (KV_LEN % 128)) % 128

        Q_PAD = Q_LEN + q_pad
        KV_PAD = KV_LEN + kv_pad

        def cross_scale_mask(b, h, query_idx, key_idx):
            valid = (query_idx < Q_LEN) & (key_idx < KV_LEN)

            q_lat = query_idx // Wq
            q_lon = query_idx % Wq

            k_lat = key_idx // Wk
            k_lon = key_idx % Wk

            # Map query location into key/value grid coordinates.
            if Hq > 1:
                q_lat_in_k = (q_lat * (Hk - 1)) // max(Hq - 1, 1)
            else:
                q_lat_in_k = torch.zeros_like(q_lat)

            q_lon_in_k = (q_lon * Wk) // Wq

            lat_dist = abs(k_lat - q_lat_in_k)

            direct_lon_dist = abs(k_lon - q_lon_in_k)
            wrap_lon_dist = Wk - direct_lon_dist
            lon_dist = torch.minimum(direct_lon_dist, wrap_lon_dist)

            in_window = (lat_dist <= rad) & (lon_dist <= rad)

            return valid & in_window

        self.block_mask = create_block_mask(
            cross_scale_mask,
            B=1,
            H=1,
            Q_LEN=Q_PAD,
            KV_LEN=KV_PAD,
            device=device,
            _compile=True,
        )

    def forward(self, q_seed: torch.Tensor, kv_input: torch.Tensor) -> torch.Tensor:
        """
        q_seed:  [B, q_dim, Hq, Wq]
        kv_input:[B, kv_dim, Hk, Wk]
        returns: [B, output_dim, Hq, Wq]
        """
        if self.block_mask is None:
            self.setup_mask(q_seed.device)

        B = q_seed.shape[0]

        Q_LEN = self.Hq * self.Wq
        KV_LEN = self.Hk * self.Wk

        q_pad = (128 - (Q_LEN % 128)) % 128
        kv_pad = (128 - (KV_LEN % 128)) % 128

        Q_PAD = Q_LEN + q_pad
        KV_PAD = KV_LEN + kv_pad

        q_base = self.q_proj_in(q_seed)
        kv_base = self.kv_proj_in(kv_input)

        q = self.q_norm(q_base)
        kv = self.kv_norm(kv_base)

        q_coords = self.q_coords_3d.to(device=q.device, dtype=q.dtype).expand(B, -1, -1, -1)
        kv_coords = self.kv_coords_3d.to(device=kv.device, dtype=kv.dtype).expand(B, -1, -1, -1)

        q = torch.cat([q, q_coords], dim=1)
        kv = torch.cat([kv, kv_coords], dim=1)

        q_tokens = self._to_tokens(q)
        kv_tokens = self._to_tokens(kv)

        if q_pad > 0:
            q_tokens = torch.cat(
                [
                    q_tokens,
                    torch.zeros(
                        B,
                        q_pad,
                        q_tokens.shape[-1],
                        device=q_tokens.device,
                        dtype=q_tokens.dtype,
                    ),
                ],
                dim=1,
            )

        if kv_pad > 0:
            kv_tokens = torch.cat(
                [
                    kv_tokens,
                    torch.zeros(
                        B,
                        kv_pad,
                        kv_tokens.shape[-1],
                        device=kv_tokens.device,
                        dtype=kv_tokens.dtype,
                    ),
                ],
                dim=1,
            )

        q = self.q_proj(q_tokens)
        k, v = self.kv_proj(kv_tokens).chunk(2, dim=-1)

        q = q.reshape(B, Q_PAD, self.num_heads, self.head_dim)
        k = k.reshape(B, KV_PAD, self.num_heads, self.head_dim)
        v = v.reshape(B, KV_PAD, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        attn = flex_attention(q, k, v, block_mask=self.block_mask)

        attn = attn.permute(0, 2, 1, 3).contiguous()
        attn = attn.reshape(B, Q_PAD, self.hidden_dim)

        if q_pad > 0:
            attn = attn[:, :Q_LEN]

        attn = self.attn_out(attn)
        attn = self._to_q_grid(attn)

        gate = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        y = q_base + gate * attn

        return self.out_proj(y)

class AttentionRestriction(nn.Module):
    """
    Learned attention-based downsampling.

    fine x:  [B, C, H, W]
    output:  [B, C, Hc, Wc]
    """

    def __init__(
        self,
        dim: int,
        fine_lat_grid: torch.Tensor,
        fine_lon_grid: torch.Tensor,
        coarse_lat_grid: torch.Tensor,
        coarse_lon_grid: torch.Tensor,
        fine_mesh: tuple[int, int],
        coarse_mesh: tuple[int, int],
        num_heads: int = 4,
        window_radius: int = 6,
        factor: int = 4,
    ):
        super().__init__()

        self.factor = factor

        # Create coarse query seed with learned anti-aliased strided conv.
        blocks = [
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.SiLU(),
        ]

        num_down = int(torch.log2(torch.tensor(float(factor))).item())
        if 2 ** num_down != factor:
            raise ValueError("factor must be a power of 2 for this implementation")

        for _ in range(num_down):
            blocks.extend(
                [
                    GeoCyclicPadding(2),
                    nn.Conv2d(dim, dim, kernel_size=5, stride=2, groups=dim, bias=False),
                    nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                    nn.SiLU(),
                ]
            )

        self.seed = nn.Sequential(*blocks)

        self.attn = CrossScaleGeoAttention(
            q_dim=dim,
            kv_dim=dim,
            output_dim=dim,
            hidden_dim=dim,
            q_lat_grid=coarse_lat_grid,
            q_lon_grid=coarse_lon_grid,
            kv_lat_grid=fine_lat_grid,
            kv_lon_grid=fine_lon_grid,
            q_mesh=coarse_mesh,
            kv_mesh=fine_mesh,
            num_heads=num_heads,
            window_radius=window_radius,
            zero_init_output=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coarse_seed = self.seed(x)
        return self.attn(coarse_seed, x)

class AttentionProlongation(nn.Module):
    """
    Learned attention-based upsampling.

    coarse x: [B, C, Hc, Wc]
    output:   [B, C, H, W]
    """

    def __init__(
        self,
        dim: int,
        fine_lat_grid: torch.Tensor,
        fine_lon_grid: torch.Tensor,
        coarse_lat_grid: torch.Tensor,
        coarse_lon_grid: torch.Tensor,
        fine_mesh: tuple[int, int],
        coarse_mesh: tuple[int, int],
        num_heads: int = 4,
        window_radius: int = 3,
    ):
        super().__init__()

        self.fine_mesh = fine_mesh

        self.seed_refine = nn.Sequential(
            GeoCyclicPadding(2),
            nn.Conv2d(dim, dim, kernel_size=5, groups=dim, bias=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.SiLU(),
        )

        self.attn = CrossScaleGeoAttention(
            q_dim=dim,
            kv_dim=dim,
            output_dim=dim,
            hidden_dim=dim,
            q_lat_grid=fine_lat_grid,
            q_lon_grid=fine_lon_grid,
            kv_lat_grid=coarse_lat_grid,
            kv_lon_grid=coarse_lon_grid,
            q_mesh=fine_mesh,
            kv_mesh=coarse_mesh,
            num_heads=num_heads,
            window_radius=window_radius,
            zero_init_output=True,
        )

    def periodic_upsample(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.fine_mesh

        x_ext = torch.cat([x, x[..., :1]], dim=-1)

        y_ext = torch.nn.functional.interpolate(
            x_ext,
            size=(H, W + 1),
            mode="bilinear",
            align_corners=True,
        )

        return y_ext[..., :-1]

    def forward(self, coarse: torch.Tensor, fine_skip: torch.Tensor | None = None) -> torch.Tensor:
        fine_seed = self.periodic_upsample(coarse)
        fine_seed = self.seed_refine(fine_seed)

        if fine_skip is not None:
            fine_seed = fine_seed + fine_skip

        return self.attn(fine_seed, coarse)


class CoarseAttentionBranch(nn.Module):
    def __init__(
        self,
        dim: int,
        fine_lat_grid: torch.Tensor,
        fine_lon_grid: torch.Tensor,
        coarse_lat_grid: torch.Tensor,
        coarse_lon_grid: torch.Tensor,
        fine_mesh: tuple[int, int],
        coarse_mesh: tuple[int, int],
        num_heads: int = 4,
        restriction_radius: int = 6,
        prolongation_radius: int = 3,
    ):
        super().__init__()

        self.restrict = AttentionRestriction(
            dim=dim,
            fine_lat_grid=fine_lat_grid,
            fine_lon_grid=fine_lon_grid,
            coarse_lat_grid=coarse_lat_grid,
            coarse_lon_grid=coarse_lon_grid,
            fine_mesh=fine_mesh,
            coarse_mesh=coarse_mesh,
            num_heads=num_heads,
            window_radius=restriction_radius,
        )

        self.processor = DRAttention(
            input_dim=dim,
            output_dim=dim,
            hidden_dim=dim,
            lat_grid=coarse_lat_grid,
            lon_grid=coarse_lon_grid,
            mesh_size=coarse_mesh,
            num_heads=num_heads,
            window_radius=8,
            zero_init_output=True,
        )

        self.prolong = AttentionProlongation(
            dim=dim,
            fine_lat_grid=fine_lat_grid,
            fine_lon_grid=fine_lon_grid,
            coarse_lat_grid=coarse_lat_grid,
            coarse_lon_grid=coarse_lon_grid,
            fine_mesh=fine_mesh,
            coarse_mesh=coarse_mesh,
            num_heads=num_heads,
            window_radius=prolongation_radius,
        )

        self.alpha = nn.Parameter(torch.full((dim,), -4.0))


    def setup_masks(self, device: torch.device):
        self.restrict.attn.setup_mask(device)
        self.processor.setup_mask(device)
        self.prolong.attn.setup_mask(device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        coarse = self.restrict(hidden)
        coarse = self.processor(coarse)
        correction = self.prolong(coarse, fine_skip=None)

        gate = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        return hidden + gate * correction