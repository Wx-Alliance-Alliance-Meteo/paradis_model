"""AMSE (Adjusted Mean Squared Error) loss function implementation.

Reference: "Fixing the Double Penalty in Data-Driven Weather Forecasting
Through a Modified Spherical Harmonic Loss Function" (Subich et al., 2025)
"""

import torch
import torch_harmonics as harmonics
import logging
from torch.cuda.amp import autocast


class AMSELoss(torch.nn.Module):
    """Adjusted Mean Squared Error loss using spherical harmonic decomposition.

    Separates amplitude errors from decorrelation errors to avoid smoothing bias.
    Handles complex coefficients from SHT properly.
    """

    def __init__(
        self,
        nlat,
        nlon,
        grid="equiangular",
        norm="backward",
    ):
        """Initialize AMSE loss with spherical harmonic transform.

        Args:
            nlat: Number of latitude points
            nlon: Number of longitude points
            grid: Grid type (see Torch Harmonic documentation)
            norm: Normalization for SHT ('backward', 'forward', or 'ortho'; see Torch Harmonic documentation)
        """
        super().__init__()

        # Initialize spherical harmonic transform
        self.sht = harmonics.RealSHT(nlat=nlat, nlon=nlon, grid=grid, norm=norm)

        # Store dimensions and numerical stability parameters
        self.nlat = nlat
        self.nlon = nlon

    def _compute_psd(self, coefficients):
        """Compute power spectral density from spherical harmonic coefficients.

        Args:
            coefficients: SH coefficients [batch, channels, nlat+1, nlon//2+1] (complex)

        Returns:
            psd: Power spectral density [batch, channels, max_k] (real)
        """
        batch_size, num_channels, nlat_spec, nlon_spec = coefficients.shape

        eps = 1e-7

        # Maximum wavenumber
        max_k = nlat_spec - 1

        # Initialize PSD tensor (real valued)
        psd = torch.zeros(
            batch_size,
            num_channels,
            max_k,
            device=coefficients.device,
            dtype=coefficients.real.dtype,
        )

        # Compute power for each wavenumber k
        # For real SHT: coefficients are complex (k, m) where k goes from 0 to nlat-1
        for k in range(max_k):
            # Sum over all m for this k (m ranges from 0 to k)
            # Power is |coefficient|^2 = real^2 + imag^2
            power_k = torch.sum(torch.abs(coefficients[:, :, k, : k + 1]) ** 2, dim=-1)
            # Account for negative wavenumbers (multiply by 2, except m=0)
            if k > 0:
                power_k = (
                    2 * power_k - torch.abs(coefficients[:, :, k, 0]) ** 2
                )  # Subtract m=0 to avoid double counting

            psd[:, :, k] = power_k + eps

        return psd

    def _compute_coherence(self, pred_coeffs, target_coeffs, pred_psd, target_psd):
        """Compute spectral coherence between prediction and target.

        Coherence_k = |pred·target| / sqrt(PSD_pred_k * PSD_target_k)

        For complex coefficients, we use the magnitude of the inner product.

        Args:
            pred_coeffs: Prediction SH coefficients [batch, channels, nlat+1, nlon//2+1] (complex)
            target_coeffs: Target SH coefficients [batch, channels, nlat+1, nlon//2+1] (complex)
            pred_psd: Prediction PSD [batch, channels, max_k]
            target_psd: Target PSD [batch, channels, max_k]

        Returns:
            coherence: Spectral coherence [batch, channels, max_k] (real, in [0, 1])
        """
        batch_size, num_channels, nlat_spec, nlon_spec = pred_coeffs.shape
        max_k = nlat_spec - 1

        eps = 1e-7

        # Initialize coherence (real valued)
        coherence = torch.zeros(
            batch_size,
            num_channels,
            max_k,
            device=pred_coeffs.device,
            dtype=pred_coeffs.real.dtype,
        )

        # Compute cross-spectrum for each wavenumber
        for k in range(max_k):
            # Cross-spectrum: sum of element-wise products of complex coefficients
            # conj(pred) * target gives us the cross-spectrum
            cross_spec_k = torch.sum(
                torch.conj(pred_coeffs[:, :, k, : k + 1])
                * target_coeffs[:, :, k, : k + 1],
                dim=-1,
            )
            # Account for negative wavenumbers
            if k > 0:
                cross_spec_k = 2 * cross_spec_k - (
                    torch.conj(pred_coeffs[:, :, k, 0]) * target_coeffs[:, :, k, 0]
                )

            # Coherence = |cross_spec| / sqrt(PSD_pred * PSD_target)
            # Use magnitude of complex cross-spectrum
            cross_spec_magnitude = torch.abs(cross_spec_k)
            denom = torch.sqrt(pred_psd[:, :, k] * target_psd[:, :, k] + eps)
            coh_k = cross_spec_magnitude / (denom + eps)

            # Clamp coherence to valid range [0, 1] (coherence is always non-negative)
            coh_k = torch.clamp(coh_k, 0.0, 1.0)
            coherence[:, :, k] = coh_k

        return coherence

    def forward(self, prediction, target, weights=None):
        """Compute AMSE loss between prediction and target.

        Args:
            prediction: Model prediction [batch, channels, lat, lon]
            target: Ground truth [batch, channels, lat, lon]
            weights: Optional feature weights [channels]

        Returns:
            loss: Scalar loss value
        """
        batch_size = prediction.shape[0]
        num_channels = prediction.shape[1]

        # Transform to spherical harmonic space (per channel)
        # Output: [batch, channels, nlat+1, nlon//2+1] (complex)
        with torch.amp.autocast('cuda', enabled=False):

            prediction_f32 = prediction.float()
            target_f32 = target.float()

            pred_coeffs = self.sht(prediction_f32)
            target_coeffs = self.sht(target_f32)

        # Compute power spectral density for each wavenumber
        pred_psd = self._compute_psd(pred_coeffs)  # [batch, channels, max_k]
        target_psd = self._compute_psd(target_coeffs)  # [batch, channels, max_k]

        # Compute spectral coherence
        coherence = self._compute_coherence(
            pred_coeffs, target_coeffs, pred_psd, target_psd
        )

        # AMSE formula: amplitude term + decorrelation term
        amplitude_term = (torch.sqrt(pred_psd) - torch.sqrt(target_psd)) ** 2

        max_psd = torch.max(pred_psd, target_psd)
        decorrelation_term = 2.0 * max_psd * (1.0 - coherence)

        # Combine terms
        amse_per_scale = amplitude_term + decorrelation_term  # [batch, channels, max_k]

        # Average over scales
        amse_per_channel = torch.mean(amse_per_scale, dim=-1)  # [batch, channels]

        # Apply feature weights if provided
        if weights is not None:
            weights = weights.to(amse_per_channel.device)
            amse_per_channel = amse_per_channel * weights.unsqueeze(0)

        # Average over batch and channels
        loss = torch.mean(amse_per_channel)

        # Safety check for NaN
        if torch.isnan(loss):
            logging.warning(
                "NaN detected in AMSE loss computation. "
                f"amplitude_term: {torch.isnan(amplitude_term).any()}, "
                f"decorrelation_term: {torch.isnan(decorrelation_term).any()}, "
                f"coherence: {torch.isnan(coherence).any()}, "
                f"pred_psd: {torch.isnan(pred_psd).any()}, "
                f"target_psd: {torch.isnan(target_psd).any()}"
            )
            # Return a fallback large loss instead of NaN to prevent training crash
            loss = torch.tensor(1e6, dtype=loss.dtype, device=loss.device)

        return loss
