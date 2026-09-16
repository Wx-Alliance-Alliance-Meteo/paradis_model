"""Lightning strategy for 2-D PARADIS training: DDP x channel tensor parallel."""

from __future__ import annotations

import logging
from collections.abc import Mapping

import torch
import torch.distributed as dist
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.core.optimizer import LightningOptimizer


def _state_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _state_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(v) for v in value)
    return value


class TensorParallelDDPStrategy(DDPStrategy):
    """Use channel TP inside replicas and DDP across replicas.

    Rank layout is [data_parallel, tensor_parallel]. TP groups are contiguous,
    which keeps TP inside a node when ``tensor_parallel_size`` divides the GPU
    count per node. DDP groups contain matching TP ranks from every replica.
    """

    def __init__(self, *, tensor_parallel_size: int, **kwargs):
        if tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        super().__init__(**kwargs)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.tp_group = None
        self.dp_group = None
        self.tp_rank = 0
        self.dp_rank = 0
        self.data_parallel_size = 1
        self.tp_src_global_rank = 0

    def setup_environment(self) -> None:
        super().setup_environment()

        world_size = self.world_size
        if world_size % self.tensor_parallel_size != 0:
            raise ValueError(
                f"world_size={world_size} must be divisible by "
                f"tensor_parallel_size={self.tensor_parallel_size}"
            )
        if self.num_processes % self.tensor_parallel_size != 0:
            raise ValueError(
                f"Per-node devices={self.num_processes} must be divisible by "
                f"tensor_parallel_size={self.tensor_parallel_size}. "
                "TP is intended to stay inside each node."
            )

        tp = self.tensor_parallel_size
        dp = world_size // tp
        rank = self.global_rank
        self.data_parallel_size = dp
        self.tp_rank = rank % tp
        self.dp_rank = rank // tp
        self.tp_src_global_rank = self.dp_rank * tp

        # All ranks must create all groups in identical order.
        for d in range(dp):
            ranks = list(range(d * tp, (d + 1) * tp))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                self.tp_group = group

        for t in range(tp):
            ranks = [d * tp + t for d in range(dp)]
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                self.dp_group = group

        if self.tp_group is None or self.dp_group is None:
            raise RuntimeError("Failed to create TP/DDP process groups")

        # Lightning's DDP wrapper now synchronizes only matching model shards.
        self._ddp_kwargs["process_group"] = self.dp_group

    @property
    def distributed_sampler_kwargs(self) -> dict[str, int]:
        # All TP ranks in a replica must receive the same weather sample.
        return {"num_replicas": self.data_parallel_size, "rank": self.dp_rank}

    def configure_ddp(self) -> None:
        assert self.lightning_module is not None
        if self.tensor_parallel_size > 1:
            self.lightning_module.enable_tensor_parallel(
                group=self.tp_group,
                rank=self.tp_rank,
                world_size=self.tensor_parallel_size,
                src_global_rank=self.tp_src_global_rank,
            )
            compile_mode = self.lightning_module.cfg.compute.get("compile", False)
            print("Compile mode:", compile_mode)
            if compile_mode is True:
                # TP conversion must happen before compilation because conversion
                # changes parameter shapes.  Compile the inner PARADIS model before
                # Lightning wraps the LightningModule in DDP.  TP collectives are
                # torch.compiler.disable graph boundaries, so fullgraph must be False.
                self.lightning_module.model.compile(
                    mode="default",
                    fullgraph=False,
                    dynamic=False,
                    backend="inductor",
                )
                if self.global_rank == 0:
                    logging.info(
                        "torch.compile enabled for TP; NCCL TP collectives execute "
                        "outside compiled regions."
                    )
            elif compile_mode == "modules":
                self.lightning_module.model._compile()
                if self.global_rank == 0:
                    logging.info(
                        "Per-module torch.compile enabled for TP; NCCL TP collectives "
                        "execute outside compiled regions."
                    )
        super().configure_ddp()

    def lightning_module_state_dict(self) -> dict[str, object]:
        assert self.lightning_module is not None
        if self.tensor_parallel_size == 1:
            return super().lightning_module_state_dict()
        # All ranks enter this method when Lightning creates a checkpoint, so TP
        # collectives are safe here. The result is a standard full-model state
        # dict and remains loadable by the serial/DDP model.
        return self.lightning_module.tensor_parallel_full_state_dict()

    def load_model_state_dict(
        self, checkpoint: Mapping[str, object], strict: bool = True
    ) -> None:
        assert self.lightning_module is not None
        if self.tensor_parallel_size == 1:
            return super().load_model_state_dict(checkpoint, strict=strict)
        localized = self.lightning_module.tensor_parallel_localize_state_dict(
            checkpoint["state_dict"]
        )
        self.lightning_module.load_state_dict(localized, strict=strict)

    def optimizer_state(self, optimizer) -> dict:
        local_state = super().optimizer_state(optimizer)
        if self.tensor_parallel_size == 1:
            return local_state

        # Optimizer moments are local to each TP parameter shard. Preserve one
        # optimizer state per TP rank inside the ordinary Lightning checkpoint.
        local_state = _state_to_cpu(local_state)
        shards = [None for _ in range(self.tensor_parallel_size)]
        dist.all_gather_object(shards, local_state, group=self.tp_group)
        return {
            "__paradis_tp_optimizer__": True,
            "tensor_parallel_size": self.tensor_parallel_size,
            "shards": shards,
        }

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, object]) -> None:
        optimizer_states = checkpoint["optimizer_states"]
        for optimizer, state in zip(self.optimizers, optimizer_states):
            if isinstance(optimizer, LightningOptimizer):
                optimizer = optimizer._optimizer
            if isinstance(state, dict) and state.get("__paradis_tp_optimizer__"):
                saved_tp = int(state["tensor_parallel_size"])
                if saved_tp != self.tensor_parallel_size:
                    raise ValueError(
                        "Restarting optimizer state with a different TP size is not "
                        f"supported: checkpoint TP={saved_tp}, current TP={self.tensor_parallel_size}."
                    )
                state = state["shards"][self.tp_rank]
            optimizer.load_state_dict(state)
            # The base strategy normally moves optimizer state after loading.
            from lightning.fabric.utilities.optimizer import _optimizer_to_device

            _optimizer_to_device(optimizer, self.root_device)
