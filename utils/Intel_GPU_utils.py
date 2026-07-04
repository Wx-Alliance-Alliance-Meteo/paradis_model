# classes for Intel GPUs
import torch
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.strategies import SingleDeviceStrategy
from typing import Any, List, Optional

# For compatibility with Intel ARC GPUs
class IntelXPUAccelerator(Accelerator):

    # Lightning uses this name property internally
    @property
    def name(self) -> str:
        return "intel_xpu"

    # create list of device indices
    @staticmethod
    def parse_devices(devices: Any) -> Optional[List[int]]:
        if isinstance(devices, int):
            return list(range(devices))  # Turns 1 into [0]
        if isinstance(devices, list):
            return devices
        return None

    # Expects a structured iterable list from parse_devices
    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    # Tells Lightning how to register the device to the runtime environment
    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise ValueError(f"Invalid device passed to XPU Accelerator: {device}")
        torch.xpu.set_device(device)

    # Tells Lightning how to clean up memory assets after execution
    def teardown(self) -> None:
        # Clear out the active XPU cache layer on exit
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    def get_device_stats(self, device: torch.device) -> dict:
        return {}

# For compatibility with Intel ARC GPUs
class SingleXPUStrategy(SingleDeviceStrategy):
    def __init__(self, device_index: int = 0):
        super().__init__(
            accelerator=IntelXPUAccelerator(),
            device=torch.device("xpu", device_index)
        )
