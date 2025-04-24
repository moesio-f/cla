from ctypes import POINTER
from dataclasses import dataclass

from pycla.bin.cla import CLA, _CUDADevice, _CUDAKernelLaunchParameters


@dataclass(frozen=True)
class CUDADevice:
    id: int
    name: str
    max_grid: tuple[int, int, int]
    max_block: tuple[int, int, int]
    max_threads_per_block: int

    def short_str(self) -> str:
        return f'CUDADevice(id={self.id}, name="{self.name}")'


@dataclass(frozen=True)
class KernelLaunchParameters:
    n_threads: tuple[int, int, int]
    n_blocks: tuple[int, int, int]


class Devices:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        # Initialize devices
        self._devices = []
        self._pointers = []
        for i in range(CLA.cuda_device_count):
            self._pointers.append(CLA.get_device_by_id(i))
            dev = self._pointers[i].contents
            self._devices.append(
                CUDADevice(
                    id=dev.id,
                    name=dev.name.decode("utf-8"),
                    max_grid=(
                        dev.max_grid_size_x,
                        dev.max_grid_size_y,
                        dev.max_grid_size_z,
                    ),
                    max_block=(
                        dev.max_block_size_x,
                        dev.max_block_size_y,
                        dev.max_block_size_z,
                    ),
                    max_threads_per_block=dev.max_threads_per_block,
                )
            )

    @property
    def count(self) -> int:
        return len(self._devices)

    @property
    def has_cuda(self) -> bool:
        return self.count > 0

    def synchronize(self):
        CLA.synchronize_devices()

    def get(self, key: int | str) -> CUDADevice:
        return self[key]

    def get_launch_parameters(
        self, device: int | str | CUDADevice
    ) -> KernelLaunchParameters:
        params = self._get_pointer(device).contents.params
        return KernelLaunchParameters(
            n_threads=(params.n_threads_x, params.n_threads_y, params.n_threads_z),
            n_blocks=(params.n_blocks_x, params.n_blocks_y, params.n_blocks_z),
        )

    def set_launch_parameters(
        self, device: int | str | CUDADevice, params: KernelLaunchParameters
    ):
        dev = self._get_pointer(device).contents
        dev.params = _CUDAKernelLaunchParameters(*params.n_threads, *params.n_blocks)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, key: int | str) -> CUDADevice:
        is_int = isinstance(key, int)
        if not (is_int or isinstance(key, str)):
            raise TypeError("Key must be either int or string.")

        if is_int and (key < 0 or key > len(self)):
            raise IndexError("Integer key is out of bounds.")

        if is_int:
            return self._devices[key]

        # Try to find device with respective name
        device = next((d for d in self._devices if d.name == key), None)
        if device is None:
            raise KeyError("No device with this name found.")
        return device

    def __str__(self) -> str:
        return f"[{','.join(self._devices)}]"

    def __repr__(self) -> str:
        return str(self)

    def _get_pointer(self, device: int | str | CUDADevice) -> POINTER(_CUDADevice):
        if isinstance(device, int) or isinstance(device, str):
            device = self[device]

        return self._pointers[device.id]
