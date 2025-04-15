from dataclasses import dataclass

from pycla.bin import CLA


@dataclass(frozen=True)
class CUDADevice:
    id: int
    name: str
    max_grid: tuple[int, int, int]
    max_block: tuple[int, int, int]
    max_threads_per_block: int


class Devices:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self):
        # Initialize devices
        self._devices = []
        for i in range(CLA.cuda_device_count):
            dev = CLA.get_device_by_id(i).contents
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

    def __len__(self) -> int:
        return len(self._devices)

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
