from dataclasses import dataclass

@dataclass(frozen=True)
class CUDADevice:
    id: int
    name: str
    max_grid: tuple[int, int, int]
    max_block: tuple[int, int, int]
    max_threads_per_block: int


class Devices:
    def __init__(self):
        pass


DEVICES: Devices = Devices()
