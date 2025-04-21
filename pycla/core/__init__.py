from .cuda_device import Devices
from .matrix import Matrix, ShareDestionationMatrix
from .vector import ShareDestionationVector, Vector

# Initialize a global Devices instance
DEVICES: Devices = Devices()
