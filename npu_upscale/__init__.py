"""
Neural x2 upscaling on the Hexagon NPU of Snapdragon X-series laptops
(X, X Plus, X Elite, X2 Plus, X2 Elite), with verified placement.
"""

from .chips import Chip, detect_chip, parse_chip
from .runtime import PlacementError, SessionInfo, create_session, npu_available
from .upscaler import Upscaler

__version__ = "0.1.0"
__all__ = ["Chip", "PlacementError", "SessionInfo", "Upscaler", "create_session",
           "detect_chip", "npu_available", "parse_chip", "__version__"]
