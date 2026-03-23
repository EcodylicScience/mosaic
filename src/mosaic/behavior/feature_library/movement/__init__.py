"""Movement library integration for mosaic.

Provides bidirectional conversion between mosaic DataFrames and movement
xarray Datasets, plus mosaic features that wrap movement's smoothing,
filtering, and interpolation functions.
"""

from .convert import from_movement_dataset, to_movement_dataset
from .filter_interp import MovementFilterInterpolate
from .smooth import MovementSmooth

__all__ = [
    "to_movement_dataset",
    "from_movement_dataset",
    "MovementSmooth",
    "MovementFilterInterpolate",
]
