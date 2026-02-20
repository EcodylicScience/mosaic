"""
Track converter library for standardizing raw tracker outputs.

This module provides a plugin architecture for converting various tracking
formats to the standardized trex_v1 parquet schema. Converters are
automatically registered on import.

Adding a New Track Converter
-----------------------------
1. Create a new file in this directory (e.g., deeplabcut.py)
2. Use track_converter_template.py as a starting point
3. Implement the converter function with signature:
   (path: Path, params: dict) -> pd.DataFrame
4. Optionally implement a sequence enumerator for multi-sequence files
5. Call register_track_converter() at module level
6. Import the module here to register it

Available Converters
--------------------
After importing, converters are registered in TRACK_CONVERTERS dict
accessible from mosaic.core.dataset module.
"""

# Import all converter modules to trigger registration.
# Each module calls register_track_converter() at module level.
from . import calms21
from . import mabe22
from . import trex

__all__ = [
    "calms21",
    "mabe22",
    "trex",
]
