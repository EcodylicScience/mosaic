"""
Track converter library for standardizing raw tracker outputs.

This module provides a plugin architecture for converting various tracking
formats to the standardized ``mosaic_v1`` parquet schema (TREx emits the
``trex_v2`` superset). Converters are
automatically registered on import.

Adding a New Track Converter
-----------------------------
1. Create a new file in this directory (e.g., deeplabcut.py), starting from
   track_converter_template.py.
2. Declare a Params model, subclass TrackConverter, implement
   convert(path, params, hints) -> pd.DataFrame, and name the output schema in
   the class variable ``output_schema``.
3. Decorate the class with @register_track_converter.
4. For one file holding several sequences, set ``enumerable = True`` and
   implement enumerate_sequences; the scan source must also declare
   ``multi_sequences_per_file=True`` or the expansion never runs.
5. Import the module here and add its name to ``__all__``.

See docs/adding-a-converter.md for the full contract.

Available Converters
--------------------
After importing, converters are registered in TRACK_CONVERTERS dict
accessible from mosaic.core.dataset module.
"""

# Import all converter modules to trigger registration.
# Each module calls register_track_converter() at module level.
from . import calms21
from . import deeplabcut
from . import sleap
from . import trex
from . import ultralytics_tracks

__all__ = [
    "calms21",
    "deeplabcut",
    "sleap",
    "trex",
    "ultralytics_tracks",
]
