"""
Template for creating new track format converters.

To create a new converter:
1. Copy this file to a new name (e.g., deeplabcut.py)
2. Implement the converter function and optional sequence enumerator
3. Register using register_track_converter() at module level
4. Import in track_library/__init__.py to auto-register
5. Test with dataset.convert_all_tracks()

Converter signature:
    (path: Path, params: dict) -> pd.DataFrame

The returned DataFrame should have at minimum:
    frame, time, id, X, Y, group, sequence

And ideally also:
    VX, VY, SPEED, ANGLE, poseX0..N, poseY0..N

Standard params keys (passed from Dataset.convert_one_track):
    - group: str — group hint from the raw tracks index
    - sequence: str — sequence hint from the raw tracks index
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from mosaic.core.dataset import register_track_converter, register_track_seq_enumerator
from mosaic.core.track_library.helpers import angle_from_two_points, angle_from_pca
from mosaic.core.schema import ensure_track_schema


def _my_format_converter(path: Path, params: dict) -> pd.DataFrame:
    """Convert a source file to a trex_v1-compatible DataFrame."""
    # 1. Load data from path
    # 2. Extract per-animal trajectories
    # 3. Compute centroid, velocity, heading angle
    # 4. Build DataFrame with standard columns
    # 5. Validate against schema
    raise NotImplementedError("Implement for your format")


# Uncomment to register:
# register_track_converter("my_format", _my_format_converter)
