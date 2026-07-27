"""Template for creating new track format converters.

To create a new converter:

1. Copy this file to a new name (e.g. ``deeplabcut.py``).
2. Declare a ``Params`` model for whatever genuinely determines the output.
3. Subclass :class:`~mosaic.core.track_converter.TrackConverter`, implement
   ``convert``, and decorate the class with ``@register_track_converter``.
4. Import the module in ``track_library/__init__.py`` so it registers.
5. Test with ``dataset.convert_all_tracks()``.

The returned DataFrame should have at minimum::

    frame, time, id, X, Y, group, sequence

and ideally also::

    VX, VY, SPEED, ANGLE, poseX0..N, poseY0..N

**What goes where.** ``params`` is the *recipe* -- it, plus ``src_format`` and
``version``, is what a tracks variant is identified by, so every field must be
something that changes the output. ``hints`` is *which entry* is being produced,
and is never hashed: putting a sequence name in params would mint one variant
per sequence where there is one recipe. Knobs that change only diagnostics or
validation are fields marked ``Annotated[T, HASH_EXCLUDE]``.

Bump ``version`` by hand when your converter's output semantics change. It is a
visible segment of the variant identity, so a bump is how you say "the same
input now means something different" without touching any existing directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd

from mosaic.core.pipeline.types import HASH_EXCLUDE
from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
)


class MyFormatParams(TrackConvertParams):
    """Everything that determines this converter's output.

    Attributes:
        fps: Frame rate used to derive ``time``. Determines the output, so it
            is part of the recipe.
        debug: Print what was found. Diagnostics only, so it is excluded from
            identity -- flipping it must not mint a second variant.
    """

    fps: float = 30.0
    debug: Annotated[bool, HASH_EXCLUDE] = False


# Decorate with ``@register_track_converter`` (imported from
# ``mosaic.core.track_converter``) once implemented. It is left off here so this
# template does not register a converter that raises.
class MyFormatConverter(TrackConverter[MyFormatParams]):
    """Convert a source file to a ``trex_v1``-compatible table."""

    src_format = "my_format"
    version = "0.1"
    # Set True and implement enumerate_sequences if one file holds several.
    enumerable = False
    Params = MyFormatParams

    def convert(
        self, path: Path, params: MyFormatParams, hints: EntryHints
    ) -> pd.DataFrame:
        # 1. Load data from path
        # 2. Extract per-animal trajectories
        # 3. Compute centroid, velocity, heading angle
        # 4. Build a DataFrame with the standard columns, taking group and
        #    sequence from `hints` (falling back to the file stem)
        df = self._build_frame(path, params, hints)
        ensure_track_schema(
            df, "trex_v1", strict=params.strict_schema, source=str(path)
        )
        return df

    def _build_frame(
        self, path: Path, params: MyFormatParams, hints: EntryHints
    ) -> pd.DataFrame:
        raise NotImplementedError("Implement for your format")

    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        """The ``(group, sequence)`` pairs *path* holds. Only if ``enumerable``."""
        raise NotImplementedError
