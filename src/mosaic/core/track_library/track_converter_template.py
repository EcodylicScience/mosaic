"""Template for creating new track format converters.

To create a new converter:

1. Copy this file to a new name (e.g. ``deeplabcut.py``).
2. Declare a ``Params`` model for whatever genuinely determines the output.
3. Subclass :class:`~mosaic.core.track_converter.TrackConverter`, implement
   ``convert``, and decorate the class with ``@register_track_converter``.
4. Import the module in ``track_library/__init__.py`` so it registers.
5. Test with ``dataset.convert_all_tracks()``.

The returned DataFrame must have at minimum::

    frame, time, id, X, Y, group, sequence

plus keypoints as ``poseX0..N`` / ``poseY0..N``, and ``poseP<k>`` where the
source reported a confidence. All spatial columns are in the space of the schema
the converter declares, and ``X`` / ``Y`` is the body center. Do **not** add
``VX``, ``VY``, ``SPEED``, ``ANGLE`` or the rest of the derived set: ``mosaic_v1``
forbids them, so writing one raises ``ForbiddenTrackColumnError``. A tracker that
genuinely measures one declares a schema that ``extends`` ``mosaic_v1`` and
``allows`` that column.

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

from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)
from mosaic.core.track_converter import (
    EntryHints,
    TimedTrackConvertParams,
    TrackConverter,
)

_DEBUG_DESCRIPTION = "Print what was found."


class MyFormatParams(TimedTrackConvertParams):
    """Everything that determines this converter's output.

    A format that records no frame rate subclasses ``TimedTrackConvertParams``,
    which declares ``fps`` and ``fallback_fps``, and times its table with
    ``conversion_rate(params.fps, params.fallback_fps, hints, str(path))``. A
    format that records its rate subclasses ``TrackConvertParams`` and reads it.
    """

    # Diagnostics only, so it is excluded from identity -- flipping it must
    # not mint a second variant.
    debug: Annotated[bool, HASH_EXCLUDE, Declared(_DEBUG_DESCRIPTION)] = False

    # Optional. A format that can write a row for every frame and individual its
    # file records declares this with the default that suits it, and drops it from
    # the variant while it holds that default, so existing conversions keep their
    # variant. ``KeepEmptyFrames`` comes from ``mosaic.core.track_converter``:
    #
    #     keep_empty_frames: KeepEmptyFrames = False
    #
    #     @override
    #     def identity_dump(self) -> dict[str, object]:
    #         return self.drop_at_default(
    #             super().identity_dump(), ("keep_empty_frames",)
    #         )


# Decorate with ``@register_track_converter`` (imported from
# ``mosaic.core.track_converter``) once implemented. It is left off here so this
# template does not register a converter that raises.
class MyFormatConverter(TrackConverter[MyFormatParams]):
    """Convert a source file to a ``mosaic_v1``-compatible table."""

    src_format = "my_format"
    version = "0.1"
    # Set True and implement enumerate_sequences if one file holds several.
    enumerable = False
    # Set True and override sequence_from_stem if several files make one
    # sequence -- one file per individual, named <sequence>_<individual>.
    merges_per_sequence = False
    # Which schema the tables answer to. Declared here and nowhere else: the
    # caller validates against this, so a converter cannot claim one schema
    # while its rows are recorded under another.
    output_schema = "mosaic_v1"
    # True when `id` follows one animal over the video, False when ids restart or
    # number detections each frame, None when the file cannot say.
    carries_tracklets = None
    Params = MyFormatParams

    def convert(
        self, path: Path, params: MyFormatParams, hints: EntryHints
    ) -> pd.DataFrame:
        # 1. Load data from path
        # 2. Extract per-animal trajectories
        # 3. Build a DataFrame with the standard columns, taking group and
        #    sequence from `hints` (falling back to the file stem)
        #
        # Do not validate here -- the caller validates every table it writes
        # against ``output_schema`` above, and a second check under a
        # separately chosen name is how the two drifted apart.
        return self._build_frame(path, params, hints)

    def _build_frame(
        self, path: Path, params: MyFormatParams, hints: EntryHints
    ) -> pd.DataFrame:
        raise NotImplementedError("Implement for your format")

    # Optional. Override when the file records its frame count: return it from the
    # file's shape, and raise OSError or ValueError on a file that cannot be read.
    # Every conversion records the count in the tracks index. The base returns
    # None, which records none.
    #
    # def frames_read(self, path: Path) -> int | None:
    #     raise NotImplementedError

    # Optional. Override when `_build_frame` reads another file beside the source:
    # return the ones that exist now, sorted, in the source's directory, and never
    # the source. Read no file's content, and take no params, because a scan runs
    # before any conversion has params. A scan records each file's checksum, and a
    # change to one marks the conversion stale. The base returns (), which lists none.
    #
    # def companion_files(self, path: Path) -> tuple[Path, ...]:
    #     sibling = path.with_name(path.stem + ".meta.json")
    #     return (sibling,) if sibling.is_file() else ()

    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        """The ``(group, sequence)`` pairs *path* holds. Only if ``enumerable``."""
        raise NotImplementedError
