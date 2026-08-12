"""Correcting what a joined conversion got wrong about time.

TRex takes its ``source`` as a ``PathArray`` and converts a session's clips into
one ``.pv`` whose frame index is continuous -- which is the whole point of
joining. What it does *not* do is notice that the clips were recorded at
different frame rates: ``VideoSource`` reads ``_framerate`` from
``_files_in_seq.front()`` and never compares it with the others. One real
session measures 30 fps, then 29.95, then 31 across seventeen clips, so a
straight conversion labels fifteen of them with a rate that is about 3% wrong.

Nor is there a per-frame timestamp to fall back on. TRex's timestamp-loading
branch for a video file is compiled out (``if(/* DISABLES CODE */ (false) &&
npz.exists())``), so ``has_timestamps()`` is false for every ``.mp4`` and its
``time`` array can only ever be an index divided by that one rate.

Two consequences reach ``tracks/`` unless something intervenes, because the
converter *prefers* what TRex exported:

* ``time`` and ``frame_rate`` are wrong for every clip but the first, and the
  error in ``time`` accumulates across the session.
* every per-second quantity TRex derived -- ``SPEED``, ``VX``, ``ANGULAR_V`` and
  their kin -- was computed against the wrong denominator.

So mosaic recomputes the first pair from the measured per-clip rates, and
**drops** the second group rather than rescaling it. Rescaling by
``fps_i / fps_0`` is exact only for a plain first-difference estimator, and
TRex's is not one this code can state -- it may smooth across a window, in which
case the ratio is wrong near every boundary anyway. ``trex_v2`` *allows* those
columns without requiring them, so the table stays schema-valid without them,
and ``speed-angvel`` derives them from ``X``/``Y`` and the corrected ``time``
with its method recorded in a run identifier. That is the standing rule -- a
tracker reports, a feature derives -- applied to the case where the tracker did
not in fact report.

A uniform-rate session keeps all of it: nothing was wrong with it.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

import numpy as np

from mosaic.core.media.timeline import ConcatenatedTimeline
from mosaic.core.track_library.helpers import column_array, column_names
from mosaic.core.track_library.trex import base_field

__all__ = ["RATE_DEPENDENT_BASES", "retime_joined_frame"]

RATE_DEPENDENT_BASES: Final[frozenset[str]] = frozenset(
    {"VX", "VY", "AX", "AY", "SPEED", "ANGULAR_V", "ANGULAR_A"}
)
"""Base fields TRex computed per *second*, and so against a single frame rate.

Matched on the base name so every ``#`` variant goes with it -- ``SPEED``,
``SPEED#wcentroid`` and ``SPEED#pcentroid`` are one quantity under three
estimators and are equally wrong.

Deliberately its own list rather than a reuse of the converter's
``DERIVED_COLUMNS``. That set also holds ``ANGLE`` and the ``#wcentroid``
positions, which are an angle and a coordinate: neither depends on a rate, and
dropping ``X#wcentroid`` would take the body centre with it, since that is where
``X``/``Y`` come from.
"""


def retime_joined_frame(
    df: pd.DataFrame, timeline: ConcatenatedTimeline
) -> pd.DataFrame:
    """Put *df* on *timeline*'s time axis, dropping what a single rate spoiled.

    A no-op for a single-segment timeline: there was one clip, one rate, and
    nothing for TRex to have got wrong.

    Args:
        df: A merged TRex export, carrying the joined ``.pv``'s global ``frame``.
        timeline: The concatenation the conversion was built from.

    Returns:
        A new frame. ``frame`` is untouched -- TRex's global index is already
        right, because ``VideoSource`` sums the clip lengths.
    """
    present = column_names(df)
    if len(timeline.segments) < 2 or "frame" not in present:
        return df

    out = df.copy()
    frames = column_array(df, "frame").astype(np.int64, copy=False)
    out["time"] = timeline.times(frames)
    if "frame_rate" in present:
        # Per row rather than per file: this names the rate in force at *that*
        # frame, which is the only reading of the column that stays true.
        out["frame_rate"] = timeline.rates(frames)

    # Minted by TRex from the index and the one rate, never measured -- see the
    # module docstring. Dropped rather than recomputed: a microsecond stamp
    # mosaic did not measure has no business being written back.
    doomed = [name for name in present if name == "timestamp"]
    if not timeline.uniform_rate:
        doomed += [name for name in present if base_field(name) in RATE_DEPENDENT_BASES]
    return out.drop(columns=doomed) if doomed else out
