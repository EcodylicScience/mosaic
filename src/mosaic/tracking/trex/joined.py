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

**A joined conversion gets two things wrong, and this module owns only the
first.** The second is that TRex's ``.pv`` is *shorter* than the media: its
``FFmpegVideoCapture`` under-counts every file it opens and then reads only as
many frames as it counted, so each clip loses its tail and the loss accumulates
across the session. Measured on a six-clip fixture carrying its own frame
numbers -- 1,800 media frames converting to 1,788, the offset stepping by two at
every boundary, constant within each clip -- and reproduced by invoking ``trex``
directly, with no mosaic in the process. The 17-clip session that prompted this
lost 70.

Nothing here can correct that: mosaic holds no map from the ``.pv`` index to the
media index, and TRex records none (``pvinfo`` prints ``Video conversion offsets:
N/A``). What mosaic does instead is *measure* it -- the bridge records the media
axis length beside the table's own extent, and
:func:`~mosaic.core.pipeline.tracks_index.frame_axis_mismatches` compares them --
and, going forward, avoid it, by handing the tool one concatenated file so there
are no clip boundaries to lose frames at. A single file converts with no drift at
all: every ``.pv`` index equals its media index, and only the tail is lost, which
costs no registration.
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
        A new frame. ``frame`` is untouched.

        **Not because it is right.** It is left on the axis TRex numbered
        because mosaic holds no map from that axis to the media's, and inventing
        one would fabricate a correspondence -- the standing rule that a tracker
        reports and a feature derives, applied to a column no one can derive.
        TRex numbers continuously across the frames it *kept*, which is fewer
        than the media holds: see the module docstring.

        ``time`` inherits that. :meth:`ConcatenatedTimeline.times` places each
        index as though the two axes agreed, and
        :meth:`ConcatenatedTimeline.segment_for_frame` clamps rather than
        raising, so a shortfall is silent here and shows up only as a late frame
        attributed to the wrong clip. The error is the accumulated loss divided
        by the local rate -- seconds at worst, against the whole-session error of
        3% that this function exists to remove, so it is still worth computing.
        The loss itself is reported by the bridge, not corrected here.
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
