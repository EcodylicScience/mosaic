"""Tracktor output -> a ``mosaic_v1`` table, one arena recording per file.

Reads the files of the DGRP looming release (Sato, Okuyama and Takahashi 2025),
which a Tracktor run writes per arena of a plate: a raw ``<stem>_tracked.csv``
(``frame, pos_x, pos_y, id, center_x, center_y``) and the corrected
``<stem>_tracked_fixed_final.tsv`` (``frame, id, pos_x, pos_y, prefix, place,
strain, sex, age, n_inds, N, timestamp, seconds_diff, seconds_total, stimuli``
and three smoothed columns). One file is one sequence. Two formats, because a
registered converter reads one file format:

- ``tracktor_corrected_tsv`` reads the corrected table, whose ids ``Fly1`` to
  ``FlyN`` follow the animals (``carries_tracklets`` is true).
- ``tracktor_raw_csv`` reads the raw table, whose ids swap between animals
  (``carries_tracklets`` is false), and takes everything that is not a position
  from the corrected table beside it.

**Time is the file's own, never ``frame / fps``.** The recording drops frames, so
its effective rate is 50 to 57 fps against a median step of 60, and one session
runs near 20 fps. ``time`` is ``seconds_total``, the seconds since the recording
began, and a frame has one value of it whatever its animals. The converters take
no frame rate. They are not ``TimedTrackConvertParams`` converters, so the
entry's measured media rate and the dataset's ``fps_default`` never reach them
(``Dataset._conversion_fps`` hands a rate only to a converter that declares
``fallback_fps``), and the table has no ``fps`` or ``frame_rate`` column. Readers
of a table look for a stated rate in such a column (``pair-position``,
``pair-egocentric``, ``approach-avoidance`` and the wavelets read ``fps``,
``resample-tracks`` reads ``frame_rate`` and then ``fps``), and a column here would
state a rate that is wrong for every frame.

**The raw table has no time column**, so it needs its sibling: the same stem with
``_tracked_fixed_final.tsv`` in place of ``_tracked.csv``, in the same directory.
Without the sibling the entry fails with an error that names it. There is no
fallback to a rate. The sibling is not part of the raw variant's provenance: the
``tracks_raw`` composition and ``source_md5`` cover ``_tracked.csv`` alone, and so
does the check that a table is current. A changed corrected table leaves the raw
variant stale without a signal, so rerun ``convert-tracks --overwrite`` after
editing one.

**Pixels.** ``X`` and ``Y`` are Tracktor's ``pos_x`` and ``pos_y``, in pixels of
the arena video, and the schema is ``mosaic_v1``. No file records the scale, about
0.19 mm per pixel for the 30 mm arena. A table stays in pixels, as every pixel
converter's does, and the scale is applied by ``scale-to-cm`` with an explicit
``cm_per_pixel`` (the release has no videos in the dataset to calibrate).

**Frames before 62 are dropped** by default (``first_frame``). The release advises
against them, and the smoothed columns of the corrected table are not defined
there. Frame numbers keep the file's numbering, which starts at 3.

**What else is carried.** ``stimuli`` is the looming indicator, ``Y`` or ``N``, on
every row, as the file spells it: it is pulsed within a window, so a reader takes
the pulses from the column. ``strain``, ``sex``, ``n_inds``, ``place`` and
``prefix`` are the recording's labels, constant text columns that must agree
within a file. The smoothed columns (``pos_x_wma``, ``pos_y_wma``,
``travelled_dist_diff``, ``angle_diff_based``) are derived and are not carried.
The raw table's arena centre (``center_x``, ``center_y``) is not carried either:
a column of pixels beside ``X`` would be left unscaled by ``scale-to-cm``.

A fly-frame whose position is ``NA`` has no row. ``id`` is the number in ``Fly<n>``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, ClassVar, Final, override

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field

from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint

__all__ = [
    "CORRECTED_SUFFIX",
    "RAW_SUFFIX",
    "TracktorParams",
    "TracktorRawCsvConverter",
    "TracktorTsvConverter",
]

RAW_SUFFIX: Final = "_tracked.csv"
"""What a raw Tracktor table's name ends in."""

CORRECTED_SUFFIX: Final = "_tracked_fixed_final.tsv"
"""What a corrected table's name ends in, and what the raw table's sibling ends in."""

_FLY_ID: Final = re.compile(r"^Fly(\d+)$")
_LABEL_COLUMNS: Final = ("strain", "sex", "n_inds", "place", "prefix")
_NUMERIC_COLUMNS: Final = ("frame", "pos_x", "pos_y", "seconds_total")
_TEXT_COLUMNS: Final = ("id", "stimuli", *_LABEL_COLUMNS)
_OUTPUT_COLUMNS: Final = (
    "frame",
    "time",
    "id",
    "X",
    "Y",
    "group",
    "sequence",
    "stimuli",
    *_LABEL_COLUMNS,
)

_FIRST_FRAME_DESCRIPTION = (
    "Rows before this frame number are dropped. The release advises against "
    "using anything earlier than 62, and its smoothed columns are not defined there."
)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class TracktorParams(TrackConvertParams):
    """Everything that decides a Tracktor conversion.

    There is no frame rate: time is read from the file.
    """

    first_frame: Annotated[
        int, Field(ge=0), Declared(_FIRST_FRAME_DESCRIPTION, unit="frames")
    ] = 62


def _read_corrected(path: Path) -> pd.DataFrame:
    """The columns of the corrected table that a conversion reads, as text or floats.

    ``NA`` is a missing number in a numeric column only.
    """
    dtypes: dict[str, type] = {name: str for name in _TEXT_COLUMNS}
    dtypes.update({name: float for name in _NUMERIC_COLUMNS})
    try:
        return pd.read_csv(
            path,
            sep="\t",
            usecols=[*_NUMERIC_COLUMNS, *_TEXT_COLUMNS],
            dtype=dtypes,
            keep_default_na=False,
            na_values={name: ["NA"] for name in _NUMERIC_COLUMNS},
        )
    except ValueError as error:
        message = f"{path} is not a corrected Tracktor table: {error}"
        raise ValueError(message) from error


def _frame_numbers(values: FloatArray, path: Path) -> IntArray:
    """The frame numbers *values* spell, refusing a number that is not whole."""
    if not np.isfinite(values).all() or (values != np.floor(values)).any():
        message = (
            f"{path}: the frame column holds a value that is not a whole frame number"
        )
        raise ValueError(message)
    return values.astype(np.int64)


def _fly_numbers(ids: pd.Series, path: Path) -> IntArray:
    """The number of each ``Fly<n>`` id of *ids*.

    Raises:
        ValueError: An id is not ``Fly`` and a number. The message names it.
    """
    names = [str(name) for name in ids.unique().tolist()]
    numbers: dict[str, int] = {}
    for name in names:
        match = _FLY_ID.match(name)
        if match is None:
            message = (
                f"{path}: id {name!r} is not a Tracktor fly id, which is Fly "
                "followed by a number"
            )
            raise ValueError(message)
        numbers[name] = int(match.group(1))
    return np.array([numbers[str(name)] for name in ids.tolist()], dtype=np.int64)


class _Clock:
    """A recording's clock and per-frame labels, read from its corrected table.

    Attributes:
        frames: The distinct frame numbers, ascending.
        seconds: ``seconds_total`` of each frame in :attr:`frames`.
        stimuli: ``stimuli`` of each frame in :attr:`frames`.
    """

    def __init__(self, table: pd.DataFrame, path: Path) -> None:
        frame = _frame_numbers(table["frame"].to_numpy(dtype=np.float64), path)
        per_frame = pd.DataFrame(
            {
                "frame": frame,
                "seconds": table["seconds_total"].to_numpy(dtype=np.float64),
                "stimuli": table["stimuli"].to_numpy(dtype=object),
            }
        ).groupby("frame", sort=True)
        spread = per_frame["seconds"].agg(["min", "max", "count"])
        disagree = spread[spread["min"] != spread["max"]]
        if len(disagree):
            first = int(disagree.index[0])
            message = (
                f"{path}: frame {first} has more than one seconds_total, and a "
                "frame is one moment"
            )
            raise ValueError(message)
        stimulus_values = per_frame["stimuli"].nunique()
        if len(stimulus_values) and int(stimulus_values.max()) > 1:
            first = int(stimulus_values[stimulus_values > 1].index[0])
            message = f"{path}: frame {first} has more than one stimuli value"
            raise ValueError(message)
        self.frames: IntArray = spread.index.to_numpy(dtype=np.int64)
        self.seconds: FloatArray = spread["min"].to_numpy(dtype=np.float64)
        self.stimuli: npt.NDArray[np.object_] = (
            per_frame["stimuli"].first().to_numpy(dtype=object)
        )


def _label(table: pd.DataFrame, column: str, path: Path) -> str:
    """The one value of *column*, which a recording keeps throughout.

    Raises:
        ValueError: The column holds more than one value.
    """
    values = sorted({str(value) for value in table[column].tolist()})
    if len(values) > 1:
        message = (
            f"{path}: {column} takes more than one value, {values[:3]}, and one "
            "file is one recording"
        )
        raise ValueError(message)
    return values[0] if values else ""


def _assemble(
    *,
    frame: IntArray,
    fly: IntArray,
    x: FloatArray,
    y: FloatArray,
    clock: _Clock,
    corrected: pd.DataFrame,
    corrected_path: Path,
    first_frame: int,
    group: str,
    sequence: str,
    positions_path: Path,
) -> pd.DataFrame:
    """The ``mosaic_v1`` table of positions, timed and labeled by the corrected table.

    Raises:
        ValueError: A kept frame is not in *clock*, or has no time.
    """
    keep = (frame >= first_frame) & np.isfinite(x) & np.isfinite(y)
    frame, fly, x, y = frame[keep], fly[keep], x[keep], y[keep]
    order = np.lexsort((fly, frame))
    frame, fly, x, y = frame[order], fly[order], x[order], y[order]

    slot = np.searchsorted(clock.frames, frame)
    slot = np.minimum(slot, max(len(clock.frames) - 1, 0))
    known = len(clock.frames) > 0
    absent = frame != clock.frames[slot] if known else np.ones(frame.shape, dtype=bool)
    if absent.any():
        first = int(frame[absent][0])
        message = (
            f"{positions_path} has frame {first}, which its sibling "
            f"{corrected_path} lacks, so the frame has no time. {int(absent.sum())} "
            "rows have a frame the sibling lacks"
        )
        raise ValueError(message)
    seconds = clock.seconds[slot] if known else np.empty(0, dtype=np.float64)
    if not np.isfinite(seconds).all():
        missing = int(frame[~np.isfinite(seconds)][0])
        message = f"{corrected_path}: frame {missing} has no seconds_total"
        raise ValueError(message)

    rows = len(frame)
    data: dict[str, object] = {
        "frame": frame,
        "time": seconds,
        "id": fly,
        "X": x,
        "Y": y,
        "group": np.full(rows, group, dtype=object),
        "sequence": np.full(rows, sequence, dtype=object),
        "stimuli": clock.stimuli[slot] if known else np.empty(0, dtype=object),
    }
    for column in _LABEL_COLUMNS:
        data[column] = np.full(
            rows, _label(corrected, column, corrected_path), dtype=object
        )
    return pd.DataFrame(data, columns=list(_OUTPUT_COLUMNS))


class _TracktorConverter(TrackConverter[TracktorParams]):
    """What the two Tracktor formats share: the recipe and the sequence name."""

    output_schema: ClassVar[str] = "mosaic_v1"
    Params: ClassVar[type[TrackConvertParams]] = TracktorParams

    @override
    def sequence_from_stem(self, stem: str) -> str:
        """The recording's name: the stem without ``_tracked`` or ``_tracked_fixed_final``.

        Both formats give the one name for a recording, so a raw and a corrected
        table of it can be joined by sequence name across datasets. They are not
        reachable in one dataset with the files side by side, because two sources on
        one directory overlap.
        """
        for ending in (CORRECTED_SUFFIX, RAW_SUFFIX):
            tail = ending.removesuffix(".tsv").removesuffix(".csv")
            if stem.endswith(tail):
                return stem[: -len(tail)]
        return stem


@register_track_converter
class TracktorTsvConverter(_TracktorConverter):
    """Corrected Tracktor ``*_tracked_fixed_final.tsv`` -> a ``mosaic_v1`` table."""

    src_format = "tracktor_corrected_tsv"
    version = "0.1"
    # The corrected ids follow the animals.
    carries_tracklets = True

    def convert(
        self, path: Path, params: TracktorParams, hints: EntryHints
    ) -> pd.DataFrame:
        table = _read_corrected(path)
        return _assemble(
            frame=_frame_numbers(table["frame"].to_numpy(dtype=np.float64), path),
            fly=_fly_numbers(table["id"], path),
            x=table["pos_x"].to_numpy(dtype=np.float64),
            y=table["pos_y"].to_numpy(dtype=np.float64),
            clock=_Clock(table, path),
            corrected=table,
            corrected_path=path,
            first_frame=params.first_frame,
            group=norm_hint(hints.group) or "",
            sequence=norm_hint(hints.sequence) or self.sequence_from_stem(path.stem),
            positions_path=path,
        )


@register_track_converter
class TracktorRawCsvConverter(_TracktorConverter):
    """Raw Tracktor ``*_tracked.csv`` -> a ``mosaic_v1`` table, timed by its sibling."""

    src_format = "tracktor_raw_csv"
    version = "0.1"
    # The raw ids swap between animals, which the corrected table undoes.
    carries_tracklets = False

    def convert(
        self, path: Path, params: TracktorParams, hints: EntryHints
    ) -> pd.DataFrame:
        if not path.name.endswith(RAW_SUFFIX):
            message = (
                f"{path} is not a raw Tracktor table, whose name ends in "
                f"{RAW_SUFFIX}. The corrected table that times it is named by "
                "that ending."
            )
            raise ValueError(message)
        sibling = path.with_name(path.name.removesuffix(RAW_SUFFIX) + CORRECTED_SUFFIX)
        if not sibling.is_file():
            message = (
                f"{path} has no time column, and its sibling {sibling} is absent. "
                "A raw Tracktor table is timed by the seconds_total of the "
                "corrected table beside it, and never by a frame rate. Put "
                f"{sibling.name} in the same directory."
            )
            raise ValueError(message)
        corrected = _read_corrected(sibling)
        try:
            raw = pd.read_csv(
                path,
                usecols=["frame", "pos_x", "pos_y", "id"],
                dtype={"frame": float, "pos_x": float, "pos_y": float, "id": str},
            )
        except ValueError as error:
            message = f"{path} is not a raw Tracktor table: {error}"
            raise ValueError(message) from error
        return _assemble(
            frame=_frame_numbers(raw["frame"].to_numpy(dtype=np.float64), path),
            fly=_fly_numbers(raw["id"], path),
            x=raw["pos_x"].to_numpy(dtype=np.float64),
            y=raw["pos_y"].to_numpy(dtype=np.float64),
            clock=_Clock(corrected, sibling),
            corrected=corrected,
            corrected_path=sibling,
            first_frame=params.first_frame,
            group=norm_hint(hints.group) or "",
            sequence=norm_hint(hints.sequence) or self.sequence_from_stem(path.stem),
            positions_path=path,
        )
