"""Build track tables, tracks variants, and raw tracker exports.

Every helper here writes what production writes: ``add_tracks_variant`` goes
through ``write_tracks_row`` rather than a hand-built CSV, and ``write_trex_npz``
carries the two fields that decide what a TREx table *means*. A fixture that
writes a shape no converter produces is measuring something that cannot occur.
"""

from __future__ import annotations

import json
import re
import zlib
from collections.abc import Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.appearance import (
    appearance_columns,
    appearance_path,
    write_appearance,
)
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    tracks_index_path,
    write_tracks_row,
)
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from tests.helpers.mock_dataset import MockDataset


TrackEntry = str | tuple[str, str]
"""A bare sequence name, whose group is empty, or a ``(group, sequence)`` pair."""


def track_table(
    group: str = "",
    sequence: str = "",
    *,
    n_rows: int = 40,
    n_ids: int = 1,
    start_frame: int = 0,
) -> pd.DataFrame:
    """One entry's track table: *n_rows* frames from *start_frame*, *n_ids* per frame.

    Frame-major, as a tracker writes one: every individual of a frame before the
    next frame. With several individuals a row offset and a frame number then
    disagree, which is what an overlap or trim defect needs in order to show.

    ``X``/``Y`` are a straight path, offset by identity so that the individuals
    are apart. ``feat_a`` is seeded from the entry's key, so one entry always
    holds the same values and two entries hold different ones. A fit over
    several entries then differs from a fit over one.
    """
    frame = np.repeat(
        np.arange(start_frame, start_frame + n_rows, dtype=np.int64), n_ids
    )
    identity = np.tile(np.arange(n_ids, dtype=np.int64), n_rows)
    seed = zlib.crc32(make_entry_key(group, sequence).encode())
    return pd.DataFrame(
        {
            "frame": frame,
            "time": frame / 30.0,
            "id": identity,
            "group": group,
            "sequence": sequence,
            "X": np.repeat(np.linspace(0.0, 10.0, n_rows), n_ids) + identity,
            "Y": np.repeat(np.linspace(10.0, 0.0, n_rows), n_ids) + identity,
            "feat_a": np.random.default_rng(seed).random(len(frame)),
        }
    )


def keypoint_rows(
    frames: Iterable[int],
    individual: int | str | None = None,
    *,
    n_keypoints: int,
    confidence: float | Sequence[float] = 0.9,
    offset: float = 0.0,
) -> pd.DataFrame:
    """Return one individual's scored keypoints, a row per frame of *frames*, in order.

    Keypoint ``k`` at frame ``f`` is at ``(p, -p)``, where
    ``p = offset + f + k / 10``. Its ``poseP`` score is *confidence*, or
    ``confidence[k]`` when *confidence* is a sequence. Each row also contains
    ``frame``, ``time`` at 30 fps, an empty ``group`` and ``sequence``, and ``id``
    unless *individual* is None.
    """
    frame = np.asarray(list(frames), dtype=np.int64)
    scores = np.broadcast_to(np.asarray(confidence, dtype=np.float64), n_keypoints)
    columns: dict[str, object] = {
        "frame": frame,
        "time": frame / 30.0,
        "group": "",
        "sequence": "",
    }
    if individual is not None:
        columns["id"] = [individual] * len(frame)
    for k in range(n_keypoints):
        columns[f"poseX{k}"] = offset + frame + k / 10
        columns[f"poseY{k}"] = -(offset + frame + k / 10)
        columns[f"poseP{k}"] = np.full(len(frame), scores[k])
    return pd.DataFrame(columns)


def world_track_table(
    group: str = "",
    sequence: str = "",
    *,
    n_rows: int = 40,
    n_ids: int = 1,
    start_frame: int = 0,
) -> pd.DataFrame:
    """One entry's ``mosaic_world_v1`` table: three keypoints, a ``Z`` and ``poseZ``.

    Laid out as :func:`track_table` lays out its rows, in meters. ``Z`` rises
    with the frame and keypoint ``k`` sits ``k`` tenths of a meter above the body
    center on every axis, so the body center is not the mean of the keypoints and
    the three axes differ.
    """
    table = track_table(
        group, sequence, n_rows=n_rows, n_ids=n_ids, start_frame=start_frame
    ).drop(columns="feat_a")
    table["Z"] = np.repeat(np.linspace(0.0, 2.0, n_rows), n_ids)
    for keypoint in range(3):
        offset = keypoint / 10
        for axis in ("X", "Y", "Z"):
            table[f"pose{axis}{keypoint}"] = table[axis] + offset
        table[f"poseP{keypoint}"] = 0.9
    return table


WORLD_POINTS_FORMAT = "test_world_points"
"""The raw format of :class:`WorldPointsConverter`."""


class WorldPointsConverter(TrackConverter[TrackConvertParams]):
    """Converts any raw file into the :func:`world_track_table` of two animals."""

    src_format: ClassVar[str] = WORLD_POINTS_FORMAT
    output_schema: ClassVar[str] = "mosaic_world_v1"

    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        return world_track_table(hints.group, hints.sequence or path.stem, n_ids=2)


@contextmanager
def registered_world_points() -> Generator[None]:
    """Register :class:`WorldPointsConverter` for the duration of the block."""
    _ = register_track_converter(WorldPointsConverter)
    try:
        yield
    finally:
        _ = TRACK_CONVERTERS.pop(WORLD_POINTS_FORMAT, None)


def raw_world_dataset(root: Path) -> Dataset:
    """A dataset indexing one raw file, ``seq.txt``, of :data:`WORLD_POINTS_FORMAT`.

    Nothing is converted. Index it inside :func:`registered_world_points`.
    """
    from tests.helpers.datasets import make_dataset

    dataset = make_dataset(root)
    raw = dataset.base_dir / "raw"
    raw.mkdir(parents=True)
    _ = (raw / "seq.txt").write_text("points\n", encoding="utf-8")
    dataset.index_tracks_raw([raw], patterns=["*.txt"], src_format=WORLD_POINTS_FORMAT)
    return dataset


ZEBRAFISH_3D_PARTS = ("head", "pec", "tail")
"""The keypoints of a release file, in its header order."""


def write_zebrafish_3d_csv(
    path: Path,
    frames: int,
    *,
    fish: Sequence[int] = (1, 2),
    parts: Sequence[str] = ZEBRAFISH_3D_PARTS,
    blank: Iterable[tuple[int, int, str | None]] = (),
) -> npt.NDArray[np.float64]:
    """Write a ``zebrafish_3d_workflow_csv`` file of *frames* contiguous frames.

    The header is ``frame_index`` and then ``fish{n}_{part}_{axis}`` for each
    fish, each of its parts and the axes ``x``, ``y``, ``z`` in that order, so the
    default 19 columns are the release's. Every cell is distinct: the value of
    fish ``n``, part ``k``, axis ``a`` at frame ``f`` is
    ``10 n + k + (a + 1) / 10 + f / 100``, in the file's unit.

    Args:
        path: Where to write the file. Missing parent directories are created.
        frames: The number of frames, numbered from 0.
        fish: The fish numbers.
        parts: The keypoint names.
        blank: Cells to leave empty, each ``(frame, fish number, part)``. A part
            of None empties every part of that fish at that frame.

    Returns:
        The values written, shaped ``(frame, fish, part, axis)``, NaN where a
        cell is empty.
    """
    n_parts = len(parts)
    frame_axis = np.arange(frames, dtype=np.float64)[:, None, None, None]
    fish_axis = np.asarray(fish, dtype=np.float64)[None, :, None, None]
    part_axis = np.arange(n_parts, dtype=np.float64)[None, None, :, None]
    axis_axis = (np.arange(3, dtype=np.float64) + 1)[None, None, None, :] / 10
    values = 10 * fish_axis + part_axis + axis_axis + frame_axis / 100
    values = np.array(np.broadcast_to(values, (frames, len(fish), n_parts, 3)))
    for frame, number, part in blank:
        fish_index = list(fish).index(number)
        if part is None:
            values[frame, fish_index] = np.nan
        else:
            values[frame, fish_index, list(parts).index(part)] = np.nan

    header = ["frame_index"] + [
        f"fish{number}_{part}_{axis}"
        for number in fish
        for part in parts
        for axis in "xyz"
    ]
    lines = [",".join(header)]
    for index in range(frames):
        cells = [str(index)] + [
            "" if np.isnan(value) else repr(float(value))
            for value in values[index].reshape(-1)
        ]
        lines.append(",".join(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


TRACKTOR_STEM = "20211117111002_900s_no12-DGRP101-female-2to4-6-1"
"""A release recording's name without its ``_tracked`` suffix."""

TRACKTOR_TSV_COLUMNS = (
    "frame",
    "id",
    "pos_x",
    "pos_y",
    "prefix",
    "place",
    "strain",
    "sex",
    "age",
    "n_inds",
    "N",
    "timestamp",
    "seconds_diff",
    "seconds_total",
    "stimuli",
    "pos_x_wma",
    "pos_y_wma",
    "travelled_dist_diff",
    "angle_diff_based",
)
"""The header of a corrected ``*_tracked_fixed_final.tsv``, in the release's order."""


@dataclass(frozen=True)
class TracktorRecording:
    """What :func:`write_tracktor_recording` wrote.

    Attributes:
        tsv: The corrected table's path.
        csv: The raw Tracktor table's path.
        frames: The frame numbers, ascending.
        seconds: The ``seconds_total`` of each frame. Its steps differ, as they do
            where a recording dropped frames.
        corrected: ``(frame, fly, 2)`` positions of the corrected table, whose
            fly ``k`` follows one animal.
        raw: The raw table's positions, in the same shape. Flies 1 and 2 trade
            places from :attr:`swap_frame` on, as the tracker's ids do.
        stimuli: ``"Y"`` or ``"N"`` for each frame.
        swap_frame: The first frame where the raw ids of flies 1 and 2 are swapped.
    """

    tsv: Path
    csv: Path
    frames: npt.NDArray[np.int64]
    seconds: npt.NDArray[np.float64]
    corrected: npt.NDArray[np.float64]
    raw: npt.NDArray[np.float64]
    stimuli: tuple[str, ...]
    swap_frame: int


def write_tracktor_recording(
    directory: Path,
    *,
    stem: str = TRACKTOR_STEM,
    frames: int = 80,
    flies: int = 6,
    first_frame: int = 3,
    write_tsv: bool = True,
    write_csv: bool = True,
    missing_position: tuple[int, int] | None = None,
) -> TracktorRecording:
    """Write one arena's Tracktor output, a raw ``.csv`` and a corrected ``.tsv``.

    The frame numbers start at *first_frame* and are contiguous. ``seconds_total``
    advances by 0.016, 0.020 and 0.0175 s in turn, so the frames are not evenly
    spaced in time and no single frame rate gives their times. Fly ``k`` is at
    ``(10 k + f / 2, 5 k + f / 4)`` in the corrected table, at frame ``f``, and the
    smoothed columns are ``NA`` before frame 62, as in the release. The looming
    ``stimuli`` is ``Y`` in two pulses.

    Args:
        directory: Where to write. Created if absent.
        stem: The recording's name without ``_tracked``.
        frames: The number of frames.
        flies: The number of flies, named ``Fly1`` onward.
        first_frame: The first frame number.
        write_tsv: Whether to write the corrected table.
        write_csv: Whether to write the raw table.
        missing_position: ``(frame, fly)`` whose corrected position is ``NA``.

    Returns:
        What was written.
    """
    frame_numbers = np.arange(first_frame, first_frame + frames, dtype=np.int64)
    steps = np.resize(np.array([0.016, 0.020, 0.0175]), frames)
    seconds = np.cumsum(steps)
    corrected = np.empty((frames, flies, 2), dtype=np.float64)
    for fly in range(1, flies + 1):
        corrected[:, fly - 1, 0] = 10 * fly + frame_numbers / 2
        corrected[:, fly - 1, 1] = 5 * fly + frame_numbers / 4
    swap_frame = first_frame + frames // 2
    raw = corrected.copy()
    if flies >= 2:
        after = frame_numbers >= swap_frame
        raw[after, 0], raw[after, 1] = corrected[after, 1], corrected[after, 0]
    stimuli = tuple(
        "Y"
        if first_frame + 30 <= frame < first_frame + 35 or frame >= first_frame + 70
        else "N"
        for frame in frame_numbers.tolist()
    )
    directory.mkdir(parents=True, exist_ok=True)
    tsv = directory / f"{stem}_tracked_fixed_final.tsv"
    csv = directory / f"{stem}_tracked.csv"
    place, strain, sex, age, n_inds, replicate = stem.split("_", 2)[2].split("-")
    prefix = stem.split("_", 1)[0]

    if write_tsv:
        lines = ["\t".join(TRACKTOR_TSV_COLUMNS)]
        for index, frame in enumerate(frame_numbers.tolist()):
            smoothed = frame >= 62
            for fly in range(1, flies + 1):
                x, y = corrected[index, fly - 1]
                missing = missing_position == (frame, fly)
                cells = [
                    str(frame),
                    f"Fly{fly}",
                    "NA" if missing else f"{x:.2f}",
                    "NA" if missing else f"{y:.2f}",
                    prefix,
                    place,
                    strain,
                    sex,
                    age,
                    n_inds,
                    replicate,
                    f"{float(prefix) + 59 + seconds[index]:.2f}",
                    f"{steps[index]:.6f}",
                    f"{seconds[index]:.6f}",
                    stimuli[index],
                    f"{x:.4f}" if smoothed else "NA",
                    f"{y:.4f}" if smoothed else "NA",
                    "0.5" if smoothed else "NA",
                    "0.1" if smoothed else "NA",
                ]
                lines.append("\t".join(cells))
        _ = tsv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if write_csv:
        lines = ["frame,pos_x,pos_y,id,center_x,center_y"]
        for index, frame in enumerate(frame_numbers.tolist()):
            for fly in range(1, flies + 1):
                x, y = raw[index, fly - 1]
                lines.append(f"{frame}.0,{float(x)!r},{float(y)!r},Fly{fly},85,88")
        _ = csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return TracktorRecording(
        tsv=tsv,
        csv=csv,
        frames=frame_numbers,
        seconds=seconds,
        corrected=corrected,
        raw=raw,
        stimuli=stimuli,
        swap_frame=swap_frame,
    )


def add_track_sequences(
    dataset: Dataset | MockDataset,
    *entries: TrackEntry,
    n_rows: int = 40,
    n_ids: int = 1,
    start_frame: int = 0,
) -> None:
    """Write a :func:`track_table` per entry and add its row to ``tracks/index.csv``.

    Entries accumulate: calling this again with a further entry leaves the
    existing parquets and rows in place, which is what lets a scenario widen a
    scope and then assert what was and was not recomputed. Naming an entry again
    replaces its table and its row.

    The parquet is ``<group>__<sequence>.parquet``, or ``<sequence>.parquet``
    for an empty group, as the composite key renders.

    The index holds three columns, ``group``, ``sequence`` and ``abs_path``: the
    shape a dataset converted before tracks variants existed has, with no
    recorded frame extent. A test that needs the extents runs
    ``backfill_frame_extents``, the migration such a dataset is given.

    ``X``/``Y`` are here because the features these scenarios run need them. Without
    them every entry's ``apply`` raised, and because a per-entity failure used to be
    swallowed, the run reported success having computed nothing -- so tests asserted
    on the ``params.json`` of a run with no outputs.
    """
    tracks = dataset.get_root("tracks")
    tracks.mkdir(parents=True, exist_ok=True)
    added: list[dict[str, str]] = []
    for entry in entries:
        group, sequence = ("", entry) if isinstance(entry, str) else entry
        path = tracks / f"{make_entry_key(group, sequence)}.parquet"
        track_table(
            group, sequence, n_rows=n_rows, n_ids=n_ids, start_frame=start_frame
        ).to_parquet(path)
        added.append({"group": group, "sequence": sequence, "abs_path": str(path)})
    index_path = tracks / "index.csv"
    replaced = {(row["group"], row["sequence"]) for row in added}
    kept: list[dict[str, str]] = []
    if index_path.exists():
        existing = pd.read_csv(index_path, dtype=str, keep_default_na=False)
        kept = [
            {str(key): str(value) for key, value in row.items()}
            for row in existing.to_dict("records")
            if (row["group"], row["sequence"]) not in replaced
        ]
    pd.DataFrame([*kept, *added]).to_csv(index_path, index=False)


def _write_marker(marker: str) -> None:
    Path(marker).write_text("unpickled")


class _UnpickleSentinel:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_marker, (str(self.marker),)


def unpickling_member(marker: Path) -> npt.NDArray[np.object_]:
    """A one-element object array whose unpickling writes the file *marker*.

    Saved into a raw export, it shows whether a reader unpickled the export: the
    marker exists exactly when the reader ran the pickle.
    """
    member = np.empty(1, dtype=object)
    member[0] = _UnpickleSentinel(marker)
    return member


def write_trex_npz(
    path: Path,
    *,
    individual: int | None = None,
    n: int = 8,
    cm_per_pixel: float = 1.0,
    columns: Mapping[str, np.ndarray] | None = None,
    omit: Sequence[str] = (),
) -> None:
    """Write a per-individual TREx export carrying what TREx always writes.

    Six near-identical builders used to sit in six test modules, and every one of
    them omitted the two fields that decide what a TREx table *means*:
    ``cm_per_pixel``, which says whether its positions are centimetres, and the
    ``#wcentroid`` pair, which is the body centre. A file without them is not a
    file TREx produces, so tests built on one were measuring a shape that cannot
    occur.

    ``cm_per_pixel`` and ``id`` are written as one-element arrays because that is
    how TREx writes them -- as ``std::vector`` of one, not as scalars -- which is
    what makes them arrive NaN-padded rather than broadcast.

    The bare ``X``/``Y`` are given the same values as ``#wcentroid`` by default.
    In a real export they differ (bare is the head), but most callers only need
    *a* position; a caller testing the head-versus-centre distinction passes them
    explicitly in *columns*, which replaces or adds fields by name. It is a
    mapping rather than keyword arguments because some TREx field names, such as
    ``X#wcentroid``, are not Python identifiers.

    ``individual`` defaults to the trailing digits of the filename, because TREx
    names each file for the individual it holds -- ``myseq_fish0.npz`` beside
    ``myseq_fish1.npz``. Defaulting it to a constant instead would give a
    sequence's several files one id and quietly collapse them into one animal.

    ``omit`` names fields to leave out, for an export lacking one that TREx
    writes today: one made before TREx recorded ``cm_per_pixel``, or one a
    converter has to refuse. Naming a field the export does not hold raises.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if individual is None:
        match = re.search(r"(\d+)$", path.stem)
        individual = int(match.group(1)) if match else 0
    centre_x = np.linspace(0.0, 1.0, n)
    centre_y = np.linspace(1.0, 0.0, n)
    fields: dict[str, np.ndarray] = {
        "frame": np.arange(n, dtype=np.int64),
        "time": np.arange(n, dtype=float) / 30.0,
        "id": np.array([individual]),
        "cm_per_pixel": np.array([cm_per_pixel]),
        "X": centre_x,
        "Y": centre_y,
        "X#wcentroid": centre_x,
        "Y#wcentroid": centre_y,
        "poseX0": centre_x,
        "poseY0": centre_y,
    }
    fields.update(columns or {})
    for name in omit:
        del fields[name]
    np.savez(path, allow_pickle=False, **fields)


type SleapPreset = Literal["matlab", "standard"]
"""The two array layouts that ``sleap-convert --format analysis`` writes."""


def write_sleap_analysis_h5(
    path: Path,
    tracks: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64] | None = None,
    *,
    preset: SleapPreset = "matlab",
    with_dims: bool = True,
    node_names: Sequence[str] = (),
    track_names: Sequence[str] = (),
) -> None:
    """Write a SLEAP analysis HDF5 from canonical arrays.

    *tracks* is ``(frame, track, node, 2)`` and *scores* is ``(frame, track,
    node)``. ``matlab`` writes the transposed layout that ``sleap-convert``
    produces by default and ``standard`` the Python-native one. Both have a
    ``dims`` attribute, which the converter reads to reorder either. *node_names*
    and *track_names*, when given, are written as the byte-string datasets that
    ``sleap-convert`` names them in.
    """
    # The import is deferred, as the converter's is, because most test modules
    # import these helpers and few of them open an HDF5 file.
    import h5py

    if preset == "matlab":
        track_array = np.transpose(tracks, (1, 3, 2, 0))
        track_dims = ["track", "xy", "node", "frame"]
        score_array = None if scores is None else np.transpose(scores, (1, 2, 0))
        score_dims = ["track", "node", "frame"]
    else:
        track_array = tracks
        track_dims = ["frame", "track", "node", "xy"]
        score_array = scores
        score_dims = ["frame", "track", "node"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as handle:
        handle["tracks"] = track_array
        if with_dims:
            handle["tracks"].attrs["dims"] = json.dumps(track_dims)
        if score_array is not None:
            handle["point_scores"] = score_array
            if with_dims:
                handle["point_scores"].attrs["dims"] = json.dumps(score_dims)
        if node_names:
            handle["node_names"] = np.array([n.encode() for n in node_names], "S")
        if track_names:
            handle["track_names"] = np.array([n.encode() for n in track_names], "S")


def write_calms21_npy(
    path: Path,
    pairs: Mapping[str, Mapping[str, int]],
    *,
    seed: int | None = None,
    n_classes: int = 1,
) -> None:
    """Write a CalMS21 ``.npy`` holding several sequences, grouped as it groups them.

    Args:
        path: The file to write. Its parent is created.
        pairs: Each in-file group (an annotator in CalMS21), mapped to its
            sequences and their frame counts. Every sequence holds two mice of
            seven keypoints and a behavior annotation per frame, so the file
            feeds both the tracks and the labels converter.
        seed: Unset, every keypoint is zero and every annotation 0. Set, the
            keypoints walk randomly from this seed and each annotation is drawn
            from ``range(n_classes)``.
        n_classes: How many behavior classes a seeded file draws from.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = None if seed is None else np.random.default_rng(seed)

    def sequence(n: int) -> dict[str, npt.NDArray[np.generic]]:
        if rng is None:
            return {
                "keypoints": np.zeros((n, 2, 2, 7), dtype=float),
                "annotations": np.zeros(n, dtype=int),
            }
        steps = rng.normal(0.0, 1.0, size=(n, 2, 2, 7))
        start = rng.uniform(100.0, 400.0, size=(1, 2, 2, 1))
        return {
            "keypoints": start + np.cumsum(steps, axis=0),
            "annotations": rng.integers(0, n_classes, size=n),
        }

    payload = {
        group: {seq: sequence(n) for seq, n in sequences.items()}
        for group, sequences in pairs.items()
    }
    np.save(path, np.array(payload, dtype=object), allow_pickle=True)


def calms21_dataset(
    base: Path,
    files: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    seed: int | None = None,
    n_classes: int = 1,
    scan: bool = True,
) -> Dataset:
    """A dataset reading CalMS21 files as both a tracks and a labels source.

    Each file's stem is the group its sequences are filed under, as a
    ``group_from="filename"`` source declares, so two files make a train group
    and a test group.

    Args:
        base: The dataset directory. Created if absent.
        files: Each file stem mapped to the ``pairs`` of
            :func:`write_calms21_npy`.
        seed: Passed to :func:`write_calms21_npy`, offset per file.
        n_classes: Passed to :func:`write_calms21_npy`.
        scan: Whether to scan both sources, which fills ``tracks_raw`` and
            ``labels_raw`` and converts nothing.

    Returns:
        The saved dataset.
    """
    from mosaic.core.manifest import LabelsScanSource, TracksScanSource
    from tests.helpers.datasets import make_dataset

    ds = make_dataset(base.resolve())
    raw = ds.base_dir / "raw"
    for offset, (stem, pairs) in enumerate(sorted(files.items())):
        write_calms21_npy(
            raw / f"{stem}.npy",
            pairs,
            seed=None if seed is None else seed + offset,
            n_classes=n_classes,
        )
    ds.add_scan_source(
        TracksScanSource(
            id="calms21",
            path=str(raw),
            patterns=("*.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from="filename",
        )
    )
    ds.add_scan_source(
        LabelsScanSource(
            id="calms21-labels",
            path=str(raw),
            patterns=("*.npy",),
            src_format="calms21_npy",
            multi_sequences_per_file=True,
            group_from="filename",
        )
    )
    if scan:
        ds.scan_tracks()
        ds.scan_labels()
    return ds


def write_dlc_csv(
    path: Path,
    bodyparts: Sequence[str],
    *,
    n_frames: int = 20,
    individuals: Sequence[str] = (),
    values: npt.NDArray[np.float64] | None = None,
    scorer: str = "DLC_model",
) -> npt.NDArray[np.float64]:
    """Write a DeepLabCut CSV, the layout that Lightning Pose also writes.

    The header rows are ``scorer``, then ``individuals`` for a multi-animal
    export, then ``bodyparts`` and ``coords``. Each row after them is a frame
    index followed by ``x``, ``y`` and ``likelihood`` for each bodypart of each
    individual.

    Args:
        path: Where to write the file. Missing parent directories are created.
        bodyparts: The bodypart names, in column order.
        n_frames: The number of frames to draw when *values* is not given.
        individuals: The individuals of a multi-animal export. Empty writes a
            single-animal one.
        values: The ``[x, y, likelihood]`` triples to write, shaped ``(frame,
            bodypart, 3)``, or ``(frame, individual, bodypart, 3)`` for a
            multi-animal export. Unset, they are drawn from a fixed seed: each
            position in 0 to 100 and each likelihood in 0.5 to 1.
        scorer: The model named in the first header row. Lightning Pose names
            ``heatmap_tracker``.

    Returns:
        The values written.
    """
    shape = (n_frames, *((len(individuals),) if individuals else ()), len(bodyparts))
    if values is None:
        rng = np.random.default_rng(0)
        values = rng.uniform(0, 100, size=(*shape, 3))
        values[..., 2] = rng.uniform(0.5, 1.0, size=shape)

    headers = [["scorer"], ["individuals"], ["bodyparts"], ["coords"]]
    for individual in individuals or ("",):
        for bodypart in bodyparts:
            headers[0] += [scorer] * 3
            headers[1] += [individual] * 3
            headers[2] += [bodypart] * 3
            headers[3] += ["x", "y", "likelihood"]
    if not individuals:
        del headers[1]

    lines = [",".join(header) for header in headers]
    for index, frame in enumerate(values):
        lines.append(",".join([str(index), *(f"{value:.6f}" for value in frame.flat)]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return values


def add_tracks_variant(
    dataset: Dataset,
    run_id: str,
    *sequences: str,
    n_rows: int = 40,
    consumed_source_roots: tuple[str, ...] = ("tracks_raw",),
    std_format: str = "trex_v1",
    producer_run_id: str = "",
    keypoints: int = 7,
    pose_prefixes: tuple[str, str] = ("poseX", "poseY"),
    source: Path | str = "",
    appearance_dim: int | None = None,
    carries_tracklets: bool | None = None,
) -> None:
    """Write a variant-addressed track table per sequence, through the real writer.

    ``std_format`` names the schema the rows claim. It defaults to the legacy
    ``trex_v1`` so existing callers are unchanged; a scenario about a dataset
    part-way through a migration sets it per call, which is the only way to build
    one index holding two schema families.

    ``consumed_source_roots`` defaults to what all three conversion writers pass,
    so a row this produces answers "which root would a change have to be under?"
    the way a converted row does. Overridable to ``()`` for a scenario about a
    row that predates the column.

    The counterpart to :func:`add_track_sequences`, which stays deliberately
    unlabelled -- it is the pre-Stage-3 dataset every existing analysis has, and
    keeping one fixture in that shape is what keeps proving that such a dataset
    still resolves and still hashes the same. This one is the shape a conversion
    writes today: tables under ``tracks/<run_id>/`` and rows naming the recipe.

    ``producer_run_id`` is the op run a tracked or inferred variant was bridged
    from, and empty for a conversion, as the real writers record it.

    ``keypoints`` sets how many keypoints each table holds, and ``pose_prefixes``
    the names of their x and y columns before the index.

    ``source`` and ``appearance_dim`` are passed to :func:`add_tracks_table` for
    every sequence.

    Uses ``write_tracks_row`` rather than a hand-built CSV, so the index it
    produces is the index production produces -- including the dedup that decides
    whether a second call adds a row or replaces one.
    """
    for sequence in sequences:
        # Two individuals and seven keypoints by default, beyond what
        # ``add_track_sequences`` writes. That is what lets a *registered*
        # feature actually run on this fixture -- including the social ones,
        # which need a sequence to hold at least two ids -- which the
        # chain-runner parity assertions depend on.
        #
        # ``#wcentroid`` holds the values of X/Y because that is what a TREx table
        # looks like: one body centre under both names. This fixture once carried
        # only the ``#wcentroid`` pair, a shape no converter produces.
        table = track_table("", sequence, n_rows=n_rows, n_ids=2)
        table["X#wcentroid"] = table["X"]
        table["Y#wcentroid"] = table["Y"]
        x_prefix, y_prefix = pose_prefixes
        for keypoint in range(keypoints):
            table[f"{x_prefix}{keypoint}"] = table["X"] + keypoint
            table[f"{y_prefix}{keypoint}"] = table["Y"] + keypoint
        add_tracks_table(
            dataset,
            run_id,
            table,
            n_rows=n_rows,
            consumed_source_roots=consumed_source_roots,
            std_format=std_format,
            producer_run_id=producer_run_id,
            source=source,
            appearance_dim=appearance_dim,
            carries_tracklets=carries_tracklets,
        )


def convert_tracks_from_raw(
    dataset: Dataset,
    run_id: str,
    *sequences: str,
    n: int = 8,
    std_format: str = "trex_v1",
) -> None:
    """Convert *sequences* into variant *run_id* from raw files of *n* frames.

    Each raw file is indexed under ``tracks_raw``, and every table records the
    source composition it was converted from. Calling it again with another *n*
    is a conversion from changed source: the table's composition moves.
    """
    raw_root = dataset.get_root("tracks_raw")
    for sequence in sequences:
        write_trex_npz(raw_root / f"{sequence}.npz", n=n)
    _ = dataset.index_tracks_raw([raw_root], patterns=["*.npz"], src_format="trex_npz")
    add_tracks_variant(dataset, run_id, *sequences, std_format=std_format)


def add_tracks_table(
    dataset: Dataset,
    run_id: str,
    table: pd.DataFrame,
    *,
    n_rows: int | None = None,
    consumed_source_roots: tuple[str, ...] = ("tracks_raw",),
    std_format: str = "trex_v1",
    producer_run_id: str = "",
    media_frames: int | None = None,
    frames_read: int | None = None,
    known_tail_loss: int | None = None,
    first_frame: int | None = None,
    consumed_composition: str | None = None,
    source: Path | str = "",
    appearance_dim: int | None = None,
    appearance: pd.DataFrame | None = None,
    carries_tracklets: bool | None = None,
) -> None:
    """Write *table*, one entry's rows, into the tracks variant *run_id*.

    The entry is the ``group`` and ``sequence`` of the table's first row, and an
    absent ``group`` column means an empty group. The row is written through
    ``write_tracks_row``, as in :func:`add_tracks_variant`, whose other arguments
    this takes.

    Args:
        dataset: The dataset to write into.
        run_id: The tracks variant.
        table: The entry's track table.
        n_rows: The row count that the index records, or ``None`` for the
            table's length.
        consumed_source_roots: As in :func:`add_tracks_variant`.
        std_format: As in :func:`add_tracks_variant`.
        producer_run_id: As in :func:`add_tracks_variant`.
        media_frames: The frame count of the entry's media that the row records,
            or ``None`` for a blank, as a conversion records it.
        frames_read: The frame count read by the tool that the row records, or
            ``None`` for a blank.
        known_tail_loss: The known tail loss that the row records, or ``None``
            for a blank.
        first_frame: The first frame that the row records the table was
            numbered from, or ``None`` for a blank, as a conversion records it.
        consumed_composition: The composition that the row records, or ``None``
            for the composition of *consumed_source_roots*.
        source: The raw file that the row records, or ``""`` for a blank.
        appearance_dim: The vector length of an appearance table drawn by
            :func:`appearance_table`, written beside the tracks table and
            recorded in the row, or ``None`` for no drawn table.
        appearance: An appearance table to write and record in place of a drawn
            one, or ``None``.
        carries_tracklets: Whether the row records ids that follow an animal, or
            ``None`` for a blank, as a conversion that was not told records it.
    """
    from mosaic.behavior.feature_library.helpers import entry_identity
    from mosaic.core.pipeline.tracks_identity import tracks_variant_root

    group, sequence = entry_identity(table)
    root = tracks_variant_root(dataset.get_root("tracks"), run_id)
    root.mkdir(parents=True, exist_ok=True)
    key = make_entry_key(group, sequence)
    out_path = root / f"{key}.parquet"
    table.to_parquet(out_path)
    recorded: Path | None = None
    if appearance is None and appearance_dim is not None:
        appearance = appearance_table(table, appearance_dim)
    if appearance is not None:
        recorded = appearance_path(root, key)
        write_appearance(recorded, appearance)
    write_tracks_row(
        dataset,
        run_id=run_id,
        group=group,
        sequence=sequence,
        out_path=out_path,
        producer=run_id.split(".")[0],
        producer_run_id=producer_run_id,
        std_format=std_format,
        n_rows=len(table) if n_rows is None else n_rows,
        consumed_source_roots=consumed_source_roots,
        media_frames=media_frames,
        frames_read=frames_read,
        known_tail_loss=known_tail_loss,
        first_frame=first_frame,
        consumed_composition=consumed_composition,
        source=source,
        appearance_path=recorded,
        carries_tracklets=carries_tracklets,
    )


def appearance_table(tracks: pd.DataFrame, dim: int) -> pd.DataFrame:
    """Return an appearance table for *tracks*: one unit vector per row, each different."""
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(tracks), dim))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    appearance = tracks[["group", "sequence", "frame", "id"]].copy()
    for position, column in enumerate(appearance_columns(dim)):
        appearance[column] = vectors[:, position].astype(np.float32)
    return appearance


def track_sequences(dataset: Dataset) -> list[str]:
    """The sequence names the tracks index currently names.

    Read from the index rather than globbed off the root, so it answers the same
    for a flat legacy layout and for variant directories.
    """
    return sorted({str(name) for name in read_tracks_index(dataset)["sequence"]})


def published_table(dataset: Dataset, producer: str) -> pd.DataFrame:
    """Return the one tracks table that *producer* published."""
    tracks = read_tracks_index(dataset)
    (row,) = [row for _, row in tracks.iterrows() if row["producer"] == producer]
    return pd.read_parquet(dataset.resolve_path(str(row["abs_path"])))


def set_tracks_cell(
    dataset: Dataset,
    column: str,
    value: str,
    *,
    entry: tuple[str, str] | None = None,
) -> None:
    """Write *value* into *column* of every tracks row, as an earlier pass might.

    *entry* restricts the write to the rows of one ``(group, sequence)``.
    """
    path = tracks_index_path(dataset)
    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    if column not in rows.columns:
        rows[column] = ""
    selected = (
        pd.Series(True, index=rows.index)
        if entry is None
        else (rows["group"] == entry[0]) & (rows["sequence"] == entry[1])
    )
    rows.loc[selected, column] = value
    rows.to_csv(path, index=False)
