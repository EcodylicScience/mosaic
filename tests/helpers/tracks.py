"""Build track tables, tracks variants, and raw tracker exports.

Every helper here writes what production writes: ``add_tracks_variant`` goes
through ``write_tracks_row`` rather than a hand-built CSV, and ``write_trex_npz``
carries the two fields that decide what a TREx table *means*. A fixture that
writes a shape no converter produces is measuring something that cannot occur.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.dataset import Dataset


def add_track_sequences(dataset: Dataset, *sequences: str, n_rows: int = 40) -> None:
    """Write a track parquet per sequence and rewrite ``tracks/index.csv``.

    Sequences accumulate: calling this again with a further name leaves the
    existing parquets in place, which is what lets a scenario widen a scope and
    then assert what was and was not recomputed.

    The group is empty, so the composite key renders as the bare sequence name
    and the parquet is ``<sequence>.parquet``.

    ``X``/``Y`` are here because the features these scenarios run need them. Without
    them every entry's ``apply`` raised, and because a per-entity failure used to be
    swallowed, the run reported success having computed nothing -- so tests asserted
    on the ``params.json`` of a run with no outputs.
    """
    tracks = dataset.get_root("tracks")
    tracks.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        frame = np.arange(n_rows, dtype=np.int64)
        pd.DataFrame(
            {
                "frame": frame,
                "time": frame / 30.0,
                "id": np.zeros(n_rows, dtype=np.int64),
                "X": np.linspace(0.0, 10.0, n_rows),
                "Y": np.linspace(10.0, 0.0, n_rows),
                "feat_a": np.linspace(0.0, 1.0, n_rows),
            }
        ).to_parquet(tracks / f"{sequence}.parquet")
    present = sorted(tracks.glob("*.parquet"))
    index = pd.DataFrame(
        {
            "group": ["" for _ in present],
            "sequence": [path.stem for path in present],
            "abs_path": [str(path) for path in present],
        }
    )
    index.to_csv(tracks / "index.csv", index=False)


def write_trex_npz(
    path: Path,
    *,
    individual: int | None = None,
    n: int = 8,
    cm_per_pixel: float = 1.0,
    **columns: np.ndarray,
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
    explicitly through *columns*.

    ``individual`` defaults to the trailing digits of the filename, because TREx
    names each file for the individual it holds -- ``myseq_fish0.npz`` beside
    ``myseq_fish1.npz``. Defaulting it to a constant instead would give a
    sequence's several files one id and quietly collapse them into one animal.
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
    fields.update(columns)
    np.savez(path, **fields)


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

    Uses ``write_tracks_row`` rather than a hand-built CSV, so the index it
    produces is the index production produces -- including the dedup that decides
    whether a second call adds a row or replaces one.
    """
    from mosaic.core.helpers import make_entry_key
    from mosaic.core.pipeline.tracks_identity import tracks_variant_root
    from mosaic.core.pipeline.tracks_index import write_tracks_row

    root = tracks_variant_root(dataset.get_root("tracks"), run_id)
    root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        # A schema-valid table with two individuals, rather than the four columns
        # ``add_track_sequences`` writes. That is what lets a *registered*
        # feature actually run on this fixture -- including the social ones,
        # which need a sequence to hold at least two ids -- which the
        # chain-runner parity assertions depend on. ``feat_a`` stays for the
        # scenario mock features that read it.
        #
        # X/Y are the body centre and every converter emits them. This fixture
        # carried only the ``#wcentroid`` pair, a shape no converter produces,
        # so tests built on it were measuring a table that cannot exist.
        # ``#wcentroid`` stays, holding the identical values, because that is
        # what a TREx table looks like: one body centre under both names.
        frame = np.tile(np.arange(n_rows, dtype=np.int64), 2)
        identity = np.repeat(np.arange(2, dtype=np.int64), n_rows)
        total = len(frame)
        centre_x = np.linspace(0.0, 10.0, total) + identity
        centre_y = np.linspace(0.0, 5.0, total) + identity
        columns: dict[str, object] = {
            "frame": frame,
            "time": frame / 30.0,
            "id": identity,
            "group": [""] * total,
            "sequence": [sequence] * total,
            "X": centre_x,
            "Y": centre_y,
            "X#wcentroid": centre_x,
            "Y#wcentroid": centre_y,
            "feat_a": np.linspace(0.0, 1.0, total),
        }
        for keypoint in range(7):
            columns[f"poseX{keypoint}"] = np.linspace(0.0, 10.0, total) + keypoint
            columns[f"poseY{keypoint}"] = np.linspace(0.0, 5.0, total) + keypoint
        out_path = root / f"{make_entry_key('', sequence)}.parquet"
        pd.DataFrame(columns).to_parquet(out_path)
        write_tracks_row(
            dataset,
            run_id=run_id,
            group="",
            sequence=sequence,
            out_path=out_path,
            producer=run_id.split(".")[0],
            std_format=std_format,
            n_rows=n_rows,
            consumed_source_roots=consumed_source_roots,
        )


def track_sequences(dataset: Dataset) -> list[str]:
    """The sequence names the tracks index currently names.

    Read from the index rather than globbed off the root, so it answers the same
    for a flat legacy layout and for variant directories.
    """
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    return sorted({str(name) for name in read_tracks_index(dataset)["sequence"]})
