"""A blank group must never become the word ``nan``.

``group`` is an optional namespace and is empty on every dataset the control
plane creates, so this is the *common* case rather than an edge. It reaches
``convert_all_tracks`` as an empty CSV cell, which pandas reads back as a float
NaN -- and ``str()`` of that is the word, which is truthy, passes
``validate_entry_name``, and round-trips through ``parse_entry_key``. Nothing
downstream can then tell ``nan__seq.parquet`` from a group genuinely named
``nan``, which is what makes the corruption durable rather than noisy.

These pin every path a raw row takes to a name: the plain one-file-one-sequence
conversion, the merge, the expansion of a file holding several sequences, and
the index writer that spells the row. They must agree, because a filename and
the index row naming it are the same identity written twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import mosaic.core.track_library  # noqa: F401  -- registers the converters
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_index import read_tracks_index

from tests.helpers import make_dataset, write_trex_npz

_BODYPARTS = ["snout", "midbody", "tailtip"]
# Only the two roots a conversion touches: raw rows in, standardized tables out.
_ROOTS = ("tracks_raw", "tracks")


def _dlc_csv(path: Path, n_frames: int = 6) -> None:
    """A single-animal DeepLabCut CSV -- a format that neither merges nor expands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scorer = ["scorer"]
    bodyparts = ["bodyparts"]
    coords = ["coords"]
    for part in _BODYPARTS:
        scorer += ["DLC_model"] * 3
        bodyparts += [part] * 3
        coords += ["x", "y", "likelihood"]
    lines = [",".join(scorer), ",".join(bodyparts), ",".join(coords)]
    for frame in range(n_frames):
        row = [str(frame)]
        for part in range(len(_BODYPARTS)):
            row += [f"{frame + part}.0", f"{frame - part}.0", "0.9"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines))


def _trex_npz(path: Path, *, individual: int, n: int = 5) -> None:
    write_trex_npz(
        path,
        individual=individual,
        n=n,
        poseX=np.stack([np.linspace(0.0, 1.0, n)] * 2, axis=1),
        poseY=np.stack([np.linspace(1.0, 0.0, n)] * 2, axis=1),
    )


def _calms21_npy(path: Path, pairs: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        group: {
            seq: {"keypoints": np.zeros((n, 2, 2, 7), dtype=float)}
            for seq, n in sequences.items()
        }
        for group, sequences in pairs.items()
    }
    np.save(path, payload, allow_pickle=True)


def _entry_names(ds: Dataset) -> list[str]:
    """Every standardized table's filename stem, across variant directories."""
    return sorted(p.stem for p in ds.get_root("tracks").rglob("*.parquet"))


# --- the four paths a raw row takes to a name --------------------------------


def test_a_blank_group_never_reaches_a_filename(tmp_path: Path) -> None:
    """The default path for a format that neither merges nor expands.

    DeepLabCut declares neither ``merges_per_sequence`` nor ``enumerable``, so
    this is the plain one-file-one-sequence conversion -- the branch that read
    the group cell a second time and spelled it ``"nan"``.
    """
    ds = make_dataset((tmp_path / "ds").resolve(), roots=_ROOTS)
    _dlc_csv(ds.base_dir / "raw" / "myseq.csv")

    ds.index_tracks_raw(
        [ds.base_dir / "raw"], patterns=["*.csv"], src_format="deeplabcut"
    )
    ds.convert_all_tracks()

    assert _entry_names(ds) == ["myseq"]
    index = read_tracks_index(ds)
    assert len(index) == 1
    assert str(index.iloc[0]["group"]) == ""


def test_a_blank_group_survives_a_merge_turned_off(tmp_path: Path) -> None:
    """``merge_per_sequence=False`` takes the early exit, converting row by row.

    A merging format asked not to merge lands in the same single-file branch, so
    the two ways of reaching it have to answer alike.
    """
    ds = make_dataset((tmp_path / "ds").resolve(), roots=_ROOTS)
    _trex_npz(ds.base_dir / "raw" / "myseq_fish0.npz", individual=0)

    ds.index_tracks_raw(
        [ds.base_dir / "raw"], patterns=["*.npz"], src_format="trex_npz"
    )
    ds.convert_all_tracks(merge_per_sequence=False)

    assert _entry_names(ds) == ["myseq"]
    assert str(read_tracks_index(ds).iloc[0]["group"]) == ""


def test_a_merge_still_reads_two_spellings_of_one_group_as_one(
    tmp_path: Path,
) -> None:
    """The merge keys its groupby on these columns, so they need the same rule.

    Not the NaN case -- the reader settles that before this loop sees the frame.
    This is the other half of what the columnwise normalization does, and the
    half that survives fixing the reader: two files of one sequence whose group
    cells differ only by padding have to be one group. Split, each iteration
    names the same output, so the first lands and the second is skipped as
    already written -- half a merge, silently.
    """
    ds = make_dataset((tmp_path / "ds").resolve(), roots=_ROOTS)
    _trex_npz(ds.base_dir / "raw" / "myseq_fish0.npz", individual=0, n=5)
    _trex_npz(ds.base_dir / "raw" / "myseq_fish1.npz", individual=1, n=5)

    ds.index_tracks_raw(
        [ds.base_dir / "raw"], patterns=["*.npz"], src_format="trex_npz"
    )
    raw_index = ds.get_root("tracks_raw") / "index.csv"
    rows = pd.read_csv(raw_index, keep_default_na=False, dtype=str)
    rows["group"] = ["cohort", " cohort "]
    rows.to_csv(raw_index, index=False)

    ds.convert_all_tracks()

    assert _entry_names(ds) == ["cohort__myseq"]
    index = read_tracks_index(ds)
    assert len(index) == 1
    assert str(index.iloc[0]["group"]) == "cohort"
    # Both files' rows are in the table: a split merge would hold only one.
    merged = pd.read_parquet(ds.resolve_path(str(index.iloc[0]["abs_path"])))
    assert len(merged) == 10
    assert int(index.iloc[0]["n_rows"]) == 10


def test_a_file_holding_several_sequences_does_not_borrow_a_blank_group(
    tmp_path: Path,
) -> None:
    """The expansion branch, where the group policy chooses between two sources.

    ``group_from="filename"`` says "prefer the row's group over the in-file one",
    and a NaN spelled ``"nan"`` is truthy -- so the policy fired on a group that
    was not there and overwrote the real one the file carries.
    """
    ds = make_dataset((tmp_path / "ds").resolve(), roots=_ROOTS)
    _calms21_npy(ds.base_dir / "raw" / "task1.npy", {"annotator0": {"seq_a": 4}})

    ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*.npy"],
        src_format="calms21_npy",
        multi_sequences_per_file=True,
    )
    ds.convert_all_tracks(group_from="filename")

    # The blank row group did not displace the group inside the file.
    assert _entry_names(ds) == ["annotator0__seq_a"]
    assert str(read_tracks_index(ds).iloc[0]["group"]) == "annotator0"


def test_an_unhashed_source_records_no_checksum(tmp_path: Path) -> None:
    """``md5`` is the same empty column, and reached the index the same way."""
    ds = make_dataset((tmp_path / "ds").resolve(), roots=_ROOTS)
    _dlc_csv(ds.base_dir / "raw" / "myseq.csv")

    ds.index_tracks_raw(
        [ds.base_dir / "raw"],
        patterns=["*.csv"],
        src_format="deeplabcut",
        compute_md5=False,
    )
    ds.convert_all_tracks()

    assert str(read_tracks_index(ds).iloc[0]["source_md5"]) == ""
