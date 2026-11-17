"""The input size of an entry, read from its input files' footers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.input_size import (
    NO_COVARIATES,
    EntryCovariates,
    EntryInputSize,
    by_rows,
    entry_covariates,
    entry_input_size,
    manifest_input_sizes,
    step_tracks_variants,
    tracks_covariates,
)
from mosaic.core.pipeline.manifest import ManifestEntry
from mosaic.core.pipeline.tracks_index import backfill_frame_extents
from mosaic.core.pipeline.types import ParquetLoadSpec
from tests.helpers import (
    MediaClip,
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
    write_media_index,
)


def _table(path: Path, rows: int, columns: int) -> Path:
    pd.DataFrame({f"c{i}": range(rows) for i in range(columns)}).to_parquet(path)
    return path


def test_an_entry_sums_rows_and_columns_over_its_inputs_and_skips_neighbors(
    tmp_path: Path,
) -> None:
    first = _table(tmp_path / "a.parquet", rows=10, columns=3)
    second = _table(tmp_path / "b.parquet", rows=4, columns=2)
    neighbor = _table(tmp_path / "n.parquet", rows=1000, columns=9)
    entry = ManifestEntry(
        file_specs=[(first, ParquetLoadSpec()), (second, ParquetLoadSpec())],
        prev_file_specs=[(neighbor, ParquetLoadSpec())],
        next_file_specs=[(neighbor, ParquetLoadSpec())],
    )

    assert entry_input_size(entry) == EntryInputSize(rows=14, columns=5)


def test_an_unreadable_footer_counts_no_rows(tmp_path: Path) -> None:
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not a parquet file")
    readable = _table(tmp_path / "a.parquet", rows=6, columns=2)
    entry = ManifestEntry(
        file_specs=[
            (broken, ParquetLoadSpec()),
            (tmp_path / "absent.parquet", ParquetLoadSpec()),
            (readable, ParquetLoadSpec()),
        ]
    )

    assert entry_input_size(entry) == EntryInputSize(rows=6, columns=2)


def test_a_manifest_is_sized_by_entry_key(tmp_path: Path) -> None:
    manifest = {
        "g__a": ManifestEntry(
            file_specs=[(_table(tmp_path / "a.parquet", 3, 1), ParquetLoadSpec())]
        ),
        "g__b": ManifestEntry(
            file_specs=[(_table(tmp_path / "b.parquet", 5, 2), ParquetLoadSpec())]
        ),
    }

    assert manifest_input_sizes(manifest) == {
        "g__a": EntryInputSize(rows=3, columns=1),
        "g__b": EntryInputSize(rows=5, columns=2),
    }


def test_entries_go_largest_first_and_ties_go_by_group_then_sequence() -> None:
    rows = {
        ("g2", "a"): 10,
        ("g1", "b"): 10,
        ("g1", "a"): 10,
        ("g0", "z"): 3,
        ("g9", "y"): 50,
    }

    assert by_rows(rows) == [
        ("g9", "y"),
        ("g1", "a"),
        ("g1", "b"),
        ("g2", "a"),
        ("g0", "z"),
    ]


# --- the covariates of an entry ---------------------------------------------

FIRST = "convert-x.0.1-aaaaaaaaaa"
SECOND = "convert-x.0.1-bbbbbbbbbb"


def _cells(**cells: str) -> dict[str, str]:
    """A tracks index row of *cells*, over a frame extent of 10 frames."""
    return {"frame_min": "10", "frame_max": "19", **cells}


def test_each_covariate_reads_from_its_tracks_row() -> None:
    row = _cells(n_rows="30", n_keypoints="7", pair_frames="12", ids_per_frame="1.5")

    assert entry_covariates(row, fps=25.0) == EntryCovariates(
        ids_per_frame=1.5,
        pair_frames=12,
        pair_frames_bound=False,
        frames=10,
        keypoints=7,
        fps=25.0,
    )


def test_a_blank_pair_frames_reads_its_bound_and_flags_it() -> None:
    covariates = entry_covariates(_cells(n_rows="25"), fps=None)

    assert covariates.pair_frames == 37
    assert covariates.pair_frames_bound
    assert covariates.ids_per_frame == pytest.approx(2.5)
    assert covariates.frames == 10


@pytest.mark.parametrize(
    "cells",
    [
        {"n_rows": ""},
        {"n_rows": "25", "frame_min": "", "frame_max": ""},
    ],
    ids=["no-rows", "no-extent"],
)
def test_a_blank_pair_frames_without_rows_or_frames_reads_no_bound(
    cells: dict[str, str],
) -> None:
    covariates = entry_covariates(_cells(**cells), fps=None)

    assert covariates.pair_frames is None
    assert covariates.ids_per_frame is None
    assert not covariates.pair_frames_bound


def test_the_bound_is_never_negative() -> None:
    covariates = entry_covariates(_cells(n_rows="4"), fps=None)

    assert covariates.pair_frames == 0
    assert covariates.pair_frames_bound
    assert covariates.ids_per_frame == pytest.approx(0.4)


def test_a_row_without_ids_reads_no_bound() -> None:
    row = _cells(n_rows="30", pair_frames="none", ids_per_frame="none")

    covariates = entry_covariates(row, fps=None)

    assert covariates.pair_frames is None
    assert covariates.ids_per_frame is None
    assert not covariates.pair_frames_bound
    assert covariates.frames == 10


def test_an_unmeasured_row_reads_every_covariate_none() -> None:
    assert entry_covariates({}, fps=None) == NO_COVARIATES


def test_an_absent_or_ambiguous_row_reads_every_covariate_none(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    assert tracks_covariates(ds, [("", "s1")], tracks_run_id=None, variants=None) == {
        ("", "s1"): NO_COVARIATES
    }

    add_tracks_variant(ds, FIRST, "both", "alone")
    add_tracks_variant(ds, SECOND, "both")
    entries = [("", "both"), ("", "alone"), ("", "missing")]

    read = tracks_covariates(ds, entries, tracks_run_id=None, variants=None)

    assert read[("", "both")] == NO_COVARIATES
    assert read[("", "missing")] == NO_COVARIATES
    assert read[("", "alone")] == EntryCovariates(
        ids_per_frame=2.0,
        pair_frames=80,
        pair_frames_bound=False,
        frames=40,
        keypoints=7,
        fps=None,
    )


def test_a_pinned_or_narrowed_variant_resolves_an_ambiguous_entry(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, FIRST, "both", n_rows=10)
    add_tracks_variant(ds, SECOND, "both", n_rows=30)
    entry = ("", "both")

    pinned = tracks_covariates(ds, [entry], tracks_run_id=SECOND, variants=(FIRST,))
    narrowed = tracks_covariates(ds, [entry], tracks_run_id=None, variants=(FIRST,))

    assert pinned[entry].frames == 30
    assert narrowed[entry].frames == 10


def test_a_dataset_without_media_reads_no_fps(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_tracks_variant(ds, FIRST, "s1")
    entry = ("", "s1")
    assert (
        tracks_covariates(ds, [entry], tracks_run_id=None, variants=None)[entry].fps
        is None
    )

    write_media_index(ds, [MediaClip(sequence="s1", fps=15.0)])

    assert tracks_covariates(ds, [entry], tracks_run_id=None, variants=None)[
        entry
    ].fps == pytest.approx(15.0)


def _upstream_run(ds_root: Path, storage: str, run_id: str, *run_ids: str) -> None:
    root = ds_root / "features" / storage / run_id
    root.mkdir(parents=True)
    edges = [
        {"where": "inputs[tracks]", "feature": "tracks", "run_id": run}
        for run in run_ids
    ]
    _ = (root / "params.json").write_text(json.dumps({"_resolved": edges}))


def test_the_variants_of_a_feature_input_come_from_its_upstream(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    base = ds.get_root("features").parent
    _upstream_run(base, "up", "r1", SECOND)
    _upstream_run(base, "other", "r2")

    own = ResolvedScope(tracks_variants=(FIRST,), feature_inputs=(("up", "r1"),))
    assert step_tracks_variants(ds, own) == tuple(sorted((FIRST, SECOND)))

    unlabeled = ResolvedScope(feature_inputs=(("other", "r2"),))
    assert step_tracks_variants(ds, unlabeled) == ()

    unknown = ResolvedScope(
        tracks_variants=(FIRST,), feature_inputs=(("up", "r1"), ("gone", "r3"))
    )
    assert step_tracks_variants(ds, unknown) is None
    assert step_tracks_variants(ds, ResolvedScope()) == ()


def test_a_step_reading_the_unlabeled_tables_resolves_to_the_unlabeled_row(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, ("", "s1"), n_rows=20)
    _ = backfill_frame_extents(ds)
    add_tracks_variant(ds, FIRST, "s1", n_rows=40)
    entry = ("", "s1")

    unlabeled_only = tracks_covariates(ds, [entry], tracks_run_id=None, variants=())
    every_variant = tracks_covariates(ds, [entry], tracks_run_id=None, variants=None)

    assert unlabeled_only[entry].frames == 20
    assert every_variant[entry].frames == 40
