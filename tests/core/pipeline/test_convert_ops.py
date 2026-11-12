"""``convert-tracks`` and ``convert-labels``: a dataset's conversions as ops.

Each names the variant its converter mints from its params, before it runs, and
a run writes exactly that variant. A file that fails is an ``entry_error`` and
fails the attempt, and a rerun converts only what is missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.convert_ops import ConvertLabelsParams, ConvertTracksParams
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.ops import OPS, ScopeRefused, run_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from tests.helpers import (
    calms21_dataset,
    entry_error_lines,
    latest_snapshot,
    make_dataset,
    tree_snapshot,
    unpickling_member,
    write_trex_npz,
)


def _dataset(tmp_path: Path) -> Dataset:
    return calms21_dataset(
        tmp_path / "ds",
        {"train": {"a": {"s1": 4, "s2": 3}}, "test": {"a": {"s3": 5}}},
    )


def _identity(ds: Dataset, kind: str, params: dict[str, object]) -> tuple[str, str]:
    op = OPS[kind]()
    identity = op.plan_identity(
        ds, op.Params.model_validate(params), ds.resolve_scope(None)
    )
    return identity.run_id, identity.tracks_variant or identity.labels_variant


def test_convert_tracks_plan_identity_is_the_converter_variant(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    before = tree_snapshot(ds.base_dir)

    run_id, variant = _identity(ds, "convert-tracks", {"source_format": "calms21_npy"})

    assert tree_snapshot(ds.base_dir) == before
    assert run_id == variant == ds.track_conversion_variant("calms21_npy")


def test_convert_tracks_run_writes_the_planned_variant(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    planned, _ = _identity(ds, "convert-tracks", {"source_format": "calms21_npy"})

    run_id = run_op(ds, "convert-tracks", {"source_format": "calms21_npy"})

    assert run_id == planned
    index = read_tracks_index(ds)
    assert set(index["run_id"]) == {planned}
    assert sorted(zip(index["group"], index["sequence"], strict=True)) == [
        ("test", "s3"),
        ("train", "s1"),
        ("train", "s2"),
    ]


def test_a_failed_file_emits_entry_error_fails_the_attempt_and_a_rerun_resumes(
    tmp_path: Path,
) -> None:
    ds = _dataset(tmp_path)
    bad = ds.base_dir / "raw" / "test.npy"
    good_bytes = bad.read_bytes()
    _ = bad.write_bytes(b"not an npy file")

    with pytest.raises(RuntimeError, match="failed on 1 raw file"):
        _ = run_op(ds, "convert-tracks", {"source_format": "calms21_npy"})

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == 1
    assert any(
        "test.npy" in line for line in entry_error_lines(ds, snapshot["execution_id"])
    )
    assert len(read_tracks_index(ds)) == 2

    _ = bad.write_bytes(good_bytes)
    _ = run_op(ds, "convert-tracks", {"source_format": "calms21_npy"})
    assert len(read_tracks_index(ds)) == 3


def test_a_crafted_trex_file_is_refused_unread_and_its_neighbor_converts(
    tmp_path: Path,
) -> None:
    """A TRex export that would unpickle on load is one failed entry, not a run.

    The refusal is an ``entry_error`` naming the file and the object member. The
    op fails the attempt, the good file's entry is written, and the member's
    unpickling never ran.
    """
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    marker = tmp_path / "unpickled"
    write_trex_npz(raw / "good_fish0.npz")
    crafted = raw / "crafted_fish0.npz"
    write_trex_npz(crafted)
    with np.load(crafted) as data:
        fields = {name: data[name] for name in data.files}
    np.savez(crafted, payload=unpickling_member(marker), **fields)
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")

    with pytest.raises(RuntimeError, match="convert-tracks failed on 1 raw file"):
        _ = run_op(ds, "convert-tracks", {"source_format": "trex_npz"})

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == 1
    [line] = entry_error_lines(ds, snapshot["execution_id"])
    assert "crafted_fish0.npz" in line
    assert "payload" in line
    assert "object" in line
    assert [
        (row["group"], row["sequence"]) for _, row in read_tracks_index(ds).iterrows()
    ] == [("", "good")]
    assert not marker.exists(), "the converter unpickled the crafted member"


def test_an_op_over_files_that_all_fail_raises_all_entries_failed(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True)
    for name in ("a", "b"):
        _ = (raw / f"{name}_fish0.npz").write_bytes(b"not an npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")

    with pytest.raises(AllEntriesFailed, match="every raw file"):
        _ = run_op(ds, "convert-tracks", {"source_format": "trex_npz"})

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "failed"
    assert snapshot["entries_written"] == 0
    assert snapshot["entries_failed"] == 2


def test_an_op_that_converts_some_files_does_not_raise_all_entries_failed(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    write_trex_npz(raw / "good_fish0.npz")
    _ = (raw / "bad_fish0.npz").write_bytes(b"not an npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")

    with pytest.raises(RuntimeError, match="failed on 1 raw file") as raised:
        _ = run_op(ds, "convert-tracks", {"source_format": "trex_npz"})

    assert not isinstance(raised.value, AllEntriesFailed)
    assert latest_snapshot(ds)["entries_written"] == 1


def test_convert_labels_plan_identity_names_the_labels_variant(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    params: dict[str, object] = {"kind": "Behavior", "source_format": "calms21_npy"}

    run_id, variant = _identity(ds, "convert-labels", params)
    written = run_op(ds, "convert-labels", params)

    assert run_id == variant == written
    assert variant == ds.label_conversion_variant("behavior", "calms21_npy")
    assert set(read_labels_index(ds, "behavior")["run_id"]) == {variant}


@pytest.mark.parametrize("kind", ["convert-tracks", "convert-labels"])
def test_convert_ops_refuse_a_scope(tmp_path: Path, kind: str) -> None:
    ds = _dataset(tmp_path)

    with pytest.raises(ScopeRefused, match="takes no entry scope"):
        _ = run_op(
            ds,
            kind,
            {"source_format": "calms21_npy"},
            scope=Scope(entries=[("train", "s1")]),
        )


def test_run_log_records_entries_written_and_variant(tmp_path: Path) -> None:
    """The run id is the variant written, and ``tracks_variant`` is empty.

    ``tracks_variant`` records the tracks variants that a run reads.

    Progress counts the two raw files, which convert into three entries.
    """
    ds = _dataset(tmp_path)

    run_id = run_op(ds, "convert-tracks", {"source_format": "calms21_npy"})

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "finished"
    assert snapshot["run_id"] == run_id
    assert snapshot["entries_written"] == 3
    assert snapshot["tracks_variant"] == ""
    assert snapshot["progress_done"] == snapshot["progress_total"] == 2


def test_convert_labels_run_log_counts_its_raw_files(tmp_path: Path) -> None:
    """Progress counts the two raw files, which convert into three entries."""
    ds = _dataset(tmp_path)
    params: dict[str, object] = {"kind": "Behavior", "source_format": "calms21_npy"}

    run_id = run_op(ds, "convert-labels", params)

    snapshot = latest_snapshot(ds)
    assert snapshot["status"] == "finished"
    assert snapshot["run_id"] == run_id
    assert snapshot["entries_written"] == 3
    assert snapshot["progress_done"] == snapshot["progress_total"] == 2


@pytest.mark.parametrize(
    ("model", "params"),
    [
        (ConvertTracksParams, {"source_format": "calms21_npy", "params": {"nope": 1}}),
        (
            ConvertTracksParams,
            {"source_format": "calms21_npy", "params": {"params_by_format": {}}},
        ),
        (ConvertLabelsParams, {"source_format": "calms21_npy", "params": {"nope": 1}}),
    ],
)
def test_a_key_the_converter_does_not_declare_is_refused(
    model: type[ConvertTracksParams | ConvertLabelsParams], params: dict[str, object]
) -> None:
    with pytest.raises(ValidationError, match="does not declare"):
        _ = model.model_validate(params)
