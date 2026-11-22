"""Copying a finished model's run root into another dataset, with its origin.

A copy holds the source's files byte for byte and a finished row whose path cells
name the copy, so the destination resolves the model by its run id. The copied
run root records where it came from. A second copy from that origin changes
nothing, and a copy from any other origin is refused before anything is written.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.markers import EXCHANGE_DIRECTORY, INFLIGHT_MARKER_NAME
from mosaic.core.pipeline.models import (
    MODEL_INDEX_PATH_COLUMNS,
    model_index_path,
    model_run_root,
)
from mosaic.tracking.model_refs import resolve_model
from mosaic.tracking.ops.copy_run_root import (
    ORIGIN_MARKER_NAME,
    CopyRefusalReason,
    CopyRefusedError,
    copy_run_root,
    read_origin_marker,
)
from mosaic.tracking.ops.trained_models import (
    TRAINED_MODEL_INDEX_COLUMNS,
    trained_model_index,
    training_is_complete,
)
from mosaic.tracking.training_provenance import training_provenance
from tests.helpers import (
    assert_tree_unchanged,
    file_digests,
    make_dataset,
    register_trained_model,
    tree_snapshot,
)

KIND: Final = "train-pose"
RUN_ID: Final = "train-pose.0.2-abcdef0123"
PREPARED_KIND: Final = "prepare-training-data"
PREPARED_ID: Final = "prepare-training-data.0.2-2222222222"
COPIED_FILES: Final = (
    ".identity_scheme",
    "train/metrics.csv",
    "train/weights/best.pt",
    "training.json",
)
"""The files of a trained run root that a copy carries, in the order it copies."""


def _weights(ds: Dataset, run_id: str = RUN_ID) -> Path:
    return model_run_root(ds, KIND, run_id) / "train" / "weights" / "best.pt"


def _train(ds: Dataset, run_id: str = RUN_ID, *, data_path: Path | None = None) -> Path:
    """Write one finished ``train-pose`` run root and register it.

    The run root holds what a training leaves, and an in-flight marker and an
    exchange directory, which a copy leaves behind. Without *data_path* the run
    records a preparation under ``models/`` as its data.

    Returns:
        The run root.
    """
    run_root = model_run_root(ds, KIND, run_id)
    weights = _weights(ds, run_id)
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    _ = (run_root / ".identity_scheme").write_text("2\n")
    _ = (run_root / "training.json").write_text('{"epochs": 1}\n')
    _ = (run_root / "train" / "metrics.csv").write_text("epoch,loss\n0,1.5\n")
    _ = (run_root / INFLIGHT_MARKER_NAME).write_text('{"execution_id": "x"}\n')
    exchange = run_root / EXCHANGE_DIRECTORY / "01JEXECUTION"
    exchange.mkdir(parents=True)
    _ = (exchange / "request.json").write_text("{}\n")
    if data_path is None:
        data_path = model_run_root(ds, PREPARED_KIND, PREPARED_ID) / "data.yaml"
        data_path.parent.mkdir(parents=True)
        _ = data_path.write_text("train: train/images\nval: valid/images\n")
    register_trained_model(ds, KIND, run_id, weights, data_path=data_path)
    return run_root


def _row(ds: Dataset, run_id: str = RUN_ID) -> dict[str, str]:
    records = index_records(trained_model_index(model_index_path(ds, KIND)).read())
    (row,) = [record for record in records if record["run_id"] == run_id]
    return row


def _carried(run_root: Path) -> dict[str, str]:
    """The digests of the files under *run_root* that a copy carries."""
    return {
        path: digest
        for path, digest in file_digests(run_root).items()
        if path not in {ORIGIN_MARKER_NAME, INFLIGHT_MARKER_NAME}
        and not path.startswith(f"{EXCHANGE_DIRECTORY}/")
    }


def _staging_left(ds: Dataset) -> list[str]:
    """The staging directories of a copy left beside the run roots of the kind."""
    holder = model_run_root(ds, KIND, RUN_ID).parent
    if not holder.is_dir():
        return []
    return sorted(p.name for p in holder.iterdir() if p.name.startswith(".copy-"))


def _content_digest(run_root: Path, files: tuple[str, ...]) -> str:
    lines = sorted(f"{name}:{file_digest(run_root / name)}" for name in files)
    return hashlib.blake2b("\n".join(lines).encode()).hexdigest()


@pytest.fixture
def src(tmp_path: Path) -> Dataset:
    """The dataset that trained the model."""
    dataset = make_dataset(tmp_path / "src", name="src")
    _ = _train(dataset)
    return dataset


@pytest.fixture
def dst(tmp_path: Path) -> Dataset:
    """A library that holds no model yet."""
    return make_dataset(tmp_path / "library", name="library")


def _refused(
    call: Callable[[], object], reason: CopyRefusalReason, untouched: Dataset
) -> None:
    before = tree_snapshot(untouched.base_dir)
    with pytest.raises(CopyRefusedError) as refused:
        _ = call()
    assert refused.value.reason == reason
    assert_tree_unchanged(untouched.base_dir, before)


# --- A dry run --------------------------------------------------------------------


def test_a_dry_run_names_what_it_would_copy_and_writes_nothing(
    src: Dataset, dst: Dataset
) -> None:
    before = tree_snapshot(dst.base_dir)

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID)

    assert report.outcome == "planned"
    assert (report.kind, report.run_id) == (KIND, RUN_ID)
    assert report.files == COPIED_FILES
    assert_tree_unchanged(dst.base_dir, before)


# --- A copy -----------------------------------------------------------------------


def test_the_copied_files_equal_the_source_without_its_markers(
    src: Dataset, dst: Dataset
) -> None:
    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    copied = model_run_root(dst, KIND, RUN_ID)
    assert report.outcome == "copied"
    assert report.files == COPIED_FILES
    assert _carried(copied) == _carried(model_run_root(src, KIND, RUN_ID))
    assert sorted(file_digests(copied)) == sorted([*COPIED_FILES, ORIGIN_MARKER_NAME])
    assert _staging_left(dst) == []


def test_the_copied_row_is_finished_when_the_source_row_was(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    source_row, copied_row = _row(src), _row(dst)
    assert copied_row["finished_at"]
    assert copied_row["finished_at"] == source_row["finished_at"]
    assert copied_row["started_at"] == source_row["started_at"]
    assert trained_model_index(model_index_path(dst, KIND)).latest_run_id() == RUN_ID
    assert training_is_complete(dst, KIND, RUN_ID)


def test_the_destination_resolves_the_run_id_to_the_copy(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    resolved = resolve_model(dst, RUN_ID, KIND)

    assert resolved.run_id == RUN_ID
    assert resolved.artifacts[0].root.resolve() == _weights(dst).resolve()


def test_the_copied_row_names_the_copy_by_relative_cells(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    source_row, row = _row(src), _row(dst)
    cells = ("abs_path", "best_model_path", "artifact_path", "data_path")
    assert [c for c in cells if Path(row[c]).is_absolute()] == []
    assert dst.resolve_path(row["abs_path"]) == model_run_root(dst, KIND, RUN_ID)
    assert dst.resolve_path(row["best_model_path"]) == _weights(dst)
    assert dst.resolve_path(row["artifact_path"]) == _weights(dst)
    assert Path(row["data_path"]) == Path(
        "models", PREPARED_KIND, PREPARED_ID, "data.yaml"
    )
    assert source_row["metrics_path"] == ""
    assert row["metrics_path"] == ""
    kept = set(source_row) - {*cells, "metrics_path", "data_yaml"}
    assert {c: row[c] for c in kept} == {c: source_row[c] for c in kept}


def test_every_cell_that_holds_no_path_is_carried_over(
    src: Dataset, dst: Dataset
) -> None:
    """Each such column is filled with its own value, so a dropped one shows."""
    named = {"run_id", "abs_path", "started_at", "finished_at", "status", "kind"}
    carried = [
        column
        for column in TRAINED_MODEL_INDEX_COLUMNS
        if column not in named and column not in MODEL_INDEX_PATH_COLUMNS
    ]
    filled = {
        column: "7" if column == "n_epochs" else f"{column}-value" for column in carried
    }
    index = trained_model_index(model_index_path(src, KIND))
    _ = index.update_cells(RUN_ID, filled)

    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    row = _row(dst)
    assert {column: row[column] for column in carried} == filled
    assert (row["kind"], row["status"]) == (KIND, "finished")


def test_an_epoch_count_written_as_a_whole_float_is_copied_as_an_integer(
    src: Dataset, dst: Dataset
) -> None:
    """An older index wrote ``100`` as ``100.0``."""
    index = trained_model_index(model_index_path(src, KIND))
    _ = index.update_cells(RUN_ID, {"n_epochs": "3.0"})

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert report.outcome == "copied"
    assert _row(dst)["n_epochs"] == "3"


@pytest.mark.parametrize("n_epochs", ["many", "2.5"])
def test_a_malformed_epoch_count_is_refused_before_anything_is_written(
    src: Dataset, dst: Dataset, n_epochs: str
) -> None:
    index = trained_model_index(model_index_path(src, KIND))
    _ = index.update_cells(RUN_ID, {"n_epochs": n_epochs})

    for dry_run in (True, False):
        _refused(
            lambda dry_run=dry_run: copy_run_root(
                src, dst, kind=KIND, run_id=RUN_ID, dry_run=dry_run
            ),
            "not_finished",
            dst,
        )


def test_a_data_path_outside_the_source_models_root_is_stored_absolute(
    tmp_path: Path, dst: Dataset
) -> None:
    exported = tmp_path / "exported" / "data.yaml"
    exported.parent.mkdir()
    _ = exported.write_text("train: train/images\n")
    src = make_dataset(tmp_path / "src", name="src")
    _ = _train(src, data_path=exported)

    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert _row(dst)["data_path"] == str(exported.resolve())


def test_the_provenance_of_a_copy_stops_at_the_preparation(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    provenance = training_provenance(dst, KIND, RUN_ID)

    assert provenance.prepared_run_id == PREPARED_ID
    assert (
        provenance.stopped_at == f"{PREPARED_KIND} run {PREPARED_ID} has no index row"
    )


def test_the_origin_marker_names_the_source_and_the_copied_content(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    copied = model_run_root(dst, KIND, RUN_ID)
    marker = read_origin_marker(copied)
    assert marker is not None
    assert (marker.source_name, marker.source_uuid) == ("src", src.uuid)
    assert marker.source_path == str(src.base_dir.resolve())
    assert (marker.run_id, marker.kind) == (RUN_ID, KIND)
    assert marker.copied_at
    assert marker.content_digest == _content_digest(copied, COPIED_FILES)


def test_an_absent_or_unreadable_marker_reads_as_none(tmp_path: Path) -> None:
    assert read_origin_marker(tmp_path) is None
    _ = (tmp_path / ORIGIN_MARKER_NAME).write_text("{not json")
    assert read_origin_marker(tmp_path) is None


# --- The same origin and another ------------------------------------------------


def test_a_second_copy_from_the_same_origin_changes_nothing(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)
    before = tree_snapshot(dst.base_dir)

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)
    planned = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID)

    assert report.outcome == "unchanged"
    assert report.files == ()
    assert planned.outcome == "unchanged"
    assert_tree_unchanged(dst.base_dir, before)


def test_a_copy_from_another_origin_of_the_run_is_refused(
    tmp_path: Path, src: Dataset, dst: Dataset
) -> None:
    other = make_dataset(tmp_path / "other", name="other")
    _ = _train(other)
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    for dry_run in (True, False):
        _refused(
            lambda dry_run=dry_run: copy_run_root(
                other, dst, kind=KIND, run_id=RUN_ID, dry_run=dry_run
            ),
            "other_origin",
            dst,
        )


def test_a_copy_into_a_dataset_that_trained_the_run_is_refused(
    src: Dataset, dst: Dataset
) -> None:
    _ = _train(dst)

    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False),
        "other_origin",
        dst,
    )


def test_a_copy_into_a_dataset_whose_index_holds_the_run_is_refused(
    src: Dataset, dst: Dataset
) -> None:
    """A row with no run root is not a copy's state, since a copy renames first."""
    _ = _train(dst)
    shutil.rmtree(model_run_root(dst, KIND, RUN_ID))

    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False),
        "other_origin",
        dst,
    )


def test_a_copy_of_a_copy_names_the_dataset_it_copied_from(
    tmp_path: Path, src: Dataset
) -> None:
    share = make_dataset(tmp_path / "share", name="share")
    library = make_dataset(tmp_path / "library", name="library")
    _ = copy_run_root(src, share, kind=KIND, run_id=RUN_ID, dry_run=False)

    report = copy_run_root(share, library, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert report.outcome == "copied"
    marker = read_origin_marker(model_run_root(library, KIND, RUN_ID))
    assert marker is not None
    assert (marker.source_name, marker.source_uuid) == ("share", share.uuid)
    assert _carried(model_run_root(library, KIND, RUN_ID)) == _carried(
        model_run_root(src, KIND, RUN_ID)
    )
    again = copy_run_root(share, library, kind=KIND, run_id=RUN_ID, dry_run=False)
    assert again.outcome == "unchanged"
    _refused(
        lambda: copy_run_root(src, library, kind=KIND, run_id=RUN_ID, dry_run=False),
        "other_origin",
        library,
    )


# --- A copy that stopped part way ---------------------------------------------


def test_a_copied_run_root_without_its_row_is_completed(
    src: Dataset, dst: Dataset
) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)
    _ = trained_model_index(model_index_path(dst, KIND)).drop_runs([RUN_ID])
    before = tree_snapshot(dst.base_dir)

    planned = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID)
    assert planned.outcome == "planned"
    assert_tree_unchanged(dst.base_dir, before)

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert report.outcome == "completed"
    assert report.files == ()
    assert _row(dst)["finished_at"] == _row(src)["finished_at"]
    assert _row(dst)["started_at"] == _row(src)["started_at"]
    assert training_is_complete(dst, KIND, RUN_ID)


def test_a_copied_row_left_unfinished_is_completed(src: Dataset, dst: Dataset) -> None:
    _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)
    index = trained_model_index(model_index_path(dst, KIND))
    _ = index.update_cells(RUN_ID, {"finished_at": "", "started_at": "2000-01-01"})

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert report.outcome == "completed"
    assert _row(dst)["finished_at"] == _row(src)["finished_at"]
    assert _row(dst)["started_at"] == _row(src)["started_at"]
    assert index.latest_run_id() == RUN_ID


def test_a_failed_file_copy_leaves_no_staging_and_no_run_root(
    src: Dataset, dst: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", fail)

    with pytest.raises(OSError, match="disk full"):
        _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert _staging_left(dst) == []
    assert not model_run_root(dst, KIND, RUN_ID).exists()
    assert not model_index_path(dst, KIND).exists()


def _lose_the_rename(
    monkeypatch: pytest.MonkeyPatch, publish: Callable[[Path, Path], None]
) -> None:
    """Make the next rename fail after *publish* fills the run root it targets."""
    real_rename = os.rename

    def lose(source: str | Path, target: str | Path) -> None:
        monkeypatch.setattr(os, "rename", real_rename)
        publish(Path(source), Path(target))
        raise OSError(f"{target} is not empty")

    monkeypatch.setattr(os, "rename", lose)


def test_a_rename_lost_to_a_finished_copy_of_the_same_origin_is_unchanged(
    src: Dataset, dst: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other_copy(_staging: Path, _target: Path) -> None:
        _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    _lose_the_rename(monkeypatch, other_copy)

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert report.outcome == "unchanged"
    assert _staging_left(dst) == []
    assert model_run_root(dst, KIND, RUN_ID).is_dir()
    assert training_is_complete(dst, KIND, RUN_ID)


def test_a_rename_lost_to_an_unfinished_copy_of_the_same_origin_completes_it(
    src: Dataset, dst: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    def other_rename(staging: Path, target: Path) -> None:
        _ = shutil.copytree(staging, target)

    _lose_the_rename(monkeypatch, other_rename)

    report = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert report.outcome == "completed"
    assert _staging_left(dst) == []
    assert model_run_root(dst, KIND, RUN_ID).is_dir()
    assert _row(dst)["finished_at"] == _row(src)["finished_at"]


def test_a_rename_lost_to_another_origin_is_refused(
    tmp_path: Path, src: Dataset, dst: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = make_dataset(tmp_path / "other", name="other")
    _ = _train(other)

    def other_copy(_staging: Path, _target: Path) -> None:
        _ = copy_run_root(other, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    _lose_the_rename(monkeypatch, other_copy)

    with pytest.raises(CopyRefusedError) as refused:
        _ = copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False)

    assert refused.value.reason == "other_origin"
    assert _staging_left(dst) == []
    assert model_run_root(dst, KIND, RUN_ID).is_dir()


# --- Refusals ---------------------------------------------------------------------


def _without_uuid(ds: Dataset) -> Dataset:
    ds.manifest = ds.manifest.model_copy(update={"uuid": None})
    return ds


@pytest.mark.parametrize("side", ["src", "dst"])
def test_a_dataset_without_a_uuid_is_refused_before_any_other_check(
    src: Dataset, dst: Dataset, side: str
) -> None:
    _ = _without_uuid(src if side == "src" else dst)

    _refused(
        lambda: copy_run_root(
            src, dst, root_key="tracks", kind="nope", run_id="nope", dry_run=False
        ),
        "no_uuid",
        dst,
    )


def test_a_root_other_than_models_is_refused(src: Dataset, dst: Dataset) -> None:
    _refused(
        lambda: copy_run_root(
            src, dst, root_key="tracks", kind=KIND, run_id=RUN_ID, dry_run=False
        ),
        "unsupported_root",
        dst,
    )


def test_a_kind_of_prepared_data_is_refused(src: Dataset, dst: Dataset) -> None:
    _refused(
        lambda: copy_run_root(
            src, dst, kind=PREPARED_KIND, run_id=PREPARED_ID, dry_run=False
        ),
        "prepared_data",
        dst,
    )


@pytest.mark.parametrize(
    "run_id", ["train-points.0.2-abcdef0123", "not-a-run", "../escape"]
)
def test_a_run_id_of_another_kind_is_refused(
    src: Dataset, dst: Dataset, run_id: str
) -> None:
    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=run_id, dry_run=False),
        "not_a_run_of_kind",
        dst,
    )


def test_a_run_the_source_does_not_register_is_refused(
    src: Dataset, dst: Dataset
) -> None:
    _refused(
        lambda: copy_run_root(
            src, dst, kind=KIND, run_id="train-pose.0.2-0000000000", dry_run=False
        ),
        "not_finished",
        dst,
    )


def test_a_run_that_did_not_finish_is_refused(src: Dataset, dst: Dataset) -> None:
    index = trained_model_index(model_index_path(src, KIND))
    _ = index.update_cells(RUN_ID, {"status": "failed"})

    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False),
        "not_finished",
        dst,
    )


def test_a_finished_run_whose_artifact_is_gone_is_refused(
    src: Dataset, dst: Dataset
) -> None:
    _weights(src).unlink()

    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False),
        "not_finished",
        dst,
    )


def test_a_source_run_root_that_is_a_link_is_refused(
    tmp_path: Path, src: Dataset, dst: Dataset
) -> None:
    run_root = model_run_root(src, KIND, RUN_ID)
    elsewhere = tmp_path / "elsewhere"
    _ = shutil.move(run_root, elsewhere)
    run_root.symlink_to(elsewhere, target_is_directory=True)

    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False),
        "run_root_is_a_link",
        dst,
    )


def test_a_copy_into_the_same_dataset_is_refused(src: Dataset) -> None:
    again = Dataset(manifest_path=src.manifest_path).load()

    _refused(
        lambda: copy_run_root(src, again, kind=KIND, run_id=RUN_ID, dry_run=False),
        "same_dataset",
        src,
    )


def test_a_destination_without_a_models_root_is_refused(
    tmp_path: Path, src: Dataset
) -> None:
    dst = make_dataset(
        tmp_path / "bare", roots=["tracks"], uuid="00000000-0000-4000-8000-000000000001"
    )

    _refused(
        lambda: copy_run_root(src, dst, kind=KIND, run_id=RUN_ID, dry_run=False),
        "no_models_root",
        dst,
    )
