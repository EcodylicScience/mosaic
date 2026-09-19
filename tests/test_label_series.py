"""An editor's saved states are versioned, and the rest of ``labels_raw`` is not.

``labels_raw`` follows the truth rule: one current state per sequence, where a
change moves a composition and blocks while derivatives exist. That is wrong for
an annotation tool that saves every time it is closed, and wrong for tying a
trained model to what it saw. A *series* under the same root follows the other
rule: each changed save is a new immutable revision, revisions coexist, and
nothing ever blocks.

The two rules live under one root, so these tests are mostly about the boundary:
that the versioned files never leak into the per-sequence index, that the rule is
declared rather than inferred, and that a revision stays what it was.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from mosaic.core.annotations.model import (
    AnnotationFrame,
    AnnotationObject,
    AnnotationSet,
    Keypoint,
    KeypointSchema,
)
from mosaic.core.annotations.projection import (
    keypoint_set_payload,
    write_keypoint_set_revision,
)
from mosaic.core.annotations.readers.coco import read_coco_keypoints
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LabelsScanSource, read_manifest
from mosaic.core.pipeline.dataset_indexes import iter_dataset_indexes
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.inventory import inventory
from mosaic.core.pipeline.label_series import (
    SERIES_MARKER,
    is_under_label_series,
    series_spec,
)
from mosaic.core.pipeline.label_series_index import (
    LabelSeriesCollisionError,
    LabelSeriesTamperedError,
    read_label_series,
    read_revision_manifest,
    write_series_revision,
)
from mosaic.core.pipeline.tracks_raw_index import iter_track_files
from tests.helpers import make_dataset

SCHEMA = KeypointSchema(names=("nose", "tail"), skeleton=((0, 1),))


def _frame(
    path: str, x: float, *, video: str = "mouse003", index: int = 7
) -> AnnotationFrame:
    obj = AnnotationObject(
        keypoints=(
            Keypoint(x=x, y=2.0, visibility=2),
            Keypoint(x=5.0, y=6.0, visibility=2),
        ),
    )
    return AnnotationFrame(
        image_path=Path(path),
        width=64,
        height=48,
        objects=(obj,),
        video=video,
        frame_index=index,
    )


def _annotations(ds: Dataset, *xs: float) -> AnnotationSet:
    frames = tuple(
        _frame(
            f"media/frames/kmeans/kmeans-d7968c97b0/mouse003/frame_{i:06d}.png",
            x,
            index=i,
        )
        for i, x in enumerate(xs)
    )
    return AnnotationSet(schema=SCHEMA, frames=frames, image_root=ds.base_dir)


def _save(ds: Dataset, *xs: float, key: str = "17-mice") -> int:
    return write_keypoint_set_revision(
        ds, set_key=key, annotations=_annotations(ds, *xs), origin={"dolt_commit": "c"}
    ).revision


# ------------------------------------------------------------------ the writer


def test_a_save_that_changed_nothing_writes_nothing(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")

    first = write_keypoint_set_revision(
        ds,
        set_key="17-mice",
        annotations=_annotations(ds, 1.0),
        origin={"dolt_commit": "a"},
    )
    again = write_keypoint_set_revision(
        ds,
        set_key="17-mice",
        annotations=_annotations(ds, 1.0),
        origin={"dolt_commit": "b"},
    )

    assert (first.revision, first.written) == (1, True)
    assert (again.revision, again.written) == (1, False), "a new origin is not a change"
    assert len(read_label_series(ds, "keypoints")) == 1


def test_a_changed_save_is_a_new_revision_and_the_old_one_is_untouched(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    first = write_keypoint_set_revision(
        ds, set_key="17-mice", annotations=_annotations(ds, 1.0), origin={}
    )
    before = first.path.read_bytes()

    second = write_keypoint_set_revision(
        ds, set_key="17-mice", annotations=_annotations(ds, 1.5), origin={}
    )

    assert second.revision == 2 and second.written
    assert first.path.read_bytes() == before, "a revision is never rewritten"
    assert second.path.parent.name == "rev2"
    assert file_digest(second.path) == second.digest


def test_a_reverted_state_is_a_new_revision_with_the_old_digest(tmp_path: Path) -> None:
    """History is kept, and identity is content: rev3 names what rev1 named."""
    ds = make_dataset(tmp_path / "52", name="project")
    one = write_keypoint_set_revision(
        ds, set_key="k", annotations=_annotations(ds, 1.0), origin={}
    )
    _ = write_keypoint_set_revision(
        ds, set_key="k", annotations=_annotations(ds, 2.0), origin={}
    )
    three = write_keypoint_set_revision(
        ds, set_key="k", annotations=_annotations(ds, 1.0), origin={}
    )

    assert three.revision == 3
    assert three.digest == one.digest


def test_the_payload_ignores_the_order_frames_were_collected_in(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    forward = _annotations(ds, 1.0, 2.0, 3.0)
    backward = forward.with_frames(tuple(reversed(forward.frames)))

    assert keypoint_set_payload(ds, forward) == keypoint_set_payload(ds, backward)


def test_a_revision_names_its_images_relative_to_the_dataset(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    saved = write_keypoint_set_revision(
        ds,
        set_key="17-mice",
        annotations=_annotations(ds, 1.0),
        origin={"dolt_commit": "abc"},
    )

    document = json.loads(saved.path.read_text())
    assert document["images"][0]["file_name"].startswith("media/frames/kmeans/")
    assert document["images"][0]["video"] == "mouse003"

    manifest = read_revision_manifest(saved.path)
    assert manifest.origin == {"dolt_commit": "abc"}
    assert manifest.dataset_uuid == ds.uuid
    assert (saved.path.parent / manifest.image_root).resolve() == ds.base_dir.resolve()

    reread = read_coco_keypoints(saved.path, saved.path.parent / manifest.image_root)
    assert reread.frames[0].video == "mouse003" and reread.frames[0].frame_index == 0


def test_two_saves_landing_together_never_share_a_number(tmp_path: Path) -> None:
    """Several annotators work in one set, so two closes can arrive at once."""
    ds = make_dataset(tmp_path / "52", name="project")
    results: list[int] = []
    errors: list[BaseException] = []

    def save(x: float) -> None:
        try:
            own = Dataset(manifest_path=ds.manifest_path).load()
            results.append(_save(own, x))
        except BaseException as exc:  # noqa: BLE001 - reported by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=save, args=(float(i),)) for i in range(1, 9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert sorted(results) == list(range(1, 9)), "eight states, eight numbers"
    rows = read_label_series(ds, "keypoints")
    assert sorted(int(r) for r in rows["revision"]) == list(range(1, 9)), (
        "and no lost row"
    )


def test_a_number_is_never_reused_after_a_lost_row(tmp_path: Path) -> None:
    """A crash between the directory and the row leaves the number occupied."""
    ds = make_dataset(tmp_path / "52", name="project")
    _ = _save(ds, 1.0)
    orphan = ds.get_root("labels_raw") / "keypoints" / "17-mice" / "rev2"
    orphan.mkdir()

    assert _save(ds, 2.0) == 3


def test_a_folder_that_already_has_the_name_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    squatter = ds.get_root("labels_raw") / "keypoints"
    squatter.mkdir(parents=True)
    _ = (squatter / "session1.csv").write_text("Behavior,Start\n")

    with pytest.raises(LabelSeriesCollisionError, match="not a label series"):
        _ = _save(ds, 1.0)
    assert not (squatter / SERIES_MARKER).exists()


@pytest.mark.parametrize("key", ["", ".", "..", "rev3", "a/b"])
def test_a_key_has_to_name_one_directory(tmp_path: Path, key: str) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    with pytest.raises(ValueError):
        _ = write_series_revision(
            ds, series="keypoints", key=key, payload=b"{}", origin={}, n_records=0
        )


def test_the_rule_is_declared_not_inferred() -> None:
    assert series_spec("keypoints").unit == "set"
    with pytest.raises(KeyError, match="reserved"):
        _ = series_spec("behavior")
    with pytest.raises(KeyError, match="unknown"):
        _ = series_spec("poses")


# ------------------------------------------------------------- the boundary


def test_a_per_sequence_scan_never_indexes_a_series_file(tmp_path: Path) -> None:
    """The leak this prevents: a projected file read as an uploaded label file."""
    ds = make_dataset(tmp_path / "52", name="project")
    _ = _save(ds, 1.0)
    labels_raw = ds.get_root("labels_raw")
    uploaded = labels_raw / "session1.json"
    _ = uploaded.write_text("{}")

    found = [
        path for path, _stat in iter_track_files([labels_raw], ["*.json", "*.csv"])
    ]

    assert found == [uploaded.resolve()] or found == [uploaded]
    assert is_under_label_series(labels_raw / "keypoints" / "index.csv")
    assert is_under_label_series(
        labels_raw / "keypoints" / "17-mice" / "rev1" / "manifest.json"
    )


def test_a_users_own_keypoints_folder_is_still_walked(tmp_path: Path) -> None:
    """Recognized by the marker, never by the name."""
    elsewhere = tmp_path / "exports" / "keypoints"
    elsewhere.mkdir(parents=True)
    mine = elsewhere / "trial.csv"
    _ = mine.write_text("x")

    found = [
        path.name for path, _stat in iter_track_files([tmp_path / "exports"], ["*.csv"])
    ]

    assert found == ["trial.csv"]
    assert not is_under_label_series(mine)


def test_a_series_index_is_visible_to_the_portability_passes(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    assert all("keypoints" not in str(i.path) for i in iter_dataset_indexes(ds))

    _ = _save(ds, 1.0)

    series = [i for i in iter_dataset_indexes(ds) if i.path.parent.name == "keypoints"]
    assert [i.root_key for i in series] == ["labels_raw"]


# ------------------------------------------------------------- claiming revisions


def _library(tmp_path: Path, project: Dataset, *files: str) -> Dataset:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    library.add_scan_source(
        LabelsScanSource(
            id="p52-17",
            path=str(project.get_root("labels_raw") / "keypoints" / "17-mice"),
            files=files,
            series="keypoints",
        )
    )
    _ = library.scan_labels()
    return library


def test_a_library_claims_exactly_the_revisions_it_names(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    for x in (1.0, 2.0, 3.0):
        _ = _save(project, x)

    library = _library(tmp_path, project, "rev2/annotations.coco.json")

    rows = read_label_series(library, "keypoints")
    assert [int(r) for r in rows["revision"]] == [2]
    assert rows.iloc[0]["origin_uuid"] == project.uuid, "which dataset it came from"
    assert rows.iloc[0]["source_id"] == "p52-17"
    assert Path(rows.iloc[0]["abs_path"]).is_absolute(), (
        "a file the library does not own"
    )
    assert (library.get_root("labels_raw") / "keypoints" / SERIES_MARKER).is_file()


def test_the_series_source_survives_a_manifest_round_trip(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    _ = _save(project, 1.0)
    library = _library(tmp_path, project, "rev1/annotations.coco.json")
    library.set_notes("the library of group 7")

    reread = read_manifest(library.manifest_path)

    assert reread.manifest_version == 3
    assert reread.sources.labels[0].series == "keypoints"
    assert reread.sources.labels[0].files == ("rev1/annotations.coco.json",)
    assert reread.notes == "the library of group 7"


def test_a_rescan_replaces_what_the_source_claims_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    for x in (1.0, 2.0):
        _ = _save(project, x)
    library = _library(tmp_path, project, "rev1/annotations.coco.json")
    _ = _save(library, 9.0, key="own-set")

    _ = library.add_source_files("labels", "p52-17", ["rev2/annotations.coco.json"])
    _ = library.scan_labels()

    rows = read_label_series(library, "keypoints")
    assert sorted((r["key"], int(r["revision"])) for _, r in rows.iterrows()) == [
        ("17-mice", 1),
        ("17-mice", 2),
        ("own-set", 1),
    ], "a row the library wrote itself sits under no source, so it survives"


def test_un_importing_a_revision_drops_its_row_from_the_series_index(
    tmp_path: Path,
) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    for x in (1.0, 2.0):
        _ = _save(project, x)
    library = _library(
        tmp_path, project, "rev1/annotations.coco.json", "rev2/annotations.coco.json"
    )

    removed = library.remove_source_files(
        "labels", "p52-17", ["rev1/annotations.coco.json"]
    )

    assert removed == 1
    assert [int(r) for r in read_label_series(library, "keypoints")["revision"]] == [2]


def test_a_revision_edited_after_the_fact_is_refused(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    _ = _save(project, 1.0)
    payload = (
        project.get_root("labels_raw") / "keypoints/17-mice/rev1/annotations.coco.json"
    )
    _ = payload.write_text(payload.read_text().replace("1.0", "1.25"))

    with pytest.raises(LabelSeriesTamperedError, match="immutable"):
        _ = _library(tmp_path, project, "rev1/annotations.coco.json")


def test_a_series_source_refuses_the_per_sequence_knobs() -> None:
    with pytest.raises(ValueError, match="do not apply"):
        _ = LabelsScanSource(
            id="a", path="/x", series="keypoints", src_format="boris_aggregated_csv"
        )
    with pytest.raises(ValueError, match="reserved"):
        _ = LabelsScanSource(id="a", path="/x", series="behavior")


def test_a_series_never_enters_a_sequence_composition(tmp_path: Path) -> None:
    """A composition that moved on every save would block every derivative."""
    ds = make_dataset(tmp_path / "52", name="project")
    sequences = ds.base_dir / "sequences.csv"
    before = sequences.read_bytes() if sequences.exists() else b""

    _ = _save(ds, 1.0)
    _ = _save(ds, 2.0)

    after = sequences.read_bytes() if sequences.exists() else b""
    assert after == before
    assert (
        not (ds.get_root("labels_raw") / "index.csv").exists()
        or "keypoints" not in (ds.get_root("labels_raw") / "index.csv").read_text()
    )


# ------------------------------------------------------------------ the inventory


def test_the_inventory_reports_a_set_and_its_revisions(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    for x in (1.0, 2.0):
        _ = _save(ds, x)

    found = inventory(ds, kinds=["label-series"])

    assert len(found.records) == 1
    record = found.records[0]
    assert record.ref.kind == "label-series"
    assert record.coverage.target == frozenset({"1", "2"})
    assert record.status == "complete"


def test_a_library_sees_a_vanished_project_as_a_row_with_no_file(
    tmp_path: Path,
) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    _ = _save(project, 1.0)
    library = _library(tmp_path, project, "rev1/annotations.coco.json")

    (
        project.get_root("labels_raw") / "keypoints/17-mice/rev1/annotations.coco.json"
    ).unlink()

    record = inventory(library, kinds=["label-series"]).records[0]
    assert record.status != "complete"
    assert record.orphan_rows == frozenset({"1"})
