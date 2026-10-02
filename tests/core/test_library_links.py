"""A model trained in one dataset is named, and found, from another.

A trained model is identified by its run identifier, which is content and carries
no location. Before library links, that name resolved only inside the dataset
that trained the model, so a second dataset could reach the weights by absolute
path alone -- and a path reference mints a different inference identifier from a
run-identifier one, for the same bytes.

A link makes the name resolve across datasets, which is what lets a shared
library serve many projects under one identity.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, LibraryMismatchError
from mosaic.core.manifest import (
    LibraryLink,
    MediaScanSource,
    read_manifest,
)
from mosaic.tracking.model_refs import (
    ModelNotFoundError,
    ModelReferenceRefusedError,
    model_id_for_ref,
    observed_model_source,
    resolve_model,
    resolve_op_model,
)
from tests.helpers import make_dataset, register_trained_model, revision_file

RUN_ID = "train-pose.0.2-abcdef0123"
KIND = "train-pose"
SLEAP_RUN_ID = "train-sleap.0.1-abcdef0123"


def _register(ds: Dataset, *, payload: bytes = b"weights") -> Path:
    """One finished ``train-pose`` run, registered the way the op registers it."""
    weights = ds.get_root("models") / KIND / RUN_ID / "train" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(payload)
    register_trained_model(ds, KIND, RUN_ID, weights)
    return weights


def _register_sleap(ds: Dataset) -> Path:
    """One finished ``train-sleap`` run, whose artifact is a model directory."""
    model_dir = ds.get_root("models") / "train-sleap" / SLEAP_RUN_ID / "model"
    model_dir.mkdir(parents=True)
    weights = model_dir / "best.ckpt"
    _ = weights.write_bytes(b"sleap-weights")
    register_trained_model(
        ds, "train-sleap", SLEAP_RUN_ID, weights, directory=model_dir
    )
    return model_dir


def _library_and_project(tmp_path: Path) -> tuple[Dataset, Dataset]:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")
    return library, project


def test_a_link_records_the_uuid_it_found(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)

    stored = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    assert stored.uuid == library.uuid, "the link pins the dataset it was made to"
    assert stored.added_at
    reread = read_manifest(project.manifest_path)
    assert [link.id for link in reread.libraries] == ["group"]
    assert reread.libraries[0].path == "../libraries/7", "relative, as declared"


def test_a_registered_model_resolves_through_the_link(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    weights = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    resolved = resolve_model(project, RUN_ID, KIND)

    assert resolved.path == weights, "resolved against the library's root, not ours"
    assert resolved.run_id == RUN_ID
    assert resolved.library_id == "group"
    assert resolved.library_uuid == library.uuid


def test_the_identity_is_the_same_from_every_dataset(tmp_path: Path) -> None:
    """The point of naming by run: one model, one identifier, wherever it is read."""
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    in_library = resolve_model(library, RUN_ID, KIND)
    from_project = resolve_model(project, RUN_ID, KIND)

    assert in_library.model_id == from_project.model_id == RUN_ID
    assert in_library.digest == from_project.digest
    assert model_id_for_ref(project, RUN_ID, KIND) == RUN_ID, "and at plan time"


def test_a_models_own_index_wins_over_a_library(tmp_path: Path) -> None:
    """A model copied in beside a library shares its run, so the nearer one serves."""
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    local = _register(project)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    resolved = resolve_model(project, RUN_ID, KIND)

    assert resolved.path == local
    assert resolved.library_id == "", "served locally, so no library is recorded"


def test_provenance_names_the_library_only_when_one_served(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    served = resolve_model(project, RUN_ID, KIND)
    local = resolve_model(library, RUN_ID, KIND)

    assert observed_model_source(served) == {
        "model_source": f"group@{library.uuid}",
        "models": RUN_ID,
    }
    assert observed_model_source(local) == {"models": RUN_ID}
    assert observed_model_source(None, local) == {"models": RUN_ID}


def test_a_run_nobody_registers_says_where_it_looked(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    with pytest.raises(KeyError, match="Linked libraries searched"):
        _ = resolve_model(project, "train-pose.0.2-0000000000", KIND)


# --- resolving an op's model by run id ---------------------------------------


def test_an_op_resolves_its_model_through_the_link(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    weights = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    resolved = resolve_op_model(project, "infer-pose", RUN_ID)

    assert resolved.path == weights
    assert resolved.run_id == RUN_ID
    assert resolved.library_id == "group"
    assert resolved.model_id == resolve_model(project, RUN_ID, KIND).model_id


def test_an_op_resolves_a_model_its_own_dataset_registered(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    weights = _register(project)

    resolved = resolve_op_model(project, "ultralytics", RUN_ID)

    assert resolved.path == weights
    assert resolved.run_id == RUN_ID
    assert resolved.library_id == ""


def test_an_op_accepting_several_kinds_resolves_under_the_references_own(
    tmp_path: Path,
) -> None:
    """TREx takes pose and point models, and a point run is in the points index."""
    project = make_dataset(tmp_path / "52", name="project")
    points = "train-points.0.2-abcdef0123"
    weights = project.get_root("models") / "train-points" / points / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"point-weights")
    register_trained_model(project, "train-points", points, weights)

    resolved = resolve_op_model(project, "trex", points)

    assert resolved.path == weights
    assert resolved.run_id == points


def test_an_op_taking_a_list_resolves_one_run_id(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    model_dir = _register_sleap(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    resolved = resolve_op_model(project, "sleap", SLEAP_RUN_ID)

    assert resolved.paths == [model_dir]
    assert resolved.run_id == SLEAP_RUN_ID
    assert resolved.library_id == "group"


def test_an_op_is_never_handed_a_path(tmp_path: Path) -> None:
    """A path reaches the filesystem, so a caller from outside may not pass one."""
    library, project = _library_and_project(tmp_path)
    weights = _register(library)

    with pytest.raises(ModelReferenceRefusedError, match="not a run id") as caught:
        _ = resolve_op_model(project, "infer-pose", str(weights))

    assert caught.value.reason == "not_a_run_id"
    assert caught.value.accepted_kinds == ("train-pose",)


def test_a_file_named_like_a_run_id_is_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the model indexes are asked, never the working directory."""
    project = make_dataset(tmp_path / "52", name="project")
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / RUN_ID).write_bytes(b"planted")

    with pytest.raises(ModelNotFoundError):
        _ = resolve_op_model(project, "infer-pose", RUN_ID)


def test_an_op_refuses_a_model_another_training_op_produced(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    points = "train-points.0.2-abcdef0123"

    with pytest.raises(ModelReferenceRefusedError, match="train-pose") as caught:
        _ = resolve_op_model(project, "ultralytics", points)

    assert caught.value.reason == "kind_not_accepted"
    assert caught.value.op_kind == "ultralytics"
    assert caught.value.reference == points
    assert caught.value.accepted_kinds == ("train-pose",)


def test_an_op_that_runs_no_model_refuses_every_reference(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")

    with pytest.raises(
        ModelReferenceRefusedError, match="runs no trained model"
    ) as caught:
        _ = resolve_op_model(project, "transcode", RUN_ID)

    assert caught.value.reason == "runs_no_model"
    assert caught.value.accepted_kinds == ()


def test_an_unregistered_run_is_not_found_rather_than_refused(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))
    alone = make_dataset(tmp_path / "53", name="alone")
    missing = "train-pose.0.2-0000000000"

    with pytest.raises(ModelNotFoundError, match="Linked libraries searched") as caught:
        _ = resolve_op_model(project, "infer-pose", missing)
    with pytest.raises(ModelNotFoundError, match="does not exist"):
        _ = resolve_op_model(alone, "infer-pose", missing)

    assert not isinstance(caught.value, ValueError)
    assert caught.value.reference == missing
    assert caught.value.model_kind == "train-pose"


def test_a_row_naming_the_wrong_shape_of_artifact_is_not_found(
    tmp_path: Path,
) -> None:
    """A SLEAP model is a directory, and a row naming a file resolves to none.

    A row written before a model could be a directory names its weights file
    alone.
    """
    project = make_dataset(tmp_path / "52", name="project")
    weights = project.get_root("models") / "train-sleap" / SLEAP_RUN_ID / "best.ckpt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"sleap-weights")
    register_trained_model(project, "train-sleap", SLEAP_RUN_ID, weights)

    with pytest.raises(ModelNotFoundError, match="not a directory") as caught:
        _ = resolve_op_model(project, "sleap", SLEAP_RUN_ID)

    assert caught.value.model_kind == "train-sleap"


def test_a_different_dataset_at_the_linked_path_is_refused(tmp_path: Path) -> None:
    """A library that moved is one repair; another dataset in its place is a fault."""
    _library, project = _library_and_project(tmp_path)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    impostor_root = tmp_path / "libraries" / "7"
    for child in impostor_root.iterdir():
        if child.is_file():
            child.unlink()
    _ = make_dataset(impostor_root, name="impostor")

    fresh = Dataset(manifest_path=project.manifest_path).load()
    with pytest.raises(LibraryMismatchError, match="was linked to dataset"):
        _ = fresh.linked_libraries()


def test_an_unreachable_library_is_skipped_not_fatal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dataset stays usable while a library is offline, as it does for a source."""
    library, project = _library_and_project(tmp_path)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))
    library.manifest_path.unlink()

    fresh = Dataset(manifest_path=project.manifest_path).load()

    assert fresh.linked_libraries() == ()
    assert "is not reachable" in capsys.readouterr().err


def test_a_link_to_nothing_fails_when_it_is_made(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")

    with pytest.raises(FileNotFoundError):
        _ = project.add_library(LibraryLink(id="group", path="../libraries/9"))
    assert project.libraries == ()


def test_a_dataset_cannot_link_itself_or_reuse_an_id(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    with pytest.raises(ValueError, match="already linked"):
        _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))
    with pytest.raises(ValueError, match="cannot link itself"):
        _ = library.add_library(LibraryLink(id="me", path="."))


def test_removing_a_link_stops_resolution_and_touches_nothing(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    weights = _register(library)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    assert project.remove_library("group") is True
    assert project.remove_library("group") is False

    assert weights.exists()
    with pytest.raises(FileNotFoundError):
        _ = resolve_model(project, RUN_ID, KIND)


def test_an_older_reader_keeps_a_links_block_it_does_not_model(tmp_path: Path) -> None:
    """``libraries`` is a top-level key, and unknown top-level keys are preserved."""
    _library, project = _library_and_project(tmp_path)
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    text = project.manifest_path.read_text()
    assert "libraries:" in text
    assert "manifest_version: " in text


def test_a_files_source_no_longer_kills_manifest_edits(tmp_path: Path) -> None:
    """The regression a library depends on: it holds files sources and keeps editing.

    Re-validating through ``model_dump()`` named every source field, defaults
    included, so a file-mode source looked as though it had declared a walk and
    every later edit was refused.
    """
    ds = make_dataset(tmp_path / "52", name="project")
    footage = tmp_path / "footage"
    footage.mkdir()
    ds.add_scan_source(
        MediaScanSource(id="pilot", path=str(footage), files=("trial_03/cam0.mp4",))
    )

    ds.set_notes("anything")
    ds.set_continuous_groups(["session"])

    reread = read_manifest(ds.manifest_path)
    assert reread.notes == "anything"
    assert reread.continuous_groups == ("session",)
    assert reread.sources.media[0].mode == "files"


def test_a_file_source_survives_a_dump_and_revalidate_round_trip() -> None:
    """The round trip itself, which is what made a file source poison a manifest.

    A dump names every field, so ``model_fields_set`` afterwards claims the source
    declared its discovery knobs. A knob still at its default is not a
    declaration, however the instance was built.
    """
    from mosaic.core.manifest import DatasetManifest, LabelsScanSource, ScanSources

    media = MediaScanSource(id="pilot", path="/footage", files=("trial_03/cam0.mp4",))
    series = LabelsScanSource(
        id="p52-17",
        path="/data/52/labels_raw/keypoints/17-mice",
        files=(revision_file(2),),
        series="keypoints",
    )
    manifest = DatasetManifest(sources=ScanSources(media=(media,), labels=(series,)))

    again = DatasetManifest.model_validate(manifest.model_dump())

    assert again.sources.media[0].mode == "files"
    assert again.sources.labels[0].series == "keypoints"


def test_a_file_source_that_really_declares_a_walk_is_still_refused() -> None:
    with pytest.raises(ValueError, match="do not apply"):
        _ = MediaScanSource(
            id="pilot", path="/footage", files=("a.mp4",), extensions=(".mkv",)
        )
