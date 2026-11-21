"""A model trained in one dataset is named, and found, from another.

A trained model is identified by its run identifier, which is content and carries
no location. Without a link, that name resolves only inside the dataset that trained
the model, so a second dataset reaches the weights by absolute path alone -- and a
path reference mints a different inference identifier from a run-identifier one, for
the same bytes.

A link makes the name resolve across datasets, which is what lets a shared
library serve many projects under one identity.
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media.transcode import Target

from mosaic.core.dataset import Dataset, LinkMismatchError
from mosaic.core.manifest import (
    MANIFEST_VERSION,
    DatasetLink,
    MediaScanSource,
    read_manifest,
)
from mosaic.core.media.facts_columns import derivative_cell, derivative_path_for_target
from mosaic.core.media.prune import LINK_COLUMNS
from mosaic.core.pipeline.artifact_kinds import LINKED_KINDS, LinkedKind
from mosaic.core.pipeline.composition import (
    labels_raw_composition,
    media_composition,
    tracks_raw_composition,
)
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.media_index import media_members_from_rows, read_media_index
from mosaic.core.pipeline.ops import ModelReference, registered_op
from mosaic.core.pipeline.sequence_index import (
    SourceRoot,
    read_sequence_index,
    read_sequence_labels,
)
from mosaic.core.pipeline.tracks_raw_index import (
    TracksRawIndexScope,
    read_tracks_raw_index,
    source_members_from_rows,
)
from mosaic.tracking import register_ops
from mosaic.tracking.model_refs import (
    ModelNotFoundError,
    ModelReferenceRefusedError,
    model_id_for_ref,
    observed_model_source,
    refuse_unless_run_ids,
    resolve_model,
    resolve_op_model,
)
from tests.helpers import (
    CONVERTED_ENTRIES,
    SUPERSEDED_RECIPE,
    MediaClip,
    assert_tree_unchanged,
    converted_source,
    copy_of,
    link_forms,
    make_dataset,
    raw_source,
    register_sleap_model,
    register_trained_model,
    revision_file,
    tree_snapshot,
    write_media_index,
)

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


def _library_and_project(tmp_path: Path) -> tuple[Dataset, Dataset]:
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    project = make_dataset(tmp_path / "52", name="project")
    return library, project


def test_a_link_records_the_uuid_it_found(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)

    stored = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    assert stored.uuid == library.uuid, "the link pins the dataset it was made to"
    assert stored.added_at
    reread = read_manifest(project.manifest_path)
    assert [link.id for link in reread.links] == ["group"]
    assert reread.links[0].path == "../libraries/7", "relative, as declared"


def test_a_registered_model_resolves_through_the_link(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    weights = _register(library)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    resolved = resolve_model(project, RUN_ID, KIND)

    assert resolved.path == weights, "resolved against the library's root, not ours"
    assert resolved.run_id == RUN_ID
    assert resolved.link_id == "group"
    assert resolved.link_uuid == library.uuid


def test_the_identity_is_the_same_from_every_dataset(tmp_path: Path) -> None:
    """The point of naming by run: one model, one identifier, wherever it is read."""
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

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
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    resolved = resolve_model(project, RUN_ID, KIND)

    assert resolved.path == local
    assert resolved.link_id == "", "served locally, so no library is recorded"


def test_provenance_names_the_library_only_when_one_served(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = _register(library)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

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
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    with pytest.raises(KeyError, match="Linked datasets searched"):
        _ = resolve_model(project, "train-pose.0.2-0000000000", KIND)


# --- resolving an op's model by run id ---------------------------------------


def test_an_op_resolves_its_model_through_the_link(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    weights = _register(library)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    resolved = resolve_op_model(project, "infer-pose", RUN_ID)

    assert resolved.path == weights
    assert resolved.run_id == RUN_ID
    assert resolved.link_id == "group"
    assert resolved.model_id == resolve_model(project, RUN_ID, KIND).model_id


def test_an_op_resolves_a_model_its_own_dataset_registered(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")
    weights = _register(project)

    resolved = resolve_op_model(project, "ultralytics", RUN_ID)

    assert resolved.path == weights
    assert resolved.run_id == RUN_ID
    assert resolved.link_id == ""


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
    model_dir = register_sleap_model(library, SLEAP_RUN_ID, b"sleap-weights")
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    resolved = resolve_op_model(project, "sleap", SLEAP_RUN_ID)

    assert resolved.paths == [model_dir]
    assert resolved.run_id == SLEAP_RUN_ID
    assert resolved.link_id == "group"


def test_an_op_is_never_handed_a_path(tmp_path: Path) -> None:
    """A path reaches the filesystem, so a caller from outside may not pass one."""
    library, project = _library_and_project(tmp_path)
    weights = _register(library)

    with pytest.raises(ModelReferenceRefusedError, match="not a run id") as caught:
        _ = resolve_op_model(project, "infer-pose", str(weights))

    assert caught.value.reason == "not_a_run_id"
    assert caught.value.accepted_kinds == ("train-pose",)


def test_the_first_reference_that_is_not_a_run_id_is_refused() -> None:
    declared = ModelReference(
        field="model", kinds=("train-pose", "train-sleap"), many=True
    )
    refs = [SLEAP_RUN_ID, "models/best.pt", "other.pt"]

    with pytest.raises(ModelReferenceRefusedError) as caught:
        refuse_unless_run_ids("predict-frames", declared, refs)

    assert str(caught.value) == (
        "'models/best.pt' is not a run id. predict-frames takes a model by the run "
        "id of the train-pose or train-sleap run that trained it."
    )
    assert caught.value.reason == "not_a_run_id"
    assert caught.value.op_kind == "predict-frames"
    assert caught.value.reference == "models/best.pt"
    assert caught.value.accepted_kinds == ("train-pose", "train-sleap")


def test_run_ids_of_any_kind_pass_the_run_id_check() -> None:
    """Whether the op accepts each kind is checked later, by ``model_kind_for``."""
    declared = ModelReference(field="model", kinds=("train-pose",), many=True)

    refuse_unless_run_ids("predict-frames", declared, [RUN_ID, SLEAP_RUN_ID])


def test_an_op_and_the_run_id_check_refuse_a_path_alike(tmp_path: Path) -> None:
    """Plan time and the run report a bare weights path as one error."""
    project = make_dataset(tmp_path / "52", name="project")
    register_ops()
    declared = registered_op("infer-pose").model_reference
    assert declared is not None
    ref = str(tmp_path / "best.pt")

    with pytest.raises(ModelReferenceRefusedError) as from_op:
        _ = resolve_op_model(project, "infer-pose", ref)
    with pytest.raises(ModelReferenceRefusedError) as from_check:
        refuse_unless_run_ids("infer-pose", declared, [ref])

    assert error_envelope(from_op.value) == error_envelope(from_check.value)
    assert from_op.value.reason == from_check.value.reason == "not_a_run_id"


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
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )
    alone = make_dataset(tmp_path / "53", name="alone")
    missing = "train-pose.0.2-0000000000"

    with pytest.raises(ModelNotFoundError, match="Linked datasets searched") as caught:
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


# --- the lookup order ---------------------------------------------------------


def _reopened(ds: Dataset) -> Dataset:
    """*ds* loaded again, with nothing opened or cached yet."""
    return Dataset(manifest_path=ds.manifest_path).load()


def _take_offline(ds: Dataset) -> Path:
    """Move *ds*'s manifest aside, as an unmounted share hides it. Returns where."""
    away = ds.manifest_path.with_name(ds.manifest_path.name + ".away")
    _ = ds.manifest_path.rename(away)
    return away


def _bring_back(ds: Dataset, away: Path) -> None:
    _ = away.rename(ds.manifest_path)


def _replace_with_impostor(ds: Dataset) -> None:
    """Put a different dataset, with its own ``uuid``, where *ds* was."""
    for child in ds.base_dir.iterdir():
        if child.is_file():
            child.unlink()
    _ = make_dataset(ds.base_dir, name="impostor")


def test_the_lookup_order_is_this_dataset_then_each_link_in_order(
    tmp_path: Path,
) -> None:
    project = make_dataset(tmp_path / "project", name="project")
    first = make_dataset(tmp_path / "first", name="first")
    second = make_dataset(tmp_path / "second", name="second")
    _ = project.add_link(DatasetLink(id="b-first", path="../first"))
    _ = project.add_link(DatasetLink(id="a-second", path="../second"))

    order = _reopened(project).lookup_order()

    assert [linked.link_id for linked in order] == ["", "b-first", "a-second"]
    assert order[0].link is None
    assert order[0].dataset.uuid == project.uuid
    assert [linked.dataset.uuid for linked in order[1:]] == [first.uuid, second.uuid]
    assert [linked.link.id for linked in order[1:] if linked.link] == [
        "b-first",
        "a-second",
    ]


def test_a_kind_drops_a_link_that_does_not_provide_it_before_opening_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    library, project = _library_and_project(tmp_path)
    _ = make_dataset(tmp_path / "source", name="source")
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )
    _ = project.add_link(DatasetLink(id="source", path="../source"))
    _ = _take_offline(library)
    fresh = _reopened(project)

    order = fresh.lookup_order("features")

    assert [linked.link_id for linked in order] == ["", "source"]
    assert capsys.readouterr().err == "", "a link not asked for is not opened"
    assert fresh.unreachable_links("features") == ()


def test_an_unreachable_link_warns_once_while_it_stays_away(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    library, project = _library_and_project(tmp_path)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models", "features"))
    )
    away = _take_offline(library)
    fresh = _reopened(project)
    (link,) = fresh.links
    warning = (
        f"[links] 'group' is not reachable at {fresh.resolve_link_path(link)}; "
        "its models, features are read from this dataset alone until it is.\n"
    )

    assert [linked.link_id for linked in fresh.lookup_order()] == [""]
    assert capsys.readouterr().err == warning
    assert fresh.unreachable_links() == (link,)
    assert fresh.unreachable_links("models") == (), "models not opened yet"

    assert [linked.link_id for linked in fresh.lookup_order("models")] == [""]
    assert capsys.readouterr().err == "", "one warning while the link stays away"
    assert fresh.unreachable_links("models") == (link,)

    _bring_back(library, away)
    assert [linked.link_id for linked in fresh.lookup_order("models")] == [
        "",
        "group",
    ]
    assert fresh.unreachable_links("models") == ()

    _ = _take_offline(library)
    assert [linked.link_id for linked in fresh.lookup_order()] == [""]
    assert capsys.readouterr().err == warning, "a later outage warns again"


def test_a_fresh_lookup_opens_again_a_link_an_earlier_lookup_kept(
    tmp_path: Path,
) -> None:
    library, project = _library_and_project(tmp_path)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )
    fresh = _reopened(project)
    (link,) = fresh.links
    assert [linked.link_id for linked in fresh.lookup_order("models")] == [
        "",
        "group",
    ]
    away = _take_offline(library)

    assert [linked.link_id for linked in fresh.lookup_order("models")] == [
        "",
        "group",
    ], "a kept lookup is not opened again"
    assert [linked.link_id for linked in fresh.lookup_order("models", fresh=True)] == [
        ""
    ]
    assert fresh.unreachable_links("models") == (link,)
    assert [linked.link_id for linked in fresh.lookup_order("models")] == [""], (
        "a fresh lookup that finds a link away drops the kept one"
    )

    _bring_back(library, away)
    assert [linked.link_id for linked in fresh.lookup_order("models", fresh=True)] == [
        "",
        "group",
    ]
    assert fresh.unreachable_links("models") == ()


@pytest.mark.parametrize("failing", ["resolve_manifest_path", "load"])
def test_any_os_error_on_opening_a_link_reads_as_unreachable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing: str,
) -> None:
    """A dead soft mount raises ``EIO`` and the like, not ``FileNotFoundError``."""
    _library, project = _library_and_project(tmp_path)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )
    fresh = _reopened(project)

    def dead_mount(*_args: object, **_kwargs: object) -> Dataset:
        raise OSError(errno.EIO, "Input/output error")

    if failing == "load":
        monkeypatch.setattr(Dataset, "load", dead_mount)
    else:
        monkeypatch.setattr("mosaic.core.dataset.resolve_manifest_path", dead_mount)

    assert [linked.link_id for linked in fresh.lookup_order("models")] == [""]
    assert [link.id for link in fresh.unreachable_links("models")] == ["group"]
    assert "'group' is not reachable" in capsys.readouterr().err


@pytest.mark.parametrize("kind", [*LINKED_KINDS, None])
def test_a_different_dataset_at_the_linked_path_is_refused(
    tmp_path: Path, kind: LinkedKind | None
) -> None:
    """A library that moved is one repair; another dataset in its place is a fault."""
    library, project = _library_and_project(tmp_path)
    _ = project.add_link(DatasetLink(id="group", path="../libraries/7"))
    _replace_with_impostor(library)
    fresh = _reopened(project)

    with pytest.raises(LinkMismatchError, match="was linked to dataset"):
        _ = fresh.lookup_order(kind)


def test_a_models_link_to_another_dataset_leaves_other_kinds_working(
    tmp_path: Path,
) -> None:
    library, project = _library_and_project(tmp_path)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )
    _replace_with_impostor(library)
    fresh = _reopened(project)

    others: list[LinkedKind] = [kind for kind in LINKED_KINDS if kind != "models"]
    for kind in others:
        assert [linked.link_id for linked in fresh.lookup_order(kind)] == [""]
    with pytest.raises(LinkMismatchError):
        _ = fresh.lookup_order("models")


def test_the_opened_set_is_kept_only_once_every_link_opened(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = project.add_link(DatasetLink(id="group", path="../libraries/7"))
    away = _take_offline(library)
    fresh = _reopened(project)
    assert len(fresh.lookup_order()) == 1

    _bring_back(library, away)
    found = fresh.lookup_order()

    assert [linked.link_id for linked in found] == ["", "group"], "found again"
    assert fresh.lookup_order() is found, "and kept once every link opened"


def test_two_datasets_that_link_each_other_read_each_others_own_data(
    tmp_path: Path,
) -> None:
    left = make_dataset(tmp_path / "left", name="left")
    right = make_dataset(tmp_path / "right", name="right")
    _ = left.add_link(DatasetLink(id="right", path="../right"))
    _ = right.add_link(DatasetLink(id="left", path="../left"))

    for ds, other in ((left, right), (right, left)):
        order = _reopened(ds).lookup_order()
        assert [linked.dataset.uuid for linked in order] == [ds.uuid, other.uuid]
        opened = order[1].dataset
        assert opened.opened_as_link
        assert [linked.dataset for linked in opened.lookup_order()] == [opened]


def test_a_linked_dataset_does_not_follow_its_own_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One level: A links B and B links C, and A's lookups never open C."""
    a = make_dataset(tmp_path / "a", name="a")
    b = make_dataset(tmp_path / "b", name="b")
    _ = make_dataset(tmp_path / "c", name="c")
    _ = b.add_link(DatasetLink(id="c", path="../c"))
    _ = a.add_link(DatasetLink(id="b", path="../b"))
    loaded: list[Path] = []
    load = Dataset.load

    def recording_load(self: Dataset, ensure_roots: bool = True) -> Dataset:
        loaded.append(self.manifest_path)
        return load(self, ensure_roots=ensure_roots)

    monkeypatch.setattr(Dataset, "load", recording_load)

    fresh = _reopened(a)
    assert not fresh.opened_as_link
    order = fresh.lookup_order()
    opened = order[1].dataset

    assert [linked.dataset.uuid for linked in order] == [a.uuid, b.uuid]
    assert opened.opened_as_link
    (alone,) = opened.lookup_order("features")
    assert alone.dataset is opened
    assert alone.link_id == ""
    assert opened.unreachable_links() == ()
    assert [path.resolve() for path in loaded] == [
        a.manifest_path.resolve(),
        b.manifest_path.resolve(),
    ], "C is never opened"


def test_the_enclosing_datasets_are_opened_once_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each manifest found is opened once, and an opened one follows no link."""
    reader = make_dataset(tmp_path / "reader", name="reader")
    owner = make_dataset(tmp_path / "owner", name="owner")
    _ = make_dataset(tmp_path / "other", name="other")
    _ = owner.add_link(DatasetLink(id="other", path="../other"))
    broken = tmp_path / "broken"
    broken.mkdir()
    _ = (broken / "dataset.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    fresh = _reopened(reader)
    loaded: list[Path] = []
    load = Dataset.load

    def recording_load(self: Dataset, ensure_roots: bool = True) -> Dataset:
        loaded.append(self.manifest_path)
        return load(self, ensure_roots=ensure_roots)

    monkeypatch.setattr(Dataset, "load", recording_load)
    first = owner.base_dir / "frames" / "a"
    second = owner.base_dir / "frames" / "b"
    own = fresh.base_dir / "frames" / "c"
    unreadable = broken / "frames"
    outside = tmp_path / "nowhere"

    found = fresh.open_enclosing([first, second, own, unreadable, outside])

    opened = found[first]
    assert opened is not None
    assert opened.uuid == owner.uuid
    assert found[second] is opened
    assert found[own] is fresh
    assert found[unreadable] is None
    assert found[outside] is None
    assert [path.resolve() for path in loaded] == [owner.manifest_path.resolve()]
    assert [linked.dataset for linked in opened.lookup_order()] == [opened]
    assert len(loaded) == 1, "the owner's link is never opened"


def test_a_link_to_nothing_fails_when_it_is_made(tmp_path: Path) -> None:
    project = make_dataset(tmp_path / "52", name="project")

    with pytest.raises(FileNotFoundError):
        _ = project.add_link(
            DatasetLink(id="group", path="../libraries/9", provides=("models",))
        )
    assert project.links == ()


def test_a_dataset_cannot_link_itself_or_reuse_an_id(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    with pytest.raises(ValueError, match="already declared"):
        _ = project.add_link(
            DatasetLink(id="group", path="../libraries/7", provides=("models",))
        )
    with pytest.raises(ValueError, match="cannot link itself"):
        _ = library.add_link(DatasetLink(id="me", path=".", provides=("models",)))


def test_a_dataset_cannot_link_itself_by_path_or_repeat_an_id_without_saving(
    tmp_path: Path,
) -> None:
    _, project = _library_and_project(tmp_path)
    _ = project.add_link(DatasetLink(id="group", path="../libraries/7"))
    saved = project.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="cannot link itself"):
        _ = project.add_link(DatasetLink(id="me", path="."))
    with pytest.raises(ValueError, match="already declared"):
        _ = project.add_link(DatasetLink(id="group", path="../libraries/7"))

    assert project.manifest_path.read_bytes() == saved
    assert [link.id for link in project.links] == ["group"]


def test_a_dataset_cannot_link_itself_by_uuid(tmp_path: Path) -> None:
    """A second dataset that records this dataset's uuid is this dataset."""
    _, project = _library_and_project(tmp_path)
    twin = make_dataset(tmp_path / "twin", name="twin")
    twin.manifest = twin.manifest.model_copy(update={"uuid": project.uuid})
    twin.save()
    saved = project.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="cannot link itself"):
        _ = project.add_link(DatasetLink(id="twin", path="../twin"))

    assert project.links == ()
    assert project.manifest_path.read_bytes() == saved


def test_a_second_link_to_one_uuid_is_refused_without_saving(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    library_uuid = library.uuid
    assert library_uuid
    mirror = make_dataset(tmp_path / "mirror", name="mirror")
    mirror.manifest = mirror.manifest.model_copy(update={"uuid": library_uuid})
    mirror.save()
    _ = project.add_link(DatasetLink(id="group", path="../libraries/7"))
    saved = project.manifest_path.read_bytes()

    with pytest.raises(ValueError, match="already linked"):
        _ = project.add_link(DatasetLink(id="again", path="../mirror"))
    with pytest.raises(ValueError, match="already linked"):
        _ = project.add_link(
            DatasetLink(id="pinned", path="../libraries/7", uuid=library_uuid)
        )

    assert [link.id for link in project.links] == ["group"]
    assert project.manifest_path.read_bytes() == saved


def test_a_relative_link_resolves_against_the_dataset(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    link = DatasetLink(id="group", path="../libraries/7")

    resolved = project.resolve_link_path(link)

    assert resolved.is_absolute()
    assert resolved.resolve() == library.base_dir.resolve()


def test_an_absolute_link_resolves_unchanged(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    link = DatasetLink(id="group", path=str(library.base_dir))

    assert project.resolve_link_path(link) == library.base_dir


def test_removing_a_link_stops_resolution_and_touches_nothing(tmp_path: Path) -> None:
    library, project = _library_and_project(tmp_path)
    weights = _register(library)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    assert project.remove_link("group") is True
    assert project.remove_link("group") is False

    assert weights.exists()
    with pytest.raises(FileNotFoundError):
        _ = resolve_model(project, RUN_ID, KIND)


def test_a_link_is_written_at_the_version_older_readers_refuse(
    tmp_path: Path,
) -> None:
    """Code older than ``links`` would compute into its own tree what a link holds."""
    _library, project = _library_and_project(tmp_path)
    _ = project.add_link(
        DatasetLink(id="group", path="../libraries/7", provides=("models",))
    )

    text = project.manifest_path.read_text()
    assert "links:" in text
    assert "libraries:" not in text
    assert f"manifest_version: {MANIFEST_VERSION}" in text


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
    write_media_index(ds, [MediaClip(group="session", sequence="s1")])
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


# --- raw rows -----------------------------------------------------------------

RAW_ROOTS: tuple[SourceRoot, ...] = ("media_raw", "tracks_raw", "labels_raw")
"""The roots whose per-file index a copy adopts, in write order."""

COMPOSITION_COLUMNS = [
    "group",
    "sequence",
    "composition",
    "member_count",
    "identity_scheme",
]
"""The cells of a ``sequences.csv`` row that a copy carries over.

``computed_at`` records when a row was written and is left out.
"""


def _index_rows(ds: Dataset, root: str) -> list[dict[str, str]]:
    path = ds.get_root(root) / "index.csv"
    if root in ("media", "media_raw"):
        return read_media_index(path)
    return read_tracks_raw_index(path)


def _assert_stored_form(
    cell: str, target: Path, anchor: Path, *, relative: bool
) -> None:
    """*cell* names *target*, relative to *anchor* under a relative link."""
    if relative:
        expected = target.relative_to(anchor.resolve(), walk_up=True).as_posix()
        assert cell == expected, "a relative link stores a path from the copy"
        assert not Path(cell).is_absolute()
    else:
        assert cell == str(target), "an absolute link stores the absolute path"


def _derivative(ds: Dataset, row: dict[str, str], target: Target) -> Path:
    path = derivative_path_for_target(row, target, ds.get_root("media"))
    assert path is not None, f"{row['name']} has no {target} derivative"
    return path.resolve()


def test_adopting_returns_the_rows_written_per_index(tmp_path: Path) -> None:
    source = raw_source(tmp_path / "source")
    copy = make_dataset(tmp_path / "copy")
    _ = copy.add_link(DatasetLink(id="source", path="../source"))

    written = copy.adopt_raw_rows(source)

    assert written == {"media_raw": 2, "tracks_raw": 2, "labels_raw": 2, "media": 4}


@link_forms
def test_adopted_raw_rows_name_the_sources_files(
    tmp_path: Path, relative: bool
) -> None:
    source = raw_source(tmp_path / "source")

    copy = copy_of(source, tmp_path / "copy", relative=relative)

    for root in RAW_ROOTS:
        theirs = _index_rows(source, root)
        ours = _index_rows(copy, root)
        assert len(ours) == len(theirs) == 2, root
        for mine, original in zip(ours, theirs, strict=True):
            target = source.resolve_path(original["abs_path"]).resolve()
            assert copy.resolve_path(mine["abs_path"]).resolve() == target
            _assert_stored_form(
                mine["abs_path"], target, copy.base_dir, relative=relative
            )
            kept = set(original) - {"abs_path", *LINK_COLUMNS}
            assert {k: mine[k] for k in kept} == {k: original[k] for k in kept}
        own = read_sequence_index(copy, root)[COMPOSITION_COLUMNS]
        recorded = read_sequence_index(source, root)[COMPOSITION_COLUMNS]
        pd.testing.assert_frame_equal(own, recorded)

    media = media_members_from_rows(
        [dict(row) for row in _index_rows(copy, "media_raw")]
    )
    tracks = source_members_from_rows(
        [dict(row) for row in _index_rows(copy, "tracks_raw")]
    )
    labels = source_members_from_rows(
        [dict(row) for row in _index_rows(copy, "labels_raw")]
    )
    recomputed: dict[SourceRoot, dict[tuple[str, str], str]] = {
        "media_raw": {key: media_composition(m).digest for key, m in media.items()},
        "tracks_raw": {
            key: tracks_raw_composition(m).digest for key, m in tracks.items()
        },
        "labels_raw": {
            key: labels_raw_composition(m).digest for key, m in labels.items()
        },
    }
    for root, digests in recomputed.items():
        recorded = read_sequence_index(source, root)
        assert digests == {
            (str(row["group"]), str(row["sequence"])): str(row["composition"])
            for _, row in recorded.iterrows()
        }, root
        assert all(digests.values()), f"{root} compositions are established"

    pd.testing.assert_frame_equal(
        read_sequence_labels(copy), read_sequence_labels(source)
    )
    assert copy.display_name("g", "s1") == "Sequence s1"


@link_forms
def test_adopted_derivative_cells_and_rows_name_the_sources_derivatives(
    tmp_path: Path, relative: bool
) -> None:
    source = raw_source(tmp_path / "source")

    copy = copy_of(source, tmp_path / "copy", relative=relative)

    live: set[Path] = set()
    targets: tuple[Target, ...] = ("analysis", "playback")
    for mine, original in zip(
        _index_rows(copy, "media_raw"), _index_rows(source, "media_raw"), strict=True
    ):
        for target in targets:
            theirs = _derivative(source, original, target)
            assert theirs.is_file()
            assert _derivative(copy, mine, target) == theirs
            _assert_stored_form(
                derivative_cell(mine, target),
                theirs,
                copy.get_root("media"),
                relative=relative,
            )
            live.add(theirs)

    rows = _index_rows(copy, "media")
    by_file = {copy.resolve_path(row["abs_path"]).resolve(): row for row in rows}
    assert set(by_file) == live, "one row per live derivative"
    assert SUPERSEDED_RECIPE not in {row["recipe_hash"] for row in rows}
    originals = {
        source.resolve_path(row["abs_path"]).resolve(): row
        for row in _index_rows(source, "media")
    }
    for path, row in by_file.items():
        assert row["recipe_hash"] == originals[path]["recipe_hash"]
        assert row["recipe_hash"]
        _assert_stored_form(row["abs_path"], path, copy.base_dir, relative=relative)


def test_a_link_stored_from_the_home_directory_counts_as_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    source = raw_source(tmp_path / "source")
    copy = make_dataset(tmp_path / "copy")
    _ = copy.add_link(DatasetLink(id="source", path="~/source"))

    _ = copy.adopt_raw_rows(source)

    cells = [row["abs_path"] for root in RAW_ROOTS for row in _index_rows(copy, root)]
    cells += [row["abs_path"] for row in _index_rows(copy, "media")]
    cells += [
        derivative_cell(row, target)
        for row in _index_rows(copy, "media_raw")
        for target in ("analysis", "playback")
    ]
    assert cells
    assert all(Path(cell).is_absolute() for cell in cells), cells


@link_forms
def test_a_raw_file_outside_the_source_stays_absolute(
    tmp_path: Path, relative: bool
) -> None:
    source = raw_source(tmp_path / "source")
    outside = tmp_path / "elsewhere" / "s3"
    outside.mkdir(parents=True)
    _ = (outside / "s3.npy").write_bytes(b"s3")
    _ = source.write_tracks_raw_index(
        [
            TracksRawIndexScope(
                directory=outside, group="g", sequence="s3", src_format="trex_npz"
            )
        ],
        patterns="*.npy",
    )
    stored = {
        row["sequence"]: row["abs_path"] for row in _index_rows(source, "tracks_raw")
    }
    assert Path(stored["s3"]).is_absolute(), "a scan source's file is stored absolute"

    copy = copy_of(source, tmp_path / "copy", relative=relative)

    cells = {
        row["sequence"]: row["abs_path"] for row in _index_rows(copy, "tracks_raw")
    }
    assert cells["s3"] == str((outside / "s3.npy").resolve())
    assert copy.resolve_path(cells["s3"]) == source.resolve_path(stored["s3"])
    assert Path(cells["s1"]).is_absolute() is not relative


def test_entries_adopt_only_the_named_sequences(tmp_path: Path) -> None:
    source = raw_source(tmp_path / "source")

    copy = copy_of(source, tmp_path / "copy", entries={("g", "s1")})

    for root in (*RAW_ROOTS, "media"):
        assert {(row["group"], row["sequence"]) for row in _index_rows(copy, root)} == {
            ("g", "s1")
        }, root
    assert len(_index_rows(copy, "media")) == 2, "the two derivatives of s1"
    for root in RAW_ROOTS:
        recorded = read_sequence_index(source, root)
        expected = recorded[recorded["sequence"] == "s1"][COMPOSITION_COLUMNS]
        pd.testing.assert_frame_equal(
            read_sequence_index(copy, root)[COMPOSITION_COLUMNS],
            expected.reset_index(drop=True),
        )
    labels = read_sequence_labels(copy)
    assert list(zip(labels["group"], labels["sequence"], strict=True)) == [("g", "s1")]


def _refuses_without_writing(copy: Dataset, source: Dataset, match: str) -> None:
    before = tree_snapshot(copy.base_dir)
    source_before = tree_snapshot(source.base_dir)

    with pytest.raises(ValueError, match=match):
        _ = copy.adopt_raw_rows(source)

    assert_tree_unchanged(copy.base_dir, before)
    assert_tree_unchanged(source.base_dir, source_before)


@pytest.mark.parametrize("root", [*RAW_ROOTS, "media"])
def test_adopting_into_an_index_that_has_a_row_is_refused(
    tmp_path: Path, root: str
) -> None:
    source = raw_source(tmp_path / "source")
    copy = make_dataset(tmp_path / "copy")
    _ = copy.add_link(DatasetLink(id="source", path="../source"))
    _ = shutil.copyfile(
        source.get_root(root) / "index.csv", copy.get_root(root) / "index.csv"
    )

    _refuses_without_writing(copy, source, "already has")


def test_adopting_from_a_dataset_with_the_same_uuid_is_refused(tmp_path: Path) -> None:
    source = raw_source(tmp_path / "source")
    copy = make_dataset(tmp_path / "copy", uuid=source.uuid)

    _refuses_without_writing(copy, source, "uuid")


def test_adopting_without_a_link_to_the_source_is_refused(tmp_path: Path) -> None:
    source = raw_source(tmp_path / "source")
    other = make_dataset(tmp_path / "other")
    copy = make_dataset(tmp_path / "copy")
    _ = copy.add_link(DatasetLink(id="other", path="../other"))
    assert other.uuid != source.uuid

    _refuses_without_writing(copy, source, "no link")


@pytest.mark.parametrize("without", ["source", "copy"])
def test_a_dataset_whose_media_root_is_not_media_raw_is_refused(
    tmp_path: Path, without: str
) -> None:
    source = raw_source(tmp_path / "source")
    copy = make_dataset(tmp_path / "copy")
    _ = copy.add_link(DatasetLink(id="source", path="../source"))
    legacy = source if without == "source" else copy
    legacy.roots = {
        key: value for key, value in legacy.roots.items() if key != "media_raw"
    }

    _refuses_without_writing(copy, source, "media_raw")


def test_a_copy_writes_no_label_series_index(tmp_path: Path) -> None:
    source = raw_source(tmp_path / "source")
    assert list(source.get_root("labels_raw").glob("*/index.csv"))

    copy = copy_of(source, tmp_path / "copy")

    assert list(copy.get_root("labels_raw").glob("*/index.csv")) == []


# --- tracks -------------------------------------------------------------------


def test_a_copy_lists_the_entries_and_groups_of_its_sources_tracks(
    tmp_path: Path,
) -> None:
    source = converted_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")

    assert copy.list_groups() == ["g"]
    assert copy.list_entries() == list(CONVERTED_ENTRIES)
    assert copy.list_entries("other") == []


def test_a_copys_sequence_metadata_names_the_sources_tables(tmp_path: Path) -> None:
    source = converted_source(tmp_path / "source")
    copy = copy_of(source, tmp_path / "copy")

    metadata = copy.get_sequence_metadata()

    expected = source.get_sequence_metadata()
    assert list(zip(metadata["group"], metadata["sequence"], strict=True)) == list(
        CONVERTED_ENTRIES
    )
    assert [copy.resolve_path(cell) for cell in metadata["abs_path"]] == [
        source.resolve_path(cell) for cell in expected["abs_path"]
    ]
    assert set(metadata["link_id"]) == {f"source-{source.uuid}"}
