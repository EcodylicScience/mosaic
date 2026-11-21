"""``dataset_inventory``: an inventory that reports the tracking kinds."""

from __future__ import annotations

import ast
from pathlib import Path

from mosaic.core.dataset_summary import dataset_summary
from mosaic.core.manifest import DatasetLink
from mosaic.core.pipeline.models import model_index_path, model_run_root
from mosaic.core.pipeline.inventory import (
    FrameRunRef,
    PreparedDatasetRef,
    TrainedModelRef,
    TracksVariantRef,
)
from mosaic.tracking import dataset_inventory
from tests.helpers import (
    REPO_ROOT,
    add_feature_run,
    add_tracks_variant,
    copy_of,
    extract_entry_camera,
    make_dataset,
    register_trained_model,
)

TRACKING_KINDS = (
    "frame-predictions",
    "frame-run",
    "frame-selection",
    "prediction-score",
    "prepared-dataset",
    "trained-model",
    "tracker-run",
)


def test_every_tracking_kind_is_reported(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "d")
    run_root = model_run_root(ds, "train-pose", "train-pose.0.1-0123456789")
    run_root.mkdir(parents=True)
    weights = run_root / "best.pt"
    weights.write_bytes(b"weights")
    register_trained_model(ds, "train-pose", "train-pose.0.1-0123456789", weights)

    found = dataset_inventory(ds, kinds=TRACKING_KINDS)

    assert found.unavailable_kinds == frozenset()
    ref = TrainedModelRef(op_kind="train-pose", run_id="train-pose.0.1-0123456789")
    assert found.status(ref) == "complete"


def _registered_kinds(source: Path) -> set[str]:
    """Return the kinds that *source* passes to ``register_inventory_contributor``."""
    return {
        node.args[0].value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_inventory_contributor"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def test_one_function_registers_every_tracking_kind() -> None:
    """Read from the source: a registration at module scope elsewhere would make
    a kind depend on which tracking modules a process had imported."""
    package = REPO_ROOT / "src" / "mosaic" / "tracking"
    registering = {
        ".".join(source.relative_to(package.parent.parent).with_suffix("").parts): kinds
        for source in sorted(package.rglob("*.py"))
        if (kinds := _registered_kinds(source))
    }

    assert set(registering) == {"mosaic.tracking.inventory"}
    assert sorted(registering["mosaic.tracking.inventory"]) == sorted(TRACKING_KINDS)


MODEL = "train-pose.0.1-0123456789"
PREPARED = "prepare-training-data.0.1-0123456789"


def _library(base: Path) -> Path:
    """A library with a trained model, a prepared dataset, a feature run and tracks."""
    library = make_dataset(base, name="library")
    run_root = model_run_root(library, "train-pose", MODEL)
    run_root.mkdir(parents=True)
    weights = run_root / "best.pt"
    _ = weights.write_bytes(b"weights")
    register_trained_model(library, "train-pose", MODEL, weights)
    prepared = model_run_root(library, "prepare-training-data", PREPARED)
    prepared.mkdir(parents=True)
    _ = (prepared / "data.yaml").write_text("names: {}\n", encoding="utf-8")
    index = model_index_path(library, "prepare-training-data")
    _ = index.write_text(
        "run_id,status,artifact_path,started_at,finished_at\n"
        f"{PREPARED},finished,{library.relative_to_root(prepared / 'data.yaml')},"
        "t0,t1\n",
        encoding="utf-8",
    )
    _ = add_feature_run(library, "speed", "0.1-1111111111")
    add_tracks_variant(library, "convert-trex.0.2-1111111111", "s1")
    return library.base_dir


def test_a_project_lists_the_models_of_the_library_it_links(tmp_path: Path) -> None:
    library = _library(tmp_path / "library")
    project = make_dataset(tmp_path / "project", name="project")
    _ = project.add_link(
        DatasetLink(id="group", path=str(library), provides=("models",))
    )

    found = dataset_inventory(project)

    model = found.record(TrainedModelRef(op_kind="train-pose", run_id=MODEL))
    prepared = found.record(
        PreparedDatasetRef(op_kind="prepare-training-data", run_id=PREPARED)
    )
    assert model is not None and prepared is not None
    assert (model.link_id, model.status) == ("group", "complete")
    assert (prepared.link_id, prepared.status) == ("group", "complete")
    assert {record.ref.kind for record in found.records if record.link_id} == {
        "trained-model",
        "prepared-dataset",
    }
    summary = dataset_summary(project, found=found)
    assert summary.counts.models == 1


def test_a_copy_names_the_entries_only_its_source_holds(tmp_path: Path) -> None:
    source = make_dataset(tmp_path / "source", name="source")
    tracker = "trex.0.1-aaaaaaaaaa"
    frames = "extract-frames.0.1-bbbbbbbbbb"
    add_tracks_variant(source, tracker, "s1", "s2", std_format="mosaic_v1")
    for sequence in ("s1", "s2"):
        _ = extract_entry_camera(
            source,
            run_id=frames,
            group="",
            sequence=sequence,
            camera="",
            sizes={0: (4, 4)},
        )
    copy = copy_of(source, tmp_path / "copy")
    add_tracks_variant(
        copy, "convert-trex.0.2-1111111111", "s3", std_format="mosaic_v1"
    )

    found = dataset_inventory(copy, kinds=["tracks-variant", "frame-run"])

    variant = found.record(TracksVariantRef(run_id=tracker))
    extracted = found.record(FrameRunRef(method="uniform", run_id=frames))
    assert variant is not None and extracted is not None
    assert variant.linked_only == {("", "s1"), ("", "s2")}
    assert extracted.linked_only == {("", "s1", ""), ("", "s2", "")}
    assert extracted.link_id == f"source-{source.uuid}"
