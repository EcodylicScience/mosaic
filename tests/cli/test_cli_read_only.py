"""Read paths leave the dataset as they found it.

Each case runs one read and compares a snapshot of the dataset tree taken before
it with one taken after. The datasets hold no ``.lock`` sidecar and no empty
directory, so a read that creates a root or takes ``index_lock`` adds a path, and
a read that rewrites an index changes one. The assertion names each.

A new read joins ``CLI_READS`` or ``LIBRARY_READS``, and also
``CLI_READS_OF_NOTHING`` when it succeeds on a dataset that holds nothing yet.
``reprobe-media`` needs indexed videos and has a test of its own. ``features
list``, ``features describe``, ``pipeline catalog``, ``catalog_index``,
``describe_step_type`` and ``step_type_detail`` take no dataset and are not listed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.dataset_discovery import discover_datasets
from mosaic.core.dataset_summary import dataset_summary
from mosaic.core.pipeline.graph import (
    CleanupPolicy,
    FileFailureStore,
    entry_errors_wire,
    list_requests,
    load_request,
    plan_pipeline,
    plan_with_cleanup,
    request_rollup,
    request_status_wire,
    run_pipeline,
    run_status_wire,
)
from mosaic.core.pipeline.inventory import FeatureRunRef
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.preview import preview_artifact
from mosaic.core.pipeline.run_log import read_runs, run_log_dir
from mosaic.runlog import JsonlRunLog, list_runs, run_log_path
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from mosaic.tracking import PrerequisitePlan, plan_prerequisites
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    SPEED_OF_CONVERTED_STEP,
    SPEED_RECIPE,
    calms21_dataset,
    speed_recipe,
    speed_recipe_stepping,
    add_media_sequence,
    add_tracks_variant,
    assert_tree_unchanged,
    invoke_ok,
    make_dataset,
    register_trained_model,
    tree_snapshot,
)

VARIANT = "convert-trex.0.2-1111111111"
MODEL_KIND = "train-pose"
MODEL_RUN_ID = "train-pose.0.1-0123456789"
MODEL_REF = f"trained-model/{MODEL_KIND}/{MODEL_RUN_ID}"
LOSSY_EXECUTION_ID = "01H00000000000000000000000"
SPEED_STORAGE = "speed-angvel__from__tracks"
LOSSY_TARGET = SPEED_STORAGE
LOSSY_RUN_ID = "0.2-abcdef0123"


@dataclass(frozen=True)
class ReadDataset:
    """A dataset under test, and the identifiers its reads name."""

    dataset: Dataset
    recipe_path: Path
    request_id: str = ""
    execution_id: str = ""
    lossy_execution_id: str = ""
    feature_run_id: str = ""

    @property
    def root(self) -> Path:
        return self.dataset.base_dir

    def reopened(self) -> Dataset:
        """The dataset opened the way a reader opens it."""
        return open_dataset(self.dataset.manifest_path, ensure_roots=False)


def _strip_to_content(root: Path) -> None:
    """Remove the lock sidecars and empty directories that a read could recreate."""
    for sidecar in root.rglob("*.lock"):
        sidecar.unlink()
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def _recipe_file(tmp_path: Path) -> Path:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(SPEED_RECIPE), encoding="utf-8")
    return path


@pytest.fixture
def populated(tmp_path: Path) -> ReadDataset:
    """Tracks, a finished feature run, its pipeline request, and a trained model."""
    dataset = make_dataset(tmp_path / "populated")
    add_tracks_variant(dataset, VARIANT, "seq_a", "seq_b", std_format="mosaic_v1")
    done = run_pipeline(dataset, speed_recipe())
    run_root = model_run_root(dataset, MODEL_KIND, MODEL_RUN_ID)
    run_root.mkdir(parents=True)
    weights = run_root / "best.pt"
    weights.write_bytes(b"weights")
    register_trained_model(dataset, MODEL_KIND, MODEL_RUN_ID, weights)
    _lose_an_entity(dataset.base_dir)
    _strip_to_content(dataset.base_dir)
    return ReadDataset(
        dataset=dataset,
        recipe_path=_recipe_file(tmp_path),
        request_id=done.request_id,
        execution_id=done.outcomes[0].execution_id,
        lossy_execution_id=LOSSY_EXECUTION_ID,
        feature_run_id=done.outcomes[0].run_id or "",
    )


def _lose_an_entity(root: Path) -> None:
    """Write an attempt that lost ``seq_b``, and the failure record that counts it."""
    path = run_log_path(root, LOSSY_EXECUTION_ID)
    with JsonlRunLog(path, LOSSY_EXECUTION_ID) as log:
        log.started(kind="feature", target=LOSSY_TARGET)
        log.set_run_id(LOSSY_RUN_ID)
        log.entry_failed("seq_b", '{"type": "ValueError", "message": "bad"}')
        log.finished()
    _ = FileFailureStore(root).note_entry_failure(
        (LOSSY_TARGET, LOSSY_RUN_ID, "", "seq_b"),
        error="bad",
        execution_id=LOSSY_EXECUTION_ID,
    )


@pytest.fixture
def with_media(tmp_path: Path) -> ReadDataset:
    """One sequence of indexed videos."""
    dataset = make_dataset(tmp_path / "with_media")
    add_media_sequence(dataset, "seq_a")
    _strip_to_content(dataset.base_dir)
    return ReadDataset(dataset=dataset, recipe_path=_recipe_file(tmp_path))


@pytest.fixture
def unpopulated(tmp_path: Path) -> ReadDataset:
    """A manifest with no root directory and no index."""
    dataset = make_dataset(tmp_path / "unpopulated")
    _strip_to_content(dataset.base_dir)
    return ReadDataset(dataset=dataset, recipe_path=_recipe_file(tmp_path))


def _other_recipe_file(fixture: ReadDataset) -> Path:
    """A recipe other than the one the fixture ran, which can retire it."""
    path = fixture.recipe_path.parent / "other.recipe.json"
    _ = path.write_text(speed_recipe_stepping(9).model_dump_json(), encoding="utf-8")
    return path


type CliArguments = Callable[[ReadDataset], list[str]]

CLI_READS: dict[str, CliArguments] = {
    "describe": lambda _: ["describe"],
    "inventory": lambda _: ["inventory"],
    "inventory --limit": lambda _: ["inventory", "--limit", "1", "--entry", ":seq_a"],
    "inventory show": lambda _: ["inventory", "show", MODEL_REF],
    "libraries list": lambda _: ["libraries", "list"],
    "measure-tracks --dry-run": lambda _: ["measure-tracks"],
    "models provenance": lambda _: ["models", "provenance", MODEL_RUN_ID],
    "notes show": lambda _: ["notes", "show"],
    "pipeline list": lambda _: ["pipeline", "list", "--limit", "1"],
    "pipeline plan": lambda f: ["pipeline", "plan", "--recipe", f"@{f.recipe_path}"],
    "pipeline plan --step-scope": lambda f: [
        "pipeline",
        "plan",
        "--recipe",
        f"@{f.recipe_path}",
        "--step-scope",
        'speed={"entries": [["", "seq_a"]]}',
    ],
    "pipeline plan --cleanup superseded": lambda f: [
        "pipeline",
        "plan",
        "--recipe",
        f"@{f.recipe_path}",
        "--cleanup",
        "superseded",
        "--min-age-hours",
        "0",
        "--keep-newest",
        "0",
    ],
    "pipeline plan --retire": lambda f: [
        "pipeline",
        "plan",
        "--recipe",
        f"@{_other_recipe_file(f)}",
        "--cleanup",
        "superseded",
        "--min-age-hours",
        "0",
        "--keep-newest",
        "0",
        "--retire",
        load_request(f.root, f.request_id).recipe_digest,
    ],
    "pipeline status": lambda f: ["pipeline", "status", "--request", f.request_id],
    "prerequisites": lambda _: ["prerequisites", "--kind", "trex"],
    "preview": lambda _: ["preview", f"tracks-variant/{VARIANT}", "--entry", "seq_a"],
    "prune-joined --dry-run": lambda _: ["prune-joined"],
    "prune-media --dry-run": lambda _: ["prune-media"],
    "reconcile --dry-run": lambda _: ["reconcile"],
    "reindex --dry-run": lambda _: ["reindex"],
    "runs": lambda _: ["runs"],
    "runs --limit": lambda _: ["runs", "--limit", "1"],
    "runs errors": lambda f: ["runs", "errors", f.lossy_execution_id],
    "sequences": lambda _: ["sequences"],
    "sources list": lambda _: ["sources", "list"],
    "status": lambda f: ["status", "--execution-id", f.execution_id, "--progress"],
    "sweep-tracking --dry-run": lambda _: ["sweep-tracking"],
    "tags list": lambda _: ["tags", "list"],
    "upgrade-tracks --dry-run": lambda _: ["upgrade-tracks"],
}
"""Each read verb's arguments, without ``--manifest`` and ``--json``."""

CLI_READS_OF_NOTHING = (
    "describe",
    "inventory",
    "inventory --limit",
    "libraries list",
    "measure-tracks --dry-run",
    "notes show",
    "pipeline list",
    "prerequisites",
    "prune-joined --dry-run",
    "prune-media --dry-run",
    "reconcile --dry-run",
    "reindex --dry-run",
    "runs",
    "runs --limit",
    "sources list",
    "sweep-tracking --dry-run",
    "tags list",
    "upgrade-tracks --dry-run",
)
"""The verbs of ``CLI_READS`` that succeed on a dataset holding nothing."""


def _invoke_read(fixture: ReadDataset, verb: str) -> None:
    arguments = CLI_READS[verb](fixture)
    manifest = str(fixture.dataset.manifest_path)
    result = invoke_ok([*arguments, "--manifest", manifest, "--json"])
    json.loads(result.stdout)


@pytest.mark.parametrize("verb", sorted(CLI_READS))
def test_a_read_verb_writes_nothing(populated: ReadDataset, verb: str) -> None:
    before = tree_snapshot(populated.root)

    _invoke_read(populated, verb)

    assert_tree_unchanged(populated.root, before)


@pytest.mark.media
def test_a_media_dry_run_writes_nothing(with_media: ReadDataset) -> None:
    before = tree_snapshot(with_media.root)

    manifest = str(with_media.dataset.manifest_path)
    result = invoke_ok(["reprobe-media", "--manifest", manifest, "--json"])
    json.loads(result.stdout)

    assert_tree_unchanged(with_media.root, before)


@pytest.mark.media
def test_planning_the_prerequisites_of_media_writes_nothing(
    with_media: ReadDataset,
) -> None:
    """The plan of a join reads the media index and the joins on disk."""
    before = tree_snapshot(with_media.root)

    plan = _plan_prerequisites(with_media)

    assert [run.kind for run in plan.runs] == ["export-joined"]
    assert_tree_unchanged(with_media.root, before)


@pytest.mark.parametrize("verb", CLI_READS_OF_NOTHING)
def test_a_read_verb_creates_no_root_or_index(
    unpopulated: ReadDataset, verb: str
) -> None:
    before = tree_snapshot(unpopulated.root)

    _invoke_read(unpopulated, verb)

    assert_tree_unchanged(unpopulated.root, before)


def test_planning_a_converting_recipe_writes_no_variant(tmp_path: Path) -> None:
    """The plan names the variants it would convert into and writes neither."""
    dataset = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})
    _strip_to_content(dataset.base_dir)
    recipe = tmp_path / "converting.json"
    document = {
        "schema_version": 1,
        "name": "converting",
        "steps": [CONVERT_TRACKS_STEP, CONVERT_LABELS_STEP, SPEED_OF_CONVERTED_STEP],
    }
    recipe.write_text(json.dumps(document), encoding="utf-8")
    before = tree_snapshot(dataset.base_dir)

    result = invoke_ok(
        [
            "pipeline",
            "plan",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            str(dataset.manifest_path),
            "--json",
        ]
    )

    assert_tree_unchanged(dataset.base_dir, before)
    steps = json.loads(result.stdout)["steps"]
    assert [step["run_id"] is not None for step in steps] == [True, True, True]


def _inventory(fixture: ReadDataset) -> object:
    from mosaic.core.pipeline.inventory import inventory

    return inventory(fixture.reopened())


def _dataset_inventory(fixture: ReadDataset) -> object:
    from mosaic.tracking import dataset_inventory

    return dataset_inventory(fixture.reopened())


def _inventory_wire(fixture: ReadDataset) -> object:
    """A narrowed, paged listing, and one artifact in depth when the dataset has it."""
    from mosaic.core.pipeline.inventory import inventory_wire
    from mosaic.tracking import dataset_inventory

    dataset = fixture.reopened()
    found = dataset_inventory(dataset)
    listed = inventory_wire(
        dataset,
        found,
        scope=Scope(entries=[("", "seq_a")]),
        tracks_run_id=VARIANT,
        status_not=["absent"],
        limit=1,
    )
    if any(record.run_id == MODEL_RUN_ID for record in found.records):
        _ = inventory_wire(dataset, found, ref=MODEL_REF)
    return listed


def _plan_prerequisites(fixture: ReadDataset) -> PrerequisitePlan:
    return plan_prerequisites(fixture.reopened(), "trex", entries=[("", "seq_a")])


def _lossy_entry_errors(fixture: ReadDataset) -> object:
    report = entry_errors_wire(fixture.root, fixture.lossy_execution_id)
    assert report is not None
    assert [entry.attempts for entry in report.entries] == [1]
    return report


def _rollup(fixture: ReadDataset) -> object:
    request = load_request(fixture.root, fixture.request_id)
    return request_rollup(fixture.root, request)


def _request_status(fixture: ReadDataset) -> object:
    request = load_request(fixture.root, fixture.request_id)
    return request_status_wire(fixture.root, request)


type LibraryRead = Callable[[ReadDataset], object]

LIBRARY_READS: dict[str, LibraryRead] = {
    "dataset_inventory": _dataset_inventory,
    "dataset_summary": lambda f: dataset_summary(f.reopened()),
    "discover_datasets": lambda f: discover_datasets(f.root),
    "entry_errors_wire": _lossy_entry_errors,
    "inventory": _inventory,
    "inventory_wire": _inventory_wire,
    "list_requests": lambda f: list_requests(f.root, limit=1),
    "list_runs": lambda f: list_runs(run_log_dir(f.root), limit=1),
    "load_tracks": lambda f: f.reopened().load_tracks("", "seq_a", auto_convert=False),
    "open_dataset": ReadDataset.reopened,
    "plan_pipeline": lambda f: plan_pipeline(f.reopened(), speed_recipe()),
    "plan_prerequisites": _plan_prerequisites,
    "plan_with_cleanup": lambda f: plan_with_cleanup(
        f.reopened(), speed_recipe(), cleanup=CleanupPolicy(min_age_hours=0)
    ),
    "preview_artifact": lambda f: preview_artifact(
        f.reopened(), FeatureRunRef(SPEED_STORAGE, f.feature_run_id), ("", "seq_a")
    ),
    "read_runs": lambda f: read_runs(run_log_dir(f.root)),
    "read_tracks_index": lambda f: read_tracks_index(f.reopened()),
    "request_rollup": _rollup,
    "request_status_wire": _request_status,
    "run_status_wire": lambda f: run_status_wire(f.root, f.execution_id, progress=True),
}


@pytest.mark.parametrize("read", sorted(LIBRARY_READS))
def test_a_library_read_writes_nothing(populated: ReadDataset, read: str) -> None:
    before = tree_snapshot(populated.root)

    LIBRARY_READS[read](populated)

    assert_tree_unchanged(populated.root, before)


@pytest.mark.parametrize(
    "read",
    [
        "dataset_inventory",
        "dataset_summary",
        "discover_datasets",
        "inventory",
        "inventory_wire",
        "list_requests",
        "list_runs",
        "open_dataset",
        "plan_prerequisites",
        "read_runs",
        "read_tracks_index",
    ],
)
def test_a_library_read_of_an_absent_index_creates_nothing(
    unpopulated: ReadDataset, read: str
) -> None:
    before = tree_snapshot(unpopulated.root)

    LIBRARY_READS[read](unpopulated)

    assert_tree_unchanged(unpopulated.root, before)
