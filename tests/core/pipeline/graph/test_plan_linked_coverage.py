"""A copy of a dataset plans what its linked dataset computed as computed.

The source runs a conversion, a tracker, an inference op and a scope-free
feature. A copy that links it plans every step complete and runs nothing. A copy
with one sequence beyond the source's asks its tracker and inference steps for
that sequence alone, because each entry's output depends on that entry alone,
and asks an assign-identities step for its whole scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.pipeline.graph import Plan, Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.scope import Scope
from tests.helpers import (
    FakeInference,
    FakeTrex,
    add_raw_track_files,
    add_tracks_variant,
    copy_of,
    install_fake_pose_inference,
    install_fake_trex,
    make_dataset,
    tree_snapshot,
    write_media_index,
)

type Document = dict[str, object]

SEQUENCES = ("s1", "s2")
"""The sequences of the source, each with a video and a raw TREx export."""

ADDED = "s3"
"""The sequence a copy holds beyond the source's."""

GROWN = Scope(entries=[("", sequence) for sequence in (*SEQUENCES, ADDED)])
"""Every entry of a copy that holds :data:`ADDED`.

A plan without a scope covers the entries that hold tracks, and the added
sequence holds none until a tracker runs over it.
"""

COPY_ROOTS = ("tracks", "features", "trex", "infer-pose", "media")
"""The roots of the copy that a step writes into."""


def _recipe(model: Path) -> Recipe:
    """A conversion, a tracker, a pose inference and a feature on the tracker."""
    steps: list[Document] = [
        {
            "id": "convert",
            "type": "op",
            "kind": "convert-tracks",
            "params": {"source_format": "trex_npz"},
        },
        {
            "id": "trex",
            "type": "op",
            "kind": "trex",
            "params": {"track_max_individuals": 2},
        },
        {
            "id": "pose",
            "type": "op",
            "kind": "infer-pose",
            "params": {"model": str(model)},
        },
        {
            "id": "speed",
            "type": "feature",
            "feature": "speed-angvel",
            "inputs": ["tracks"],
            "tracks": {"step": "trex"},
        },
    ]
    return Recipe.model_validate({"steps": steps})


def _assign_identities(variant: str) -> Document:
    """An assign-identities step that relabels *variant* with two identities."""
    return {
        "id": "identities",
        "type": "op",
        "kind": "assign-identities",
        "params": {
            "source_tracks_run_id": variant,
            "references": [
                {"identity": "left", "entry": {"group": "", "sequence": "s1"}},
                {"identity": "right", "entry": {"group": "", "sequence": "s2"}},
            ],
        },
    }


@pytest.fixture
def model(tmp_path: Path) -> Path:
    """A pose model file outside every dataset."""
    path = tmp_path / "weights" / "best.pt"
    path.parent.mkdir(parents=True)
    _ = path.write_bytes(b"weights")
    return path


@dataclass(frozen=True)
class Tools:
    """The fake tracker and pose inference, which record each video they read."""

    trex: FakeTrex
    pose: FakeInference


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> Tools:
    """Install the fake tracker and pose inference."""
    return Tools(
        trex=install_fake_trex(monkeypatch),
        pose=install_fake_pose_inference(monkeypatch),
    )


@pytest.fixture
def source(tmp_path: Path, model: Path, tools: Tools) -> Dataset:
    """A dataset over :data:`SEQUENCES` on which every step of the recipe ran."""
    _ = tools
    dataset = make_dataset(tmp_path / "source")
    write_media_index(
        dataset,
        list(SEQUENCES),
        uids={sequence: f"uid-{sequence}" for sequence in SEQUENCES},
    )
    add_raw_track_files(dataset, trex=list(SEQUENCES))
    _ = run_pipeline(dataset, _recipe(model))
    return dataset


def _add_sequence(dataset: Dataset, sequence: str) -> None:
    """Index one more stub video in *dataset*, beside the rows it holds."""
    index = dataset.get_root(dataset.resolve_media_root()) / "index.csv"
    rows: list[dict[str, object]] = [dict(row) for row in read_media_index(index)]
    write_media_index(dataset, [sequence], uids={sequence: f"uid-{sequence}"})
    rows.extend(dict(row) for row in read_media_index(index))
    write_media_index_rows(index, frame_from_rows(rows))


def _asked(plan: Plan, step_id: str) -> list[Entry]:
    """The entries the plan asks *step_id* for."""
    return sorted(plan.step(step_id).spec.entries.entry_pairs or ())


def test_a_copy_plans_every_step_its_source_ran_as_complete(
    source: Dataset, model: Path, tmp_path: Path
) -> None:
    copy = copy_of(source, tmp_path / "copy")

    plan = plan_pipeline(copy, _recipe(model))

    assert {step.step_id: step.status for step in plan.steps} == {
        "convert": "complete",
        "trex": "complete",
        "pose": "complete",
        "speed": "complete",
    }
    assert plan.is_complete


def test_running_the_recipe_on_a_copy_runs_no_step_and_writes_no_output(
    source: Dataset, model: Path, tmp_path: Path
) -> None:
    copy = copy_of(source, tmp_path / "copy")
    roots = {name: copy.get_root(name) for name in COPY_ROOTS}
    before = {
        name: tree_snapshot(root) if root.exists() else None
        for name, root in roots.items()
    }

    run = run_pipeline(copy, _recipe(model))

    assert run.ran == ()
    after = {
        name: tree_snapshot(root) if root.exists() else None
        for name, root in roots.items()
    }
    assert after == before
    assert read_tracks_index(copy).empty


def test_a_tracker_and_an_inference_on_a_grown_copy_are_asked_for_the_added_sequence(
    source: Dataset, model: Path, tmp_path: Path
) -> None:
    copy = copy_of(source, tmp_path / "copy")
    _add_sequence(copy, ADDED)

    plan = plan_pipeline(copy, _recipe(model), scope=GROWN)

    assert _asked(plan, "trex") == [("", ADDED)]
    assert _asked(plan, "pose") == [("", ADDED)]


def test_a_tracker_and_an_inference_on_a_grown_copy_read_the_added_video_alone(
    source: Dataset, model: Path, tools: Tools, tmp_path: Path
) -> None:
    copy = copy_of(source, tmp_path / "copy")
    _add_sequence(copy, ADDED)
    tools.trex.converted.clear()
    tools.pose.videos.clear()

    _ = run_pipeline(copy, _recipe(model), scope=GROWN)

    added = f"{ADDED}.mp4"
    assert [video.name for video in tools.trex.converted] == [added]
    assert [video.name for video in tools.pose.videos] == [added]
    held = read_tracks_index(copy)
    assert set(held["sequence"]) == {ADDED}


def test_an_assign_identities_step_on_a_grown_copy_is_asked_for_its_whole_scope(
    source: Dataset, model: Path, tmp_path: Path
) -> None:
    trex = plan_pipeline(source, _recipe(model)).step("trex")
    assert trex.tracks_variant is not None
    recipe = Recipe.model_validate({"steps": [_assign_identities(trex.tracks_variant)]})
    planned = plan_pipeline(source, recipe).step("identities")
    assert planned.tracks_variant is not None
    add_tracks_variant(
        source, planned.tracks_variant, *SEQUENCES, std_format="mosaic_v1"
    )
    copy = copy_of(source, tmp_path / "copy")
    _add_sequence(copy, ADDED)

    plan = plan_pipeline(copy, recipe, scope=GROWN)

    assert plan.step("identities").status == "partial"
    assert _asked(plan, "identities") == sorted(GROWN.entry_pairs or ())
