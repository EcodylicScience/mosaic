"""``manage_dataset``: datasets created, sourced, scanned and tagged through the MCP.

The operator's ``dataset_roots`` and ``source_roots`` bound every path, resolved
through symbolic links and ``..``. Creating a dataset or changing its sources asks
to confirm. A created dataset is served at once, and no action touches ``splits``.
"""

from __future__ import annotations

import os
from pathlib import Path

import anyio
import pytest

from mosaic.agent.mcp.local import LocalBackend
from mosaic.agent.mcp.spec import DatasetChange, DatasetList, call_tool
from mosaic.core.arenas import ArenaLayout
from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.dataset_summary import DatasetSummary
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import PlanWire
from tests.helpers import (
    add_track_sequences,
    CONVERT_TRACKS_STEP,
    SPEED_OF_CONVERTED_STEP,
    RecordingStarter,
    calms21_dataset,
    write_calms21_npy,
)

CALMS21 = {
    "kind": "tracks",
    "patterns": ["*.npy"],
    "src_format": "calms21_npy",
    "multi_sequences_per_file": True,
    "group_from": "filename",
}
"""The fields of a CalMS21 tracks source."""


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    """A dataset root, and a source root holding two CalMS21 files."""
    datasets = tmp_path / "datasets"
    sources = tmp_path / "sources"
    datasets.mkdir()
    write_calms21_npy(
        sources / "calms" / "train.npy", {"a": {"s1": 20, "s2": 20}}, seed=1
    )
    write_calms21_npy(sources / "calms" / "test.npy", {"a": {"s3": 20}}, seed=2)
    return datasets, sources


@pytest.fixture
def backend(roots: tuple[Path, Path]) -> LocalBackend:
    datasets, sources = roots
    return LocalBackend(
        [],
        warm=False,
        start_driver=RecordingStarter(),
        dataset_roots=[datasets],
        source_roots=[sources],
    )


def _tool[T](
    backend: LocalBackend, expected: type[T], tool: str, /, **arguments: object
) -> T:
    answered = call_tool(backend, tool, arguments)
    assert isinstance(answered, expected), answered
    return answered


def _call[T](backend: LocalBackend, expected: type[T], /, **arguments: object) -> T:
    return _tool(backend, expected, "manage_dataset", **arguments)


def _created(backend: LocalBackend, name: str = "study") -> DatasetChange:
    return _call(
        backend, DatasetChange, action="create", path=name, name=name, confirm=True
    )


def _sourced(backend: LocalBackend, uuid: str, **extra: object) -> DatasetChange:
    return _call(
        backend,
        DatasetChange,
        action="add_source",
        dataset=uuid,
        path="calms",
        source_id="calms21",
        confirm=True,
        **{**CALMS21, **extra},
    )


def test_creating_a_dataset_asks_and_serves_it_at_once(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    asked = _call(backend, ErrorEnvelope, action="create", path="study", name="study")
    created = _created(backend)
    listed = _tool(backend, DatasetList, "list_datasets")

    assert asked.reason == "confirmation_required"
    assert asked.detail is not None
    assert asked.detail["conditions"] == ["create_dataset"]
    assert not (roots[0] / "study").exists() or created.dataset
    assert created.manifest == str(roots[0] / "study" / "dataset.yaml")
    assert [item.uuid for item in listed.items] == [created.dataset]
    assert created.next_steps and "declare a source" in created.next_steps[0]


def test_a_dataset_made_below_a_root_by_another_process_is_served(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    other = calms21_dataset(
        roots[0] / "made-elsewhere", {"g": {"a": {"s": 20}}}, seed=3
    )

    listed = _tool(backend, DatasetList, "list_datasets")

    assert [item.uuid for item in listed.items] == [other.manifest.uuid]


def test_scanning_a_declared_source_leaves_a_convert_step_to_do_and_a_plan_reaches_it(
    backend: LocalBackend,
) -> None:
    """The acceptance: a folder of .npy files and a root reach a plan, no operator."""
    uuid = _created(backend).dataset
    sourced = _sourced(backend, uuid)
    scanned = _call(backend, DatasetChange, action="scan", dataset=uuid)
    described = _tool(backend, DatasetSummary, "describe_dataset", dataset=uuid)
    plan = _tool(
        backend,
        PlanWire,
        "plan_recipe",
        dataset=uuid,
        recipe={
            "schema_version": 1,
            "name": "speed",
            "steps": [CONVERT_TRACKS_STEP, SPEED_OF_CONVERTED_STEP],
        },
    )

    assert sourced.next_steps and "scan" in sourced.next_steps[0]
    assert scanned.scanned == {"tracks": 2}
    assert scanned.next_steps and "convert-tracks" in scanned.next_steps[0]
    assert described.next_steps == scanned.next_steps
    assert plan.problems == ()
    assert plan.fingerprint


def test_the_whole_session_runs_over_the_mcp_with_the_user_asked(
    backend: LocalBackend,
) -> None:
    from mcp.client import Client, ClientRequestContext
    import mcp_types as types

    from mosaic.agent.mcp.server import build_server

    asked: list[str] = []

    async def accept(
        context: ClientRequestContext, params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        _ = context
        asked.append(params.message.splitlines()[-1])
        return types.ElicitResult(action="accept", content={"confirm": True})

    async def run() -> list[types.CallToolResult]:
        async with Client(build_server(backend), elicitation_callback=accept) as client:
            created = await client.call_tool(
                "manage_dataset", {"action": "create", "path": "study"}
            )
            uuid = created.structured_content["dataset"]  # pyright: ignore[reportOptionalSubscript]
            sourced = await client.call_tool(
                "manage_dataset",
                {"action": "add_source", "dataset": uuid, "path": "calms", **CALMS21},
            )
            scanned = await client.call_tool(
                "manage_dataset", {"action": "scan", "dataset": uuid}
            )
            return [created, sourced, scanned]

    results = anyio.run(run)

    assert [result.is_error for result in results] == [False, False, False]
    assert asked == [
        "[mosaic-confirm tool=manage_dataset conditions=create_dataset]",
        "[mosaic-confirm tool=manage_dataset conditions=add_source]",
    ]


@pytest.mark.parametrize("path", ["../escaped", "/tmp", "study/../../escaped"])
def test_a_dataset_outside_the_roots_is_refused(
    backend: LocalBackend, path: str
) -> None:
    refused = _call(backend, ErrorEnvelope, action="create", path=path, confirm=True)

    assert refused.reason == "outside_roots"


def test_a_symbolic_link_out_of_a_root_is_refused(
    backend: LocalBackend, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    (outside / "calms").mkdir(parents=True)
    os.symlink(outside, roots[0] / "link")
    os.symlink(outside / "calms", roots[1] / "linked")
    uuid = _created(backend).dataset

    created = _call(
        backend, ErrorEnvelope, action="create", path="link/inner", confirm=True
    )
    sourced = _call(
        backend,
        ErrorEnvelope,
        action="add_source",
        dataset=uuid,
        path="linked",
        confirm=True,
        **CALMS21,
    )

    assert created.reason == sourced.reason == "outside_roots"


def test_without_roots_nothing_is_created_or_declared(
    roots: tuple[Path, Path], tmp_path: Path
) -> None:
    ds = calms21_dataset(tmp_path / "served", {"g": {"a": {"s": 20}}}, seed=3)
    backend = LocalBackend([ds.manifest_path], warm=False)

    created = _call(
        backend, ErrorEnvelope, action="create", path=str(roots[0] / "x"), confirm=True
    )
    sourced = _call(
        backend,
        ErrorEnvelope,
        action="add_source",
        dataset=ds.manifest.uuid,
        path=str(roots[1] / "calms"),
        confirm=True,
        **CALMS21,
    )

    assert created.reason == sourced.reason == "roots_not_configured"


def test_a_directory_that_is_not_empty_is_not_made_a_dataset(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    (roots[0] / "taken").mkdir()
    _ = (roots[0] / "taken" / "file.txt").write_text("x")

    refused = _call(backend, ErrorEnvelope, action="create", path="taken", confirm=True)

    assert refused.reason == "dataset_exists"


def test_a_source_change_asks_and_a_removed_source_keeps_its_rows(
    backend: LocalBackend,
) -> None:
    uuid = _created(backend).dataset
    asked = _call(
        backend,
        ErrorEnvelope,
        action="add_source",
        dataset=uuid,
        path="calms",
        **CALMS21,
    )
    _ = _sourced(backend, uuid)
    _ = _call(backend, DatasetChange, action="scan", dataset=uuid)
    asked_again = _call(
        backend,
        ErrorEnvelope,
        action="remove_source",
        dataset=uuid,
        kind="tracks",
        source_id="calms21",
    )
    removed = _call(
        backend,
        DatasetChange,
        action="remove_source",
        dataset=uuid,
        kind="tracks",
        source_id="calms21",
        confirm=True,
    )

    assert asked.reason == asked_again.reason == "confirmation_required"
    assert removed.orphaned_rows == 2
    assert not open_dataset(Path(removed.manifest)).manifest.sources.tracks


def test_a_field_of_another_kind_is_refused(backend: LocalBackend) -> None:
    uuid = _created(backend).dataset

    refused = _call(
        backend,
        ErrorEnvelope,
        action="add_source",
        dataset=uuid,
        path="calms",
        confirm=True,
        **{**CALMS21, "extensions": [".mp4"]},
    )

    assert refused.reason == "invalid_source"
    assert "extensions" in refused.message


def test_tags_are_defined_by_their_value_set_and_removed(backend: LocalBackend) -> None:
    uuid = _created(backend).dataset

    _ = _call(
        backend,
        DatasetChange,
        action="set_tags",
        dataset=uuid,
        tags={"species": "mouse", "animals": 2, "pilot": None},
    )
    _ = _call(
        backend,
        DatasetChange,
        action="set_tags",
        dataset=uuid,
        tags={"animals": 3},
        remove_tags=["pilot"],
    )
    mistyped = _call(
        backend, ErrorEnvelope, action="set_tags", dataset=uuid, tags={"animals": "x"}
    )
    summary = _tool(backend, DatasetSummary, "describe_dataset", dataset=uuid)

    assert {(tag.name, tag.type, tag.value) for tag in summary.tags} == {
        ("species", "text", "mouse"),
        ("animals", "int", 3),
    }
    assert mistyped.reason == "invalid_tag"


def test_set_arenas_declares_a_recording_s_layout(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _created(backend).dataset
    dataset = open_dataset(roots[0] / "study" / "dataset.yaml")
    add_track_sequences(dataset, ("", "pilot"))
    layout = ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 100))

    changed = _call(
        backend,
        DatasetChange,
        action="set_arenas",
        dataset=uuid,
        recording="pilot",
        arena_layout=layout.model_dump(mode="json"),
    )
    unknown = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        recording="nowhere",
        arena_layout=layout.model_dump(mode="json"),
    )
    summary = _tool(backend, DatasetSummary, "describe_dataset", dataset=uuid)

    assert changed.message == "set 2 arenas of 'pilot'"
    assert unknown.reason == "invalid_arenas"
    assert [
        (item.recording, item.camera, item.arenas, item.digest)
        for item in summary.arena_layouts
    ] == [("pilot", "", ("A1", "B1"), layout.digest())]


def test_no_action_moves_the_operators_splits(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _created(backend).dataset
    manifest = roots[0] / "study" / "dataset.yaml"
    Dataset(manifest).load().set_splits("sealed", ["test"])

    _ = _sourced(backend, uuid)
    _ = _call(backend, DatasetChange, action="scan", dataset=uuid)
    _ = _call(backend, DatasetChange, action="set_tags", dataset=uuid, tags={"a": 1})

    assert open_dataset(manifest).manifest.splits == {"test": "sealed"}


def test_a_managed_dataset_is_not_changed(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _created(backend).dataset
    with (roots[0] / "study" / "dataset.yaml").open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")

    refused = _call(
        backend, ErrorEnvelope, action="set_tags", dataset=uuid, tags={"a": 1}
    )

    assert refused.reason == "dataset_managed"


def test_an_action_without_its_field_names_it(backend: LocalBackend) -> None:
    refused = _call(backend, ErrorEnvelope, action="scan")
    no_path = _call(backend, ErrorEnvelope, action="create", confirm=True)

    assert (refused.reason, refused.detail) == (
        "field_required",
        {"action": "scan", "field": "dataset"},
    )
    assert no_path.reason == "field_required"


def test_a_plan_over_a_dataset_with_no_tracks_says_to_convert_first(
    backend: LocalBackend,
) -> None:
    uuid = _created(backend).dataset
    _ = _sourced(backend, uuid)
    _ = _call(backend, DatasetChange, action="scan", dataset=uuid)

    plan = _tool(
        backend,
        PlanWire,
        "plan_recipe",
        dataset=uuid,
        recipe={
            "schema_version": 1,
            "name": "speed",
            "steps": [{"id": "speed", "type": "feature", "feature": "speed-angvel"}],
        },
    )

    assert any("no converted tracks" in caution.message for caution in plan.cautions)
    assert not any("covers all 0" in caution.message for caution in plan.cautions)


@pytest.mark.parametrize(
    "extra",
    [
        {"patterns": ["../../secret/*.npy"]},
        {"patterns": ["/etc/*.npy"]},
        {"patterns": ["sub/../../x.npy"]},
    ],
)
def test_a_pattern_that_reaches_outside_the_source_is_refused(
    backend: LocalBackend, extra: dict[str, object]
) -> None:
    uuid = _created(backend).dataset

    refused = _call(
        backend,
        ErrorEnvelope,
        action="add_source",
        dataset=uuid,
        path="calms",
        confirm=True,
        **{**CALMS21, **extra},
    )

    assert refused.reason == "outside_roots"


def test_a_scan_by_source_id_scans_the_kind_that_declares_it(
    backend: LocalBackend,
) -> None:
    uuid = _created(backend).dataset
    _ = _sourced(backend, uuid)
    _ = _call(
        backend,
        DatasetChange,
        action="add_source",
        dataset=uuid,
        path="calms",
        source_id="calms-labels",
        confirm=True,
        **{**CALMS21, "kind": "labels"},
    )

    scanned = _call(
        backend, DatasetChange, action="scan", dataset=uuid, source_id="calms-labels"
    )
    unknown = _call(
        backend, ErrorEnvelope, action="scan", dataset=uuid, source_id="nope"
    )

    assert scanned.scanned == {"labels": 2}
    assert unknown.reason == "unknown_source"


def test_a_path_that_is_a_file_is_not_made_a_dataset(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    _ = (roots[0] / "file").write_text("x")

    refused = _call(backend, ErrorEnvelope, action="create", path="file", confirm=True)

    assert refused.reason == "dataset_exists"


def test_two_creates_of_one_path_make_one_dataset(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    import threading

    answers: list[object] = []

    def create() -> None:
        answers.append(
            call_tool(
                backend,
                "manage_dataset",
                {"action": "create", "path": "same", "confirm": True},
            )
        )

    threads = [threading.Thread(target=create) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    made = [answer for answer in answers if isinstance(answer, DatasetChange)]
    refused = [answer for answer in answers if isinstance(answer, ErrorEnvelope)]
    assert len(made) == 1
    assert {envelope.reason for envelope in refused} == {"dataset_exists"}
    assert (
        open_dataset(roots[0] / "same" / "dataset.yaml").manifest.uuid
        == made[0].dataset
    )
