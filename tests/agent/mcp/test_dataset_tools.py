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
from mosaic.agent.mcp.spec import (
    MAX_RESULT_CHARS,
    DatasetChange,
    DatasetList,
    ManageDatasetInput,
    call_tool,
)
from mosaic.core.annotations import sequence_tags
from mosaic.core.annotations.sequence_tag_wire import (
    SequenceTagImport,
    SequenceTagListing,
)
from mosaic.core.annotations.sequence_tags import (
    SequenceTagCsvInvalid,
    SequenceTagImportRefused,
    SequenceTagsMoved,
    SequenceTagTableUnavailable,
    list_sequence_tags,
)
from mosaic.core.arenas import ArenaLayout
from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.dataset_summary import DatasetSummary
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.graph import PlanWire
from mosaic.core.regions import CircleShape, RectShape, Roi
from tests.helpers import (
    add_track_sequences,
    CONVERT_TRACKS_STEP,
    SPEED_OF_CONVERTED_STEP,
    RecordingStarter,
    calms21_dataset,
    invoke_json,
    tag_table,
    write_calms21_npy,
    write_tags,
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


def test_set_arenas_declares_a_layout_with_a_boundary_and_regions(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _created(backend).dataset
    dataset = open_dataset(roots[0] / "study" / "dataset.yaml")
    add_track_sequences(dataset, ("", "pilot"))
    layout = ArenaLayout.single(
        CircleShape(cx=50.0, cy=50.0, r=40.0),
        rois=(Roi(name="nest", shape=RectShape(x0=0.0, y0=0.0, x1=20.0, y1=20.0)),),
        width=100,
        height=100,
    )

    changed = _call(
        backend,
        DatasetChange,
        action="set_arenas",
        dataset=uuid,
        recording="pilot",
        arena_layout=layout.model_dump(mode="json"),
    )

    assert changed.message == "set 1 arenas of 'pilot'"
    found = open_dataset(roots[0] / "study" / "dataset.yaml").arena_layout("", "pilot")
    assert found is not None and found.layout == layout


PLACES: dict[tuple[str, str], dict[str, str | int]] = {
    ("g", "a"): {"place": 3},
    ("g", "b"): {"place": 4},
    ("h", "c"): {"place": 3},
    ("rec", "s1"): {"place": 3},
    ("rec", "s2"): {"place": 4},
}
"""The ``place`` of each entry of :func:`_tagged`."""

GRID = ArenaLayout.grid(xs=(0, 50, 100), ys=(0, 100))


def _tagged(backend: LocalBackend, roots: tuple[Path, Path]) -> tuple[str, Dataset]:
    """A served dataset of three undivided recordings and one divided, holding ``place``."""
    uuid = _created(backend).dataset
    dataset = open_dataset(roots[0] / "study" / "dataset.yaml")
    add_track_sequences(
        dataset, ("g", "a"), ("g", "b"), ("h", "c"), ("rec", "s1"), ("rec", "s2")
    )
    dataset.set_continuous_groups(["rec"])
    _ = write_tags(dataset, tag_table(PLACES, {"place": "int"}))
    return uuid, dataset


def _arena_keys(roots: tuple[Path, Path]) -> set[tuple[str, str, str]]:
    found = open_dataset(roots[0] / "study" / "dataset.yaml").arena_layouts()
    return {(key.group, key.sequence, key.camera) for key in found}


def test_set_arenas_declares_the_layout_of_each_recording_the_sequence_tags_select(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid, _ = _tagged(backend, roots)

    changed = _call(
        backend,
        DatasetChange,
        action="set_arenas",
        dataset=uuid,
        sequence_tags={"place": [4, "3"]},
        camera="top",
        arena_layout=GRID.model_dump(mode="json"),
    )

    assert _arena_keys(roots) == {
        ("g", "a", "top"),
        ("g", "b", "top"),
        ("h", "c", "top"),
        ("rec", "", "top"),
    }
    assert changed.message == (
        "set 2 arenas of 4 recordings selected by sequence_tags of camera 'top'"
    )


def test_set_arenas_declares_a_divided_recording_the_sequence_tags_cover_whole(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid, dataset = _tagged(backend, roots)
    _ = write_tags(
        dataset,
        tag_table(
            {("rec", "s1"): {"place": 3}, ("rec", "s2"): {"place": 3}},
            {"place": "int"},
        ),
    )

    _ = _call(
        backend,
        DatasetChange,
        action="set_arenas",
        dataset=uuid,
        sequence_tags={"place": [3]},
        arena_layout=GRID.model_dump(mode="json"),
    )

    assert _arena_keys(roots) == {("rec", "", "")}


def test_set_arenas_refuses_a_recording_with_sequence_tags_and_one_of_neither(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid, _ = _tagged(backend, roots)
    wire = GRID.model_dump(mode="json")

    both = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        recording="g:a",
        sequence_tags={"place": [3]},
        arena_layout=wire,
    )
    neither = _call(
        backend, ErrorEnvelope, action="set_arenas", dataset=uuid, arena_layout=wire
    )

    assert both.kind == "invalid"
    assert "recording or sequence_tags, not both" in both.message
    assert (neither.kind, neither.reason) == ("invalid", "field_required")
    assert _arena_keys(roots) == set()


def test_set_arenas_refuses_a_divided_recording_the_sequence_tags_cover_in_part(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid, _ = _tagged(backend, roots)

    refused = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        sequence_tags={"place": [3]},
        arena_layout=GRID.model_dump(mode="json"),
    )

    assert refused.kind == "invalid"
    assert "'rec'" in refused.message
    assert "s1=3" in refused.message and "s2=4" in refused.message
    assert refused.detail is not None
    assert refused.detail["tags"] == {"place": [3]}
    assert _arena_keys(roots) == set()


@pytest.mark.parametrize(
    ("sequence_tags", "message"),
    [
        ({"place": [9]}, "no entry holds"),
        ({"nowhere": [1]}, "not a sequence tag"),
        ({}, "at least one tag"),
    ],
    ids=["no entry", "unknown tag", "no tag"],
)
def test_set_arenas_refuses_sequence_tags_that_select_nothing_usable(
    backend: LocalBackend,
    roots: tuple[Path, Path],
    sequence_tags: dict[str, list[int]],
    message: str,
) -> None:
    uuid, _ = _tagged(backend, roots)

    refused = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        sequence_tags=sequence_tags,
        arena_layout=GRID.model_dump(mode="json"),
    )

    assert refused.kind == "invalid"
    assert message in refused.message
    assert refused.detail == {"revision": 1, "tags": sequence_tags}
    assert _arena_keys(roots) == set()


def test_set_arenas_refuses_a_key_the_dataset_does_not_hold_before_any_write(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _created(backend).dataset
    dataset = open_dataset(roots[0] / "study" / "dataset.yaml")
    add_track_sequences(dataset, ("g", "a"))
    _ = write_tags(
        dataset,
        tag_table(
            {("g", "a"): {"place": 3}, ("g", "ghost"): {"place": 3}}, {"place": "int"}
        ),
    )

    refused = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        sequence_tags={"place": [3]},
        arena_layout=GRID.model_dump(mode="json"),
    )

    assert refused.reason == "invalid_arenas"
    assert _arena_keys(roots) == set()


@pytest.mark.parametrize(
    ("broken", "field"),
    [
        pytest.param(
            {"boundary": {"cx": 50.0, "cy": 50.0, "r": 40.0}},
            "boundary",
            id="a boundary without its kind",
        ),
        pytest.param(
            {"boundary": {"kind": "circle", "cx": 50.0, "cy": 50.0, "r": -1.0}},
            "boundary",
            id="a circle with a negative radius",
        ),
        pytest.param(
            {"rois": [{"name": "None", "shape": {"kind": "circle", "cx": 1, "cy": 1}}]},
            "rois",
            id="a malformed region",
        ),
        pytest.param(
            {
                "rois": [
                    {
                        "name": "nest",
                        "shape": {"kind": "circle", "cx": 1, "cy": 1, "r": 1},
                    },
                    {
                        "name": "nest",
                        "shape": {"kind": "circle", "cx": 5, "cy": 5, "r": 1},
                    },
                ]
            },
            "rois",
            id="a repeated region name",
        ),
    ],
)
def test_a_malformed_boundary_or_region_answers_invalid_arenas_naming_the_field(
    backend: LocalBackend,
    roots: tuple[Path, Path],
    broken: dict[str, JsonValue],
    field: str,
) -> None:
    uuid = _created(backend).dataset
    dataset = open_dataset(roots[0] / "study" / "dataset.yaml")
    add_track_sequences(dataset, ("", "pilot"))
    arena: dict[str, JsonValue] = {
        "name": "A1",
        "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
    }
    wire: dict[str, JsonValue] = {
        "width": 100,
        "height": 100,
        "arenas": [{**arena, **broken}],
    }

    refused = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        recording="pilot",
        arena_layout=wire,
    )

    assert refused.reason == "invalid_arenas"
    assert refused.detail is not None
    errors = refused.detail["errors"]
    assert isinstance(errors, list) and errors
    assert any(
        isinstance(item, dict) and field in f"{item['field']} {item['message']}"
        for item in errors
    )
    assert not open_dataset(roots[0] / "study" / "dataset.yaml").arena_layouts()


def test_an_invalid_layout_answers_a_short_message_and_a_field_message_list(
    backend: LocalBackend,
) -> None:
    uuid = _created(backend).dataset
    wire: dict[str, JsonValue] = {
        "width": 100,
        "height": 100,
        "arenas": [
            {
                "name": "A1",
                "region": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
                "boundary": {"kind": "circle", "cx": 50.0, "cy": 50.0, "r": -1.0},
            }
        ],
    }

    refused = _call(
        backend,
        ErrorEnvelope,
        action="set_arenas",
        dataset=uuid,
        recording="pilot",
        arena_layout=wire,
    )

    assert refused.reason == "invalid_arenas"
    assert refused.message == "the arena layout is invalid: 1 problem(s)"
    assert refused.detail == {
        "recording": "pilot",
        "camera": "",
        "errors": [
            {
                "field": "arenas.0.boundary.circle.r",
                "message": "Input should be greater than 0",
            }
        ],
    }


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


def test_a_scan_of_a_missing_source_path_returns_the_finding(
    backend: LocalBackend,
) -> None:
    """An agent reads the result, not stderr."""
    uuid = _created(backend).dataset
    _ = _call(
        backend,
        DatasetChange,
        action="add_source",
        dataset=uuid,
        path="ghost",
        source_id="ghost",
        confirm=True,
        **CALMS21,
    )

    scanned = _call(backend, DatasetChange, action="scan", dataset=uuid)

    (finding,) = scanned.findings
    assert (finding.kind, finding.source, finding.problem) == (
        "tracks",
        "ghost",
        "path_missing",
    )
    assert finding.path is not None
    assert finding.path.endswith("ghost")


def test_a_path_pattern_groups_a_source_declared_through_the_tool(
    backend: LocalBackend,
) -> None:
    created = _created(backend)
    uuid = created.dataset
    _ = _sourced(
        backend,
        uuid,
        multi_sequences_per_file=False,
        group_from=None,
        group_path_pattern=r"^([^.]+)\.",
    )

    scanned = _call(backend, DatasetChange, action="scan", dataset=uuid)

    assert scanned.scanned == {"tracks": 2}
    assert scanned.findings == ()
    held = open_dataset(Path(created.manifest))
    assert sorted(row["group"] for row in held.read_tracks_raw_index()) == [
        "test",
        "train",
    ]


def test_a_dataset_root_that_does_not_exist_yet_holds_none_and_takes_the_first(
    tmp_path: Path,
) -> None:
    later = tmp_path / "later"
    backend = LocalBackend(
        [], warm=False, start_driver=RecordingStarter(), dataset_roots=[later]
    )

    before = _tool(backend, DatasetList, "list_datasets")
    created = _created(backend)
    after = _tool(backend, DatasetList, "list_datasets")

    assert before.items == ()
    assert [item.uuid for item in after.items] == [created.dataset]
    assert (later / "study" / "dataset.yaml").is_file()


_TAG_ENTRIES = (("g1", "s1"), ("g1", "s2"), ("g2", "s3"))


def _tagged_study(backend: LocalBackend, roots: tuple[Path, Path]) -> str:
    """A served dataset of three entries, with a CSV of tags in the source root."""
    uuid = _created(backend).dataset
    dataset = open_dataset(roots[0] / "study" / "dataset.yaml")
    add_track_sequences(dataset, *_TAG_ENTRIES)
    return uuid


def _tags_csv(roots: tuple[Path, Path], text: str, name: str = "tags.csv") -> str:
    _ = (roots[1] / name).write_text(text, encoding="utf-8")
    return name


def _study(roots: tuple[Path, Path]) -> Dataset:
    return open_dataset(roots[0] / "study" / "dataset.yaml")


def test_importing_sequence_tags_writes_a_revision_and_moves_the_findings_up(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site\ng1:s1,a\ng1:s2,b\ng9:nope,c\n")

    changed = _call(
        backend,
        DatasetChange,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
    )

    assert changed.action == "import_sequence_tags"
    assert changed.tag_import is not None
    assert (changed.tag_import.revision, changed.tag_import.written) == (1, True)
    assert [change.name for change in changed.tag_import.changes] == ["site"]
    assert [finding.problem for finding in changed.findings] == ["row_unmatched"]
    assert changed.tag_import.findings == ()
    listing = list_sequence_tags(_study(roots))
    assert [(tag.name, tag.entries) for tag in listing.tags] == [("site", 2)]


def test_a_path_outside_the_source_roots_is_refused_and_a_server_without_roots_refuses(
    backend: LocalBackend, roots: tuple[Path, Path], tmp_path: Path
) -> None:
    uuid = _tagged_study(backend, roots)
    outside = tmp_path / "elsewhere.csv"
    _ = outside.write_text("entry,site\ng1:s1,a\n", encoding="utf-8")
    ds = _study(roots)
    unconfigured = LocalBackend([ds.manifest_path], warm=False)

    refused = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=str(outside),
    )
    without = _call(
        unconfigured,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=str(outside),
    )

    assert refused.reason == "outside_roots"
    assert without.reason == "roots_not_configured"
    assert list_sequence_tags(_study(roots)).revision is None


def test_replace_asks_to_confirm_and_an_import_without_it_asks_nothing(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site\ng1:s1,a\n")
    other = _tags_csv(roots, "entry,age\ng1:s1,3\n", "age.csv")
    _ = _call(
        backend, DatasetChange, action="import_sequence_tags", dataset=uuid, path=name
    )

    plain = _call(
        backend, DatasetChange, action="import_sequence_tags", dataset=uuid, path=other
    )
    asked = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
        replace=True,
    )
    confirmed = _call(
        backend,
        DatasetChange,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
        replace=True,
        confirm=True,
    )

    assert plain.tag_import is not None and plain.tag_import.written
    assert asked.reason == "confirmation_required"
    assert asked.detail is not None
    assert asked.detail["conditions"] == ["replace_sequence_tags"]
    assert confirmed.tag_import is not None
    assert [
        (change.name, change.change) for change in confirmed.tag_import.changes
    ] == [("age", "removed"), ("site", "unchanged")]


def test_a_dry_run_reports_the_changes_and_writes_nothing(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site\ng1:s1,a\n")

    dry = _call(
        backend,
        DatasetChange,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
        dry_run=True,
    )

    assert dry.tag_import is not None
    assert (dry.tag_import.dry_run, dry.tag_import.written) == (True, False)
    assert [change.change for change in dry.tag_import.changes] == ["added"]
    assert list_sequence_tags(_study(roots)).revision is None


def test_the_key_and_the_types_reach_the_import(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "name,code\ng1:s1,03\ng1:s2,04\n")

    changed = _call(
        backend,
        DatasetChange,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
        key="name=entry",
        types={"code": "text"},
    )

    assert changed.tag_import is not None
    assert [(change.name, change.type) for change in changed.tag_import.changes] == [
        ("code", "text")
    ]


def test_a_refusal_of_the_library_is_an_invalid_answer_with_its_message(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    broken = _tags_csv(roots, "entry,n\ng1:s1,1\ng1:s2,two\n")
    missing = _tags_csv(roots, "entry,site\ng1:s1,a\n", "present.csv")
    (roots[1] / missing).unlink()

    typed = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=broken,
        types={"n": "int"},
    )
    bad_key = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=broken,
        key="n",
    )
    absent = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=missing,
    )
    no_path = _call(backend, ErrorEnvelope, action="import_sequence_tags", dataset=uuid)

    assert (typed.kind, typed.reason) == ("invalid", "sequence_tags_invalid")
    assert "line 3" in typed.message
    assert (bad_key.kind, bad_key.reason) == ("invalid", "sequence_tags_invalid")
    assert "NAME=" in bad_key.message
    assert (absent.kind, absent.reason) == ("not_found", "csv_not_found")
    assert "present.csv" in absent.message
    assert (no_path.reason, no_path.detail) == (
        "field_required",
        {"action": "import_sequence_tags", "field": "path"},
    )


def test_a_managed_dataset_lists_its_tags_and_refuses_an_import_even_when_allowed(
    roots: tuple[Path, Path],
) -> None:
    datasets, sources = roots
    backend = LocalBackend(
        [],
        warm=False,
        start_driver=RecordingStarter(),
        dataset_roots=[datasets],
        source_roots=[sources],
        allow_managed=True,
    )
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site\ng1:s1,a\n")
    _ = _call(
        backend, DatasetChange, action="import_sequence_tags", dataset=uuid, path=name
    )
    with (datasets / "study" / "dataset.yaml").open("a", encoding="utf-8") as manifest:
        _ = manifest.write("managed_by: mosaic-api\n")
    stricter = LocalBackend(
        [(datasets / "study" / "dataset.yaml")], warm=False, source_roots=[sources]
    )

    listed = _call(backend, DatasetChange, action="list_sequence_tags", dataset=uuid)
    refused = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
    )
    listed_by_default = _call(
        stricter, DatasetChange, action="list_sequence_tags", dataset=uuid
    )
    refused_by_default = _call(
        stricter,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
    )

    assert listed.sequence_tags is not None
    assert [tag.name for tag in listed.sequence_tags.tags] == ["site"]
    assert listed_by_default.sequence_tags == listed.sequence_tags
    assert (refused.kind, refused.reason) == ("conflict", "sequence_tags_refused")
    assert "managed" in refused.message
    assert refused_by_default.reason == "dataset_managed"
    assert list_sequence_tags(_study(roots)).revision == 1


def test_a_listing_above_the_result_budget_is_refused_and_names_the_command_line(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _created(backend).dataset
    entries = tuple(("g", f"s{index:04d}") for index in range(1200))
    add_track_sequences(_study(roots), *entries)
    rows = "".join(
        f"g:s{index:04d},{'v' * 40}{index:04d}\n" for index in range(len(entries))
    )
    name = _tags_csv(roots, f"entry,site\n{rows}")
    _ = _call(
        backend, DatasetChange, action="import_sequence_tags", dataset=uuid, path=name
    )

    refused = _call(backend, ErrorEnvelope, action="list_sequence_tags", dataset=uuid)

    assert (refused.kind, refused.reason) == ("invalid", "result_too_large")
    assert refused.detail is not None
    assert refused.detail["tool"] == "manage_dataset"
    assert refused.detail["budget"] == MAX_RESULT_CHARS
    assert refused.message.count("CLI") + refused.message.count("mosaic ") == 1
    assert "mosaic sequence-tags list --json" in refused.message


def test_other_actions_name_no_command_for_a_whole_result() -> None:
    listing = ManageDatasetInput(action="list_sequence_tags", dataset="d")
    scan = ManageDatasetInput(action="scan", dataset="d")

    assert "mosaic sequence-tags list --json" in listing.whole_result_advice()
    assert scan.whole_result_advice() == ""
    assert listing.reads_only(False)
    assert not scan.reads_only(False)


def test_the_listing_equals_the_command_line_json(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site,age\ng1:s1,a,3\ng1:s2,b,4\ng2:s3,a,5\n")
    _ = _call(
        backend, DatasetChange, action="import_sequence_tags", dataset=uuid, path=name
    )

    listed = _call(backend, DatasetChange, action="list_sequence_tags", dataset=uuid)
    printed = invoke_json(
        [
            "sequence-tags",
            "list",
            "-m",
            str(roots[0] / "study" / "dataset.yaml"),
            "--json",
        ]
    )

    assert listed.sequence_tags == SequenceTagListing.model_validate(printed)
    assert listed.sequence_tags is not None and listed.sequence_tags.tags
    assert listed.tag_import is None


def test_a_dry_run_with_replace_asks_nothing_and_leaves_the_revision(
    backend: LocalBackend, roots: tuple[Path, Path]
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site\ng1:s1,a\n")
    other = _tags_csv(roots, "entry,age\ng1:s1,3\n", "age.csv")
    _ = _call(
        backend, DatasetChange, action="import_sequence_tags", dataset=uuid, path=name
    )
    before = list_sequence_tags(_study(roots))

    dry = _call(
        backend,
        DatasetChange,
        action="import_sequence_tags",
        dataset=uuid,
        path=other,
        replace=True,
        dry_run=True,
    )

    assert dry.tag_import is not None
    assert (dry.tag_import.dry_run, dry.tag_import.written) == (True, False)
    assert [(c.name, c.change) for c in dry.tag_import.changes] == [
        ("age", "added"),
        ("site", "removed"),
    ]
    assert list_sequence_tags(_study(roots)) == before


@pytest.mark.parametrize(
    ("raised", "kind", "reason"),
    [
        (SequenceTagImportRefused("elsewhere"), "conflict", "sequence_tags_refused"),
        (SequenceTagsMoved("moved"), "conflict", "sequence_tags_moved"),
        (SequenceTagCsvInvalid("bad"), "invalid", "sequence_tags_invalid"),
    ],
)
def test_an_import_refusal_is_answered_by_the_kind_of_its_cause(
    backend: LocalBackend,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    raised: ValueError,
    kind: str,
    reason: str,
) -> None:
    uuid = _tagged_study(backend, roots)
    name = _tags_csv(roots, "entry,site\ng1:s1,a\n")

    def refuse(*_args: object, **_kwargs: object) -> SequenceTagImport:
        raise raised

    monkeypatch.setattr(sequence_tags, "import_sequence_tags", refuse)

    answered = _call(
        backend,
        ErrorEnvelope,
        action="import_sequence_tags",
        dataset=uuid,
        path=name,
    )

    assert (answered.kind, answered.reason) == (kind, reason)
    assert answered.message == str(raised)


def test_an_unreadable_tag_table_fails_the_listing_as_a_conflict(
    backend: LocalBackend,
    roots: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uuid = _tagged_study(backend, roots)

    def unavailable(*_args: object, **_kwargs: object) -> SequenceTagListing:
        msg = "several datasets are indexed"
        raise SequenceTagTableUnavailable(msg)

    monkeypatch.setattr(sequence_tags, "list_sequence_tags", unavailable)

    answered = _call(
        backend,
        ErrorEnvelope,
        action="list_sequence_tags",
        dataset=uuid,
    )

    assert (answered.kind, answered.reason) == ("conflict", "sequence_tags_unavailable")
