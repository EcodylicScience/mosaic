"""A feature definition across an edit of its source and an upgrade of the toolkit.

An edit stores a second version beside the first. The bare name then plans and
runs the second, and the first version's run stays readable, refittable and
plannable by its digest. A toolkit model's default that a newer toolkit changes
reaches a definition that declared ``PoseConfig()`` as it reaches a built-in, and
leaves one that set ``pose_n`` where it was.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.scope import (
    clear_definition_caches,
    definition_scope,
)
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionToolkitModelMissing,
    list_versions,
    read_current,
)
from mosaic.core.pipeline.graph import (
    Recipe,
    check_recipe,
    editor_catalog,
    lane_for_step,
    plan_pipeline,
    run_pipeline,
)
from mosaic.core.pipeline.graph.lanes import DEFAULT_LANE
from mosaic.core.pipeline.inventory.model import FeatureRunRef
from mosaic.core.pipeline.preview import preview_artifact
from mosaic.core.pipeline.types.data_config import PoseConfig
from mosaic.core.pipeline.verify import verify_plan
from tests.helpers import (
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    POSE_NAME,
    POSE_SOURCE,
    defined_dataset,
    forget_imported_definitions,
    imported_definitions,
    tree_snapshot,
)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _recipe(*steps: tuple[str, str]) -> Recipe:
    """A recipe of one feature step over the tracks for each ``(id, feature)``."""
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "acceptance",
            "steps": [
                {
                    "id": step_id,
                    "type": "feature",
                    "feature": feature,
                    "inputs": ["tracks"],
                }
                for step_id, feature in steps
            ],
        }
    )


def _run_ids(dataset: Dataset, recipe: Recipe) -> dict[str, str | None]:
    """The run id each step of *recipe* plans to."""
    plan = plan_pipeline(dataset, recipe)
    return {step.step_id: step.run_id for step in plan.steps}


# --- edit ----------------------------------------------------------------------------

GLOBAL_SLUG: Final = f"def:{GLOBAL_NAME}"
GLOBAL_STORAGE: Final = f"def.{GLOBAL_NAME}__from__tracks"
EDITED_SOURCE: Final = GLOBAL_SOURCE.replace('"centered"', '"offset"')
"""``GLOBAL_SOURCE`` writing its column under another name, of the same params."""


def test_an_edit_runs_a_new_version_and_leaves_the_first_run_readable(
    tmp_path: Path,
) -> None:
    dataset = defined_dataset(tmp_path, {GLOBAL_NAME: GLOBAL_SOURCE})
    bare = _recipe(("d", GLOBAL_SLUG))
    (ran,) = run_pipeline(dataset, bare).outcomes
    first = ran.run_id
    assert first is not None

    record = define_feature(dataset, GLOBAL_NAME, EDITED_SOURCE)

    directory = dataset.get_root(DEFINITIONS_ROOT) / GLOBAL_NAME
    assert not record.reused
    assert record.previous_digest == _digest(GLOBAL_SOURCE)
    assert sorted(version.digest for version in list_versions(directory)) == sorted(
        (_digest(GLOBAL_SOURCE), _digest(EDITED_SOURCE))
    )
    assert sorted(path.suffix for path in directory.glob("*_*.*")) == [
        ".json",
        ".json",
        ".py",
        ".py",
    ]
    current = read_current(directory)
    assert current is not None
    assert current.digest == _digest(EDITED_SOURCE)
    (planned,) = _run_ids(dataset, bare).values()
    assert planned != first
    (second,) = run_pipeline(dataset, bare).outcomes
    assert second.run_id == planned

    # The first version's run, read after the edit.
    forget_imported_definitions()
    clear_definition_caches()
    before = tree_snapshot(dataset.base_dir)
    preview = preview_artifact(
        dataset, FeatureRunRef(GLOBAL_STORAGE, first), ("g", "s1")
    )
    report = dataset.reconcile(only=("features",))
    assert tree_snapshot(dataset.base_dir) == before
    columns = [column.name for column in preview.columns]
    assert "centered" in columns
    assert "offset" not in columns
    assert preview.rows
    (finding,) = (each for each in report.findings if each.old_address == first)
    assert finding.verdict == "ok", finding.reason
    assert not imported_definitions()

    for pin in (_digest(GLOBAL_SOURCE), _digest(GLOBAL_SOURCE)[:12]):
        pinned = _recipe(("d", f"{GLOBAL_SLUG}@{pin}"))
        with definition_scope(dataset, load=False):
            assert check_recipe(pinned) == ()
            assert lane_for_step(f"{GLOBAL_SLUG}@{pin}") == DEFAULT_LANE
        assert _run_ids(dataset, pinned) == {"d": first}

    plan = plan_pipeline(
        dataset, _recipe(("d", f"{GLOBAL_SLUG}@{_digest(GLOBAL_SOURCE)}"))
    )
    verified = verify_plan(dataset, plan)
    (verdict,) = verified.steps
    assert (verdict.status, verdict.role) == ("ok", "fit"), verdict.reason
    assert tree_snapshot(dataset.base_dir) == before


# --- toolkit upgrade -----------------------------------------------------------------

POSE_PARAMS: Final = (
    "        pose: PoseConfig = Field(default_factory=PoseConfig)\n"
    "        keypoints: list[int | str] = [0]\n"
)
"""The params of ``POSE_SOURCE``, which each upgrade source replaces."""

HOLDER_MODEL: Final = '''from mosaic.core.params import DeclaredModel


class Holder(DeclaredModel):
    """A model of the definition's own file holding a toolkit model."""

    pose: PoseConfig = PoseConfig()


'''


def _pose_source(name: str, params: str, models: str = "") -> str:
    """``POSE_SOURCE`` named *name*, of *params*, after the file's *models*."""
    assert POSE_PARAMS in POSE_SOURCE
    return (
        POSE_SOURCE.replace(f'name = "{POSE_NAME}"', f'name = "{name}"')
        .replace(POSE_PARAMS, f"{params}        keypoints: list[int | str] = [0]\n")
        .replace("class PoseCount:", f"{models}class PoseCount:")
    )


DEFAULT_NAME: Final = "pose-default"
FIVE_NAME: Final = "pose-five"
HELD_NAME: Final = "pose-held"
LISTED_NAME: Final = "pose-listed"

UPGRADE_SOURCES: Final = {
    DEFAULT_NAME: _pose_source(
        DEFAULT_NAME, "        pose: PoseConfig = PoseConfig()\n"
    ),
    FIVE_NAME: _pose_source(
        FIVE_NAME, "        pose: PoseConfig = PoseConfig(pose_n=5)\n"
    ),
    HELD_NAME: _pose_source(
        HELD_NAME, "        holder: Holder = Holder()\n", HOLDER_MODEL
    ),
    LISTED_NAME: _pose_source(
        LISTED_NAME,
        "        poses: list[PoseConfig] = [PoseConfig(), PoseConfig(pose_n=5)]\n",
    ),
}

BUILT_IN: Final = "track-subsample"
"""A built-in feature whose params hold ``PoseConfig()``."""

UPGRADE_RECIPE: Final = _recipe(
    *((name, f"def:{name}") for name in UPGRADE_SOURCES), (BUILT_IN, BUILT_IN)
)

OLD_POSE_N: Final = PoseConfig().pose_n
NEW_POSE_N: Final = OLD_POSE_N + 2


@contextmanager
def _pose_n_default(value: int) -> Generator[None]:
    """Change ``PoseConfig.pose_n``'s default, as a newer toolkit would."""
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(PoseConfig.model_fields["pose_n"], "default", value)
            _ = PoseConfig.model_rebuild(force=True)
            clear_definition_caches()
            yield
    finally:
        _ = PoseConfig.model_rebuild(force=True)
        clear_definition_caches()
    assert PoseConfig().pose_n == OLD_POSE_N


@pytest.fixture(scope="module")
def upgraded(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A dataset defining each upgrade source."""
    return defined_dataset(tmp_path_factory.mktemp("upgrade"), UPGRADE_SOURCES)


def _schema_default(dataset: Dataset, name: str, field: str) -> JsonValue:
    """The default of *field* in the catalog's params schema of ``def:<name>``."""
    with definition_scope(dataset, load=False):
        (entry,) = (
            each for each in editor_catalog().entries if each.name == f"def:{name}"
        )
    schema = entry.params_schema
    assert schema is not None
    properties = schema["properties"]
    assert isinstance(properties, dict)
    described = properties[field]
    assert isinstance(described, dict)
    return described["default"]


def test_a_newer_toolkit_default_reaches_a_definition_as_it_reaches_a_built_in(
    upgraded: Dataset,
) -> None:
    before = _run_ids(upgraded, UPGRADE_RECIPE)

    with _pose_n_default(NEW_POSE_N):
        with definition_scope(upgraded, load=False):
            assert check_recipe(UPGRADE_RECIPE) == ()
        after = _run_ids(upgraded, UPGRADE_RECIPE)
        defaults = {
            DEFAULT_NAME: _schema_default(upgraded, DEFAULT_NAME, "pose"),
            FIVE_NAME: _schema_default(upgraded, FIVE_NAME, "pose"),
            HELD_NAME: _schema_default(upgraded, HELD_NAME, "holder"),
            LISTED_NAME: _schema_default(upgraded, LISTED_NAME, "poses"),
        }

    moved = {step for step in before if before[step] != after[step]}
    assert moved == {DEFAULT_NAME, HELD_NAME, LISTED_NAME, BUILT_IN}
    assert json.dumps(defaults, sort_keys=True) == json.dumps(
        {
            DEFAULT_NAME: PoseConfig(pose_n=NEW_POSE_N).model_dump(),
            FIVE_NAME: PoseConfig(pose_n=5).model_dump(),
            HELD_NAME: {"pose": PoseConfig(pose_n=NEW_POSE_N).model_dump()},
            LISTED_NAME: [
                PoseConfig(pose_n=NEW_POSE_N).model_dump(),
                PoseConfig(pose_n=5).model_dump(),
            ],
        },
        sort_keys=True,
    )
    assert not imported_definitions()


def test_a_definition_runs_under_a_newer_toolkit_default(upgraded: Dataset) -> None:
    recipe = _recipe(*((name, f"def:{name}") for name in UPGRADE_SOURCES))

    with _pose_n_default(NEW_POSE_N):
        planned = _run_ids(upgraded, recipe)
        ran = run_pipeline(upgraded, recipe).outcomes

    assert {outcome.step_id: outcome.run_id for outcome in ran} == planned


def test_a_sidecar_naming_a_removed_toolkit_model_is_refused(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {FIVE_NAME: UPGRADE_SOURCES[FIVE_NAME]})
    ref = f"{PoseConfig.__module__}:{PoseConfig.__qualname__}"
    (sidecar,) = (dataset.get_root(DEFINITIONS_ROOT) / FIVE_NAME).glob("*.json")
    text = sidecar.read_text(encoding="utf-8")
    assert ref in text
    _ = sidecar.write_text(text.replace(ref, f"{ref}Removed"), encoding="utf-8")
    clear_definition_caches()

    with pytest.raises(DefinitionToolkitModelMissing) as refused:
        _ = plan_pipeline(dataset, _recipe(("d", f"def:{FIVE_NAME}")))

    assert refused.value.error_reason() == "definition_toolkit_model_missing"
