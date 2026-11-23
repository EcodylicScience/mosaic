"""A definition copied from a feature template, reading another feature."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from mosaic.behavior.feature_library import (
    feature_template__global as global_template,
)
from mosaic.behavior.feature_library import feature_template__per_sequence as template
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.scope import definition_scope
from mosaic.core.pipeline.graph import Recipe, check_recipe, declaration_catalog
from tests.helpers import defined_dataset

NAME: Final = "template-copy"
GLOBAL_COPY_NAME: Final = "global-template-copy"

_ABSOLUTE_IMPORT: Final = "from mosaic.behavior.feature_library.helpers import"


def _copied_source(template_file: str, template_name: str, name: str) -> str:
    """A template's text with the three edits that the write-your-own guide names."""
    source = Path(template_file).read_text(encoding="utf-8")
    return source.replace("from .helpers import", _ABSOLUTE_IMPORT).replace(
        f'name = "{template_name}"', f'name = "{name}"'
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return defined_dataset(
        tmp_path_factory.mktemp("template"),
        {NAME: _copied_source(template.__file__, "my-new-feature", NAME)},
    )


def _recipe(*upstream: str) -> Recipe:
    steps: list[dict[str, object]] = [
        {"id": step, "type": "feature", "feature": step, "inputs": ["tracks"]}
        for step in upstream
    ]
    steps.append(
        {
            "id": "mine",
            "type": "feature",
            "feature": f"def:{NAME}",
            "inputs": [{"step": step} for step in upstream] or ["tracks"],
        }
    )
    return Recipe.model_validate({"schema_version": 1, "steps": steps})


def test_the_template_validates_over_tracks(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        assert check_recipe(_recipe()) == ()


def test_the_template_validates_over_another_features_output(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        assert check_recipe(_recipe("keypoint-kinematics")) == ()


def test_the_template_declares_the_planar_track_spaces(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        declared = declaration_catalog().get(f"def:{NAME}")

    assert declared is not None
    assert declared.track_spaces == ("image_px", "plane_cm")


@pytest.fixture(scope="module")
def global_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return defined_dataset(
        tmp_path_factory.mktemp("global-template"),
        {
            GLOBAL_COPY_NAME: _copied_source(
                global_template.__file__, "my-global-feature", GLOBAL_COPY_NAME
            )
        },
    )


def test_the_global_template_validates_as_a_fit_and_its_apply(
    global_dataset: Dataset,
) -> None:
    kinematics: dict[str, object] = {
        "id": "kinematics",
        "type": "feature",
        "feature": "keypoint-kinematics",
        "inputs": ["tracks"],
    }
    fit: dict[str, object] = {
        "id": "fit",
        "type": "feature",
        "feature": f"def:{GLOBAL_COPY_NAME}",
        "inputs": [{"step": "kinematics"}],
    }
    apply: dict[str, object] = {
        "id": "apply",
        "type": "feature",
        "feature": f"def:{GLOBAL_COPY_NAME}",
        "inputs": [{"step": "kinematics"}],
        "params": {"model": {"step": "fit"}},
    }
    recipe = Recipe.model_validate(
        {"schema_version": 1, "steps": [kinematics, fit, apply]}
    )

    with definition_scope(global_dataset, load=False):
        assert check_recipe(recipe) == ()
