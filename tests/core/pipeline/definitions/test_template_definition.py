"""A definition copied from the per-sequence feature template, reading another feature."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from mosaic.behavior.feature_library import feature_template__per_sequence as template
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.scope import definition_scope
from mosaic.core.pipeline.graph import Recipe, check_recipe
from tests.helpers import defined_dataset

NAME: Final = "template-copy"

_ABSOLUTE_IMPORT: Final = "from mosaic.behavior.feature_library.helpers import"


def _definition_source() -> str:
    """The template's text with the three edits that the write-your-own guide names."""
    source = Path(template.__file__).read_text(encoding="utf-8")
    return source.replace("from .helpers import", _ABSOLUTE_IMPORT).replace(
        'name = "my-new-feature"', f'name = "{NAME}"'
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    return defined_dataset(
        tmp_path_factory.mktemp("template"), {NAME: _definition_source()}
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
