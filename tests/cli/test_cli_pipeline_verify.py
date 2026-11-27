"""``mosaic pipeline verify``: the exit status and the ``--json`` report of a verification.

The library's cases are in ``tests/core/pipeline/test_verify_plan.py``. These
check what the command adds: the plan options it shares with ``pipeline plan``,
the exit code, the refusals, and that it leaves the dataset alone.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.params import Params
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, run_pipeline
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.types import Inputs, InputStream, Result, TrackInput
from mosaic.core.scope import Scope
from tests.helpers import (
    add_tracks_variant,
    assert_tree_unchanged,
    calms21_dataset,
    classifier_recipe,
    make_dataset,
    registered_features,
    tree_snapshot,
)

runner = CliRunner()

SCOPES = {"labeled": Scope(groups=["train"])}


def _recipe() -> Recipe:
    base = classifier_recipe("clf").model_dump(mode="json")
    evaluation = {
        "id": "score",
        "type": "feature",
        "feature": "evaluate-labels",
        "inputs": [{"step": "clf"}],
        "params": {"labels": {"step": "labels"}, "perspective": 0},
    }
    return Recipe.model_validate({**base, "steps": [*base["steps"], evaluation]})


@pytest.fixture
def study(tmp_path: Path) -> Dataset:
    ds = calms21_dataset(
        tmp_path / "study",
        {"train": {"a": {"s1": 90, "s2": 90}}, "other": {"a": {"s3": 90}}},
        seed=11,
        n_classes=2,
    )
    _ = run_pipeline(ds, _recipe(), step_scopes=SCOPES)
    return ds


@pytest.fixture
def recipe_file(tmp_path: Path) -> str:
    path = tmp_path / "recipe.json"
    _ = path.write_text(_recipe().model_dump_json())
    return f"@{path}"


def _verify(study: Dataset, recipe_file: str, *flags: str) -> Any:
    return runner.invoke(
        app,
        [
            "pipeline",
            "verify",
            "--manifest",
            str(study.manifest_path),
            "--recipe",
            recipe_file,
            "--step-scope",
            'labeled={"groups": ["train"]}',
            *flags,
        ],
    )


def test_a_matching_recipe_exits_zero_and_reports_every_step(
    study: Dataset, recipe_file: str
) -> None:
    result = _verify(study, recipe_file, "--json")

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert report["atol"] == 1e-5
    assert report["n_threads"] == 1
    by_step = {step["step"]: step for step in report["steps"]}
    assert by_step["clf"]["status"] == "ok"
    assert by_step["clf"]["argmax_agreement"] == 1.0
    assert by_step["score"]["role"] == "evaluation"


def test_step_limits_the_verification_to_the_steps_named(
    study: Dataset, recipe_file: str
) -> None:
    result = _verify(study, recipe_file, "--step", "clf", "--json")

    assert result.exit_code == 0, result.stderr
    by_step = {
        step["step"]: step["status"] for step in json.loads(result.stdout)["steps"]
    }
    assert by_step["clf"] == "ok"
    assert by_step["labeled"] == by_step["score"] == "skipped"


def test_step_may_be_repeated(study: Dataset, recipe_file: str) -> None:
    result = _verify(study, recipe_file, "--step", "clf", "--step", "score", "--json")

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["verified"] == 2


def test_a_step_the_recipe_lacks_exits_one_and_names_it(
    study: Dataset, recipe_file: str
) -> None:
    result = _verify(study, recipe_file, "--step", "nope")

    assert result.exit_code == 1
    assert "nope" in result.stderr


def test_a_changed_output_exits_one_and_the_text_report_names_the_entry(
    study: Dataset, recipe_file: str
) -> None:
    planned = plan_pipeline(study, _recipe(), step_scopes=SCOPES).step("clf")
    root = study.get_root("features") / planned.storage_name / str(planned.run_id)
    [path] = sorted(root.glob("train__s1*.parquet"))
    stored = pd.read_parquet(path)
    stored["prob_1"] = stored["prob_1"] + 0.5
    stored.to_parquet(path)

    result = _verify(study, recipe_file)

    assert result.exit_code == 1
    assert "1 step(s) differ." in result.stderr
    assert "train/s1 differs" in result.stderr
    assert "clf\txgboost\tfit\tdiffers" in result.stdout


def test_a_step_scope_that_the_run_did_not_use_verifies_nothing_and_exits_one(
    study: Dataset, recipe_file: str
) -> None:
    """Without the step scope the plan is another run, which has not been made."""
    result = runner.invoke(
        app,
        [
            "pipeline",
            "verify",
            "--manifest",
            str(study.manifest_path),
            "--recipe",
            recipe_file,
            "--json",
        ],
    )

    assert result.exit_code == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["verified"] == 0
    by_step = {step["step"]: step for step in report["steps"]}
    assert by_step["labeled"]["status"] == "skipped"
    assert "no complete run" in by_step["labeled"]["reason"]
    assert report["unfinished"] > 0


def test_a_scratch_directory_inside_the_dataset_exits_one(
    study: Dataset, recipe_file: str
) -> None:
    result = _verify(study, recipe_file, "--scratch", str(study.base_dir / "scratch"))

    assert result.exit_code == 1
    assert "inside the dataset" in result.stderr


def test_verifying_writes_nothing_under_the_dataset(
    study: Dataset, recipe_file: str
) -> None:
    before = tree_snapshot(study.base_dir)

    result = _verify(study, recipe_file, "--json")

    assert result.exit_code == 0, result.stderr
    assert_tree_unchanged(study.base_dir, before)


def test_an_unfinished_step_is_named_as_the_cause_and_not_a_difference(
    study: Dataset, recipe_file: str
) -> None:
    result = runner.invoke(
        app,
        [
            "pipeline",
            "verify",
            "--manifest",
            str(study.manifest_path),
            "--recipe",
            recipe_file,
        ],
    )

    assert result.exit_code == 1
    assert "have no complete run" in result.stderr
    assert "differ" not in result.stderr.split("verified 0 step(s)")[-1]


class _ArenaAlpha:
    """Writes ``alpha`` and an ``arena`` for each ``(frame, id)`` of its tracks."""

    name = "test-verify-alpha"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    column: ClassVar[str] = "alpha"

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _ArenaAlpha.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs or self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "arena": "a",
                self.column: df["X"],
            }
        )


class _ArenaBeta(_ArenaAlpha):
    name = "test-verify-beta"
    column = "beta"


class _ArenaCentered:
    """Fits the mean of two inputs' payloads and centers their sum on it."""

    name = "test-verify-centered"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    fits = True
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _ArenaCentered.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs or self.Inputs(())
        self.params = self.Params.from_overrides(params)
        self._mean = 0.0

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        stored = run_root / "state.json"
        if not stored.exists():
            return False
        self._mean = float(json.loads(stored.read_text())["mean"])
        return True

    def fit(self, inputs: InputStream) -> None:
        totals = [
            float(value)
            for _key, df in inputs()
            for value in (df["alpha"] + df["beta"]).tolist()
        ]
        self._mean = sum(totals) / len(totals)

    def save_state(self, run_root: Path) -> None:
        _ = (run_root / "state.json").write_text(json.dumps({"mean": self._mean}))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame": df["frame"],
                "id": df["id"],
                "arena": df["arena"],
                "centered": df["alpha"] + df["beta"] - self._mean,
            }
        )


_ARENA_FEATURES = (_ArenaAlpha, _ArenaBeta, _ArenaCentered)


@pytest.fixture
def arena_features() -> Iterator[None]:
    """Register the stand-in features for one test."""
    with registered_features(*_ARENA_FEATURES):
        yield


def _arena_recipe() -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "joined on the arena",
            "steps": [
                {
                    "id": "alpha",
                    "type": "feature",
                    "feature": _ArenaAlpha.name,
                    "inputs": ["tracks"],
                },
                {
                    "id": "beta",
                    "type": "feature",
                    "feature": _ArenaBeta.name,
                    "inputs": ["tracks"],
                },
                {
                    "id": "centered",
                    "type": "feature",
                    "feature": _ArenaCentered.name,
                    "inputs": [{"step": "alpha"}, {"step": "beta"}],
                },
            ],
        }
    )


def test_a_run_whose_first_step_joined_on_the_arena_is_verified(
    tmp_path: Path, arena_features: None
) -> None:
    ds = make_dataset(tmp_path / "joined")
    add_tracks_variant(
        ds, "convert-trex.0.2-1111111111", "seq_a", "seq_b", std_format="mosaic_v1"
    )
    recipe = _arena_recipe()
    _ = run_pipeline(ds, recipe)
    planned = plan_pipeline(ds, recipe).step("centered")
    index = feature_index(feature_index_path(ds, planned.storage_name)).read()
    assert set(index["consumed_summary_join"]) == {"arena"}
    path = tmp_path / "arena-recipe.json"
    _ = path.write_text(recipe.model_dump_json())

    result = runner.invoke(
        app,
        [
            "pipeline",
            "verify",
            "--manifest",
            str(ds.manifest_path),
            "--recipe",
            f"@{path}",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    by_step = {step["step"]: step["status"] for step in report["steps"]}
    assert by_step["centered"] == "ok"
