"""``mosaic pipeline score``: a finished recipe scored on held-out data added afterwards.

The study holds the ``train`` group only. A recipe was run on it with its
fitting step scoped to that group, and the held-out file is added to a copy.
What the command promises is that adding the file moves no run id, so the
classifier it scores is the one the study holds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    Recipe,
    ScoreAnswers,
    ScoreProtocol,
    plan_pipeline,
    run_pipeline,
)
from mosaic.core.pipeline.graph.score import SCORES_JSON, SCORES_TXT, md5_of_file
from mosaic.core.scope import Scope
from tests.helpers import (
    assert_tree_unchanged,
    calms21_dataset,
    classifier_recipe,
    tree_snapshot,
    write_calms21_npy,
)

runner = CliRunner()

TRAIN = Scope(groups=["train"])
SCOPES = {"labeled": TRAIN}
HELDOUT_SEQUENCES = {"a": {"t1": 90, "t2": 90}}
PROTOCOL = ["--perspective", "0", "--average-over", "0", "--average-over", "1"]


def _study(base: Path, *, test_inside: bool = False) -> Dataset:
    """A dataset holding the ``train`` group, and optionally a ``test`` group too."""
    files = {"train": {"a": {"s1": 90, "s2": 90, "s3": 90}}}
    if test_inside:
        files["test"] = HELDOUT_SEQUENCES
    return calms21_dataset(base, files, seed=11, n_classes=2)


@pytest.fixture
def study(tmp_path: Path) -> Dataset:
    """The study, with the two-seed classifier recipe run over the training group."""
    ds = _study(tmp_path / "study")
    _ = run_pipeline(ds, classifier_recipe("clf_a", "clf_b"), step_scopes=SCOPES)
    return ds


@pytest.fixture
def heldout(tmp_path: Path) -> Path:
    """The held-out file, outside the study."""
    path = tmp_path / "heldout" / "test.npy"
    write_calms21_npy(path, HELDOUT_SEQUENCES, seed=29, n_classes=2)
    return path


def _write(tmp_path: Path, recipe: Recipe, models: dict[str, list[str]]) -> list[str]:
    """The recipe and answers files, as ``score`` arguments."""
    recipe_file = tmp_path / "recipe.json"
    _ = recipe_file.write_text(recipe.model_dump_json(), encoding="utf-8")
    answers_file = tmp_path / "answers.json"
    _ = answers_file.write_text(json.dumps({"models": models}), encoding="utf-8")
    return ["--recipe", f"@{recipe_file}", "--answers", str(answers_file)]


def _score(
    study: Dataset,
    tmp_path: Path,
    *flags: str,
    recipe: Recipe | None = None,
    models: dict[str, list[str]] | None = None,
    scopes: bool = True,
) -> Any:
    """Invoke ``pipeline score --json`` and return the click result."""
    arguments = [
        "pipeline",
        "score",
        "--manifest",
        str(study.manifest_path),
        *_write(
            tmp_path,
            recipe or classifier_recipe("clf_a", "clf_b"),
            models or {"final": ["clf_a", "clf_b"]},
        ),
        *PROTOCOL,
        *(["--step-scope", 'labeled={"groups": ["train"]}'] if scopes else []),
        *flags,
        "--json",
    ]
    return runner.invoke(app, arguments)


def test_a_held_out_file_is_scored_on_a_copy_and_the_study_is_left_alone(
    study: Dataset, heldout: Path, tmp_path: Path
) -> None:
    out = tmp_path / "scored"
    before = tree_snapshot(study.base_dir)

    result = _score(
        study,
        tmp_path,
        "--heldout",
        str(heldout),
        "--src-format",
        "calms21_npy",
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--expect-md5",
        md5_of_file(heldout),
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.stderr
    assert_tree_unchanged(study.base_dir, before)
    report = json.loads(result.stdout)
    assert report["moved"] == []
    assert report["heldout_scope"] == {"groups": ["test"]}
    assert report["heldout_entries"] == 2
    [model] = report["models"]
    assert model["name"] == "final"
    assert [member["step"] for member in model["members"]] == ["clf_a", "clf_b"]
    for member in model["members"]:
        assert member["n_scored"] > 0
        assert 0.0 <= member["macro_f1"] <= 1.0
    assert model["macro_f1_sd"] is not None
    assert "mean ± sd" in report["table"]
    assert json.loads((out / SCORES_JSON).read_text())["models"] == report["models"]
    assert "[final]" in (out / SCORES_TXT).read_text()
    assert not (study.base_dir / SCORES_JSON).exists()


def test_the_held_out_data_enters_the_copy_and_never_the_study(
    study: Dataset, heldout: Path, tmp_path: Path
) -> None:
    out = tmp_path / "scored"

    result = _score(
        study,
        tmp_path,
        "--heldout",
        str(heldout),
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.stderr
    from mosaic.core.dataset import open_dataset

    copy = open_dataset(out / study.manifest_path.name)
    assert {s.id for s in copy.scan_sources("tracks")} == {"calms21", "heldout"}
    assert {s.id for s in study.scan_sources("tracks")} == {"calms21"}
    planned = plan_pipeline(
        open_dataset(study.manifest_path, ensure_roots=False),
        classifier_recipe("clf_a", "clf_b"),
        step_scopes=SCOPES,
        refuse=False,
    )
    assert planned.step("clf_a").coverage.target == {
        ("train", "s1"),
        ("train", "s2"),
        ("train", "s3"),
    }


def test_a_fit_over_every_group_moves_run_ids_and_nothing_is_run(
    tmp_path: Path, heldout: Path
) -> None:
    """With no step scope the templates fit on whatever groups the dataset holds."""
    ds = _study(tmp_path / "study")
    _ = run_pipeline(ds, classifier_recipe("clf_a"))
    out = tmp_path / "scored"

    result = _score(
        ds,
        tmp_path,
        "--heldout",
        str(heldout),
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--out",
        str(out),
        recipe=classifier_recipe("clf_a"),
        models={"final": ["clf_a"]},
        scopes=False,
    )

    assert result.exit_code == 1
    assert "RUN IDS CHANGED: labeled" in result.stderr
    assert "RUN IDS CHANGED: clf_a" in result.stderr
    report = json.loads(result.stdout)
    assert {moved["step"] for moved in report["moved"]} == {"labeled", "clf_a"}
    assert report["models"] == []
    assert not (out / SCORES_JSON).exists()


def test_a_dry_run_stops_at_the_comparison_and_the_next_run_continues_from_it(
    study: Dataset, heldout: Path, tmp_path: Path
) -> None:
    out = tmp_path / "scored"
    held = [
        "--heldout",
        str(heldout),
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--out",
        str(out),
    ]
    before = tree_snapshot(study.base_dir)

    dry = _score(study, tmp_path, *held, "--dry-run")

    assert dry.exit_code == 0, dry.stderr
    assert_tree_unchanged(study.base_dir, before)
    planned = json.loads(dry.stdout)
    assert planned["dry_run"] is True
    assert planned["moved"] == []
    assert planned["models"] == []
    assert not (out / SCORES_JSON).exists()
    assert (out / "scoring" / "scoring-recipe.json").exists()

    refused = _score(study, tmp_path, *held)
    assert refused.exit_code == 1
    assert "--reuse" in refused.stderr

    stale = _score(
        study, tmp_path, *held, "--reuse", "--expect-plan", "not-the-fingerprint"
    )
    assert stale.exit_code == 1
    assert not (out / SCORES_JSON).exists()

    final = _score(
        study,
        tmp_path,
        *held,
        "--reuse",
        "--expect-plan",
        planned["plan_fingerprint"],
    )
    assert final.exit_code == 0, final.stderr
    assert json.loads(final.stdout)["models"][0]["name"] == "final"
    assert (out / SCORES_JSON).exists()


def test_a_reused_copy_still_compares_with_the_plan_made_before_the_data_joined(
    tmp_path: Path, heldout: Path
) -> None:
    ds = _study(tmp_path / "study")
    _ = run_pipeline(ds, classifier_recipe("clf_a"))
    out = tmp_path / "scored"
    held = [
        "--heldout",
        str(heldout),
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--out",
        str(out),
    ]

    def dry_run(*extra: str) -> Any:
        return _score(
            ds,
            tmp_path,
            *held,
            *extra,
            "--dry-run",
            recipe=classifier_recipe("clf_a"),
            models={"final": ["clf_a"]},
            scopes=False,
        )

    first = dry_run()
    again = dry_run("--reuse")

    assert first.exit_code == again.exit_code == 1
    assert json.loads(again.stdout)["moved"] == json.loads(first.stdout)["moved"] != []


def test_a_test_group_in_the_dataset_is_planned_in_place_by_a_dry_run(
    tmp_path: Path,
) -> None:
    ds = _study(tmp_path / "study", test_inside=True)
    ds.set_splits("train", ["train"])
    ds.set_splits("test", ["test"])
    _ = run_pipeline(ds, classifier_recipe("clf_a"), step_scopes=SCOPES)
    before = tree_snapshot(ds.base_dir)

    result = _score(
        ds,
        tmp_path,
        "--heldout",
        "test",
        "--out",
        str(tmp_path / "scored"),
        "--dry-run",
        recipe=classifier_recipe("clf_a"),
        models={"final": ["clf_a"]},
    )

    assert result.exit_code == 0, result.stderr
    assert_tree_unchanged(ds.base_dir, before)
    assert not (tmp_path / "scored").exists()
    assert json.loads(result.stdout)["heldout_scope"] == {"groups": ["test"]}


def test_a_test_group_in_the_dataset_is_scored_without_adding_anything(
    tmp_path: Path,
) -> None:
    ds = _study(tmp_path / "study", test_inside=True)
    ds.set_splits("train", ["train"])
    ds.set_splits("test", ["test"])
    _ = run_pipeline(ds, classifier_recipe("clf_a"), step_scopes=SCOPES)
    out = tmp_path / "scored"

    result = _score(
        ds,
        tmp_path,
        "--heldout",
        "test",
        "--out",
        str(out),
        recipe=classifier_recipe("clf_a"),
        models={"final": ["clf_a"]},
    )

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["heldout_entries"] is None
    assert report["models"][0]["members"][0]["n_scored"] > 0
    assert (out / SCORES_JSON).exists()


def test_a_fit_over_the_test_group_is_refused_by_the_planner_before_anything_is_added(
    tmp_path: Path,
) -> None:
    ds = _study(tmp_path / "study", test_inside=True)
    _ = ds.convert_all_tracks()
    ds.set_splits("train", ["train"])
    ds.set_splits("test", ["test"])

    result = _score(
        ds,
        tmp_path,
        "--heldout",
        "test",
        "--out",
        str(tmp_path / "scored"),
        "--dry-run",
        recipe=classifier_recipe("clf_a"),
        models={"final": ["clf_a"]},
        scopes=False,
    )

    assert result.exit_code == 1
    assert "held-out group(s) 'test' (test)" in result.stderr


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--heldout", "nowhere"], "neither an existing path nor a group"),
        (["--heldout", "{heldout}", "--expect-md5", "0" * 32], "expected md5"),
        (["--heldout", "{heldout}", "--src-format", "boris_csv"], "differs"),
    ],
)
def test_a_heldout_that_does_not_fit_is_refused(
    study: Dataset,
    heldout: Path,
    tmp_path: Path,
    flags: list[str],
    message: str,
) -> None:
    out = tmp_path / "scored"
    given = [flag.replace("{heldout}", str(heldout)) for flag in flags]

    result = _score(study, tmp_path, *given, "--out", str(out))

    assert result.exit_code == 1
    assert message in result.stderr


def test_answers_must_name_a_feature_step_of_the_recipe(
    study: Dataset, heldout: Path, tmp_path: Path
) -> None:
    result = _score(
        study,
        tmp_path,
        "--heldout",
        str(heldout),
        "--out",
        str(tmp_path / "scored"),
        models={"final": ["no_such_step"]},
    )

    assert result.exit_code == 1
    assert "'no_such_step'" in result.stderr
    assert not (tmp_path / "scored").exists()


def test_the_answers_name_at_least_one_step_per_model() -> None:
    with pytest.raises(ValueError, match="lists no step"):
        _ = ScoreAnswers.model_validate({"models": {"final": []}})
    with pytest.raises(ValueError, match="no model"):
        _ = ScoreAnswers.model_validate({"models": {}})


def test_a_rerun_after_a_partly_failed_conversion_scores_every_held_out_file(
    study: Dataset, tmp_path: Path
) -> None:
    folder = tmp_path / "held"
    write_calms21_npy(folder / "test1.npy", {"a": {"t1": 90}}, seed=29, n_classes=2)
    _ = (folder / "test2.npy").write_bytes(b"not an array")
    out = tmp_path / "scored"
    flags = [
        "--heldout",
        str(folder),
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--out",
        str(out),
    ]

    first = _score(study, tmp_path, *flags)
    write_calms21_npy(folder / "test2.npy", {"a": {"t2": 90}}, seed=31, n_classes=2)
    again = _score(study, tmp_path, *flags, "--reuse")

    assert first.exit_code == 1
    assert "did not convert" in first.stderr
    assert again.exit_code == 0, again.stderr
    report = json.loads(again.stdout)
    assert report["heldout_scope"] == {"groups": ["test1", "test2"]}
    assert report["heldout_entries"] == 2
    assert {m["n_scored"] for m in report["models"][0]["members"]} == {180}


def test_a_reused_copy_is_refused_for_other_held_out_data(
    study: Dataset, heldout: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other" / "other.npy"
    write_calms21_npy(other, HELDOUT_SEQUENCES, seed=3, n_classes=2)
    out = tmp_path / "scored"
    shared = ["--multi-sequences-per-file", "--group-from", "filename"]

    first = _score(
        study,
        tmp_path,
        "--heldout",
        str(heldout),
        *shared,
        "--out",
        str(out),
        "--dry-run",
    )
    again = _score(
        study, tmp_path, "--heldout", str(other), *shared, "--out", str(out), "--reuse"
    )

    assert first.exit_code == 0, first.stderr
    assert again.exit_code == 1
    assert "other --heldout data or step scopes" in again.stderr


def test_a_recipe_step_that_only_scores_does_not_stop_the_scoring(
    tmp_path: Path, heldout: Path
) -> None:
    own_evaluation: dict[str, Any] = {
        "id": "cv_eval",
        "type": "feature",
        "feature": "evaluate-labels",
        "inputs": [{"step": "clf_a"}],
        "params": {"labels": {"step": "labels"}, "perspective": 0},
    }
    base = classifier_recipe("clf_a")
    recipe = Recipe.model_validate(
        {
            **base.model_dump(mode="json"),
            "steps": [*base.model_dump(mode="json")["steps"], own_evaluation],
        }
    )
    ds = _study(tmp_path / "study")
    _ = run_pipeline(ds, recipe, step_scopes=SCOPES)

    result = _score(
        ds,
        tmp_path,
        "--heldout",
        str(heldout),
        "--multi-sequences-per-file",
        "--group-from",
        "filename",
        "--out",
        str(tmp_path / "scored"),
        recipe=recipe,
        models={"final": ["clf_a"]},
    )

    assert result.exit_code == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["moved"] == []
    assert any("cv_eval" in warning for warning in report["warnings"])
    assert report["models"][0]["members"][0]["n_scored"] > 0


def test_the_protocol_reaches_each_evaluation() -> None:
    protocol = ScoreProtocol(
        perspective="pooled",
        average_over=[0, 1],
        classes=[0, 1, 2],
        prediction_column="label",
    )

    assert protocol.evaluation_params("labels") == {
        "labels": {"step": "labels"},
        "perspective": "pooled",
        "average_over": [0, 1],
        "classes": [0, 1, 2],
        "prediction_column": "label",
    }
    assert ScoreProtocol().evaluation_params("labels") == {"labels": {"step": "labels"}}
