"""A plan advises a GPU for a large xgboost fit on the CPU when the host has one."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, plan_pipeline, plan_wire, run_pipeline
from mosaic.core.pipeline.graph import cpu_fit_advice
from tests.helpers import (
    CONVERT_LABELS_STEP,
    CONVERT_TRACKS_STEP,
    calms21_dataset,
    classifier_recipe,
    modules_loaded_by,
)

SEQUENCES = {"train": {"a": {"s1": 90, "s2": 90, "s3": 90}}}


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Three converted entries, so the plan asks each step for them."""
    ds = calms21_dataset(tmp_path / "study", SEQUENCES, seed=11, n_classes=2)
    converting = Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [CONVERT_TRACKS_STEP, CONVERT_LABELS_STEP],
        }
    )
    _ = run_pipeline(ds, converting)
    return ds


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cpu_fit_advice, "LARGE_CPU_FIT_ENTRIES", 2)


def _on_gpu_host(monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    monkeypatch.setattr(cpu_fit_advice, "host_has_gpu", lambda: present)


def _recipe(**params: object) -> Recipe:
    document = classifier_recipe("clf").model_dump(mode="json")
    for step in document["steps"]:
        if step["id"] == "clf":
            step["params"].update(params)
    return Recipe.model_validate(document)


def _advice(dataset: Dataset, recipe: Recipe) -> list[str]:
    plan = plan_pipeline(dataset, recipe, refuse=False)
    return [
        str(caution) for caution in plan_wire(plan).cautions if caution.step == "clf"
    ]


def test_a_large_cpu_fit_on_a_host_with_a_gpu_is_cautioned(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu_host(monkeypatch, True)

    [caution] = _advice(dataset, _recipe())

    assert "clf" in caution
    assert "device" in caution
    assert "cuda" in caution


def test_a_fit_on_cuda_is_not_cautioned(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu_host(monkeypatch, True)

    assert _advice(dataset, _recipe(device="cuda")) == []


def test_a_host_without_a_gpu_is_not_cautioned(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu_host(monkeypatch, False)

    assert _advice(dataset, _recipe()) == []


def test_a_small_fit_is_not_cautioned(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu_host(monkeypatch, True)
    monkeypatch.setattr(cpu_fit_advice, "LARGE_CPU_FIT_ENTRIES", 1000)

    assert _advice(dataset, _recipe()) == []


def test_the_advice_changes_neither_the_fingerprint_nor_the_lane(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu_host(monkeypatch, False)
    quiet = plan_wire(plan_pipeline(dataset, _recipe(), refuse=False))
    _on_gpu_host(monkeypatch, True)
    advised = plan_wire(plan_pipeline(dataset, _recipe(), refuse=False))

    assert advised.fingerprint == quiet.fingerprint
    assert [step.lane for step in advised.steps] == [step.lane for step in quiet.steps]


@pytest.mark.parametrize(
    ("device", "lane"), [("cuda", "gpu-train"), ("cpu", "feature-compute")]
)
def test_a_classifier_plans_the_lane_of_its_device(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch, device: str, lane: str
) -> None:
    _on_gpu_host(monkeypatch, True)

    plan = plan_pipeline(dataset, _recipe(device=device), refuse=False)

    [classifier] = [step for step in plan.steps if step.step_id == "clf"]
    assert classifier.lane == lane


def test_the_caution_says_that_cuda_moves_the_step_to_a_gpu_lane(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    _on_gpu_host(monkeypatch, True)

    [caution] = _advice(dataset, _recipe())

    assert "GPU lane" in caution
    assert "stays on its lane" not in caution


@pytest.mark.parametrize(
    ("visible", "device_present", "expected"),
    [
        ("0", False, True),
        ("1,2", False, True),
        ("-1", True, False),
        ("", True, False),
        (None, True, True),
        (None, False, False),
    ],
)
def test_a_gpu_is_visible_through_the_environment_or_the_device_file(
    monkeypatch: pytest.MonkeyPatch,
    visible: str | None,
    device_present: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(cpu_fit_advice, "_torch_installed", lambda: True)
    monkeypatch.setattr(cpu_fit_advice, "_nvidia_device_exists", lambda: device_present)
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    assert cpu_fit_advice.host_has_gpu() is expected


def test_a_host_without_torch_has_no_gpu_to_advise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cpu_fit_advice, "_torch_installed", lambda: False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert cpu_fit_advice.host_has_gpu() is False


def test_planning_does_not_import_torch() -> None:
    loaded = modules_loaded_by(
        "import mosaic.core.pipeline.graph.cpu_fit_advice as advice\n"
        "advice.host_has_gpu()",
        ["torch"],
    )

    assert loaded == []
