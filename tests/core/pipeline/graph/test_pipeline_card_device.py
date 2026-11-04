"""A step leased one card refuses a device that names another, and a plan cautions it."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import (
    Plan,
    Recipe,
    StepRefused,
    plan_pipeline,
    plan_wire,
    submit_request,
)
from mosaic.core.pipeline.graph import cpu_fit_advice
from mosaic.core.pipeline.graph.cpu_fit_advice import card_device_cautions
from mosaic.core.pipeline.graph.gpu_lease import (
    GPU_CARD_VAR,
    leases_cards_here,
    names_another_card,
)
from mosaic.core.pipeline.graph.step_preflight import preflight
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    TWO_CARDS,
    add_tracks_variant,
    install_fake_nvidia_smi,
    make_dataset,
)

runner = CliRunner()

CARD = "0:GPU-11111111-1111-1111-1111-111111111111"
STEP = "track"


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two schema-valid track tables, so the plan resolves."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(
        dataset,
        "convert-trex.0.2-1111111111",
        "seq_a",
        "seq_b",
        std_format="mosaic_v1",
    )
    return dataset


def _recipe(tmp_path: Path, **params: object) -> Recipe:
    """One tracker step on a GPU lane, whose weights are a file that exists."""
    weights = tmp_path / "weights.pt"
    _ = weights.write_bytes(b"weights")
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "steps": [
                {
                    "id": STEP,
                    "type": "op",
                    "kind": "ultralytics",
                    "params": {"model_path": str(weights), **params},
                }
            ],
        }
    )


def _plan(dataset: Dataset, tmp_path: Path, **params: object) -> Plan:
    return plan_pipeline(dataset, _recipe(tmp_path, **params), refuse=False)


@pytest.mark.parametrize(
    ("device", "other"),
    [
        (None, False),
        ("0", False),
        ("cuda:0", False),
        ("gpu:0", False),
        ("cuda", False),
        ("gpu", False),
        ("cpu", False),
        ("auto", False),
        ("mps", False),
        ("", False),
        ("1", True),
        ("cuda:1", True),
        ("gpu:1", True),
        ("0,1", True),
        ("0, 1", True),
        ("cuda:0,cuda:1", True),
        ("0,0", True),
        ("0,cpu", False),
    ],
)
def test_a_device_names_another_card_unless_it_is_card_zero_or_no_index(
    device: str | None, other: bool
) -> None:
    assert names_another_card(device) is other


def test_a_non_string_device_names_no_card() -> None:
    assert names_another_card(1) is False


def test_a_host_leases_cards_when_it_has_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert leases_cards_here() is False

    _ = install_fake_nvidia_smi(monkeypatch, tmp_path, TWO_CARDS)

    assert leases_cards_here() is True


@pytest.mark.parametrize("device", ["1", "0,1", "cuda:1"])
def test_a_leased_step_naming_another_card_is_refused(
    tracked: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.setenv(GPU_CARD_VAR, CARD)
    plan = _plan(tracked, tmp_path, device=device)

    with pytest.raises(StepRefused) as refused:
        preflight(tracked, plan, STEP)

    assert refused.value.reason == "gpu_card_not_leased"
    assert refused.value.step_id == STEP
    message = str(refused.value)
    assert CARD in message
    assert repr(device) in message
    assert "0 names that card" in message


def test_the_card_refusal_holds_whatever_allow_partial_says(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GPU_CARD_VAR, CARD)
    plan = _plan(tracked, tmp_path, device="1")

    with pytest.raises(StepRefused) as refused:
        preflight(tracked, plan, STEP, allow_partial=True)

    assert refused.value.reason == "gpu_card_not_leased"


@pytest.mark.parametrize("params", [{"device": "0"}, {"device": "cuda"}, {}])
def test_a_leased_step_naming_card_zero_or_no_card_runs(
    tracked: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    params: dict[str, object],
) -> None:
    monkeypatch.setenv(GPU_CARD_VAR, CARD)

    preflight(tracked, _plan(tracked, tmp_path, **params), STEP)


@pytest.mark.parametrize("device", ["1", "0,1", "0"])
def test_a_step_without_a_lease_is_not_refused(
    tracked: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    monkeypatch.delenv(GPU_CARD_VAR, raising=False)

    preflight(tracked, _plan(tracked, tmp_path, device=device), STEP)


def test_the_cli_exits_with_the_reserved_code_and_logs_the_card_refusal(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(GPU_CARD_VAR, CARD)
    request = submit_request(tracked, _recipe(tmp_path, device="1")).request

    result = runner.invoke(
        app,
        [
            "run",
            "--json",
            "--manifest",
            str(tracked.manifest_path),
            "--graph-request",
            request.request_id,
            "--step",
            STEP,
            "--execution-id",
            request.execution_of(STEP),
        ],
    )

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "gpu_card_not_leased"
    logged = read_run(run_log_dir(tracked.base_dir), request.execution_of(STEP))
    assert logged is not None
    assert logged["status"] == "failed"
    assert json.loads(logged["error_json"])["reason"] == "gpu_card_not_leased"


def _cautions(plan: Plan) -> list[str]:
    return [
        str(caution)
        for caution in plan_wire(plan).cautions
        if caution.step == STEP and caution.where == "params.device"
    ]


def test_a_plan_cautions_a_gpu_step_naming_another_card_on_a_host_that_leases(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpu_fit_advice, "leases_cards_here", lambda: True)

    [caution] = _cautions(_plan(tracked, tmp_path, device="1"))

    assert "'1'" in caution
    assert "one card" in caution
    assert "0" in caution


def test_a_plan_does_not_caution_card_zero_or_a_host_that_leases_nothing(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpu_fit_advice, "leases_cards_here", lambda: True)
    assert _cautions(_plan(tracked, tmp_path, device="0")) == []
    assert _cautions(_plan(tracked, tmp_path)) == []

    monkeypatch.setattr(cpu_fit_advice, "leases_cards_here", lambda: False)
    assert _cautions(_plan(tracked, tmp_path, device="1")) == []


def test_a_cpu_lane_step_is_not_cautioned_for_its_device(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cpu_fit_advice, "leases_cards_here", lambda: True)
    plan = _plan(tracked, tmp_path, device="1")
    on_cpu = tuple(replace(step, lane="feature-compute") for step in plan.steps)

    assert card_device_cautions(on_cpu) == ()
    assert len(card_device_cautions(plan.steps)) == 1


def test_the_caution_changes_neither_the_fingerprint_nor_the_lanes(
    tracked: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tracked, tmp_path, device="1")
    monkeypatch.setattr(cpu_fit_advice, "leases_cards_here", lambda: False)
    quiet = plan_wire(plan)
    monkeypatch.setattr(cpu_fit_advice, "leases_cards_here", lambda: True)
    cautioned = plan_wire(plan)

    assert len(cautioned.cautions) == len(quiet.cautions) + 1
    assert cautioned.fingerprint == quiet.fingerprint
    assert [step.lane for step in cautioned.steps] == [
        step.lane for step in quiet.steps
    ]
