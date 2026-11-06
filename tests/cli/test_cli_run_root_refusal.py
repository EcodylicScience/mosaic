"""How the verbs report a training or a fit that another execution already holds.

A one-shot op takes its run root with an exclusive create before it writes, and a
feature fit does the same before it fits. When a live peer holds the root, the op
refuses with ``run_root_held`` rather than returning a model another execution is
mid-write. Every verb that runs an op prints that as a refusal and exits with the
code reserved for one, and the attempt's run-log records it as a failed attempt
carrying the holder.

The refusal passes through the real op, the real claim and the real verb.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mosaic.cli.run as cli_run
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.json_object import JSON_OBJECT
from mosaic.core.pipeline.graph import Recipe, submit_request
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.markers import new_inflight, write_inflight
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.core.pipeline.run import resolve_feature_identity
from mosaic.core.pipeline.storage_names import derive_storage_name
from mosaic.core.scope import Scope
from mosaic.runlog import run_log_path

from tests.helpers import (
    FakeTrainer,
    FitControl,
    FittingFeature,
    add_track_sequences,
    make_dataset,
    write_data_yaml,
)

runner = CliRunner()

_PEER = "SOMEONE-ELSE"
"""The execution whose live claim holds the run root."""


def _recipe(params: dict[str, object]) -> Recipe:
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "train a pose model",
            "steps": [
                {"id": "train", "type": "op", "kind": "train-pose", "params": params}
            ],
        }
    )


class _Held:
    """A dataset whose planned model root a live peer holds."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.trainer = FakeTrainer()
        self.trainer.install(monkeypatch)
        self.ds: Dataset = make_dataset(tmp_path / "ds")
        self.params: dict[str, object] = {
            "data": str(write_data_yaml(tmp_path)),
            "epochs": 2,
            "device": "cpu",
        }
        self.recipe = _recipe(self.params)
        self.manifest = str(self.ds.manifest_path)
        planned = submit_request(self.ds, self.recipe).plan.step("train")
        assert planned.model_run_id, "a training step names its model before it runs"
        write_inflight(
            model_run_root(self.ds, "train-pose", planned.model_run_id),
            new_inflight(
                execution_id=_PEER,
                host="otherhost",
                pid=4242,
                phase=None,
                idle_seconds=3600.0,
            ),
        )

    def command(self, verb: str, tmp_path: Path) -> list[str]:
        if verb == "run":
            return [
                "run",
                "--kind",
                "train-pose",
                "-m",
                self.manifest,
                "--params",
                json.dumps(self.params),
            ]
        if verb == "pipeline":
            recipe = tmp_path / "recipe.json"
            _ = recipe.write_text(self.recipe.model_dump_json())
            return [
                "pipeline",
                "run",
                "--recipe",
                f"@{recipe}",
                "--manifest",
                self.manifest,
            ]
        submitted = submit_request(self.ds, self.recipe)
        return [
            "run",
            "-m",
            self.manifest,
            "--graph-request",
            submitted.request.request_id,
            "--step",
            "train",
        ]


@pytest.fixture
def held(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Held:
    return _Held(tmp_path, monkeypatch)


_VERBS = ["run", "pipeline", "graph-step"]


@pytest.mark.parametrize("verb", _VERBS)
def test_a_held_run_root_is_a_refusal_not_a_traceback(
    verb: str, held: _Held, tmp_path: Path
) -> None:
    result = runner.invoke(app, held.command(verb, tmp_path))

    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == REFUSED_EXIT_CODE
    assert "[mosaic] refused (run_root_held)" in result.stderr
    assert _PEER in result.stderr
    assert "not training it again" in result.stderr
    assert "Traceback" not in result.stderr
    assert held.trainer.calls == 0


@pytest.mark.parametrize("verb", _VERBS)
def test_a_held_run_root_is_a_json_refusal_naming_the_holder(
    verb: str, held: _Held, tmp_path: Path
) -> None:
    result = runner.invoke(app, [*held.command(verb, tmp_path), "--json"])

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    assert payload["status"] == "refused"
    assert payload["reason"] == "run_root_held"
    assert payload["detail"]["holder_execution_id"] == _PEER
    assert payload["detail"]["holder_host"] == "otherhost"
    assert payload["detail"]["holder_pid"] == 4242


@pytest.mark.parametrize("verb", _VERBS)
def test_a_held_run_root_ends_the_attempt_as_a_recorded_refusal(
    verb: str, held: _Held, tmp_path: Path
) -> None:
    """The last event is ``failed``, with the refusal that ``--json`` prints."""
    result = runner.invoke(app, [*held.command(verb, tmp_path), "--json"])

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    lines = run_log_path(held.ds.base_dir, payload["execution_id"]).read_text()
    last = JSON_OBJECT.validate_json(lines.splitlines()[-1])
    assert last["ev"] == "failed"
    error = last["error"]
    assert isinstance(error, str)
    recorded = json.loads(error)
    assert recorded["reason"] == "run_root_held"
    assert recorded["holder_execution_id"] == _PEER
    assert recorded == json.loads(payload["error_json"])


def test_a_held_feature_fit_is_a_refusal_naming_the_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fit takes its run root as a training does, and a verb refuses the same way."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, "seq_a")
    control = FitControl()
    feature = FittingFeature(control)
    run_id, _resolved = resolve_feature_identity(
        ds, feature, scope=Scope(entries=[("", "seq_a")])
    )
    root = feature_run_root(
        ds, derive_storage_name(feature.name, feature.inputs.storage_suffix()), run_id
    )
    root.mkdir(parents=True)
    write_inflight(
        root,
        new_inflight(
            execution_id=_PEER,
            host="otherhost",
            pid=4242,
            phase=None,
            idle_seconds=3600.0,
        ),
    )

    def build_feature(*_args: object, **_kwargs: object) -> FittingFeature:
        return feature

    monkeypatch.setattr(cli_run, "build_feature", build_feature)
    result = runner.invoke(
        app,
        ["run", "--feature", feature.name, "-m", str(ds.manifest_path), "--json"],
    )

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    assert payload["reason"] == "run_root_held"
    assert payload["detail"]["holder_execution_id"] == _PEER
    assert control.fits == 0
