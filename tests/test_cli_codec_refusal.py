"""How ``mosaic run`` and ``mosaic track`` report a tracker's refusal of a codec.

Lightning Pose reads AV1 on no GPU, so every Lightning Pose run on AV1 media is
refused before its environment starts. Both verbs print the refusal as a message
and exit with the code reserved for a refusal. Under ``--json`` the refusal is the one JSON value on
stdout, in the shape of a pipeline step's refusal. ``mosaic pipeline run`` does the
same, naming the step that refused and its attempt. The attempt's run-log records
the same refusal, so a reader of the ledger can tell it from a crash.

The refusal passes through the real code from the op to the verb.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.pipeline.refusal import REFUSED_EXIT_CODE
from mosaic.core.pipeline.run_log import read_run, run_log_dir

from tests.helpers import (
    index_media_sequence,
    install_fake_litpose,
    make_dataset,
    write_litpose_model,
)

runner = CliRunner()

_DATASET = "ds"
"""The dataset directory under ``tmp_path``, where each attempt's run-log is."""

_REASON = "That reader decodes av1 in no environment, so it is not tested."
"""Why Lightning Pose is refused an AV1 file, in the refusal's own words."""


@pytest.fixture
def refused_command(
    tmp_path: Path,
    write_cfr_mp4: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[str]]:
    """Each verb's command line for a Lightning Pose run that its codec refuses."""
    ds = make_dataset(tmp_path / _DATASET)
    write_cfr_mp4(ds.get_root("media_raw") / "s" / "clip0.mp4")
    index_media_sequence(ds, "s", ["clip0.mp4"])
    _ = install_fake_litpose(monkeypatch)
    model = str(write_litpose_model(tmp_path / "model"))
    manifest = str(ds.manifest_path)
    recipe = tmp_path / "recipe.json"
    _ = recipe.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "track with Lightning Pose",
                "steps": [
                    {
                        "id": "track",
                        "type": "op",
                        "kind": "litpose",
                        "params": {"model_path": model},
                    }
                ],
            }
        )
    )
    return {
        "run": [
            "run",
            "--kind",
            "litpose",
            "-m",
            manifest,
            "--params",
            json.dumps({"model_path": model}),
        ],
        "track": ["track", "litpose", "-m", manifest, "--set", f"model_path={model}"],
        "pipeline": [
            "pipeline",
            "run",
            "--recipe",
            f"@{recipe}",
            "--manifest",
            manifest,
        ],
    }


_STEP = {"run": "", "track": "", "pipeline": "track"}
"""The step each verb's refusal names: only a pipeline's run has one."""


@pytest.mark.parametrize("verb", ["run", "track", "pipeline"])
def test_a_codec_refusal_is_printed_as_a_message(
    verb: str, refused_command: dict[str, list[str]]
) -> None:
    result = runner.invoke(app, refused_command[verb])

    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == REFUSED_EXIT_CODE
    assert "[mosaic] refused (undecodable_codec)" in result.stderr
    assert "(, s) resolves to clip0.mp4, which is av1" in result.stderr
    assert _REASON in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("verb", ["run", "track", "pipeline"])
def test_a_codec_refusal_is_a_json_refusal_under_json(
    verb: str, refused_command: dict[str, list[str]]
) -> None:
    result = runner.invoke(app, [*refused_command[verb], "--json"])

    assert isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == REFUSED_EXIT_CODE
    assert "(, s) resolves to clip0.mp4, which is av1" in result.stderr
    payload = json.loads(result.stdout)
    assert set(payload) == {"execution_id", "status", "reason", "step", "error_json"}
    assert payload["status"] == "refused"
    assert payload["reason"] == "undecodable_codec"
    assert payload["step"] == _STEP[verb]
    assert payload["execution_id"]
    recorded = json.loads(payload["error_json"])
    assert recorded["reason"] == "undecodable_codec"
    assert recorded["step"] == _STEP[verb]
    assert _REASON in recorded["message"]


@pytest.mark.parametrize("verb", ["run", "track", "pipeline"])
def test_a_codec_refusal_is_recorded_as_a_refusal_in_the_run_log(
    verb: str, refused_command: dict[str, list[str]], tmp_path: Path
) -> None:
    """The attempt is recorded as failed, with the refusal that ``--json`` prints."""
    result = runner.invoke(app, [*refused_command[verb], "--json"])

    assert result.exit_code == REFUSED_EXIT_CODE
    payload = json.loads(result.stdout)
    logged = read_run(run_log_dir(tmp_path / _DATASET), payload["execution_id"])
    assert logged is not None
    assert logged["status"] == "failed"
    recorded = json.loads(logged["error_json"])
    assert recorded["reason"] == "undecodable_codec"
    assert recorded["step"] == _STEP[verb]
    assert recorded == json.loads(payload["error_json"])
