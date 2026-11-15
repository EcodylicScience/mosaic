"""``mosaic features define``: store a feature's source in a dataset."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.pipeline.definitions.store import DefinitionRecord
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from tests.helpers import (
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    invoke_json,
    invoke_ok,
    make_dataset,
)


class _Finding(BaseModel):
    check: str
    field: str | None
    message: str


class _InvalidDetail(BaseModel):
    name: str
    findings: list[_Finding]


def _define(tmp_path: Path, manifest: Path, name: str, source: str) -> list[str]:
    path = tmp_path / f"{name}.py"
    _ = path.write_text(source, encoding="utf-8")
    return ["features", "define", name, str(path), "--manifest", str(manifest)]


def _refused(args: list[str]) -> ErrorEnvelope:
    """Run *args*, assert a failure, and return the error on standard output."""
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    return ErrorEnvelope.model_validate_json(result.stdout)


def test_json_prints_a_record_that_validates(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = _define(tmp_path, manifest, MINIMAL_NAME, MINIMAL_SOURCE)

    payload = invoke_json([*args, "--json"])

    record = DefinitionRecord.model_validate(payload)
    assert record.name == MINIMAL_NAME
    assert not record.reused
    assert len(record.digest) == 64


def test_defining_the_same_source_again_prints_reused(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = [*_define(tmp_path, manifest, MINIMAL_NAME, MINIMAL_SOURCE), "--json"]

    first = invoke_json(args)
    second = invoke_json(args)

    assert second["reused"] is True
    assert second["digest"] == first["digest"]


def test_the_sentence_names_the_name_the_digest_prefix_and_the_date(
    tmp_path: Path,
) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = _define(tmp_path, manifest, MINIMAL_NAME, MINIMAL_SOURCE)

    first = invoke_ok(args).stdout
    second = invoke_ok(args).stdout
    record = DefinitionRecord.model_validate(invoke_json([*args, "--json"]))

    for sentence in (first, second):
        assert MINIMAL_NAME in sentence
        assert record.digest[:12] in sentence
        assert record.digest[:13] not in sentence
        assert record.defined_on.isoformat() in sentence
    assert "Reused" not in first
    assert "Reused" in second


def test_a_refused_source_exits_with_its_findings(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = _define(tmp_path, manifest, MINIMAL_NAME, "raise RuntimeError('boom')\n")

    error = _refused([*args, "--json"])

    assert error.kind == "invalid"
    assert error.reason == "definition_invalid"
    detail = _InvalidDetail.model_validate(error.detail)
    assert detail.name == MINIMAL_NAME
    assert detail.findings
    assert any("boom" in finding.message for finding in detail.findings)


def test_a_refused_source_prints_each_finding_on_standard_error(
    tmp_path: Path,
) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = _define(tmp_path, manifest, MINIMAL_NAME, "raise RuntimeError('boom')\n")

    result = CliRunner().invoke(app, args)

    assert result.exit_code == 1
    assert MINIMAL_NAME in result.stderr
    assert "boom" in result.stderr


def test_a_built_in_name_is_definition_name_taken(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = _define(tmp_path, manifest, "speed-angvel", MINIMAL_SOURCE)

    error = _refused([*args, "--json"])

    assert error.kind == "conflict"
    assert error.reason == "definition_name_taken"


def test_a_name_outside_the_grammar_is_definition_name_invalid(
    tmp_path: Path,
) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path
    args = _define(tmp_path, manifest, "Not_Kebab", MINIMAL_SOURCE)

    error = _refused([*args, "--json"])

    assert error.reason == "definition_name_invalid"


def test_a_missing_file_exits_without_defining(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path / "dataset").manifest_path

    result = CliRunner().invoke(
        app,
        [
            "features",
            "define",
            MINIMAL_NAME,
            str(tmp_path / "absent.py"),
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code != 0
    assert "absent.py" in result.output
