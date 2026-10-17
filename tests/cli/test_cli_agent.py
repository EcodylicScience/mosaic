"""``mosaic agent``: printing the agent guide and pointing a project at it."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from mosaic.agent import (
    BLOCK_BEGIN,
    BLOCK_END,
    MCP_CONFIG,
    SKILL_DIRECTORY,
    Installation,
    guide_text,
    skill_stub,
)
from mosaic.cli import app
from mosaic.core.json_value import JsonValue
from tests.helpers import make_dataset

runner = CliRunner()


def frontmatter(text: str) -> dict[str, object]:
    """The YAML mapping between a file's opening ``---`` lines."""
    assert text.startswith("---\n"), text[:80]
    header, separator, _body = text[len("---\n") :].partition("\n---\n")
    assert separator, "the frontmatter is never closed"
    parsed: dict[str, object] = yaml.safe_load(header)
    return parsed


def test_python_dash_m_mosaic_runs_the_cli() -> None:
    """The stub and the pointer fall back on this command when no script exists."""
    completed = subprocess.run(
        [sys.executable, "-m", "mosaic", "agent", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "install" in completed.stdout


def test_the_packaged_guide_is_a_skill() -> None:
    fields = frontmatter(guide_text())
    assert fields["name"] == "mosaic"
    assert isinstance(fields["description"], str) and fields["description"]


class TestGuide:
    def test_it_prints_the_guide_then_this_installation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["agent", "guide"])
        assert result.exit_code == 0, result.output
        assert result.output.startswith(guide_text().rstrip())
        installation = result.output[len(guide_text().rstrip()) :]
        assert "## This installation" in installation
        assert "Python interpreter:" in installation
        assert "Documentation:" in installation
        assert "not inside one" in installation

    def test_it_names_the_enclosing_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset = tmp_path / "study"
        assert runner.invoke(app, ["init", str(dataset)]).exit_code == 0
        inside = dataset / "media"
        inside.mkdir(exist_ok=True)
        monkeypatch.chdir(inside)
        result = runner.invoke(app, ["agent", "guide"])
        assert result.exit_code == 0, result.output
        assert f"(manifest {dataset.resolve() / 'dataset.yaml'})" in result.output


def blocks_in(path: Path) -> int:
    return path.read_text(encoding="utf-8").count(BLOCK_BEGIN)


class TestInstall:
    def test_it_writes_a_stub_that_runs_the_guide(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code == 0, result.output
        stub = (tmp_path / SKILL_DIRECTORY / "SKILL.md").read_text(encoding="utf-8")
        fields = frontmatter(stub)
        assert fields["name"] == "mosaic"
        assert "imports mosaic" in str(fields["description"])
        assert "agent guide" in stub
        assert "-m mosaic agent guide" in stub
        assert "Never edit the installed mosaic package" in stub
        assert str(tmp_path / SKILL_DIRECTORY) in result.output
        assert "To remove" in result.output

    def test_a_stub_without_a_script_names_one_command(self, tmp_path: Path) -> None:
        installation = Installation(
            executable=None,
            interpreter=Path(sys.executable),
            docs=None,
            manifest=None,
        )
        stub = skill_stub(installation)
        assert stub.count(installation.fallback_command) == 1
        assert "not found" not in stub

    def test_it_creates_claude_md_when_no_agent_file_exists(
        self, tmp_path: Path
    ) -> None:
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert blocks_in(tmp_path / "CLAUDE.md") == 1
        assert not (tmp_path / "AGENTS.md").exists()

    def test_a_second_run_replaces_the_block(self, tmp_path: Path) -> None:
        claude = tmp_path / "CLAUDE.md"
        _ = claude.write_text("# Project\n\nOur own notes.\n", encoding="utf-8")
        first = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert first.exit_code == 0, first.output
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code == 0, result.output
        text = claude.read_text(encoding="utf-8")
        assert text.startswith("# Project\n\nOur own notes.\n")
        assert text.count(BLOCK_BEGIN) == 1
        assert text.count(BLOCK_END) == 1
        assert "unchanged:" in result.output

    def test_a_stale_block_is_rewritten_in_place(self, tmp_path: Path) -> None:
        claude = tmp_path / "CLAUDE.md"
        stale = f"before\n{BLOCK_BEGIN}\nold pointer\n{BLOCK_END}\nafter\n"
        _ = claude.write_text(stale, encoding="utf-8")
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code == 0, result.output
        text = claude.read_text(encoding="utf-8")
        assert "old pointer" not in text
        assert text.startswith(f"before\n{BLOCK_BEGIN}\n")
        assert text.endswith(f"{BLOCK_END}\nafter\n")

    def test_both_agent_files_get_the_block(self, tmp_path: Path) -> None:
        for name in ("CLAUDE.md", "AGENTS.md"):
            _ = (tmp_path / name).write_text(f"# {name}\n", encoding="utf-8")
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert blocks_in(tmp_path / "CLAUDE.md") == 1
        assert blocks_in(tmp_path / "AGENTS.md") == 1

    def test_only_agents_md_is_not_joined_by_claude_md(self, tmp_path: Path) -> None:
        _ = (tmp_path / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert blocks_in(tmp_path / "AGENTS.md") == 1
        assert not (tmp_path / "CLAUDE.md").exists()

    def test_no_memory_leaves_the_agent_files_alone(self, tmp_path: Path) -> None:
        agents = tmp_path / "AGENTS.md"
        _ = agents.write_text("# Agents\n", encoding="utf-8")
        result = runner.invoke(app, ["agent", "install", str(tmp_path), "--no-memory"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / SKILL_DIRECTORY / "SKILL.md").is_file()
        assert agents.read_text(encoding="utf-8") == "# Agents\n"
        assert not (tmp_path / "CLAUDE.md").exists()

    def test_no_memory_names_an_earlier_block_for_removal(self, tmp_path: Path) -> None:
        assert runner.invoke(app, ["agent", "install", str(tmp_path)]).exit_code == 0
        result = runner.invoke(app, ["agent", "install", str(tmp_path), "--no-memory"])
        assert result.exit_code == 0, result.output
        removal = result.output.split("To remove", 1)[1]
        assert str(tmp_path.resolve() / "CLAUDE.md") in removal

    def test_an_unclosed_block_is_refused(self, tmp_path: Path) -> None:
        claude = tmp_path / "CLAUDE.md"
        _ = claude.write_text(f"{BLOCK_BEGIN}\nno end\n", encoding="utf-8")
        result = runner.invoke(app, ["agent", "install", str(tmp_path)])
        assert result.exit_code != 0
        assert BLOCK_END in result.output
        assert claude.read_text(encoding="utf-8") == f"{BLOCK_BEGIN}\nno end\n"

    def test_a_missing_directory_is_refused(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent"
        result = runner.invoke(app, ["agent", "install", str(missing)])
        assert result.exit_code != 0
        assert "Directory not found" in result.output
        assert not missing.exists()


def mcp_servers(directory: Path) -> dict[str, dict[str, object]]:
    """The ``mcpServers`` object of *directory*'s ``.mcp.json``."""
    document: dict[str, dict[str, dict[str, object]]] = json.loads(
        (directory / MCP_CONFIG).read_text(encoding="utf-8")
    )
    return document["mcpServers"]


class TestInstallMcp:
    def test_it_registers_the_server_over_the_datasets_below(
        self, tmp_path: Path
    ) -> None:
        dataset = make_dataset(tmp_path / "study", name="study")
        result = runner.invoke(app, ["agent", "install", str(tmp_path), "--mcp"])
        assert result.exit_code == 0, result.output
        entry = mcp_servers(tmp_path)["mosaic"]
        assert entry["type"] == "stdio"
        assert isinstance(entry["command"], str) and entry["command"]
        args = entry["args"]
        assert isinstance(args, list)
        assert args[-4:] == [
            "agent",
            "serve",
            "--dataset",
            str(dataset.manifest_path.resolve()),
        ]
        assert (
            str(tmp_path.resolve() / MCP_CONFIG) in result.output.split("To remove")[1]
        )

    def test_it_keeps_the_other_servers_and_replaces_its_own(
        self, tmp_path: Path
    ) -> None:
        other: dict[str, JsonValue] = {"command": "other-server", "args": []}
        existing: JsonValue = {
            "mcpServers": {"other": other, "mosaic": {"command": "old"}}
        }
        _ = (tmp_path / MCP_CONFIG).write_text(json.dumps(existing), encoding="utf-8")
        named = tmp_path / "named"
        result = runner.invoke(
            app,
            ["agent", "install", str(tmp_path), "--mcp", "--dataset", str(named)],
        )
        assert result.exit_code == 0, result.output
        servers = mcp_servers(tmp_path)
        assert servers["other"] == other
        args = servers["mosaic"]["args"]
        assert isinstance(args, list)
        assert args[-4:] == ["agent", "serve", "--dataset", str(named.resolve())]

    def test_a_directory_without_a_dataset_is_refused(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["agent", "install", str(tmp_path), "--mcp"])
        assert result.exit_code != 0
        assert "--dataset" in result.output
        assert not (tmp_path / MCP_CONFIG).exists()

    def test_a_dataset_without_mcp_is_refused(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["agent", "install", str(tmp_path), "--dataset", str(tmp_path)]
        )
        assert result.exit_code != 0
        assert "--mcp" in result.output

    @pytest.mark.parametrize("config", ["[]", "{not json"])
    def test_a_config_it_cannot_merge_is_refused_before_anything_is_written(
        self, tmp_path: Path, config: str
    ) -> None:
        _ = (tmp_path / MCP_CONFIG).write_text(config, encoding="utf-8")
        result = runner.invoke(
            app,
            ["agent", "install", str(tmp_path), "--mcp", "--dataset", str(tmp_path)],
        )
        assert result.exit_code != 0
        assert "install failed" in result.output
        assert (tmp_path / MCP_CONFIG).read_text(encoding="utf-8") == config
        assert not (tmp_path / SKILL_DIRECTORY).exists()
        assert not (tmp_path / "CLAUDE.md").exists()


class TestServe:
    def test_without_a_dataset_it_refuses_to_start(self) -> None:
        result = runner.invoke(app, ["agent", "serve"])
        assert result.exit_code != 0
        assert "No dataset to serve" in result.output

    def test_an_invalid_config_is_refused(self, tmp_path: Path) -> None:
        config = tmp_path / "server.yaml"
        _ = config.write_text("datasets: 3\n", encoding="utf-8")
        result = runner.invoke(app, ["agent", "serve", "--config", str(config)])
        assert result.exit_code != 0
        assert "is invalid" in result.output

    def test_without_the_sdk_it_names_the_extra(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "mcp", None)
        result = runner.invoke(app, ["agent", "serve", "--dataset", str(tmp_path)])
        assert result.exit_code != 0
        assert 'pip install "mosaic-behavior[mcp]"' in result.output


CONFIRMATION = {
    "hook_event_name": "Elicitation",
    "mcp_server_name": "mosaic",
    "message": (
        "mosaic asks you to confirm submit_pipeline: step 'pose' runs on lane "
        "gpu-infer. Proceed?\n\n"
        "[mosaic-confirm tool=submit_pipeline conditions=gpu_lane]"
    ),
    "mode": "form",
    "requested_schema": {
        "type": "object",
        "properties": {"confirm": {"type": "boolean"}},
        "required": ["confirm"],
    },
}
"""The input of a Claude Code Elicitation hook for one confirmation."""


class TestConfirmHook:
    def test_it_prints_the_answer_for_its_stdin(self) -> None:
        stdin = json.dumps(CONFIRMATION)

        accepted = runner.invoke(
            app, ["agent", "confirm-hook", "--approve", "gpu_lane"], input=stdin
        )
        declined = runner.invoke(
            app, ["agent", "confirm-hook", "--approve", "bind"], input=stdin
        )
        asked = runner.invoke(
            app,
            ["agent", "confirm-hook", "--approve", "bind", "--otherwise", "ask"],
            input=stdin,
        )

        assert accepted.exit_code == declined.exit_code == asked.exit_code == 0
        assert json.loads(accepted.stdout)["hookSpecificOutput"] == {
            "hookEventName": "Elicitation",
            "action": "accept",
            "content": {"confirm": True},
        }
        assert json.loads(declined.stdout)["hookSpecificOutput"]["action"] == "decline"
        assert asked.stdout == ""

    @pytest.mark.parametrize("stdin", ["", "not json", "[1, 2]"])
    def test_malformed_stdin_prints_nothing(self, stdin: str) -> None:
        result = runner.invoke(
            app, ["agent", "confirm-hook", "--approve", "gpu_lane"], input=stdin
        )

        assert result.exit_code == 0
        assert result.stdout == ""

    def test_an_unknown_condition_is_refused(self) -> None:
        result = runner.invoke(
            app,
            ["agent", "confirm-hook", "--approve", "everything"],
            input=json.dumps(CONFIRMATION),
        )

        assert result.exit_code != 0
        assert "Unknown confirmation condition(s): everything" in result.output
        assert "gpu_lane" in result.output
