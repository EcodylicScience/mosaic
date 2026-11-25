"""``mosaic examples``: listing the catalog and fetching an example.

Each fetch runs against the local server of ``tests.helpers``, with the CLI's catalog
and fetch function replaced by ones bound to a generated example, so none reaches
the network.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.cli import examples as examples_cli
from mosaic.examples import fetch as fetch_module
from mosaic.examples.catalog import Example
from mosaic.examples.fetch import (
    ArchiveWithoutDataset,
    ChecksumMismatch,
    ChecksumMissing,
    DownloadFailed,
    ExampleFetchError,
    FetchConflict,
    UnsafeArchive,
    default_directory,
    fetch_example,
)
from tests.helpers import ExampleServer, example_fixture, serve_example

runner = CliRunner()

PICKLE_REASON = (
    "Its raw files load only by unpickling, which can run code in the file; "
    "use these settings only for datasets from sources you trust."
)


@pytest.fixture
def tiny(example_server: ExampleServer, monkeypatch: pytest.MonkeyPatch) -> Example:
    """The generated example, served locally and the only entry of the CLI's catalog."""
    fixture = example_fixture()
    serve_example(example_server, fixture)
    catalog_of(monkeypatch, fixture.example)
    return fixture.example


def catalog_of(monkeypatch: pytest.MonkeyPatch, example: Example) -> None:
    """Make *example* the only entry of the CLI's catalog, fetched from the server."""
    catalog = (example,)
    monkeypatch.setattr(examples_cli, "CATALOG", catalog)
    monkeypatch.setattr(
        examples_cli, "fetch_example", functools.partial(fetch_example, catalog=catalog)
    )


def at_terminal(monkeypatch: pytest.MonkeyPatch, *, is_terminal: bool) -> None:
    """Make the CLI see standard input as a terminal, or not."""
    monkeypatch.setattr(examples_cli, "stdin_is_terminal", lambda: is_terminal)


class TestList:
    def test_it_prints_one_block_per_example_in_catalog_order(self) -> None:
        result = runner.invoke(app, ["examples", "list"])
        assert result.exit_code == 0, result.stderr
        names = [example.name for example in examples_cli.CATALOG]
        positions = [result.stdout.index(name) for name in names]
        assert positions == sorted(positions)
        for example in examples_cli.CATALOG:
            assert example.license in result.stdout
            assert example.summary in result.stdout
            assert example.notebook in result.stdout

    def test_it_prints_each_size_to_one_decimal_in_megabytes_or_gigabytes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        small = example_fixture(name="small").example
        archive = dataclasses.replace(small.archives[0], download_bytes=12_613_112)
        large = dataclasses.replace(
            small,
            name="large",
            archives=(dataclasses.replace(archive, download_bytes=1_900_000_000),),
        )
        monkeypatch.setattr(
            examples_cli,
            "CATALOG",
            (dataclasses.replace(small, archives=(archive,)), large),
        )
        result = runner.invoke(app, ["examples", "list"])
        assert "12.6 MB" in result.stdout
        assert "1.9 GB" in result.stdout

    def test_json_lists_every_field(self) -> None:
        result = runner.invoke(app, ["examples", "list", "--json"])
        assert result.exit_code == 0, result.stderr
        listed = json.loads(result.stdout)
        assert [entry["name"] for entry in listed] == [
            example.name for example in examples_cli.CATALOG
        ]
        first = examples_cli.CATALOG[0]
        assert listed[0] == {
            "name": first.name,
            "repository": first.repository,
            "revision": first.revision,
            "license": first.license,
            "summary": first.summary,
            "notebook": first.notebook,
            "raw_needs_unpickling": first.raw_needs_unpickling,
            "download_bytes": first.download_bytes,
            "archives": [
                {
                    "file": archive.file,
                    "unpacks_to": archive.unpacks_to,
                    "download_bytes": archive.download_bytes,
                    "dataset": archive.dataset,
                }
                for archive in first.archives
            ],
        }
        assert any(entry["raw_needs_unpickling"] for entry in listed)

    def test_it_makes_no_request(self, example_server: ExampleServer) -> None:
        result = runner.invoke(app, ["examples", "list"])
        assert result.exit_code == 0, result.stderr
        assert example_server.requests == []

    def test_the_group_help_lists_both_verbs(self) -> None:
        result = runner.invoke(app, ["examples", "--help"])
        assert result.exit_code == 0
        assert "Download example datasets to try mosaic on." in result.stdout
        assert "list" in result.stdout
        assert "fetch" in result.stdout


class TestFetch:
    def test_stdout_holds_the_manifest_alone_and_stderr_the_hint(
        self, tiny: Example, tmp_path: Path
    ) -> None:
        directory = tmp_path / "out"
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(directory)]
        )
        assert result.exit_code == 0, result.stderr
        manifest = directory / tiny.manifest_relative
        assert result.stdout == f"{manifest}\n"
        assert manifest.is_file()
        quoted = shlex.quote(str(manifest))
        assert f"mosaic agent install --desktop --dataset {quoted}" in result.stderr
        assert f"mosaic agent install --mcp --dataset {quoted}" in result.stderr

    def test_an_example_without_pickled_files_gets_the_plain_commands(
        self, tiny: Example, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 0, result.stderr
        assert "--trust-pickled-files" not in result.stderr
        assert "MOSAIC_TRUST_PICKLED_FILES" not in result.stderr
        assert "unpickling" not in result.stderr

    def test_an_example_with_pickled_files_gets_the_trust_settings(
        self,
        tiny: Example,
        example_server: ExampleServer,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        raw = dataclasses.replace(tiny, raw_needs_unpickling=True)
        catalog_of(monkeypatch, raw)
        directory = tmp_path / "out"
        result = runner.invoke(
            app, ["examples", "fetch", raw.name, "--dir", str(directory)]
        )
        assert result.exit_code == 0, result.stderr
        quoted = shlex.quote(str(directory / raw.manifest_relative))
        lines = result.stderr.splitlines()
        desktop = (
            f"mosaic agent install --desktop --trust-pickled-files --dataset {quoted}"
        )
        mcp = f"mosaic agent install --mcp --dataset {quoted}"
        export = "  export MOSAIC_TRUST_PICKLED_FILES=1"
        assert desktop in result.stderr
        assert lines.index(export) < lines.index(f"  {mcp}")
        assert PICKLE_REASON in lines
        assert result.stdout == f"{directory / raw.manifest_relative}\n"

    def test_on_windows_the_manifest_is_quoted_for_cmd_and_the_variable_is_set(
        self,
        tiny: Example,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        raw = dataclasses.replace(tiny, raw_needs_unpickling=True)
        catalog_of(monkeypatch, raw)
        monkeypatch.setattr(examples_cli, "is_windows", lambda: True)
        directory = tmp_path / "my out"
        result = runner.invoke(
            app, ["examples", "fetch", raw.name, "--dir", str(directory)]
        )
        assert result.exit_code == 0, result.stderr
        quoted = f'"{directory / raw.manifest_relative}"'
        lines = result.stderr.splitlines()
        mcp = f"  mosaic agent install --mcp --dataset {quoted}"
        assert (
            f"  mosaic agent install --desktop --trust-pickled-files --dataset {quoted}"
            in lines
        )
        assert lines.index("  set MOSAIC_TRUST_PICKLED_FILES=1") < lines.index(mcp)
        assert "export" not in result.stderr
        assert "'" not in result.stderr

    def test_on_windows_a_plain_example_has_no_variable_line(
        self,
        tiny: Example,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(examples_cli, "is_windows", lambda: True)
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(tmp_path / "my out")]
        )
        assert result.exit_code == 0, result.stderr
        assert "MOSAIC_TRUST_PICKLED_FILES" not in result.stderr
        assert f'--dataset "{tmp_path / "my out" / tiny.manifest_relative}"' in (
            result.stderr
        )

    def test_elsewhere_a_path_with_a_space_is_quoted_for_a_posix_shell(
        self, tiny: Example, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(examples_cli, "is_windows", lambda: False)
        directory = tmp_path / "my out"
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(directory)]
        )
        assert result.exit_code == 0, result.stderr
        quoted = shlex.quote(str(directory / tiny.manifest_relative))
        assert quoted.startswith("'")
        assert f"mosaic agent install --mcp --dataset {quoted}" in result.stderr

    @pytest.mark.parametrize("raw_needs_unpickling", [False, True])
    def test_the_hint_commands_are_accepted_by_mosaic_agent_install(
        self,
        tiny: Example,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        raw_needs_unpickling: bool,
    ) -> None:
        monkeypatch.delenv("MOSAIC_TRUST_PICKLED_FILES", raising=False)
        example = dataclasses.replace(tiny, raw_needs_unpickling=raw_needs_unpickling)
        catalog_of(monkeypatch, example)
        fetched = runner.invoke(
            app, ["examples", "fetch", example.name, "--dir", str(tmp_path / "out")]
        )
        assert fetched.exit_code == 0, fetched.stderr
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)
        configuration = tmp_path / "claude_desktop_config.json"
        commands = [
            shlex.split(line)
            for line in fetched.stderr.splitlines()
            if line.startswith("  mosaic agent install")
        ]
        assert len(commands) == 2
        for command in commands:
            arguments = command[1:]
            if "--desktop" in arguments:
                arguments += ["--desktop-config", str(configuration)]
            result = runner.invoke(app, arguments)
            assert result.exit_code == 0, (command, result.output)
        assert configuration.is_file()
        assert (project / ".mcp.json").is_file()

    def test_json_prints_the_paths_and_actions_alone(
        self, tiny: Example, tmp_path: Path
    ) -> None:
        directory = tmp_path / "out"
        arguments = ["examples", "fetch", tiny.name, "--dir", str(directory), "--json"]
        first = runner.invoke(app, arguments)
        assert first.exit_code == 0, first.stderr
        archive = tiny.archives[0]
        assert json.loads(first.stdout) == {
            "name": tiny.name,
            "directory": str(directory),
            "dataset": str(directory / archive.unpacks_to),
            "manifest": str(directory / tiny.manifest_relative),
            "archives": [
                {
                    "file": archive.file,
                    "path": str(directory / archive.unpacks_to),
                    "action": "downloaded",
                }
            ],
        }
        second = runner.invoke(app, arguments)
        assert json.loads(second.stdout)["archives"][0]["action"] == "reused"

    def test_the_default_directory_is_under_the_home_directory(
        self, tiny: Example, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        result = runner.invoke(app, ["examples", "fetch", tiny.name])
        assert result.exit_code == 0, result.stderr
        expected = tmp_path / "mosaic-examples" / tiny.name / tiny.manifest_relative
        assert result.stdout == f"{expected}\n"
        assert default_directory(tiny.name) == tmp_path / "mosaic-examples" / tiny.name

    def test_a_second_fetch_reports_the_reuse_and_prints_the_same_path(
        self, tiny: Example, tmp_path: Path
    ) -> None:
        arguments = ["examples", "fetch", tiny.name, "--dir", str(tmp_path / "out")]
        first = runner.invoke(app, arguments)
        second = runner.invoke(app, arguments)
        assert second.exit_code == 0, second.stderr
        assert second.stdout == first.stdout


class TestWithoutName:
    def test_it_exits_2_naming_the_list_verb_without_a_terminal(
        self, tiny: Example, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at_terminal(monkeypatch, is_terminal=False)
        result = runner.invoke(app, ["examples", "fetch"])
        assert result.exit_code == 2
        assert result.stdout == ""
        assert "mosaic examples list" in result.stderr

    def test_json_exits_2_even_on_a_terminal(
        self, tiny: Example, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at_terminal(monkeypatch, is_terminal=True)
        result = runner.invoke(app, ["examples", "fetch", "--json"], input="1\n")
        assert result.exit_code == 2
        assert result.stdout == ""
        assert "mosaic examples list" in result.stderr

    @pytest.mark.parametrize("answer", ["1", "tiny"])
    def test_a_terminal_is_asked_and_accepts_a_number_or_a_name(
        self,
        tiny: Example,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        answer: str,
    ) -> None:
        at_terminal(monkeypatch, is_terminal=True)
        directory = tmp_path / "out"
        result = runner.invoke(
            app, ["examples", "fetch", "--dir", str(directory)], input=f"{answer}\n"
        )
        assert result.exit_code == 0, result.stderr
        assert result.stdout.endswith(f"{directory / tiny.manifest_relative}\n")
        assert "1" in result.stderr and tiny.name in result.stderr
        assert tiny.summary in result.stderr

    def test_the_prompt_refuses_an_unknown_answer(
        self, tiny: Example, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        at_terminal(monkeypatch, is_terminal=True)
        result = runner.invoke(app, ["examples", "fetch"], input="nope\n")
        assert result.exit_code == 2
        assert tiny.name in result.stderr


FAILURES: list[type[ExampleFetchError]] = [
    FetchConflict,
    DownloadFailed,
    ChecksumMissing,
    ChecksumMismatch,
    UnsafeArchive,
    ArchiveWithoutDataset,
]


class TestFailures:
    def test_an_unknown_name_exits_2_naming_the_known_names(
        self, tiny: Example
    ) -> None:
        result = runner.invoke(app, ["examples", "fetch", "nope"])
        assert result.exit_code == 2
        assert result.stdout == ""
        assert tiny.name in result.stderr

    def test_an_unknown_name_under_json_exits_2_with_empty_stdout(
        self, tiny: Example
    ) -> None:
        result = runner.invoke(app, ["examples", "fetch", "nope", "--json"])
        assert result.exit_code == 2
        assert result.stdout == ""
        assert "nope" in result.stderr and tiny.name in result.stderr

    @pytest.mark.parametrize("as_json", [False, True])
    @pytest.mark.parametrize("error", FAILURES)
    def test_each_fetch_error_exits_1_with_empty_stdout(
        self,
        tiny: Example,
        monkeypatch: pytest.MonkeyPatch,
        error: type[ExampleFetchError],
        as_json: bool,
    ) -> None:
        def refuse(name: str, directory: Path | None = None, **_: object) -> None:
            raise error(f"{error.__name__} happened")

        monkeypatch.setattr(examples_cli, "fetch_example", refuse)
        arguments = ["examples", "fetch", tiny.name] + (["--json"] if as_json else [])
        result = runner.invoke(app, arguments)
        assert result.exit_code == 1
        assert result.stdout == ""
        assert f"{error.__name__} happened" in result.stderr

    def test_a_revision_the_server_lacks_exits_1_naming_it(
        self, tiny: Example, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        absent = dataclasses.replace(tiny, revision="f" * 40)
        catalog_of(monkeypatch, absent)
        result = runner.invoke(
            app, ["examples", "fetch", absent.name, "--dir", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        assert result.stdout == ""
        assert "f" * 40 in result.stderr

    def test_an_existing_lock_exits_1_with_empty_stdout(
        self, tiny: Example, tmp_path: Path
    ) -> None:
        directory = tmp_path / "out"
        directory.mkdir()
        (directory / fetch_module.LOCK_FILE).write_text("{}")
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(directory), "--json"]
        )
        assert result.exit_code == 1
        assert result.stdout == ""
        assert str(directory) in result.stderr

    def test_a_disk_error_exits_1_with_empty_stdout(
        self, tiny: Example, tmp_path: Path
    ) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a directory")
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(blocker / "out")]
        )
        assert result.exit_code == 1
        assert result.stdout == ""
        assert str(blocker) in result.stderr

    def test_a_disk_error_under_the_default_directory_names_that_directory(
        self, tiny: Example, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.write_text("a file, not a directory")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        result = runner.invoke(app, ["examples", "fetch", tiny.name])
        assert result.exit_code == 1
        assert result.stdout == ""
        assert str(home / "mosaic-examples" / tiny.name) in result.stderr

    def test_ctrl_c_exits_130_and_removes_the_lock(
        self, tiny: Example, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def interrupt(example: Example) -> dict[str, str]:
            raise KeyboardInterrupt

        monkeypatch.setattr(fetch_module, "fetch_checksums", interrupt)
        directory = tmp_path / "out"
        result = runner.invoke(
            app, ["examples", "fetch", tiny.name, "--dir", str(directory)]
        )
        assert result.exit_code == 130
        assert result.stdout == ""
        assert "Aborted" not in result.output
        assert not (directory / fetch_module.LOCK_FILE).exists()
