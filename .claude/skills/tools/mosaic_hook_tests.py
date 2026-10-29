"""Tests of the mosaic PreToolUse hooks, run from this directory:

python3 -B -m unittest mosaic_hook_tests
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from json_value import Json, parse_json
from mosaic_commands import parse_step

TOOLS = Path(__file__).resolve().parent
REPOSITORY = TOOLS.parent.parent.parent


def answer(
    hook: str,
    command: str,
    cwd: Path = REPOSITORY,
    environment: dict[str, str] | None = None,
) -> dict[str, Json]:
    """Run `hook` on a Bash payload and return its hookSpecificOutput, empty
    when it printed nothing."""
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "cwd": str(cwd),
        "session_id": "test",
    }
    completed = subprocess.run(
        [sys.executable, "-B", str(TOOLS / hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **(environment or {})},
    )
    if not completed.stdout.strip():
        return {}
    output: Json = parse_json(completed.stdout)
    assert isinstance(output, dict)
    specific = output["hookSpecificOutput"]
    assert isinstance(specific, dict)
    return specific


def denied(specific: dict[str, Json]) -> bool:
    return specific.get("permissionDecision") == "deny"


class ParseStepTest(unittest.TestCase):
    def test_a_quoted_assignment_and_a_venv_interpreter_read_as_pytest(self) -> None:
        step = parse_step(
            'PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q tests/core'
        )
        self.assertEqual(step.program, "pytest")
        self.assertEqual(step.arguments, ["-q", "tests/core"])
        self.assertEqual(step.assignments, {"PYTHONPATH": "$PWD/src"})

    def test_heavy_and_its_log_option_are_a_wrapper(self) -> None:
        step = parse_step("heavy --log /tmp/x.log .venv/bin/pytest -x")
        self.assertEqual(
            (step.wrappers, step.program, step.arguments), (["heavy"], "pytest", ["-x"])
        )

    def test_remote_with_env_wraps_the_program(self) -> None:
        step = parse_step(
            "remote env PATH=/usr/bin .venv/bin/pytest -m 'slow or not slow'"
        )
        self.assertEqual((step.wrappers, step.program), (["remote", "env"], "pytest"))


class SuiteHookTest(unittest.TestCase):
    hook = "mosaic-suite-hook.py"
    clear = {"MOSAIC_LOCAL_SUITE": ""}

    def test_full_suites_are_denied(self) -> None:
        for command in (
            'PYTHONPATH="$PWD/src" .venv/bin/python -m pytest -q -m "slow or not slow" -p no:cacheprovider',
            "heavy --log x.log .venv/bin/pytest -q",
            ".venv/bin/pytest tests/",
            "cd tests && pytest -n auto | tail -5",
            ".venv/bin/pytest -m 'not media'",
            ".venv/bin/pytest --log-cli-level INFO",
            "cd src && ../.venv/bin/pytest ../tests",
            ".claude/skills/tools/mosaic-suite",
        ):
            with self.subTest(command=command):
                self.assertTrue(
                    denied(answer(self.hook, command, environment=self.clear))
                )

    def test_narrowed_and_offloaded_runs_pass(self) -> None:
        for command in (
            "PYTHONPATH=src .venv/bin/python -m pytest -n0 tests/core/test_params.py::test_x",
            ".venv/bin/pytest tests/core/pipeline -q",
            ".venv/bin/pytest -k arena",
            ".venv/bin/pytest -m tracker",
            ".venv/bin/pytest --collect-only -q",
            "remote --log x.log .claude/skills/tools/mosaic-suite",
            "remote env PATH=/usr/bin .venv/bin/pytest -m 'slow or not slow'",
            "MOSAIC_LOCAL_SUITE=1 heavy .venv/bin/pytest -m 'slow or not slow'",
            'git commit -m "run pytest on the suite"',
            "grep -rn pytest tests/CLAUDE.md",
            ".venv/bin/pytest -kfoo",
            "cd /tmp && pytest -n auto",
            "cd tests/core && pytest -q",
        ):
            with self.subTest(command=command):
                self.assertFalse(
                    denied(answer(self.hook, command, environment=self.clear))
                )

    def test_a_run_in_another_repository_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "-q", directory], check=True)
            specific = answer(
                self.hook,
                ".venv/bin/pytest",
                cwd=Path(directory),
                environment=self.clear,
            )
        self.assertFalse(denied(specific))

    def test_the_session_override_lets_a_full_suite_run(self) -> None:
        specific = answer(
            self.hook, ".venv/bin/pytest", environment={"MOSAIC_LOCAL_SUITE": "1"}
        )
        self.assertFalse(denied(specific))


class WorktreeHookTest(unittest.TestCase):
    hook = "mosaic-worktree-hook.py"

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = (
            Path(self.directory.name).resolve()
            / "repo"
            / ".claude"
            / "worktrees"
            / "branch"
        )
        (self.root / "src" / "mosaic").mkdir(parents=True)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_a_python_run_in_a_worktree_imports_its_source(self) -> None:
        (self.root / ".venv").mkdir()
        specific = answer(
            self.hook,
            ".venv/bin/python -m pytest -q tests/x.py",
            cwd=self.root / "tests",
        )
        updated = specific["updatedInput"]
        assert isinstance(updated, dict)
        command = updated["command"]
        assert isinstance(command, str)
        self.assertTrue(command.startswith(f"export PYTHONPATH={self.root / 'src'}"))
        self.assertTrue(command.endswith(".venv/bin/python -m pytest -q tests/x.py"))

    def test_a_leading_cd_into_a_worktree_counts(self) -> None:
        (self.root / ".venv").mkdir()
        specific = answer(
            self.hook, f"cd {self.root} && python3 -c 'import mosaic'", cwd=Path("/")
        )
        self.assertIn("updatedInput", specific)

    def test_a_chores_call_and_a_python_run_keep_both_prefixes(self) -> None:
        (self.root / ".venv").mkdir()
        specific = answer(
            self.hook,
            "chores done 2 && .venv/bin/python -m pytest tests/x.py",
            cwd=self.root,
        )
        updated = specific["updatedInput"]
        assert isinstance(updated, dict)
        command = str(updated["command"])
        self.assertTrue(command.startswith("export PYTHONPATH="))
        self.assertIn("export CHORES_FILE=", command)

    def test_a_chores_call_outside_a_worktree_is_still_rewritten(self) -> None:
        specific = answer(self.hook, "chores list", cwd=Path("/"))
        updated = specific["updatedInput"]
        assert isinstance(updated, dict)
        self.assertTrue(str(updated["command"]).startswith("export CHORES_FILE="))

    def test_an_absolute_environment_path_is_not_denied(self) -> None:
        specific = answer(
            self.hook,
            f"{REPOSITORY.parent}/.venv/bin/python -m pytest tests/x.py",
            cwd=self.root,
        )
        self.assertFalse(denied(specific))
        self.assertIn("updatedInput", specific)

    def test_a_missing_venv_is_denied_with_the_link_to_make(self) -> None:
        specific = answer(self.hook, ".venv/bin/pytest tests/x.py", cwd=self.root)
        self.assertTrue(denied(specific))
        self.assertIn("ln -sfn", str(specific["permissionDecisionReason"]))

    def test_other_commands_pass_untouched(self) -> None:
        (self.root / ".venv").mkdir()
        for command in (
            "git status",
            "PYTHONPATH=src python -m pytest",
            "remote .venv/bin/pytest tests/x.py",
        ):
            with self.subTest(command=command):
                self.assertEqual(answer(self.hook, command, cwd=self.root), {})

    def test_the_main_checkout_passes_untouched(self) -> None:
        self.assertEqual(
            answer(self.hook, ".venv/bin/pytest tests/x.py", cwd=REPOSITORY.parent), {}
        )


# Built at run time, so that the commit hook does not refuse this file.
ADDRESS = ".".join(["10", "4", "2", "17"])


class SpecificHostNameTest(unittest.TestCase):
    def test_only_names_prose_does_not_spell_are_searched(self) -> None:
        from importlib import util

        spec = util.spec_from_file_location(
            "commit_hook", TOOLS / "mosaic-commit-hook.py"
        )
        assert spec is not None and spec.loader is not None
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        specific: object = getattr(module, "specific")
        assert callable(specific)
        for name, expected in (
            ("Recording", False),
            ("dorado", False),
            ("box-02", True),
            ("UniName", True),
            ("box.lan", True),
            ("*", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(specific(name), expected)


class CommitHookTest(unittest.TestCase):
    hook = "mosaic-commit-hook.py"

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "scripts" / "gen_docs_reference.py").write_text(
            "import sys\nsys.exit(0)\n"
        )
        (self.root / "src" / "mosaic").mkdir(parents=True)
        for arguments in (
            ["init", "-q"],
            ["add", "."],
            ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
        ):
            subprocess.run(
                ["git", "-C", str(self.root), *arguments],
                check=True,
                capture_output=True,
            )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def stage(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        subprocess.run(["git", "-C", str(self.root), "add", name], check=True)

    def test_a_home_path_in_an_added_line_is_denied(self) -> None:
        self.stage("docs/notes.md", f"see {Path.home()}/data\n")
        specific = answer(self.hook, 'git commit -m "notes"', cwd=self.root)
        self.assertTrue(denied(specific))
        self.assertIn("docs/notes.md", str(specific["permissionDecisionReason"]))

    def fake_home(self) -> dict[str, str]:
        """Return an environment whose home has an ssh config naming one host."""
        home = self.root / "home"
        (home / ".ssh").mkdir(parents=True, exist_ok=True)
        (home / ".ssh" / "config").write_text(f"Host box-02\n  HostName {ADDRESS}\n")
        return {"HOME": str(home)}

    def test_a_configured_host_and_pattern_are_denied(self) -> None:
        for text, environment in (
            (f"host {ADDRESS}\n", self.fake_home()),
            ("ssh box-02 ls\n", self.fake_home()),
            ("the Acme deal\n", {"MOSAIC_PRIVATE_PATTERNS": "acme"}),
        ):
            with self.subTest(text=text):
                self.stage("docs/notes.md", text)
                self.assertTrue(
                    denied(
                        answer(
                            self.hook,
                            "git commit -am wip",
                            cwd=self.root,
                            environment=environment,
                        )
                    )
                )

    def test_versions_and_example_addresses_pass(self) -> None:
        self.stage(
            "uv.lock", 'version = "13.0.3.0"\nscipy-stubs>=1.17.1.1\nhost 100.64.0.1\n'
        )
        specific = answer(
            self.hook,
            'git commit -m "Refresh the lock"',
            cwd=self.root,
            environment=self.fake_home(),
        )
        self.assertFalse(denied(specific))

    def test_a_private_commit_message_is_denied(self) -> None:
        self.stage("docs/notes.md", "plain\n")
        command = f"git commit -F - <<'EOM'\nCopied from {Path.home()}/data\nEOM"
        specific = answer(self.hook, command, cwd=self.root)
        self.assertTrue(denied(specific))
        self.assertIn("the commit message", str(specific["permissionDecisionReason"]))

    def test_an_absolute_path_outside_the_message_passes(self) -> None:
        # The repository and the message file lie under this home.
        home = {"HOME": str(self.root.parent)}
        self.stage("docs/notes.md", "plain\n")
        command = f"git -C {self.root} commit -m x"
        self.assertFalse(
            denied(answer(self.hook, command, cwd=Path("/"), environment=home))
        )
        message = self.root.parent / "message.txt"
        message.write_text("plain message\n")
        command = f"git commit -F {message}"
        self.assertFalse(
            denied(answer(self.hook, command, cwd=self.root, environment=home))
        )

    def test_a_private_message_value_or_file_is_denied(self) -> None:
        self.stage("docs/notes.md", "plain\n")
        specific = answer(
            self.hook, f'git commit -m "data on {Path.home()}/x"', cwd=self.root
        )
        self.assertTrue(denied(specific))
        message = self.root.parent / "message.txt"
        message.write_text(f"copied from {Path.home()}/x\n")
        self.assertTrue(
            denied(answer(self.hook, f"git commit -F {message}", cwd=self.root))
        )

    def test_a_failing_generator_is_a_note_and_not_a_denial(self) -> None:
        (self.root / "scripts" / "gen_docs_reference.py").write_text(
            "import sys\nsys.exit('ImportError: no pandas')\n"
        )
        self.stage("src/mosaic/feature.py", "@register_feature\nclass F:\n    pass\n")
        specific = answer(self.hook, "git commit -m feature", cwd=self.root)
        self.assertFalse(denied(specific))
        self.assertIn("could not run", str(specific.get("additionalContext")))

    def test_clean_lines_and_the_override_pass(self) -> None:
        self.stage("docs/notes.md", "version 1.2 of the guide, run at 127.0.0.1\n")
        self.assertFalse(
            denied(answer(self.hook, 'git commit -m "notes"', cwd=self.root))
        )
        self.stage("docs/notes.md", f"see {Path.home()}\n")
        self.assertFalse(
            denied(
                answer(
                    self.hook, 'MOSAIC_COMMIT_CHECKS=0 git commit -m "x"', cwd=self.root
                )
            )
        )

    def test_a_commit_outside_mosaic_passes(self) -> None:
        (self.root / "scripts" / "gen_docs_reference.py").unlink()
        self.stage("docs/notes.md", f"see {Path.home()}\n")
        self.assertEqual(answer(self.hook, 'git commit -m "x"', cwd=self.root), {})

    def test_a_stale_reference_is_denied_for_a_registering_module(self) -> None:
        (self.root / "scripts" / "gen_docs_reference.py").write_text(
            "import sys\nprint('features out of date')\nsys.exit(1)\n"
        )
        self.stage("src/mosaic/feature.py", "@register_feature\nclass F:\n    pass\n")
        specific = answer(self.hook, "git commit -m feature", cwd=self.root)
        self.assertTrue(denied(specific))
        self.assertIn(
            "docs/reference/ is stale", str(specific["permissionDecisionReason"])
        )


if __name__ == "__main__":
    unittest.main()
