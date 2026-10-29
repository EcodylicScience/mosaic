"""The lint and generated-reference checks shared by mosaic-check and the
mosaic commit hook."""

import os
import shutil
import subprocess
from pathlib import Path

TIMEOUT_SECONDS = 100
LINT_DIRECTORIES = ("src/", "tests/", "tools/", "scripts/")
CLI_DIRECTORY = "src/mosaic/cli/"

# A module that calls one of these registers into a registry that
# scripts/gen_docs_reference.py renders.
REGISTRAR_MARKERS = (
    "register_feature",
    "register_op",
    "register_media_step",
    "register_track_converter",
    "register_label_converter",
)

# The generator prints this when a page is stale, and not when it fails.
STALE_MARKER = "out of date"


def git_toplevel(directory: Path) -> Path | None:
    """Return the toplevel of the git checkout holding `directory`, None
    outside one."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
            timeout=TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    toplevel = completed.stdout.strip()
    return Path(toplevel) if completed.returncode == 0 and toplevel else None


def mosaic_toplevel(directory: Path) -> Path | None:
    """Return the toplevel of the mosaic checkout holding `directory`, None
    for any other directory."""
    root = git_toplevel(directory)
    return root if root is not None and is_mosaic_toplevel(root) else None


def is_mosaic_toplevel(root: Path) -> bool:
    return (root / "scripts" / "gen_docs_reference.py").is_file() and (
        root / "src" / "mosaic"
    ).is_dir()


def tool(root: Path, name: str) -> str | None:
    """Return the checkout environment's `name`, else the one on PATH."""
    local = root / ".venv" / "bin" / name
    return str(local) if local.exists() else shutil.which(name)


def run(
    root: Path, command: list[str], timeout: float | None = TIMEOUT_SECONDS
) -> tuple[int, str]:
    """Run `command` in `root` importing that checkout's source, and return
    its status and combined output."""
    environment = {**os.environ, "PYTHONPATH": str(root / "src")}
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return -1, str(error)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.split("\n")[-lines:])


def lint_files(files: list[str]) -> list[str]:
    return [
        name
        for name in files
        if name.endswith(".py") and name.startswith(LINT_DIRECTORIES)
    ]


def lint_findings(root: Path, files: list[str]) -> list[str]:
    """Return a finding for each of `ruff check` and `ruff format --check`
    that fails on the Python files among `files`."""
    python_files = lint_files(files)
    ruff = tool(root, "ruff")
    if not python_files or ruff is None:
        return []
    findings: list[str] = []
    status, output = run(root, [ruff, "check", "--no-fix", "--quiet", *python_files])
    if status != 0:
        findings.append(f"ruff check failed:\n{tail(output)}")
    status, output = run(root, [ruff, "format", "--check", "--quiet", *python_files])
    if status != 0:
        findings.append(
            f"ruff format would reformat (run `ruff format` on them):\n{tail(output)}"
        )
    return findings


def touches_generated_reference(root: Path, files: list[str]) -> bool:
    """True when a file among `files` can change a page of docs/reference/."""
    for name in files:
        if not (name.startswith("src/mosaic/") and name.endswith(".py")):
            continue
        if name.startswith(CLI_DIRECTORY):
            return True
        text = (root / name).read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in REGISTRAR_MARKERS):
            return True
    return False


def docs_findings(root: Path, files: list[str]) -> tuple[list[str], str | None]:
    """Check docs/reference/ when `files` can change it. Return a finding when
    the generator reports a stale page, and a note when it could not run."""
    if not touches_generated_reference(root, files):
        return [], None
    python = tool(root, "python")
    if python is None:
        return [], "the docs reference was not checked: no python in .venv or on PATH"
    status, output = run(root, [python, "scripts/gen_docs_reference.py", "--check"])
    if status == 0:
        return [], None
    if STALE_MARKER not in output:
        return [], f"the docs reference check could not run:\n{tail(output, 10)}"
    return [
        "docs/reference/ is stale. Run `python scripts/gen_docs_reference.py --write` "
        f"and stage the result:\n{tail(output, 10)}"
    ], None
