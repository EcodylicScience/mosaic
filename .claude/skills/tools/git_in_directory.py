"""Run git in a given directory, ignoring the repository that the environment
names."""

import os
import subprocess
from pathlib import Path

from lossless_text import substitution_text
from program_start import run_program

# git answers from the environment before its -C directory. GIT_DIR,
# GIT_WORK_TREE, GIT_COMMON_DIR and GIT_NAMESPACE name another checkout to read
# and to put refs in. The other three redirect the files that git writes. Every
# one is exported to every git hook, and GIT_OBJECT_DIRECTORY is set during
# object quarantine. Runs from a hook, from `git rebase -x` or from `git
# submodule foreach` meet them.
REDIRECTING_VARIABLES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
)

# Makes git read every pathspec as a file name, so a magic pathspec such as
# `:(top,exclude)docs/` matches nothing. A literal lookup passes
# --literal-pathspecs itself.
LITERAL_PATHSPEC_VARIABLES = ("GIT_LITERAL_PATHSPECS",)

# Prints the absolute common git directory, whose parent is the main worktree.
COMMON_DIRECTORY_QUERY = ("rev-parse", "--path-format=absolute", "--git-common-dir")


def git_environment(index_file: Path | None = None) -> dict[str, str]:
    """Return this process's environment clear of every redirecting variable
    and of GIT_LITERAL_PATHSPECS.

    `index_file` alone may name the index, such as a scratch index.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in REDIRECTING_VARIABLES and name not in LITERAL_PATHSPEC_VARIABLES
    }
    if index_file is not None:
        environment["GIT_INDEX_FILE"] = str(index_file)
    return environment


def run_git(
    directory: Path,
    arguments: list[str],
    *,
    index_file: Path | None = None,
    stdout: int | None = None,
    stderr: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run git in `directory` in the environment that `git_environment`
    returns."""
    return run_program(
        ["git", "-C", directory, *arguments], environment=git_environment(index_file), stdout=stdout, stderr=stderr
    )


def git_text(directory: Path, arguments: list[str], *, index_file: Path | None = None) -> tuple[int, str]:
    """Return the status of git run as `run_git` runs it, and its output as a
    command substitution yields it. Its standard error passes through."""
    completed = run_git(directory, arguments, index_file=index_file, stdout=subprocess.PIPE)
    return completed.returncode, substitution_text(completed.stdout)


def nul_separated_names(output: bytes) -> list[str]:
    """Return the names in the output of a git command run with -z, without
    the empty entry after the last NUL."""
    return [os.fsdecode(name) for name in output.split(b"\0") if name]


def repository_toplevel(directory: Path) -> Path | None:
    """Return the root of the working tree that contains `directory`, or None
    outside one."""
    completed = run_git(
        directory, ["rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    return Path(substitution_text(completed.stdout)) if completed.returncode == 0 else None


def resolved_commit(directory: Path, revision: str) -> str | None:
    """Return the id of the commit that `revision` names in the repository of
    `directory`, or None when it names no commit."""
    status, commit = git_text(directory, ["rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"])
    return commit if status == 0 else None
