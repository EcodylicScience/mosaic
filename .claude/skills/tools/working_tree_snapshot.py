"""Snapshot a working tree's tracked state and list its untracked files."""

import shlex
import shutil
from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from git_in_directory import git_text
from lossless_text import nonempty_lines
from quiet_paths import present

UNTRACKED_LIST_CAP = 10
# Specs, plans and issues sit untracked in these toplevel directories before
# implementation starts, and a run needs none of them.
UNTRACKED_EXEMPT_DIRECTORIES = ("docs/specs/", "docs/plans/", "docs/issues/")
# Select every untracked file from the toplevel except those in the exempt
# directories. git matches these before it quotes a name for output.
UNTRACKED_PATHSPECS = (":/", *(f":(top,exclude){directory}" for directory in UNTRACKED_EXEMPT_DIRECTORIES))

Step = Literal[
    "index-path",
    "index-missing",
    "index-copy",
    "seed",
    "stage",
    "write-tree",
    "empty-tree",
    "commit",
    "untracked",
]


class GitRunner(Protocol):
    def __call__(self, arguments: list[str], *, index_file: Path | None = None) -> tuple[int, str]:
        """Run git with `arguments` in the caller's repository, with
        `index_file` as its index when given and its stderr passed through.
        Return the exit status and the output as `git_text` returns it."""
        ...


def git_runner(directory: Path) -> GitRunner:
    """Return the runner of git in `directory`."""

    def run(arguments: list[str], *, index_file: Path | None = None) -> tuple[int, str]:
        return git_text(directory, arguments, index_file=index_file)

    return run


class IndexSeed(Enum):
    CALLER_INDEX = "the caller's index"


# Seeds the scratch index with a copy of the caller's index. Every other seed
# is a commit id, and the empty string seeds an empty index.
CALLER_INDEX = IndexSeed.CALLER_INDEX


class SnapshotError(Exception):
    """Report a failed step. `permanent` is whether a retry cannot change the
    outcome, `git_status` is git's exit status for a git step, and
    `index_path` is the caller's index once its lookup succeeded."""

    def __init__(
        self,
        step: Step,
        message: str,
        *,
        permanent: bool,
        git_status: int | None = None,
        index_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.step: Step = step
        self.message: str = message
        self.permanent: bool = permanent
        self.git_status: int | None = git_status
        self.index_path: Path | None = index_path


def build_snapshot_commit(
    runner: GitRunner,
    scratch_index: Path,
    seed: IndexSeed | str,
    add_arguments: list[str],
    message: str,
    parent: str | None,
    empty_tree_refuses: bool,
) -> str:
    """Commit what `git add <add_arguments>` stages onto `seed`, and return
    the commit id.

    `scratch_index` stages it, and the caller owns and removes it. The
    caller's index, working tree and refs stay untouched.
    """
    index_path: Path | None = None

    def failure(step: Step, failure_message: str, permanent: bool, git_status: int | None = None) -> SnapshotError:
        return SnapshotError(
            step, failure_message, permanent=permanent, git_status=git_status, index_path=index_path
        )

    def git(step: Step, failure_message: str, arguments: list[str], *, index_file: Path | None = None,
            permanent: bool = False) -> str:
        status, output = runner(arguments, index_file=index_file)
        if status != 0:
            raise failure(step, failure_message, permanent, status)
        return output

    if seed is CALLER_INDEX:
        # --path-format=absolute (git 2.31) keeps the path independent of where
        # this process stands, and the snapshot reads the index of this
        # repository.
        index_path = Path(
            git(
                "index-path",
                "cannot resolve the git index path",
                ["rev-parse", "--path-format=absolute", "--git-path", "index"],
                permanent=True,
            )
        )
        if not present(index_path.is_file):
            raise failure("index-missing", f"the git index does not exist at {index_path}", True)
        try:
            # The copy keeps the index's mtime, which git's racy-entry check
            # compares with each entry's mtime.
            _ = shutil.copy2(index_path, scratch_index)
        except OSError:
            raise failure("index-copy", "cannot copy the git index", False) from None
    else:
        # `git add` reads an empty scratch file as a truncated index.
        _ = git("seed", "cannot seed the snapshot index", ["read-tree", seed or "--empty"], index_file=scratch_index)
    _ = git(
        "stage",
        "cannot stage working-tree changes into the snapshot index",
        ["add", *add_arguments],
        index_file=scratch_index,
    )
    tree = git("write-tree", "cannot write the snapshot tree", ["write-tree"], index_file=scratch_index)
    if empty_tree_refuses:
        # The check reads empty output instead of comparing with the empty
        # tree's hash. Failing gits produce empty output too, and broken
        # helpers then refuse instead of passing.
        _, listing = runner(["ls-tree", tree])
        if not listing:
            raise failure("empty-tree", "the snapshot tree is empty", True)
    # -m keeps commit-tree from reading the message from stdin and hanging. -p
    # makes a push of the snapshot a delta on the parent's history rather than
    # every object.
    parent_arguments = [] if parent is None else ["-p", parent]
    return git("commit", "cannot create the snapshot commit", ["commit-tree", tree, *parent_arguments, "-m", message])


def untracked_files(runner: GitRunner) -> list[str]:
    """Return the working tree's files that git neither tracks nor ignores,
    leaving out those in UNTRACKED_EXEMPT_DIRECTORIES.

    `runner` runs at the toplevel, so each path is relative to it.
    """
    status, output = runner(["ls-files", "--others", "--exclude-standard", "--", *UNTRACKED_PATHSPECS])
    if status != 0:
        raise SnapshotError(
            "untracked", "cannot list the working tree's untracked files", permanent=False, git_status=status
        )
    return nonempty_lines(output)


def untracked_listing(paths: list[str]) -> list[str]:
    """Return the lines that list `paths`, at most UNTRACKED_LIST_CAP of them.

    Each path is a `:/` pathspec, which pastes from any subdirectory.
    """
    lines = [f"  :/{path}" for path in paths[:UNTRACKED_LIST_CAP]]
    if len(paths) > UNTRACKED_LIST_CAP:
        lines.append(
            f"  ... and {len(paths) - UNTRACKED_LIST_CAP} more;"
            + " list them with git ls-files --others --exclude-standard -- "
            + shlex.join(UNTRACKED_PATHSPECS)
        )
    return lines
