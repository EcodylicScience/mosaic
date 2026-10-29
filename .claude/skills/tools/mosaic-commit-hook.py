#!/usr/bin/env -S python3 -B
"""Check a `git commit` in the mosaic repository before it runs, as a
PreToolUse hook.

The hook denies the commit, naming each finding, when:

  * `ruff check` or `ruff format --check` fails on a committed Python file
    under `src`, `tests`, `tools` or `scripts`;
  * a committed module registers into a registry the docs reference is
    generated from, or lives under `src/mosaic/cli/`, and
    `scripts/gen_docs_reference.py --check` reports a stale page;
  * an added line or the commit message names something private to this
    machine: the user's home directory, a host alias, host name or address
    from `~/.ssh/config`, or a match of the `MOSAIC_PRIVATE_PATTERNS` regular
    expression from the session's environment. The repository is public, and
    the list of private names stays outside it. An address-shaped string that
    is no configured host's address (a version such as `13.0.3.0`, an example
    address in a test) is not refused.

The checks read the working tree's files, which equal the staged content in
the usual `git add` then `git commit` flow. A commit outside the mosaic
repository passes untouched. `MOSAIC_COMMIT_CHECKS=0`, on the command or in
the session's environment, skips every check. That override is the user's to
give.
"""

import getpass
import os
import re
import subprocess
from pathlib import Path

from hook_payload import read_payload
from mosaic_commands import Step, steps, target_directory
from mosaic_repository import (
    TIMEOUT_SECONDS,
    docs_findings,
    mosaic_toplevel,
    lint_findings,
)
from pre_tool_use_hook import add_context, bash_command, deny

OVERRIDE = "MOSAIC_COMMIT_CHECKS"

# git options that take a separate value before the subcommand.
GIT_VALUE_OPTIONS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})

# `git commit` options that take a separate value.
COMMIT_VALUE_OPTIONS = frozenset(
    {
        "-m",
        "--message",
        "-F",
        "--file",
        "-C",
        "--reuse-message",
        "-c",
        "--reedit-message",
        "--author",
        "--date",
        "--fixup",
        "--squash",
        "-t",
        "--template",
        "--cleanup",
        "--trailer",
        "--pathspec-from-file",
    }
)

# Short `git commit` flags that take a value, attached or as the next argument.
SHORT_VALUE_FLAGS = frozenset("mFCct")

PUBLIC_HOSTS = frozenset({"github.com", "gitlab.com", "bitbucket.org", "localhost"})
MAX_FINDINGS = 20


class CommitRequest:
    """The repository, the files and the message of one `git commit` step."""

    def __init__(self, directory: Path, arguments: list[str]) -> None:
        self.directory = directory
        self.all_tracked = False
        self.paths: list[str] = []
        self.messages: list[str] = []
        self.message_files: list[str] = []
        index = 0
        while index < len(arguments):
            token = arguments[index]
            following = arguments[index + 1] if index + 1 < len(arguments) else ""
            if token == "--":
                self.paths.extend(arguments[index + 1 :])
                break
            if token.startswith("--"):
                name, has_value, attached = token.partition("=")
                takes_next = not has_value and name in COMMIT_VALUE_OPTIONS
                self.all_tracked = self.all_tracked or name == "--all"
                self.read_value(name, attached if has_value else following)
                index += 2 if takes_next else 1
                continue
            if token.startswith("-") and len(token) > 1:
                index += 1 + self.read_short_flags(token[1:], following)
                continue
            self.paths.append(token)
            index += 1

    def read_value(self, option: str, value: str) -> None:
        """Keep `value` when `option` gives the message or its file."""
        if option in ("-m", "--message"):
            self.messages.append(value)
        elif option in ("-F", "--file"):
            self.message_files.append(value)

    def read_short_flags(self, letters: str, following: str) -> int:
        """Read a cluster of short flags such as `am`, and return how many
        following arguments it consumes: one when it ends in a flag that
        takes a value, as `-am "message"` does."""
        for position, letter in enumerate(letters):
            if letter == "a":
                self.all_tracked = True
            if letter in SHORT_VALUE_FLAGS:
                last = position == len(letters) - 1
                self.read_value(
                    f"-{letter}", following if last else letters[position + 1 :]
                )
                return 1 if last else 0
        return 0

    def message(self, step: Step) -> str:
        """Return the commit message: the `-m` values, the contents of a `-F`
        file, and the heredoc body of `-F -`."""
        parts = list(self.messages)
        for name in self.message_files:
            if name == "-":
                parts.append(step.raw.partition("\n")[2])
                continue
            try:
                parts.append(
                    (self.directory / os.path.expanduser(name)).read_text(
                        encoding="utf-8", errors="replace"
                    )
                )
            except OSError:
                continue
        return "\n".join(parts)


def commit_request(step: Step, cwd: Path) -> CommitRequest | None:
    """Return the request of a `git commit` step, None for any other step."""
    if step.program != "git":
        return None
    directory = cwd
    arguments = step.arguments
    index = 0
    while index < len(arguments) and arguments[index].startswith("-"):
        token = arguments[index]
        if token == "-C" and index + 1 < len(arguments):
            directory = directory / os.path.expanduser(arguments[index + 1])
        index += 2 if token in GIT_VALUE_OPTIONS else 1
    if index >= len(arguments) or arguments[index] != "commit":
        return None
    return CommitRequest(directory, arguments[index + 1 :])


def git(directory: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(directory), *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=TIMEOUT_SECONDS,
    )
    return completed.stdout if completed.returncode == 0 else ""


def committed_files(root: Path, request: CommitRequest) -> list[str]:
    names = set(
        git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR").split("\n")
    )
    if request.all_tracked:
        names.update(git(root, "diff", "--name-only", "--diff-filter=ACMR").split("\n"))
    for path in request.paths:
        names.update(git(root, "ls-files", "--", path).split("\n"))
    return sorted(name for name in names if name and (root / name).is_file())


def added_lines(root: Path, request: CommitRequest) -> list[tuple[str, str]]:
    diffs = [git(root, "diff", "--cached", "-U0", "--no-color", "--no-ext-diff")]
    if request.all_tracked or request.paths:
        diffs.append(
            git(
                root, "diff", "-U0", "--no-color", "--no-ext-diff", "--", *request.paths
            )
        )
    lines: list[tuple[str, str]] = []
    for diff in diffs:
        path = ""
        for line in diff.split("\n"):
            if line.startswith("+++ "):
                path = line[6:] if line.startswith("+++ b/") else ""
            elif line.startswith("+") and path:
                lines.append((path, line[1:]))
    return lines


def specific(name: str) -> bool:
    """True for a host name that a word of prose does not spell: one with a
    digit, a `-` or a `.`, or a capital past its first letter."""
    if any(character in name for character in "*?!"):
        return False
    return any(character.isdigit() or character in "-." for character in name) or any(
        character.isupper() for character in name[1:]
    )


def ssh_names(config: Path) -> set[str]:
    """Return the host aliases and host names of an ssh config that are
    specific enough to search for."""
    names: set[str] = set()
    try:
        text = config.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return names
    for line in text.split("\n"):
        words = line.split()
        if len(words) >= 2 and words[0].lower() in ("host", "hostname"):
            names.update(word for word in words[1:] if specific(word))
    return {name for name in names if name.lower() not in PUBLIC_HOSTS}


def private_patterns() -> list[re.Pattern[str]]:
    home = Path.home()
    user = getpass.getuser()
    literal = {str(home), f"/Users/{user}", f"/home/{user}"}
    patterns = [re.compile(re.escape(text)) for text in sorted(literal)]
    patterns.extend(
        re.compile(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", re.IGNORECASE)
        for name in sorted(ssh_names(home / ".ssh" / "config"))
    )
    extra = os.environ.get("MOSAIC_PRIVATE_PATTERNS", "")
    if extra:
        try:
            patterns.append(re.compile(extra, re.IGNORECASE))
        except re.error:
            pass
    return patterns


def sensitive_findings(lines: list[tuple[str, str]], message: str) -> list[str]:
    """Return a finding naming each added line, and the commit message, that
    matches a private pattern. `lines` are (path, text) pairs."""
    patterns = private_patterns()
    hits: list[str] = []
    for path, line in [*lines, ("the commit message", message)]:
        for pattern in patterns:
            match = pattern.search(line)
            if match is not None:
                hits.append(f"{path}: {match.group(0)!r}")
                break
        if len(hits) >= MAX_FINDINGS:
            break
    if not hits:
        return []
    listed = "\n".join(hits)
    return [
        "Added lines name something private to this machine, and this repository "
        f"is public. Remove or generalize them:\n{listed}"
    ]


def overridden(step: Step) -> bool:
    return step.assignments.get(OVERRIDE) == "0" or os.environ.get(OVERRIDE) == "0"


def main() -> None:
    payload = read_payload()
    if payload is None:
        return
    command = bash_command(payload)
    cwd = payload.get("cwd")
    if command is None or not isinstance(cwd, str) or "commit" not in command:
        return
    command_steps = steps(command)
    directory = Path(cwd)
    for step in command_steps:
        if step.program == "cd":
            directory = target_directory(directory, [step])
            continue
        request = commit_request(step, directory)
        if request is None or overridden(step):
            continue
        root = mosaic_toplevel(request.directory)
        if root is None:
            continue
        try:
            files = committed_files(root, request)
            stale, note = docs_findings(root, files)
            findings = [
                *sensitive_findings(added_lines(root, request), request.message(step)),
                *lint_findings(root, files),
                *stale,
            ]
        except (OSError, subprocess.TimeoutExpired) as error:
            add_context(f"mosaic-commit-hook could not check this commit: {error}")
            return
        if findings:
            joined = "\n\n".join(findings)
            deny(
                f"The commit check found problems:\n\n{joined}\n\nFix them, stage the fixes and commit again."
            )
        elif note is not None:
            add_context(f"mosaic-commit-hook: {note}")
        return


if __name__ == "__main__":
    main()
