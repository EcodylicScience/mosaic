"""Spell paths as a shell spells them.

Paths are normalized lexically and taken from the logical working directory.
"""

import os
from pathlib import Path


def lexically_normal(path: Path) -> Path:
    """Return `path` with each `..` removed together with the component before
    it.

    Symbolic links are not followed, and `link/..` is the directory that
    contains `link`. `..` components at the root are dropped, and those leading
    a relative path are kept.
    """
    parts: list[str] = []
    for part in path.parts:
        if part != "..":
            parts.append(part)
        elif parts and parts[-1] not in (path.anchor, ".."):
            _ = parts.pop()
        elif not path.anchor:
            parts.append(part)
    return Path(*parts)


def normal_join(base: Path, path: str | Path) -> Path:
    """Return `path` taken from `base`, lexically normalized."""
    return lexically_normal(base / path)


def appended_to_name(path: Path, suffix: str) -> Path:
    """Return `path` with `suffix` appended to its last component."""
    return path.parent / f"{path.name}{suffix}"


def working_directory() -> Path:
    """Return the current directory as a shell names it.

    Return $PWD while it names this directory. Paths entered through a symbolic
    link then keep that spelling.
    """
    named = os.environ.get("PWD", "")
    if named.startswith("/"):
        try:
            if Path(named).samefile("."):
                return lexically_normal(Path(named))
        except OSError:
            pass
    return Path.cwd()


def from_working_directory(path: str | Path) -> Path:
    """Return `path` taken from the working directory as a shell names it.

    `..` is kept, and opening the path resolves it past symbolic links as
    opening the relative path does.
    """
    return working_directory() / path


def temporary_directory() -> Path:
    """Return TMPDIR, or /tmp without it, as an absolute path.

    Relative TMPDIR values are taken from the working directory.
    """
    return from_working_directory(os.environ.get("TMPDIR") or "/tmp")
