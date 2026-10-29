"""Run path operations without raising.

Each answers as the shell's `test`, `rm -f` or `mktemp` does.
"""

import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from lossless_text import encoded


def present(probe: Callable[[], bool]) -> bool:
    """Return the answer of a path probe, with paths that cannot be examined
    counted as absent.

    pathlib raises for a path below a directory that cannot be searched, where
    `test` answers false.
    """
    try:
        return probe()
    except OSError:
        return False


def nonempty(path: Path) -> bool:
    """Return whether `path` is nonempty, as `test -s` answers."""
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def read_quietly(path: Path) -> bytes | None:
    """Return the content of the file at `path`, None when it cannot be
    read."""
    try:
        return path.read_bytes()
    except OSError:
        return None


def write_quietly(path: Path, text: str) -> bool:
    """Write `text` to `path`, False when it cannot."""
    try:
        _ = path.write_bytes(encoded(text))
    except OSError:
        return False
    return True


def new_file(directory: Path, prefix: str, suffix: str = "") -> Path | None:
    """Create a new empty file in `directory`, or return None when the
    directory cannot be written."""
    try:
        descriptor, name = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=directory)
    except OSError:
        return None
    os.close(descriptor)
    return Path(name)


def remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def linked_file(path: Path) -> Path | None:
    """Return the file that `path` names through a symbolic link, `path` itself
    when it is not a link, or None when the link cannot be resolved."""
    if not present(path.is_symlink):
        return path
    try:
        return path.resolve(strict=True)
    except OSError:
        return None


@contextmanager
def sibling_file(target: Path, prefix: str, suffix: str = "") -> Generator[Path | None, None, None]:
    """Yield a new empty file beside `target`, to be renamed over it, or None
    when the directory cannot be written. The file is removed on exit unless it
    was renamed.

    Pass the file that a link names (`linked_file`), because a rename over a
    link replaces the link.
    """
    temporary = new_file(target.parent, prefix, suffix)
    try:
        yield temporary
    finally:
        if temporary is not None:
            remove_quietly(temporary)
