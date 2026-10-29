"""Print a program's diagnostics, each line under the program's name."""

import os
import sys
from typing import NoReturn, TextIO

from lossless_text import print_error


def discard(stream: TextIO) -> None:
    """Point a broken stream at the null device. The flush at exit then cannot
    fail."""
    null = os.open(os.devnull, os.O_WRONLY)
    _ = os.dup2(null, stream.fileno())
    os.close(null)


def say(program: str, text: str) -> None:
    """Write `program: text` to standard error. Once standard error is gone,
    this line and every later one are dropped."""
    try:
        print_error(f"{program}: {text}")
    except OSError:
        discard(sys.stderr)


def refuse(program: str, text: str, status: int) -> NoReturn:
    """Write `program: text` as `say` does, then exit with `status`."""
    say(program, text)
    sys.exit(status)
