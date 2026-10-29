"""Read, write and print text without loss, as UTF-8 that keeps every other
byte."""

import io
import os
import subprocess
import sys
from pathlib import Path


def decoded(data: bytes) -> str:
    """Return the text of `data`. Bytes that are not UTF-8 survive into
    `encoded`."""
    return data.decode("utf-8", "surrogateescape")


def encoded(text: str) -> bytes:
    return text.encode("utf-8", "surrogateescape")


def substitution_text(output: bytes) -> str:
    """Return captured output as a command substitution yields it: NUL bytes
    and trailing newlines dropped."""
    return os.fsdecode(output.replace(b"\0", b"").rstrip(b"\n"))


def read_text(path: Path) -> str:
    """Return the text of the file at `path` as `decoded` reads it, carriage
    returns included."""
    return decoded(path.read_bytes())


def split_lines(text: str) -> list[str]:
    """Split `text` on newline only, keeping an unterminated last line."""
    lines = text.split("\n")
    if lines[-1] == "":
        _ = lines.pop()
    return lines


def nonempty_lines(text: str) -> list[str]:
    """Split `text` on newline, dropping empty lines."""
    return [line for line in text.split("\n") if line]


def pass_undecodable_bytes() -> None:
    """Write the bytes of an argument or a variable, decodable or not, as a
    shell does.

    Python decodes such bytes into surrogate escapes. The default error handler
    prints them as backslash sequences or raises.
    """
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="surrogateescape")


def write_error(data: bytes) -> None:
    """Write `data` to standard error unchanged, and skip the write when
    standard error is closed.

    Python sets sys.stderr to None for a closed descriptor 2, and print then
    writes to standard output. Descriptor 2 itself is left alone, because a
    file opened since by this process may have that number.
    """
    stream = sys.stderr
    if stream is None:
        return
    _ = stream.flush()
    _ = stream.buffer.write(data)
    _ = stream.buffer.flush()


def print_error(text: str) -> None:
    """Write `text` and a newline as `write_error` does."""
    write_error(encoded(f"{text}\n"))


def error_descriptor() -> int:
    """Return the destination for the standard error of a child: this process's
    standard error, or the null device when it is closed."""
    if sys.stderr is None:
        return subprocess.DEVNULL
    _ = sys.stderr.flush()
    return sys.stderr.fileno()


def flush_output() -> None:
    """Flush standard output and standard error, each unless it is closed."""
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            _ = stream.flush()
