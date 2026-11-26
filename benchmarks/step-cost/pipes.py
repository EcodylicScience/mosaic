"""Run one ``mosaic`` command in this process and count the bytes on its pipes.

Usage: pipes.py OUT LABEL -- MOSAIC_ARGS...

Runs the command through the CLI's entry function with
``multiprocessing.connection.Connection._send_bytes`` and ``_recv_bytes``
wrapped, so every message this process sends to its worker processes or
receives from them is counted. A process pool's call queue carries each task to
the workers from its feeder thread, and its result queue carries each outcome
back to the executor's manager thread.

It also wraps ``multiprocessing.reduction.dump``, which pickles the data a
spawned worker is sent at its start: the preparation data, then the process
object, which holds the pool's initializer and its arguments. Each dump writes
through a file that counts its bytes.

Appends ``LABEL<TAB>bytes sent<TAB>bytes received<TAB>wall seconds<TAB>exit
status<TAB>workers spawned<TAB>spawn bytes`` to ``OUT/pipes.tsv``, writes the
messages and bytes of each direction per thread of this process to
``OUT/LABEL.pipes.txt``, and writes the preparation and process-object bytes of
each spawned worker to ``OUT/LABEL.spawn.txt``. A command that raises instead
of exiting is recorded with status 1 and a wall of ``nan``. Worker processes
import this file under spawn and keep their connections unwrapped.
"""

import io
import math
import sys
import threading
from collections import Counter
from collections.abc import Callable
from multiprocessing import reduction
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Literal, Protocol

from cli_run import run_mosaic

type Direction = Literal["sent", "received"]


class Writable(Protocol):
    def write(self, data: bytes, /) -> object: ...


class CountingFile:
    """A write-only file that passes each write on to *target* and counts its bytes."""

    def __init__(self, target: Writable) -> None:
        self.target = target
        self.size = 0

    def write(self, data: bytes, /) -> object:
        self.size += len(data)
        return self.target.write(data)


def main() -> int:
    out = Path(sys.argv[1])
    label = sys.argv[2]
    args = sys.argv[sys.argv.index("--") + 1 :]
    total_bytes: Counter[Direction] = Counter()
    thread_bytes: Counter[tuple[Direction, str]] = Counter()
    thread_messages: Counter[tuple[Direction, str]] = Counter()
    lock = threading.Lock()
    # The standard library offers no public hook on the bytes a connection moves.
    send_bytes: Callable[[Connection, memoryview], None] = getattr(
        Connection, "_send_bytes"
    )
    recv_bytes: Callable[[Connection, int | None], io.BytesIO] = getattr(
        Connection, "_recv_bytes"
    )

    def record(direction: Direction, size: int) -> None:
        thread = threading.current_thread()
        key = (direction, f"{type(thread).__name__} {thread.name}")
        with lock:
            total_bytes[direction] += size
            thread_bytes[key] += size
            thread_messages[key] += 1

    def counted_send(connection: Connection, buffer: memoryview) -> None:
        send_bytes(connection, buffer)
        record("sent", buffer.nbytes)

    def counted_recv(connection: Connection, maxsize: int | None = None) -> io.BytesIO:
        received = recv_bytes(connection, maxsize)
        with received.getbuffer() as view:
            record("received", view.nbytes)
        return received

    dump: Callable[[object, Writable, int | None], None] = reduction.dump
    # Per spawning thread: the preparation bytes dumped before the process object.
    preparation: dict[int, int] = {}
    spawned: list[tuple[str, int, int]] = []

    def counted_dump(obj: object, file: Writable, protocol: int | None = None) -> None:
        counting = CountingFile(file)
        dump(obj, counting, protocol)
        thread = threading.get_ident()
        with lock:
            if isinstance(obj, BaseProcess):
                prepared = preparation.pop(thread, 0)
                spawned.append((obj.name, prepared, counting.size))
            else:
                preparation[thread] = counting.size

    setattr(Connection, "_send_bytes", counted_send)
    setattr(Connection, "_recv_bytes", counted_recv)
    setattr(reduction, "dump", counted_dump)
    status, wall = 1, math.nan
    try:
        status, wall = run_mosaic(args)
    finally:
        out.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        with lock:
            for (direction, thread), size in sorted(thread_bytes.items()):
                messages = thread_messages[(direction, thread)]
                lines.append(
                    f"{direction}\t{messages} messages\t{size} bytes\t{thread}"
                )
            sent, received = total_bytes["sent"], total_bytes["received"]
            workers = [
                f"{name}\t{prepared} preparation bytes\t{process} process bytes"
                for name, prepared, process in spawned
            ]
            spawn_bytes = sum(prepared + process for _, prepared, process in spawned)
        (out / f"{label}.pipes.txt").write_text("\n".join(lines) + "\n")
        (out / f"{label}.spawn.txt").write_text("\n".join(workers) + "\n")
        with (out / "pipes.tsv").open("a") as handle:
            handle.write(
                f"{label}\t{sent}\t{received}\t{wall:.2f}\t{status}"
                f"\t{len(spawned)}\t{spawn_bytes}\n"
            )
        print(
            f"{label}: sent {sent} bytes, received {received} bytes,"
            f" {len(spawned)} workers spawned with {spawn_bytes} bytes,"
            f" wall {wall:.2f} s"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
