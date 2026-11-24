"""Run one ``mosaic`` command in this process and count the bytes on its pipes.

Usage: pipes.py OUT LABEL -- MOSAIC_ARGS...

Runs the command through the CLI's entry function with
``multiprocessing.connection.Connection._send_bytes`` and ``_recv_bytes``
wrapped, so every message this process sends to its worker processes or
receives from them is counted. A process pool's call queue carries each task to
the workers from its feeder thread, and its result queue carries each outcome
back to the executor's manager thread. Appends
``LABEL<TAB>bytes sent<TAB>bytes received<TAB>wall seconds<TAB>exit status`` to
``OUT/pipes.tsv``, and writes the messages and bytes of each direction per
thread of this process to ``OUT/LABEL.pipes.txt``. A command that raises
instead of exiting is recorded with status 1 and a wall of ``nan``. Worker
processes import this file under spawn and keep their connections unwrapped.
"""

import io
import math
import sys
import threading
from collections import Counter
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal

from cli_run import run_mosaic

type Direction = Literal["sent", "received"]


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

    setattr(Connection, "_send_bytes", counted_send)
    setattr(Connection, "_recv_bytes", counted_recv)
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
        (out / f"{label}.pipes.txt").write_text("\n".join(lines) + "\n")
        with (out / "pipes.tsv").open("a") as handle:
            handle.write(f"{label}\t{sent}\t{received}\t{wall:.2f}\t{status}\n")
        print(
            f"{label}: sent {sent} bytes, received {received} bytes, wall {wall:.2f} s"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
