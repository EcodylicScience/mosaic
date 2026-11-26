"""The growth of peak memory over one conversion, measured in a fresh interpreter.

A conversion test indexes a warm-up file and a measured file of one source format.
:func:`conversion_growth` runs the measuring program in a fresh interpreter. The
program converts the warm-up file, which loads every module and the pyarrow pool, then
converts the measured file, and reads the growth of the child's peak resident
memory over the second conversion. The program serves every source format that a
pattern and a ``src_format`` name.

A large conversion takes :func:`one_large_child`, the exclusive lock on
``<tempdir>/mosaic-synthetic-memory.lock``, so one such child runs per machine at a
time. The lock file keeps its name across branches, so two checkouts running a
large conversion share it.

On Linux a process starts with the peak resident memory of the process that spawned
it. A program started from the large test process reads zero growth.
The launcher is small, and its child starts from the launcher's peak.
"""

from __future__ import annotations

import fcntl
import json
import signal
import sys
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from tests.helpers.fresh_interpreter import run_fresh
from tests.helpers.processes import run_child

__all__ = ["conversion_growth", "one_large_child"]

_CHILD_TIMEOUT_SECONDS: Final = 30 * 60.0
_LAUNCHER_MARGIN_SECONDS: Final = 30.0
_SMALL_TIMEOUT_SECONDS: Final = 120.0

_LAUNCHER: Final = """
import os
import subprocess
import sys

timeout = float(sys.argv[1])
argv = [sys.executable, "-c", sys.argv[2], *sys.argv[3:]]
code = subprocess.run(argv, timeout=timeout).returncode
if code < 0:
    os.kill(os.getpid(), -code)
raise SystemExit(code)
"""
"""Runs the program in a grandchild interpreter and passes its exit status on.

The first argument is the seconds the grandchild has. A longer run kills the
grandchild and fails the launcher. No conversion outlives the lock the test holds.
"""

_PROGRAM: Final = """
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.memory_watch import max_rss_bytes
from mosaic.core.pipeline.tracks_index import read_tracks_index

manifest, measured_raw, pattern, src_format, read_table = sys.argv[1:6]
ds = Dataset(manifest_path=Path(manifest)).load()
warm = ds.convert_all_tracks()
assert (warm.converted, warm.failed) == (1, 0), warm
before = max_rss_bytes()
_ = ds.index_tracks_raw([Path(measured_raw)], patterns=[pattern], src_format=src_format)
measured = ds.convert_all_tracks()
after = max_rss_bytes()
assert (measured.converted - measured.current, measured.failed) == (1, 0), measured
written = ds.resolve_path(str(read_tracks_index(ds).iloc[-1]["abs_path"]))
metadata = pq.ParquetFile(written).metadata
parquet_bytes = sum(
    metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)
)
table_bytes = 0
if read_table == "1":
    table_bytes = int(pd.read_parquet(written).memory_usage(deep=True).sum())
print(
    json.dumps(
        {
            "growth": after - before,
            "table_bytes": table_bytes,
            "parquet_bytes": parquet_bytes,
        }
    )
)
"""
"""Converts a warm-up entry, then the measured one, and prints the measurements.

The arguments are the manifest, the directory of the measured file, its pattern, its
source format and ``1`` to read the written table back for its bytes. The last
line of the output is a JSON object of ``growth``, ``table_bytes`` and
``parquet_bytes``.
"""


@contextmanager
def one_large_child() -> Generator[None]:
    """Take the machine's lock for large conversions, one child at a time."""
    lock_path = Path(tempfile.gettempdir()) / "mosaic-synthetic-memory.lock"
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _measurements(stdout: str) -> dict[str, int]:
    """The program's last line: ``growth``, ``table_bytes`` and ``parquet_bytes``."""
    payload: dict[str, int] = json.loads(stdout.strip().splitlines()[-1])
    return payload


def conversion_growth(
    manifest_path: Path,
    measured: Path,
    *,
    pattern: str,
    src_format: str,
    read_table: bool,
    large: bool,
) -> dict[str, int]:
    """Convert the measured file in a fresh interpreter and return its measurements.

    A small conversion has 2 minutes. A large one has 30 minutes and runs under
    :func:`one_large_child`, and its exit status is checked, so a child that a
    signal killed is named.

    Args:
        manifest_path: The manifest of a dataset that indexes the warm-up file.
        measured: The directory of the measured file.
        pattern: The glob that finds the measured file.
        src_format: The source format of the measured file.
        read_table: Whether the program reads the written table back for its bytes.
            It is nonzero in the result only then, and it costs the table's memory.
        large: Whether the conversion runs under the machine's lock for large ones.

    Returns:
        ``growth`` (bytes of peak resident memory over the measured conversion),
        ``table_bytes`` (zero unless *read_table*) and ``parquet_bytes`` (the
        uncompressed size of the written file).

    Raises:
        AssertionError: The child exits nonzero, is killed by a signal, or runs past
            its timeout.
    """
    timeout = _CHILD_TIMEOUT_SECONDS if large else _SMALL_TIMEOUT_SECONDS
    args = [
        str(timeout - _LAUNCHER_MARGIN_SECONDS),
        _PROGRAM,
        str(manifest_path),
        str(measured),
        pattern,
        src_format,
        "1" if read_table else "0",
    ]
    if not large:
        return _measurements(run_fresh(_LAUNCHER, args=args, timeout=timeout))
    with one_large_child():
        completed = run_child([sys.executable, "-c", _LAUNCHER, *args], timeout=timeout)
    code = completed.returncode
    assert code >= 0, (
        f"the conversion child was killed by {signal.Signals(-code).name}; "
        f"stderr: {completed.stderr}"
    )
    assert code == 0, completed.stderr
    return _measurements(completed.stdout)
