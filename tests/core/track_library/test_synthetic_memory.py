"""A synthetic entry converts in memory proportional to its table.

The converter returns one table and the write path holds a few copies of it, so the
spec reader caps a file at ``MAX_TABLE_CELLS`` cells. Each test converts one file in a
fresh interpreter, after a 600-row warm-up file has loaded every module and the pyarrow
pool, and reads the growth of the child's peak resident memory over the conversion.

The two slow tests are the largest entries a 16 GiB machine converts: 10 million rows
with 5 keypoints, and 30 million rows at the cell cap. Each runs under an exclusive
lock, so one large child runs per machine at a time.
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

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.track_library.synthetic_spec import (
    MAX_TABLE_CELLS,
    read_synthetic_spec,
)
from tests.helpers import make_dataset, run_child, run_fresh, write_synthetic_spec

GIB: Final = 1024**3
MIB: Final = 1024**2
CHILD_TIMEOUT_SECONDS: Final = 30 * 60.0
LAUNCHER_MARGIN_SECONDS: Final = 30.0
SMALL_TIMEOUT_SECONDS: Final = 120.0
WARM_UP_FRAMES: Final = 6
INDIVIDUALS: Final = 100
KEYPOINTS: Final = 5
LONG_FRAMES: Final = 300_000

LAUNCHER: Final = """
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

The first argument is the seconds the grandchild has. A longer run kills the grandchild
and fails the launcher, so no conversion outlives the lock the test holds.

On Linux a process starts with the peak resident memory of the process that spawned
it, so a program started from the large test process would read its own growth as zero.
The launcher is small, and its child starts from the launcher's peak.
"""

PROGRAM: Final = """
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.memory_watch import max_rss_bytes
from mosaic.core.pipeline.tracks_index import read_tracks_index

manifest, measured_raw, read_table = sys.argv[1:4]
ds = Dataset(manifest_path=Path(manifest)).load()
warm = ds.convert_all_tracks()
assert (warm.converted, warm.failed) == (1, 0), warm
before = max_rss_bytes()
_ = ds.index_tracks_raw([Path(measured_raw)], patterns=["*.toml"], src_format="synthetic")
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


def _overrides(frames: int, keypoints: int) -> dict[str, object]:
    return {
        "frames": frames,
        "individuals": INDIVIDUALS,
        "zones": {"repulsion": 15.0, "orientation": 60.0, "attraction": 200.0},
        "body": {"keypoints": keypoints, "length": 20.0, "keypoint_sd": 1.0},
    }


def _prepare(tmp_path: Path, frames: int, keypoints: int) -> tuple[Dataset, Path]:
    """A dataset indexing the warm-up file, and the directory of the measured file."""
    ds = make_dataset(tmp_path / "ds")
    warm = ds.base_dir / "warm"
    _ = write_synthetic_spec(
        warm / "warm.toml", seed=1, **_overrides(WARM_UP_FRAMES, keypoints)
    )
    _ = ds.index_tracks_raw([warm], patterns=["*.toml"], src_format="synthetic")
    measured = ds.base_dir / "measured"
    path = write_synthetic_spec(
        measured / "measured.toml", seed=2, **_overrides(frames, keypoints)
    )
    spec = read_synthetic_spec(path)
    assert spec.frames * spec.individuals * spec.numeric_columns <= MAX_TABLE_CELLS
    return ds, measured


def _measurements(stdout: str) -> dict[str, int]:
    """The program's last line: ``growth``, ``table_bytes`` and ``parquet_bytes``."""
    payload: dict[str, int] = json.loads(stdout.strip().splitlines()[-1])
    return payload


@contextmanager
def _one_large_child() -> Generator[None]:
    """Hold the machine's lock for large conversions, so one child runs at a time."""
    lock_path = Path(tempfile.gettempdir()) / "mosaic-synthetic-memory.lock"
    with lock_path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _slow_growth(tmp_path: Path, frames: int, keypoints: int) -> int:
    ds, measured = _prepare(tmp_path, frames, keypoints)
    argv = [
        sys.executable,
        "-c",
        LAUNCHER,
        str(CHILD_TIMEOUT_SECONDS - LAUNCHER_MARGIN_SECONDS),
        PROGRAM,
        str(ds.manifest_path),
        str(measured),
        "0",
    ]
    with _one_large_child():
        completed = run_child(argv, timeout=CHILD_TIMEOUT_SECONDS)
    code = completed.returncode
    assert code >= 0, (
        f"the conversion child was killed by {signal.Signals(-code).name}; "
        f"stderr: {completed.stderr}"
    )
    assert code == 0, completed.stderr
    measurements = _measurements(completed.stdout)
    growth = measurements["growth"]
    print(
        f"{frames * INDIVIDUALS} rows, {keypoints} keypoints: growth "
        f"{growth / GIB:.2f} GiB, parquet {measurements['parquet_bytes'] / GIB:.2f} "
        "GiB uncompressed"
    )
    return growth


def test_a_200k_row_entry_grows_memory_by_a_few_tables(tmp_path: Path) -> None:
    """Growth stays under a few tables' bytes plus 32 MiB.

    The bound is 4 tables on Linux and 6 on macOS, where the allocator returns less
    between steps. The measured ratio of growth to table bytes is 2.9 on Linux and
    between 4.3 and 5.5 on macOS. A table copy added to the write path moves the ratio
    by about 1, so the Linux bound catches the second added copy and the macOS bound
    one to two.
    """
    ds, measured = _prepare(tmp_path, 2_000, KEYPOINTS)
    argv = [
        str(SMALL_TIMEOUT_SECONDS - LAUNCHER_MARGIN_SECONDS),
        PROGRAM,
        str(ds.manifest_path),
        str(measured),
        "1",
    ]
    stdout = run_fresh(LAUNCHER, args=argv, timeout=SMALL_TIMEOUT_SECONDS)
    measurements = _measurements(stdout)
    growth, table_bytes = measurements["growth"], measurements["table_bytes"]
    print(
        f"growth {growth} bytes, table {table_bytes} bytes, ratio {growth / table_bytes:.2f}"
    )
    tables = 4 if sys.platform == "linux" else 6
    assert growth <= tables * table_bytes + 32 * MIB, (
        f"growth {growth} bytes over a table of {table_bytes} bytes"
    )


@pytest.mark.slow
def test_a_10m_row_entry_converts_within_8_gib(tmp_path: Path) -> None:
    growth = _slow_growth(tmp_path, 100_000, KEYPOINTS)

    assert growth <= 8 * GIB, f"growth {growth / GIB:.2f} GiB"


@pytest.mark.slow
def test_a_30m_row_entry_converts_within_12_gib(tmp_path: Path) -> None:
    growth = _slow_growth(tmp_path, LONG_FRAMES, KEYPOINTS)

    assert growth <= 12 * GIB, f"growth {growth / GIB:.2f} GiB"
