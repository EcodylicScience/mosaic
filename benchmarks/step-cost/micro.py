"""Time the per-entry operations of a step one at a time.

Usage: micro.py DATASET_DIR SCRATCH_DIR

Over the first 20 converted tracks tables of DATASET_DIR, prints the median time
of each operation that a speed-angvel entry costs: the input read, the apply,
the pickling a process worker adds, the output write, and the measures the run
takes per entry. Writes its outputs under a new directory in SCRATCH_DIR.
Prints the host's process count first, since the measures walk the process
tree.
"""

import pickle
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import psutil

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.pipeline.memory_watch import MemoryWatch
from mosaic.core.pipeline.writers import write_parquet_atomic

ENTRIES = 20
INPUT_COLUMNS = ["frame", "time", "id", "X", "Y"]


def main() -> None:
    dataset = Path(sys.argv[1])
    scratch = Path(tempfile.mkdtemp(dir=sys.argv[2]))
    files = sorted((dataset / "tracks").glob("*/*.parquet"))[:ENTRIES]
    feature = SpeedAngvel()
    watch = MemoryWatch()
    watch.begin()
    me = psutil.Process()
    timings: dict[str, list[float]] = {}

    def timed[T](name: str, work: Callable[[], T]) -> T:
        started = time.perf_counter()
        result = work()
        timings.setdefault(name, []).append(time.perf_counter() - started)
        return result

    print(f"processes on host: {len(psutil.pids())}")
    for path in files:
        table = timed("read input parquet, all columns", lambda: pd.read_parquet(path))
        timed(
            "read input parquet, five columns",
            lambda: pd.read_parquet(path, columns=INPUT_COLUMNS),
        )
        output = timed("apply speed-angvel", lambda: feature.apply(table))
        pickled_input = timed("pickle input", lambda: pickle.dumps(table, protocol=5))
        timed("unpickle input", lambda: pickle.loads(pickled_input))
        pickled_output = timed(
            "pickle output", lambda: pickle.dumps(output, protocol=5)
        )
        timed("unpickle output", lambda: pickle.loads(pickled_output))
        timed(
            "write_parquet_atomic output",
            lambda: write_parquet_atomic(output, scratch / path.name),
        )
        timed("MemoryWatch.cpu_seconds", watch.cpu_seconds)
        timed("MemoryWatch.sample", watch.sample)
        timed("psutil children(recursive=True)", lambda: me.children(recursive=True))
    first = pd.read_parquet(files[0])
    print(
        f"rows per entry {len(first)}, {first.memory_usage(deep=True).sum() / 1e6:.1f} MB"
    )
    for name, seconds in timings.items():
        print(f"{statistics.median(seconds) * 1000:8.1f} ms median  {name}")


if __name__ == "__main__":
    main()
