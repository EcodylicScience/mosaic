"""Print the bench's results and the figures that its targets name.

Usage: summary.py OUT [ENTRIES WORKERS]

Reads OUT/results.tsv, which timed.py appends to, and prints each run's wall
time. It then prints the target figures: the wall of one cached step process
(``mosaic run --graph-request`` and ``pipeline settle``), the 61-step recipe
cached in one process, and the parent's cost per entry of the cold speed-angvel
step in thread and in process mode at WORKERS (default 8). That cost is the cold
run's wall less the shorter of the cached runs of the same mode, over ENTRIES
(default 160). Last it prints the bytes per entry that the process-mode run of
pipes.py sent to its workers and received from them, from OUT/pipes.tsv. A run
that is missing leaves its figure out.
"""

import statistics
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    entries = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    walls: dict[str, float] = {}
    for line in (out / "results.tsv").read_text().splitlines():
        label, wall, status = line.split("\t")
        walls[label] = float(wall)
        marker = "" if status == "0" else f"  exit {status}"
        print(f"{float(wall):8.2f} s  {label}{marker}")

    def matching(prefix: str) -> list[float]:
        return [wall for label, wall in walls.items() if label.startswith(prefix)]

    print()
    steps = matching("step process cached") + matching("settle process cached")
    if steps:
        print(
            f"one cached step process: median {statistics.median(steps):.2f} s,"
            f" range {min(steps):.2f} to {max(steps):.2f} s over {len(steps)}"
        )
    wide = matching("wide cached, one process")
    if wide:
        print(f"61 cached steps in one process: {min(wide):.2f} s")
    for mode, cached_prefix in (
        ("thread", "speed-angvel recipe cached, one process"),
        ("process", "speed-angvel recipe cached, process"),
    ):
        cold = walls.get(f"speed-angvel cold, {mode}{workers}")
        cached = matching(cached_prefix)
        if cold is not None and cached:
            per_entry = (cold - min(cached)) / entries * 1000
            print(f"parent per-entry cost, cold {mode} {workers}: {per_entry:.1f} ms")
    pipes = out / "pipes.tsv"
    if pipes.is_file():
        for line in pipes.read_text().splitlines():
            label, sent, received, wall, status = line.split("\t")
            marker = "" if status == "0" else f", exit {status}"
            print(
                f"pipe bytes per entry, {label}: {int(sent) / entries:,.0f} to the"
                f" workers, {int(received) / entries:,.0f} from them"
                f" (wall {wall} s{marker})"
            )


if __name__ == "__main__":
    main()
