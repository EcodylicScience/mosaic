"""Print the bench's results and the figures that its targets name.

Usage: summary.py OUT [ENTRIES WORKERS]

Reads OUT/results.tsv, which timed.py appends to, and prints each run's wall
time. It then prints the target figures: the wall of one cached step process
(``mosaic run --graph-request`` and ``pipeline settle``), the 61-step recipe
cached in one process, and the parent's cost per entry of the cold speed-angvel
step in thread and in process mode at WORKERS (default 8). That cost is the cold
run's wall less the shorter of the cached runs of the same mode, over ENTRIES
(default 160). It prints the serial cold wall over the thread-mode cold wall at
WORKERS, and for the cold process-mode run the mean CPU of its WORKERS busiest
child processes and the CPU seconds of the parent and of those workers, from its
``.cpu.txt``. Last it prints the bytes per entry that the process-mode run of
pipes.py sent to its workers and received from them, from OUT/pipes.tsv. A run
that is missing leaves its figure out.
"""

import re
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
    serial = walls.get("speed-angvel cold, serial")
    thread = walls.get(f"speed-angvel cold, thread{workers}")
    if serial is not None and thread is not None:
        print(f"cold thread {workers} speed over serial: {serial / thread:.2f}")
    cpu = out / f"speed-angvel cold, process{workers}.cpu.txt"
    if cpu.is_file():
        # The lines rank each process by its summed samples, busiest first.
        processes = [
            (match.group(1), float(match.group(2)), float(match.group(3)))
            for line in cpu.read_text().splitlines()
            if (match := re.match(r"(\w+) \d+: .* mean (\d+)% .* cpu_s ([\d.]+)", line))
        ]
        children = [process for process in processes if process[0] == "child"][:workers]
        parent = [process for process in processes if process[0] == "parent"]
        if children and parent:
            print(
                f"cold process {workers}: mean CPU of a worker"
                f" {statistics.mean(mean for _, mean, _ in children):.1f}%"
                f" over {len(children)}, CPU seconds of the parent {parent[0][2]:.1f},"
                f" of the workers {sum(seconds for _, _, seconds in children):.1f}"
            )
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
