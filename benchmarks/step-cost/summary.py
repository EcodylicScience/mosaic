"""Print the bench's results and the figures that its targets name.

Usage: summary.py OUT [ENTRIES WORKERS]

Reads OUT/results.tsv, which timed.py appends to, and prints each run's wall
time. It then prints the target figures: the wall of one cached step process
(``mosaic run --graph-request`` and ``pipeline settle``), the 61-step recipe
cached in one process, and the parent's cost per entry of the cold speed-angvel
step and of the widening and pair-filter recipes' stack steps in thread and in
process mode at WORKERS (default 8). That cost is the cold run's wall less the
shorter of the cached runs of the same mode, over ENTRIES (default 160). It
prints the serial cold wall over the thread-mode cold wall at WORKERS, and for
each cold process-mode run the mean CPU of its WORKERS busiest child processes
and the CPU seconds of the parent and of those workers, from its ``.cpu.txt``,
and for the speed-angvel run the build and apply seconds of its entries from its
``.builds.txt``. Last it prints the bytes per entry that the process-mode runs
of pipes.py sent to its workers and received from them, and the mean bytes each
worker was sent at its spawn, from OUT/pipes.tsv. A run that is missing leaves
its figure out.
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
    for step, cold_prefix, cached_prefix, cached_suffixes in (
        (
            "speed-angvel",
            "speed-angvel cold, ",
            "speed-angvel recipe cached, ",
            {"thread": "one process", "process": "process"},
        ),
        *(
            (
                f"{recipe} stack",
                f"{recipe} cold, ",
                f"{recipe} cached, ",
                {"thread": f"thread{workers}", "process": f"process{workers}"},
            )
            for recipe in ("widening", "pairfilter")
        ),
    ):
        for mode, cached_suffix in cached_suffixes.items():
            cold = walls.get(f"{cold_prefix}{mode}{workers}")
            cached = matching(f"{cached_prefix}{cached_suffix}")
            if cold is not None and cached:
                per_entry = (cold - min(cached)) / entries * 1000
                print(
                    f"parent per-entry cost, cold {step} {mode} {workers}:"
                    f" {per_entry:.1f} ms"
                )
        serial = walls.get(f"{cold_prefix}serial")
        thread = walls.get(f"{cold_prefix}thread{workers}")
        if serial is not None and thread is not None:
            print(
                f"cold {step} thread {workers} speed over serial: {serial / thread:.2f}"
            )
    for step, cold_prefix in (
        ("speed-angvel", "speed-angvel cold, "),
        ("widening stack", "widening cold, "),
        ("pairfilter stack", "pairfilter cold, "),
    ):
        cpu = out / f"{cold_prefix}process{workers}.cpu.txt"
        if not cpu.is_file():
            continue
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
                f"cold {step} process {workers}: mean CPU of a worker"
                f" {statistics.mean(mean for _, mean, _ in children):.1f}%"
                f" over {len(children)}, CPU seconds of the parent {parent[0][2]:.1f},"
                f" of the workers {sum(seconds for _, _, seconds in children):.1f}"
            )
    builds = out / f"speed-angvel cold, process{workers}.builds.txt"
    if builds.is_file():
        for line in builds.read_text().splitlines():
            print(f"cold speed-angvel process {workers}: {line}")
    pipes = out / "pipes.tsv"
    if pipes.is_file():
        for line in pipes.read_text().splitlines():
            label, sent, received, wall, status, spawned, spawn_bytes = line.split("\t")
            marker = "" if status == "0" else f", exit {status}"
            print(
                f"pipe bytes per entry, {label}: {int(sent) / entries:,.0f} to the"
                f" workers, {int(received) / entries:,.0f} from them"
                f" (wall {wall} s{marker})"
            )
            if int(spawned):
                print(
                    f"spawn bytes per worker, {label}:"
                    f" {int(spawn_bytes) / int(spawned):,.0f} over {spawned} workers"
                )


if __name__ == "__main__":
    main()
