"""Run a command, sample the CPU of its process tree, and record its wall time.

Usage: timed.py OUT_DIR LABEL -- COMMAND...

Appends one line, ``LABEL<TAB>wall seconds<TAB>exit status``, to
``OUT_DIR/results.tsv``. Writes the per-process CPU samples to
``OUT_DIR/LABEL.cpu.tsv`` (seconds since start, pid, role, CPU percent) and a
summary per process to ``OUT_DIR/LABEL.cpu.txt``. The command's standard output
and error go to ``OUT_DIR/LABEL.out`` and ``OUT_DIR/LABEL.err``. Exits with the
command's status.
"""

import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil

SAMPLE_SECONDS = 0.5


def _cpu_seconds(process: psutil.Process) -> float:
    times = process.cpu_times()
    return times.user + times.system


def main() -> int:
    out_dir = Path(sys.argv[1])
    label = sys.argv[2]
    command = sys.argv[sys.argv.index("--") + 1 :]
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (
        (out_dir / f"{label}.out").open("w") as stdout,
        (out_dir / f"{label}.err").open("w") as stderr,
    ):
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        root = psutil.Process(child.pid)
        last: dict[int, tuple[float, float]] = {}
        samples: list[tuple[float, int, str, float]] = []
        done = threading.Event()

        def sample() -> None:
            while not done.wait(SAMPLE_SECONDS):
                try:
                    tree = [root, *root.children(recursive=True)]
                except psutil.Error:
                    return
                now = time.perf_counter() - started
                for process in tree:
                    try:
                        cpu = _cpu_seconds(process)
                    except psutil.Error:
                        continue
                    if process.pid in last:
                        then, cpu_then = last[process.pid]
                        role = "parent" if process.pid == child.pid else "child"
                        percent = 100 * (cpu - cpu_then) / max(now - then, 1e-6)
                        samples.append((now, process.pid, role, percent))
                    last[process.pid] = (now, cpu)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        status = child.wait()
        wall = time.perf_counter() - started
        done.set()
        sampler.join()
    with (out_dir / f"{label}.cpu.tsv").open("w") as handle:
        for now, pid, role, percent in samples:
            handle.write(f"{now:.2f}\t{pid}\t{role}\t{percent:.1f}\n")
    by_process: defaultdict[tuple[int, str], list[float]] = defaultdict(list)
    for _, pid, role, percent in samples:
        by_process[(pid, role)].append(percent)
    lines = [f"wall {wall:.2f}s exit {status}"]
    ranked = sorted(by_process.items(), key=lambda item: -sum(item[1]))
    for (pid, role), percents in ranked[:20]:
        mean = sum(percents) / len(percents)
        lines.append(
            f"{role} {pid}: samples {len(percents)} mean {mean:.0f}%"
            f" max {max(percents):.0f}% cpu_s {last[pid][1]:.1f}"
        )
    (out_dir / f"{label}.cpu.txt").write_text("\n".join(lines) + "\n")
    with (out_dir / "results.tsv").open("a") as handle:
        handle.write(f"{label}\t{wall:.2f}\t{status}\n")
    print(f"{label}: wall {wall:.2f} s, exit {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
