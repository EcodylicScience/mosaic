"""Print the build and apply seconds of each feature run in one dataset.

Usage: builds.py DATASET_DIR

Reads every run-log of DATASET_DIR through ``read_run_cost(...).entries`` and
prints one line per run with computed entries: how many entries record
``build_seconds`` (the entry whose process worker built the feature), the mean,
largest and least of those seconds, and the mean ``apply_seconds`` of the
entries with a build, of those without and of every entry that records it. A
tree whose ``EntryCost`` has no ``build_seconds`` counts every entry as without
a build.
"""

import statistics
import sys
from pathlib import Path

from mosaic.run_cost import EntryCost, read_run_cost


def _build_seconds(entry: EntryCost) -> float | None:
    # Trees from before the field have no attribute.
    seconds: float | None = getattr(entry, "build_seconds", None)
    return seconds


def main() -> None:
    runs = Path(sys.argv[1]) / ".mosaic" / "runs"
    for log in sorted(runs.glob("*.jsonl")):
        cost = read_run_cost(runs, log.stem)
        if cost is None or not cost.entries:
            continue
        built = [entry for entry in cost.entries if _build_seconds(entry) is not None]
        unbuilt = [entry for entry in cost.entries if _build_seconds(entry) is None]
        line = (
            f"{cost.feature} {log.stem}: {len(cost.entries)} entries,"
            f" {len(built)} with build_seconds"
        )
        builds = [
            seconds for entry in built if (seconds := _build_seconds(entry)) is not None
        ]
        if builds:
            line += (
                f", build mean {statistics.mean(builds):.3f} s,"
                f" max {max(builds):.3f} s, min {min(builds):.3f} s"
            )
        for label, entries in (("with", built), ("without", unbuilt)):
            applies = [
                entry.apply_seconds
                for entry in entries
                if entry.apply_seconds is not None
            ]
            if applies:
                separator = "; " if label == "with" else ", "
                line += (
                    f"{separator}apply mean {label} build"
                    f" {statistics.mean(applies):.3f} s"
                )
        applies = [
            entry.apply_seconds
            for entry in cost.entries
            if entry.apply_seconds is not None
        ]
        if applies:
            line += f"; apply mean over all {statistics.mean(applies):.3f} s"
        print(line)


if __name__ == "__main__":
    main()
