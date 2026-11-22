"""List the costliest functions of a cProfile file by cumulative time.

Usage: pstat.py PROFILE COUNT [PATH_SUBSTRING]

Prints COUNT rows of cumulative seconds, own seconds, calls and function,
keeping only functions whose file path contains PATH_SUBSTRING when given.
"""

import pstats
import sys


def main() -> None:
    stats = pstats.Stats(sys.argv[1])
    count = int(sys.argv[2])
    path_filter = sys.argv[3] if len(sys.argv) > 3 else ""
    profile = stats.get_stats_profile()
    rows: list[tuple[float, float, str, str]] = []
    for name, function in profile.func_profiles.items():
        if path_filter not in function.file_name:
            continue
        short = function.file_name.split("/src/")[-1].split("site-packages/")[-1]
        rows.append(
            (
                function.cumtime,
                function.tottime,
                function.ncalls,
                f"{short}:{function.line_number}({name})",
            )
        )
    rows.sort(reverse=True)
    print(f"total {profile.total_tt:.2f}s")
    for cumulative, own, calls, label in rows[:count]:
        print(f"{cumulative:8.2f} {own:7.2f} {calls:>9}  {label}")


if __name__ == "__main__":
    main()
