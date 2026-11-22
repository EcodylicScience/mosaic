"""Write the bench's synthetic track files.

Usage: make_raw.py DIRECTORY COUNT

Writes COUNT files ``g<i mod 4>/s<i>.toml`` under DIRECTORY. Each describes 9,000
frames of 4 individuals with 5 keypoints, 36,000 rows once converted, seeded by
its index.
"""

import sys
from pathlib import Path

TEMPLATE = """format = "mosaic-synthetic/1"
seed = {seed}
dimensions = 2
frames = 9000
fps = 30.0
individuals = 4
[arena]
shape = "box"
width = 1000.0
height = 1000.0
[motion]
mean_speed = 100.0
speed_sd = 30.0
[body]
keypoints = 5
length = 30.0
keypoint_sd = 1.0
"""


def main() -> None:
    root = Path(sys.argv[1])
    count = int(sys.argv[2])
    for index in range(count):
        directory = root / f"g{index % 4}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"s{index:03d}.toml").write_text(TEMPLATE.format(seed=index))


if __name__ == "__main__":
    main()
