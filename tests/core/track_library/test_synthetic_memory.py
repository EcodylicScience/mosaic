"""A synthetic entry converts in memory proportional to its table.

The converter returns one table and the write path keeps a few copies of it in
memory. The spec reader caps a file at ``MAX_TABLE_CELLS`` cells. Each test converts one file in a
fresh interpreter, after a 600-row warm-up file has loaded every module and the pyarrow
pool, and reads the growth of the child's peak resident memory over the conversion.

The two slow tests are the largest entries a 16 GiB machine converts: 10 million rows
with 5 keypoints, and 30 million rows at the cell cap. Each runs under an exclusive
lock that allows one large child per machine at a time. The launcher, the program and
the lock are in ``tests/helpers/conversion_memory.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.track_library.synthetic_spec import (
    MAX_TABLE_CELLS,
    read_synthetic_spec,
)
from tests.helpers import conversion_growth, make_dataset, write_synthetic_spec

GIB: Final = 1024**3
MIB: Final = 1024**2
WARM_UP_FRAMES: Final = 6
INDIVIDUALS: Final = 100
KEYPOINTS: Final = 5
LONG_FRAMES: Final = 300_000


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


def _slow_growth(tmp_path: Path, frames: int, keypoints: int) -> int:
    ds, measured = _prepare(tmp_path, frames, keypoints)
    measurements = conversion_growth(
        ds.manifest_path,
        measured,
        pattern="*.toml",
        src_format="synthetic",
        read_table=False,
        large=True,
    )
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
    by about 1. The Linux bound catches the second added copy and the macOS bound
    one to two.
    """
    ds, measured = _prepare(tmp_path, 2_000, KEYPOINTS)
    measurements = conversion_growth(
        ds.manifest_path,
        measured,
        pattern="*.toml",
        src_format="synthetic",
        read_table=True,
        large=False,
    )
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
