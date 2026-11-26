"""A Live Mouse Tracker database converts in memory proportional to its table.

``lmt_sqlite`` reads ``DETECTION`` in chunks of ``read_chunk_rows`` rows into
preallocated column arrays, sorts them once and returns one table, which the write
path copies a few times. Each test converts one database in a fresh interpreter,
after a database of a few frames has loaded every module and the pyarrow pool, and
reads the growth of the child's peak resident memory over the conversion.

The slow test converts the database of a 71 hour recording of 4 mice at 30 fps,
30 million detection rows, under the lock that allows one large child per machine
at a time. Its database is about 3 GB on disk.

Measured on the task machine at the default ``read_chunk_rows`` of 50 000: the
200 000-row database grows peak memory by 82.6 MB over a table of 25.2 MB, a ratio
of 3.28. The 30.7 million row database grows it by 10.41 GiB, about 2.9 times a
table of about 3.6 GiB (126 bytes a row); the written file is 1.41 GiB of encoded
parquet. A chunk of 100 000 rows measured a ratio of 5.43 on the 200 000-row
database, where the Python row tuples of one chunk outweigh the table. At 800 000
rows the ratio is 2.94 with a chunk of 100 000 or of 25 000.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from tests.helpers import (
    LmtAnimal,
    LmtFrame,
    conversion_growth,
    lmt_detection_grid,
    make_dataset,
    write_lmt_sqlite,
)

GIB: Final = 1024**3
MIB: Final = 1024**2
ANIMALS: Final = 4
WARM_UP_FRAMES: Final = 6
FIRST_TIMESTAMP: Final = 1515570902366
FRAMES_PER_SECOND: Final = 30
SMALL_FRAMES: Final = 50_000
LONG_FRAMES: Final = 71 * 3600 * FRAMES_PER_SECOND


def _write_recording(path: Path, frames: int) -> None:
    """Write a database of *frames* frames at 30 fps, one detection per mouse and frame."""
    animals = [
        LmtAnimal(
            id=animal, rfid=f"{animal:012d}", genotype="WT", name=f"mouse {animal}"
        )
        for animal in range(1, ANIMALS + 1)
    ]
    timeline = [
        LmtFrame(number, FIRST_TIMESTAMP + number * 1000 // FRAMES_PER_SECOND)
        for number in range(frames)
    ]
    _ = write_lmt_sqlite(
        path,
        animals=animals,
        frames=timeline,
        events=[],
        detections=lmt_detection_grid([animal.id for animal in animals], range(frames)),
    )


def _prepare(tmp_path: Path, frames: int) -> tuple[Dataset, Path]:
    """A dataset indexing the warm-up database, and the directory of the measured one."""
    ds = make_dataset(tmp_path / "ds")
    warm = ds.base_dir / "warm"
    warm.mkdir(parents=True)
    _write_recording(warm / "warm.sqlite", WARM_UP_FRAMES)
    _ = ds.index_tracks_raw([warm], patterns=["*.sqlite"], src_format="lmt_sqlite")
    measured = ds.base_dir / "measured"
    measured.mkdir()
    _write_recording(measured / "measured.sqlite", frames)
    return ds, measured


def test_a_200k_row_database_grows_memory_by_a_few_tables(tmp_path: Path) -> None:
    """Growth stays under a few tables' bytes plus 32 MiB.

    The bound is 4 tables on Linux and 6 on macOS, where the allocator returns less
    between steps, as for the synthetic converter.
    """
    ds, measured = _prepare(tmp_path, SMALL_FRAMES)
    measurements = conversion_growth(
        ds.manifest_path,
        measured,
        pattern="*.sqlite",
        src_format="lmt_sqlite",
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
def test_a_30m_row_database_converts_within_12_gib(tmp_path: Path) -> None:
    ds, measured = _prepare(tmp_path, LONG_FRAMES)
    measurements = conversion_growth(
        ds.manifest_path,
        measured,
        pattern="*.sqlite",
        src_format="lmt_sqlite",
        read_table=False,
        large=True,
    )
    growth = measurements["growth"]
    print(
        f"{LONG_FRAMES * ANIMALS} rows: growth {growth / GIB:.2f} GiB, "
        f"parquet {measurements['parquet_bytes'] / GIB:.2f} GiB uncompressed"
    )

    assert growth <= 12 * GIB, f"growth {growth / GIB:.2f} GiB"
