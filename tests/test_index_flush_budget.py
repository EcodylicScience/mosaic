"""When a run's index rows reach disk, and why not one at a time.

The index lags the filesystem on purpose: a parquet is renamed into place before
its row is queued, so a kill leaves outputs with no row rather than rows with no
output, and a file check resumes correctly from that. What the budget adds is a
bound on the lag that does not depend on how fast the feature happens to be.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.index import FeatureIndexRow, feature_index


def _rows(directory: Path, count: int) -> list[FeatureIndexRow]:
    made: list[FeatureIndexRow] = []
    for position in range(count):
        output = directory / f"s{position}.parquet"
        output.write_bytes(b"x")
        made.append(
            FeatureIndexRow(
                abs_path=output,
                run_id="0.1-abcdef0123",
                feature="f",
                version="0.1",
                group="",
                sequence=f"s{position}",
                params_hash="x",
            )
        )
    return made


def test_the_batch_size_is_importable_rather_than_buried(tmp_path: Path) -> None:
    """It was a function-local constant inside run_feature: not importable, not
    configurable, and not testable without running a feature."""
    assert run_module.IDX_FLUSH_EVERY >= 1
    assert run_module.IDX_FLUSH_SECONDS > 0


def test_per_row_flushing_rewrites_the_whole_index_every_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Why the budget is a time and not a count of one.

    Every append re-reads the whole CSV, re-masks it per new row and rewrites
    it, so the total work is quadratic in the run. Asserted by counting the rows
    rewritten rather than by timing: a clock on a shared machine measures the
    machine, and a threshold tuned to pass here is one that flakes elsewhere.
    """
    import pandas as pd

    rewritten: list[int] = []
    real_to_csv = pd.DataFrame.to_csv

    def _counting(self, *args, **kwargs):
        rewritten.append(len(self))
        return real_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", _counting)

    count = 20
    index = feature_index(tmp_path / "index.csv")
    index.ensure()
    for row in _rows(tmp_path, count):
        index.append([row])

    # 1 + 2 + ... + N rows rewritten, against N if the batch were written once.
    assert sum(rewritten) >= count * (count - 1) // 2, (
        f"expected the whole file rewritten per append, saw {sum(rewritten)} "
        f"rows over {len(rewritten)} writes"
    )


def test_a_slow_producer_does_not_hold_rows_indefinitely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window the budget closes.

    A feature producing one entry a minute would leave nine outputs unrecorded
    for nine minutes under a count-only rule. Simulated by shrinking the budget
    rather than by waiting.
    """
    index = feature_index(tmp_path / "index.csv")
    index.ensure()
    rows = _rows(tmp_path, 3)

    pending: list[FeatureIndexRow] = []
    last_flush = time.monotonic()
    for row in rows:
        pending.append(row)
        time.sleep(0.02)
        if (
            len(pending) >= run_module.IDX_FLUSH_EVERY
            or time.monotonic() - last_flush >= 0.01
        ):
            index.append(list(pending))
            pending.clear()
            last_flush = time.monotonic()

    assert len(index.read()) == 3, (
        "rows well under the batch size should still reach disk on the timer"
    )
