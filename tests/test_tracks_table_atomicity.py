"""A torn tracks table must not be published, nor adopted as an empty one.

Two halves of one defect.

**The write.** Feature outputs go through ``atomic_write`` -- temp file, then
rename -- so an interrupted write never leaves a torn file at the addressed path.
The tracker bridge, all three conversion paths and both inference writers instead
called ``df.to_parquet(final_path)`` directly, so a kill mid-write left a
half-written parquet exactly where a whole one belongs.

**The adoption.** ``existing_counts`` then read that path back and caught
``(OSError, ValueError, KeyError)`` -- which includes pyarrow's ``ArrowInvalid`` on
a truncated file -- returning ``BridgeCounts(0, 0)``. That is indistinguishable
from the *legitimate* zero: a video in which the tracker found no individuals,
which the marker rules explicitly declare reusable. So a torn table was reused as
a valid empty result, and its zero was written into an index row.

The two must be fixed together and tested together. Making the write atomic
without teaching the reader to distinguish unreadable from empty would leave every
table torn by a pre-fix crash, or by an external tool, silently adopted. Teaching
the reader without making the write atomic would turn every interrupted run into a
recompute of work that had in fact finished.

The counter-test is load-bearing: a genuinely empty table must stay reusable, or
the fix trades a silent wrong answer for a tracker that re-runs forever on any
video with nothing in it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.tracking.common.bridge import (
    BridgeCounts,
    readable_tracks_table,
)


def _tracks_frame(n_rows: int = 4) -> pd.DataFrame:
    """A minimal standardized frame: enough columns for the schema to accept it."""
    return pd.DataFrame(
        {
            "frame": range(n_rows),
            "time": [f / 30.0 for f in range(n_rows)],
            "id": [0, 1] * (n_rows // 2) if n_rows else [],
            "group": [""] * n_rows,
            "sequence": ["s1"] * n_rows,
            "poseX0": [1.0] * n_rows,
            "poseY0": [2.0] * n_rows,
        }
    )


def test_a_truncated_table_is_not_reused_as_an_empty_one(tmp_path: Path) -> None:
    path = tmp_path / "g__s1.parquet"
    _tracks_frame(4).to_parquet(path, index=False)
    whole = path.read_bytes()
    assert readable_tracks_table(path) == BridgeCounts(n_rows=4, n_ids=2)

    # Truncate to half its bytes: the footer is gone, so pyarrow cannot read it.
    _ = path.write_bytes(whole[: len(whole) // 2])

    # The distinction the defect erased: unreadable is not zero rows.
    assert readable_tracks_table(path) is None


def test_a_genuinely_empty_table_is_still_reusable(tmp_path: Path) -> None:
    """A tracker that found no individuals produced a real, reusable result."""
    path = tmp_path / "g__empty.parquet"
    _tracks_frame(0).to_parquet(path, index=False)

    assert readable_tracks_table(path) == BridgeCounts(n_rows=0, n_ids=0)


def test_an_absent_table_is_not_readable(tmp_path: Path) -> None:
    assert readable_tracks_table(tmp_path / "nothing-here.parquet") is None


def test_a_failed_write_leaves_no_table_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted publish leaves the addressed path absent, not torn.

    Modelled on ``tests/test_pipeline_writers.py``, which pins the same property
    for feature outputs -- the tracker side had no equivalent.
    """
    from mosaic.core.pipeline import writers

    path = tmp_path / "out" / "g__s1.parquet"

    def boom(self: object, target: object, **kw: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)

    with pytest.raises(OSError, match="disk full"):
        writers.write_parquet_atomic(_tracks_frame(4), path)

    assert not path.exists()
    assert list(path.parent.glob("*")) == []


def test_a_failed_rewrite_preserves_the_existing_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed overwrite must not destroy the table that was already there."""
    from mosaic.core.pipeline import writers

    path = tmp_path / "g__s1.parquet"
    writers.write_parquet_atomic(_tracks_frame(4), path)
    before = path.read_bytes()

    def boom(self: object, target: object, **kw: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(OSError, match="disk full"):
        writers.write_parquet_atomic(_tracks_frame(2), path)

    assert path.read_bytes() == before
    assert list(path.parent.glob("*.tmp")) == []


def test_every_tracks_writer_goes_through_the_atomic_one() -> None:
    """No writer may address a final path directly.

    A structural check rather than a behavioural one, because the failure mode is
    a *new* call site added later: the four bridge sites and the three conversion
    paths were each written independently, which is how they came to disagree.
    """
    import mosaic

    root = Path(mosaic.__file__).parent
    offenders: list[str] = []
    for source in sorted(root.rglob("*.py")):
        if source.name == "writers.py":
            continue  # the one legitimate definition site
        for number, line in enumerate(source.read_text().splitlines(), start=1):
            if ".to_parquet(" in line and "write_parquet_atomic" not in line:
                # A temp path handed to us by atomic_write is fine.
                if "(p," in line or "(temp," in line or "(tmp," in line:
                    continue
                offenders.append(f"{source.relative_to(root)}:{number}")
    assert offenders == [], (
        "these write a parquet without going through write_parquet_atomic: "
        + ", ".join(offenders)
    )


def test_a_truncated_output_does_not_satisfy_coverage(tmp_path: Path) -> None:
    """A step is not complete because a torn file is sitting where output belongs.

    The coverage gate tested presence, and a half-written parquet is present. A
    level-triggered runner would chain past it; a human only notices when the
    next step fails on unreadable input.
    """
    from mosaic.core.pipeline.pipeline import _run_is_complete

    run_root = tmp_path / "run"
    run_root.mkdir()
    target = {("g1", "s1")}
    _tracks_frame(4).to_parquet(run_root / "g1__s1.parquet", index=False)
    assert _run_is_complete(run_root, target) is True

    whole = (run_root / "g1__s1.parquet").read_bytes()
    _ = (run_root / "g1__s1.parquet").write_bytes(whole[: len(whole) // 2])

    assert _run_is_complete(run_root, target) is False
