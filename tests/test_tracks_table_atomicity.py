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


def _inside_a_virtualenv(source: Path, root: Path) -> bool:
    """True when *source* is installed third-party code rather than mosaic's own.

    Two directories under ``src/mosaic/`` are where a user builds an environment
    for an external tool -- the keypoint-MoSeq runner's and the Ultralytics
    tracking runner's -- so a walk of the package tree can reach a whole
    site-packages. It finds hits that are not writers at all: ``pandas``'s own
    ``frame.py`` carries ``df.to_parquet("df.parquet.gzip", ...)`` in a docstring
    example, which reads exactly like a final path.

    Detected two ways because a virtualenv directory can be called anything: a
    ``site-packages`` component *below the package root*, or a ``pyvenv.cfg`` in a
    directory between *source* and *root*.

    Both tests are deliberately relative to *root*. Under a non-editable install
    the package root is itself ``.../site-packages/mosaic``, so an absolute
    ``"site-packages" in source.parts`` is true of every file in the walk --
    which excludes the whole of mosaic, leaves the caller asserting against an
    empty list, and turns the guard green having checked nothing. The property
    being tested is where a file sits inside the package, never where the
    package was installed.
    """
    if "site-packages" in source.relative_to(root).parts:
        return True
    for parent in source.parents:
        if parent == root:
            return False
        if (parent / "pyvenv.cfg").is_file():
            return True
    return False


def test_every_tracks_writer_goes_through_the_atomic_one() -> None:
    """No writer may address a final path directly.

    A structural check rather than a behavioural one, because the failure mode is
    a *new* call site added later: the four bridge sites and the three conversion
    paths were each written independently, which is how they came to disagree.
    """
    import mosaic

    root = Path(mosaic.__file__).parent
    offenders: list[str] = []
    scanned: set[str] = set()
    reached_the_definition_site = False
    for source in sorted(root.rglob("*.py")):
        if source.name == "writers.py":
            reached_the_definition_site = True
            continue  # the one legitimate definition site
        if _inside_a_virtualenv(source, root):
            continue
        scanned.add(source.relative_to(root).as_posix())
        for number, line in enumerate(source.read_text().splitlines(), start=1):
            if ".to_parquet(" in line and "write_parquet_atomic" not in line:
                # A temp path handed to us by atomic_write is fine.
                if "(p," in line or "(temp," in line or "(tmp," in line:
                    continue
                offenders.append(f"{source.relative_to(root)}:{number}")

    # A structural guard that can pass by scanning nothing is worse than no
    # guard at all: it reports a green invariant it never checked, and the
    # exclusion above is one over-broad predicate away from that. So the walk is
    # made to prove it happened before its finding is believed -- that it
    # reached the sanctioned writer it deliberately skips, and that it read
    # ordinary mosaic modules beside it. The sibling guards in
    # test_read_target_gate.py carry the same kind of check for the same reason.
    assert reached_the_definition_site, (
        "the walk never reached core/pipeline/writers.py, so it scanned no part "
        "of mosaic and its verdict below means nothing"
    )
    assert "core/dataset.py" in scanned, (
        "the walk did not read core/dataset.py, so it is not covering mosaic's "
        f"own tree; it scanned {len(scanned)} files"
    )

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
    from mosaic.core.pipeline.inventory.scan import run_covers

    run_root = tmp_path / "run"
    run_root.mkdir()
    target = frozenset({("g1", "s1")})
    _tracks_frame(4).to_parquet(run_root / "g1__s1.parquet", index=False)
    assert run_covers(run_root, target).is_satisfied is True

    whole = (run_root / "g1__s1.parquet").read_bytes()
    _ = (run_root / "g1__s1.parquet").write_bytes(whole[: len(whole) // 2])

    coverage = run_covers(run_root, target)
    assert coverage.is_satisfied is False
    # And it says which entry, where the predicate this replaced said only
    # "not complete" and left the caller to re-glob to find out.
    assert coverage.missing == target
