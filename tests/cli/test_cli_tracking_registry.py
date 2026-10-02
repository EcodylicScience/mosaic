"""The CLI commands that read a tracker index must first fill the registry.

A tracker root's index is opened through ``register_reconcilable_index``, which
each tracker fills as a side effect of being imported. ``core`` does not import
``tracking``, so a command that only reaches ``Dataset`` finds the registry
empty -- and both consumers fail *quietly* rather than loudly:

* ``reindex`` skips every ``_tracking`` root, so a working directory deleted by
  hand keeps its row forever, which is the exact case the method exists for;
* ``sweep-tracking`` reads every directory as ``unrowed``, and unrowed is
  refused, so it reclaims nothing while reporting a well-formed result that
  blames the index.

**Driven through a subprocess, and that is the point.** The defect is a question
about what a fresh interpreter has imported, and by the time a ``CliRunner``
test runs, the session has already imported ``mosaic.tracking`` for other
reasons -- so an in-process run passes against the broken code. Only a new
interpreter can observe it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.markers import PhaseMarker, write_phase_marker
from mosaic.runlog import now_iso
from mosaic.tracking.trex.dataset_runs import (
    TRexIndexRow,
    trex_index,
    trex_index_path,
    trex_run_root,
)

RUN_ID = "trex.0.1-aaaaaaaaaa"
ENTRY = "g__vid1"


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the mosaic CLI in a fresh interpreter."""
    return subprocess.run(
        [sys.executable, "-c", "from mosaic.cli import app; app()", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def dataset_with_a_tracker_run(tmp_path: Path) -> tuple[Path, Dataset, Path]:
    """A dataset carrying one rowed trex working directory."""
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    ds = Dataset(manifest_path=manifest).load()

    work_dir = trex_run_root(ds, RUN_ID) / ENTRY
    work_dir.mkdir(parents=True)
    (work_dir / "vid1.pv").write_bytes(b"pv")
    tmp_video = tmp_path / "vid1.mp4"
    tmp_video.write_bytes(b"video")

    # Both declared phases, so the sweep gets past "carries no mosaic marker"
    # and actually reaches the question this file is about. Without them the
    # directory is refused as ``foreign`` and the rowed check never runs.
    for phase in ("convert", "track"):
        write_phase_marker(
            work_dir,
            PhaseMarker(
                phase=phase,
                run_id=RUN_ID,
                completed_at=now_iso(),
                recorded_output=ds.relative_to_root(work_dir / "vid1.pv"),
            ),
        )

    index = trex_index(trex_index_path(ds))
    index.ensure()
    index.append(
        [
            TRexIndexRow(
                run_id=RUN_ID,
                group="g",
                sequence="vid1",
                abs_path=Path(ds.relative_to_root(work_dir)),
                video_abs_path=ds.relative_to_root(tmp_video),
                params_hash="aaaaaaaaaa",
                pv_path=ds.relative_to_root(work_dir / "vid1.pv"),
            )
        ]
    )
    return manifest, ds, work_dir


def test_reindex_drops_the_row_of_a_deleted_tracker_directory(
    dataset_with_a_tracker_run: tuple[Path, Dataset, Path],
) -> None:
    """The case ``reindex``'s own docstring promises to cover."""
    manifest, ds, work_dir = dataset_with_a_tracker_run
    import shutil

    shutil.rmtree(work_dir)

    result = _cli("reindex", "-m", str(manifest), "--apply", "--json")

    assert result.returncode == 0, result.stderr
    remaining = trex_index(trex_index_path(ds)).read()
    assert len(remaining) == 0, (
        "the row of a hand-deleted tracker directory survived reindex; the "
        "reconcilable-index registry was empty, so the root was skipped"
    )


def test_sweep_does_not_call_a_rowed_directory_unrowed(
    dataset_with_a_tracker_run: tuple[Path, Dataset, Path],
) -> None:
    """A directory the index names must never read as ``unrowed``.

    Asserted on the verdict rather than on a deletion, so the test says what is
    wrong rather than only that nothing was reclaimed: a fresh run is inside its
    retention window and is *correctly* held, and "held for age" and "not in the
    index" are opposite diagnoses that both reclaim nothing.
    """
    import json

    manifest, _ds, work_dir = dataset_with_a_tracker_run

    result = _cli("sweep-tracking", "-m", str(manifest), "--dry-run", "--json")

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert str(work_dir) not in report["unrowed"], (
        "a directory named by the trex index read as unrowed, so the sweep "
        "refused it; the reconcilable-index registry was empty"
    )
