"""The tracker sweeper -- item 8.4, and M5's second gate.

The gate is *the sweeper does not delete in-flight work*, and it is asserted
twice on purpose: once as the rule (``inflight`` is not in the deletable set,
which covers every path) and once as behaviour (a live claim survives
``apply=True`` with the window set to zero, which proves the rule is wired to
something). Neither alone is enough -- the first cannot see a bug in the walk,
the second only covers the arrangement it builds.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.markers import (
    PhaseMarker,
    new_inflight,
    write_inflight,
    write_phase_marker,
)
from mosaic.core.pipeline.sweep import (
    _DELETABLE,
    SweepClass,
    deletable,
    retention_days,
)

from tests.helpers import make_dataset

_NOW = datetime.datetime(2026, 7, 29, 12, 0, tzinfo=datetime.timezone.utc)


def _entry(
    ds: Dataset,
    *,
    root_key: str = "trex",
    run_id: str = "trex.1.0-aaaa",
    sequence: str = "seq_a",
    group: str = "",
    finished_days_ago: float | None = None,
    rowed: bool = True,
    claim_expires_in: float | None = None,
) -> Path:
    """A tracker working directory in a chosen state.

    Built through the real marker writers rather than by hand: a fixture that
    writes its own JSON would keep passing after the marker schema moved.
    """
    seq_dir = ds.get_root(root_key) / run_id / make_entry_key(group, sequence)
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "out.pv").write_bytes(b"x" * 16)

    if finished_days_ago is not None:
        stamp = (_NOW - datetime.timedelta(days=finished_days_ago)).isoformat()
        from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

        for phase in TRACKING_ROOTS[root_key].phases:
            write_phase_marker(
                seq_dir,
                PhaseMarker(phase=phase, run_id=run_id, completed_at=stamp),
            )
    if claim_expires_in is not None:
        claim = new_inflight(
            execution_id="other-exec",
            host="other-host",
            pid=1,
            phase=None,
            idle_seconds=claim_expires_in,
        )
        write_inflight(seq_dir, claim)
    if rowed:
        _row(ds, root_key, run_id, sequence, seq_dir, group=group)
    return seq_dir


def _row(
    ds: Dataset,
    root_key: str,
    run_id: str,
    sequence: str,
    path: Path,
    *,
    group: str = "",
) -> None:
    from mosaic.tracking.trex.dataset_runs import (
        TRexIndexRow,
        trex_index,
        trex_index_path,
    )

    from mosaic.tracking.trex.conversion_cache import (
        CONVERT_KIND,
        ConversionIndexRow,
        conversion_index,
        conversion_index_path,
    )

    if root_key == CONVERT_KIND:
        cache = conversion_index(conversion_index_path(ds))
        cache.ensure()
        cache.append(
            [
                ConversionIndexRow(
                    run_id=run_id,
                    group=group,
                    sequence=sequence,
                    abs_path=Path(ds.relative_to_root(path)),
                    video_abs_path="",
                    params_hash="",
                )
            ]
        )
        return

    assert root_key == "trex", "only the trex row shape is built here"
    idx = trex_index(trex_index_path(ds))
    idx.ensure()
    idx.append(
        [
            TRexIndexRow(
                run_id=run_id,
                group=group,
                sequence=sequence,
                abs_path=Path(ds.relative_to_root(path)),
                video_abs_path="",
                params_hash="",
            )
        ]
    )


# --- The shared conversion cache ---------------------------------------------


def _slot(
    ds: Dataset,
    *,
    uid: str = "uid-a",
    run_id: str = "trex-convert.0.1-cccc",
    finished_days_ago: float | None = 90.0,
    rowed: bool = True,
) -> Path:
    """A published conversion slot, aged well past any window by default."""
    from mosaic.tracking.trex.conversion_cache import CONVERT_KIND

    return _entry(
        ds,
        root_key=CONVERT_KIND,
        run_id=run_id,
        sequence=uid,
        finished_days_ago=finished_days_ago,
        rowed=rowed,
    )


def _reader(ds: Dataset, slot: Path, *, finished_days_ago: float | None = 1.0) -> Path:
    """A trex working directory whose convert marker names *slot*'s conversion."""
    seq_dir = _entry(ds, finished_days_ago=finished_days_ago)
    stamp = (
        (_NOW - datetime.timedelta(days=finished_days_ago)).isoformat()
        if finished_days_ago is not None
        else ""
    )
    write_phase_marker(
        seq_dir,
        PhaseMarker(
            phase="convert",
            run_id="trex.1.0-aaaa",
            completed_at=stamp,
            recorded_output=ds.relative_to_root(slot / "conversion.pv"),
        ),
    )
    return seq_dir


def test_every_retention_class_has_a_default_window() -> None:
    """A class with no window raises inside the walk and aborts the whole sweep.

    ``retention_days`` indexes the table directly, and the failure is not local:
    the exception escapes the loop, so one unwindowed root stops every other
    root being swept. basedpyright cannot see it -- the table is a ``Mapping``,
    not a ``TypedDict``, so a literal missing a key still type-checks.
    """
    from typing import get_args

    from mosaic.core.pipeline.sweep import _DEFAULT_RETENTION_DAYS
    from mosaic.core.pipeline.tracking_roots import RetentionClass

    assert set(_DEFAULT_RETENTION_DAYS) == set(get_args(RetentionClass))


def test_the_conversion_window_is_settable_and_changes_the_verdict(
    tmp_path: Path,
) -> None:
    """A window nobody can set is a window silently ignored.

    ``Dataset.sweep_tracking`` filters the overrides it accepts, so a class
    missing from that filter is dropped in silence -- the call succeeds, the
    number has no effect, and nothing says so. Asserting the verdict *moves* is
    what catches that; asserting the call returns does not.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds, finished_days_ago=5.0)

    default = ds.sweep_tracking(apply=False, now=_NOW)
    assert [e.verdict for e in default.entries] == ["complete_young"]

    narrowed = ds.sweep_tracking(
        apply=False, retention_overrides={"conversion": 1.0}, now=_NOW
    )
    assert [e.verdict for e in narrowed.entries] == ["complete_aged"]
    assert slot.exists(), "neither call applied anything"


def test_a_pinned_conversion_survives_an_applied_sweep_with_no_window(
    tmp_path: Path,
) -> None:
    """The gate for the expensive artifact, as behaviour.

    A slot 90 days old, a zero window and ``apply=True`` is the most aggressive
    call the surface admits. It must still be there, because a tracker directory
    that survives this pass names it as its input.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds)
    _ = _reader(ds, slot)

    report = ds.sweep_tracking(
        apply=True, retention_overrides={"conversion": 0.0}, now=_NOW
    )

    assert slot.exists()
    assert (slot / "out.pv").exists()
    verdicts = {e.root_key: e.verdict for e in report.entries}
    assert verdicts["trex-convert"] == "pinned"


def test_a_slot_pinned_by_an_unrowed_reader_is_refused(tmp_path: Path) -> None:
    """The pin reads markers, never rows.

    Several producers append their index rows only after the whole batch, so
    mid-run a live tracking run has written its convert marker and no row at all.
    Keying the pin on the index would reclaim that run's conversion while it is
    reading it.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds)
    reader = _reader(ds, slot)
    _drop_rows(ds, "trex")

    report = ds.sweep_tracking(
        apply=True, retention_overrides={"conversion": 0.0}, now=_NOW
    )

    assert reader.exists(), "an unrowed tracker directory is refused too"
    assert slot.exists()
    assert {e.verdict for e in report.entries} == {"unrowed", "pinned"}


def test_an_unpinned_aged_slot_is_reclaimed(tmp_path: Path) -> None:
    """A cache nothing can reclaim grows without bound."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert not slot.exists()
    assert slot in report.removed


def test_an_unpinned_young_slot_is_held(tmp_path: Path) -> None:
    """The window still applies once the last reader is gone."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds, finished_days_ago=1.0)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert slot.exists()
    assert [e.verdict for e in report.entries] == ["complete_young"]


def test_a_slot_whose_last_reader_goes_this_pass_goes_with_it(
    tmp_path: Path,
) -> None:
    """One pass, not two: the cascade must not need a second run to catch up."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds)
    reader = _reader(ds, slot, finished_days_ago=90.0)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert not reader.exists()
    assert not slot.exists()
    assert set(report.removed) == {reader, slot}


def test_sweeping_only_the_cache_still_reads_the_tracker_root_for_pins(
    tmp_path: Path,
) -> None:
    """``--root trex-convert`` must not reclaim conversions that are in use.

    The pin scan runs over every tracker root regardless of the narrowing, or
    restricting the sweep to the cache would delete all of it.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds)
    _ = _reader(ds, slot)

    report = ds.sweep_tracking(
        apply=True,
        roots=["trex-convert"],
        retention_overrides={"conversion": 0.0},
        now=_NOW,
    )

    assert slot.exists()
    assert [e.verdict for e in report.entries] == ["pinned"]


def test_a_slot_with_no_index_row_is_refused(tmp_path: Path) -> None:
    """Forgetting the index registration must cost disk, never data."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds, rowed=False)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert slot.exists()
    assert [e.verdict for e in report.entries] == ["unrowed"]


def test_dropping_a_slot_row_matches_its_directory_name(tmp_path: Path) -> None:
    """The row goes with the directory, or reindex has to repair every sweep."""
    from mosaic.tracking.trex.conversion_cache import (
        conversion_index_path,
    )

    ds = make_dataset(tmp_path / "ds", name="sweep")
    _ = _slot(ds)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert report.rows_dropped == 1
    assert len(pd.read_csv(conversion_index_path(ds))) == 0


def test_a_reclaimed_slot_takes_its_staging_tree_with_it(tmp_path: Path) -> None:
    """The sweep removes the slot whole, debris included.

    A *surviving* slot's stale staging is a different problem, and it is the
    conversion path that clears it rather than this one -- see
    ``tests/test_trex_conversion_cache.py``. What is pinned here is that the
    declared clear globs name it, so a re-conversion removes it, and that an
    aged unpinned slot does not leave a partial `.pv` behind.
    """
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
    from mosaic.tracking.trex.conversion_cache import CONVERT_KIND

    assert ".incoming-*" in TRACKING_ROOTS[CONVERT_KIND].clear_globs("convert")

    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds)
    staging = slot / ".incoming-dead"
    staging.mkdir()
    (staging / "conversion.pv").write_bytes(b"partial")

    _ = ds.sweep_tracking(apply=True, now=_NOW)
    assert not slot.exists()
    assert not staging.exists()


def test_a_pinned_conversion_survives_a_stale_claim(tmp_path: Path) -> None:
    """A claim is evidence about an attempt; a pin is evidence about the artifact.

    A conversion outlives the run that made it, so a claim that run left behind
    when it died says nothing about the runs reading the conversion now. Since
    ``expired_claim`` is deletable and is reached before completeness, consulting
    it before the pin would reclaim a 28 GB `.pv` that surviving tracker
    directories still name as their input.

    A claim's expiry has a floor of half an hour, so -- as elsewhere in this
    file -- the claim is aged out by sweeping from far in the future rather than
    by writing a short one. The reader is stamped relative to that same instant
    so that it survives the pass and can do the pinning.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    later = _NOW + datetime.timedelta(days=365)

    slot = _slot(ds, finished_days_ago=1.0)
    write_inflight(
        slot,
        new_inflight(
            execution_id="dead-exec",
            host="dead-host",
            pid=1,
            phase="convert",
            idle_seconds=1.0,
        ),
    )
    # Finished the day before the sweep, so it is inside its own window.
    _ = _reader(ds, slot, finished_days_ago=-364.0)

    report = ds.sweep_tracking(apply=True, now=later)

    verdicts = {e.root_key: e.verdict for e in report.entries}
    assert verdicts["trex"] == "complete_young", "the reader must survive to pin"
    assert verdicts["trex-convert"] == "pinned"
    assert slot.exists(), "a pinned conversion is never deleted over a stale claim"


def test_an_unpinned_slot_with_a_stale_claim_is_still_reclaimed(
    tmp_path: Path,
) -> None:
    """The pin is the only thing that outranks a claim, not the root itself."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    slot = _slot(ds, finished_days_ago=1.0)
    write_inflight(
        slot,
        new_inflight(
            execution_id="dead-exec",
            host="dead-host",
            pid=1,
            phase="convert",
            idle_seconds=1.0,
        ),
    )

    report = ds.sweep_tracking(apply=True, now=_NOW + datetime.timedelta(days=365))

    assert not slot.exists()
    assert [e.verdict for e in report.entries] == ["expired_claim"]


def _drop_rows(ds: Dataset, root_key: str) -> None:
    """Empty one root's index, leaving the file and its header in place."""
    path = ds.get_root(root_key) / "index.csv"
    frame = pd.read_csv(path)
    frame.iloc[0:0].to_csv(path, index=False)


# --- The gate ----------------------------------------------------------------


def test_in_flight_is_not_a_deletable_class() -> None:
    """The rule, stated once and covering every path through the walk."""
    assert "inflight" not in _DELETABLE
    assert not deletable("inflight")


def test_a_live_claim_survives_an_applied_sweep_with_no_window(tmp_path: Path) -> None:
    """The gate as behaviour: the rule is wired to something.

    ``min`` window and ``apply=True`` is the most aggressive call the surface
    admits, and a directory finished long ago but *claimed* must still be there
    afterwards. This is the overnight-batch case -- a run whose earlier entries
    are finished while a later one is still going.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    held = _entry(ds, finished_days_ago=90.0, claim_expires_in=3600.0)

    report = ds.sweep_tracking(
        apply=True, retention_overrides={"tracker": 0.0}, now=_NOW
    )

    assert held.exists(), "the sweeper deleted work another execution holds"
    assert (held / "out.pv").exists()
    assert [e.verdict for e in report.entries] == ["inflight"]
    assert report.removed == []


def test_an_unrowed_directory_is_refused(tmp_path: Path) -> None:
    """Mid-batch, most finished directories are unrowed and all are live work.

    Several producers append their index rows only after the whole batch, so
    "no row names it" is the *normal* state of a healthy run in progress --
    the opposite of what it sounds like.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    unrowed = _entry(ds, finished_days_ago=90.0, rowed=False)

    report = ds.sweep_tracking(
        apply=True, retention_overrides={"tracker": 0.0}, now=_NOW
    )

    assert unrowed.exists()
    assert [e.verdict for e in report.entries] == ["unrowed"]


def test_a_directory_with_no_marker_is_foreign_and_refused(tmp_path: Path) -> None:
    """A root pointed somewhere that is not a tracker root reclaims nothing."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    stranger = ds.get_root("trex") / "not-a-run" / "not-an-entry"
    stranger.mkdir(parents=True)
    (stranger / "somebody.txt").write_text("mine")

    report = ds.sweep_tracking(
        apply=True, retention_overrides={"tracker": 0.0}, now=_NOW
    )

    assert (stranger / "somebody.txt").exists()
    assert [e.verdict for e in report.entries] == ["foreign"]


# --- What it does reclaim ----------------------------------------------------


def test_a_finished_aged_entry_goes_with_its_row(tmp_path: Path) -> None:
    """The ordinary reclaim: files removed, row dropped, bytes reported."""
    import pandas as pd

    from mosaic.tracking.trex.dataset_runs import trex_index_path

    ds = make_dataset(tmp_path / "ds", name="sweep")
    old = _entry(ds, finished_days_ago=30.0)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert not old.exists()
    assert report.removed == [old]
    assert report.rows_dropped == 1
    assert report.bytes_reclaimed > 0
    assert len(pd.read_csv(trex_index_path(ds))) == 0


def test_a_finished_entry_inside_its_window_is_held_and_said_so(
    tmp_path: Path,
) -> None:
    """ "Would delete 0" must not read as "there was nothing here"."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    recent = _entry(ds, finished_days_ago=1.0)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert recent.exists()
    assert [e.verdict for e in report.entries] == ["complete_young"]
    assert report.held_for_age == 1


def test_a_half_finished_trex_run_is_not_complete(tmp_path: Path) -> None:
    """Convert done, track killed: one marker, and it is not a finished run.

    Reading whichever markers happen to be present would take the convert stamp
    as the answer and reclaim the directory at its age -- throwing away a
    conversion someone is still using. The registry declares both phases, so
    the question asked is "are all of them there".
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    half = ds.get_root("trex") / "trex.1.0-bbbb" / "seq_a"
    half.mkdir(parents=True)
    (half / "out.pv").write_bytes(b"x" * 16)
    write_phase_marker(
        half,
        PhaseMarker(
            phase="convert",
            run_id="trex.1.0-bbbb",
            completed_at=(_NOW - datetime.timedelta(days=90)).isoformat(),
        ),
    )
    _row(ds, "trex", "trex.1.0-bbbb", "seq_a", half)

    report = ds.sweep_tracking(apply=False, now=_NOW)

    assert [e.verdict for e in report.entries] == ["incomplete"]


def test_an_expired_claim_is_reclaimable(tmp_path: Path) -> None:
    """The only cross-host authority is the expiry the claim carries itself."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    _ = _entry(ds, finished_days_ago=None, claim_expires_in=1.0)

    report = ds.sweep_tracking(apply=False, now=_NOW + datetime.timedelta(days=365))

    assert [e.verdict for e in report.entries] == ["expired_claim"]


# --- Gates, retention, determinism -------------------------------------------


def test_a_legacy_layout_declines_rather_than_deleting(tmp_path: Path) -> None:
    """A tracker root still inside tracks_raw holds user uploads beneath it."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    ds.roots["trex"] = "tracks_raw/trex"

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert not report.considered
    assert report.declined == "legacy-layout"
    assert report.removed == []


def test_a_root_outside_the_dataset_declines(tmp_path: Path) -> None:
    """Item 9.1's rule, enforced where being wrong costs files."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    ds.roots["trex"] = str(outside)

    report = ds.sweep_tracking(apply=True, now=_NOW)

    assert not report.considered
    assert report.declined == "root-outside-dataset"


def test_inference_output_is_kept_for_less_time_than_tracker_output() -> None:
    """Item 8.4's table is data on the registry, not a branch per tool."""
    assert retention_days("infer-pose") < retention_days("trex")


def test_a_dry_run_removes_nothing(tmp_path: Path) -> None:
    """Dry-run is the default and must not be one in name only."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    old = _entry(ds, finished_days_ago=30.0)

    report = ds.sweep_tracking(apply=False, now=_NOW)

    assert old.exists()
    assert report.removed == []
    assert [e.verdict for e in report.entries] == ["complete_aged"]


def test_two_runs_agree(tmp_path: Path) -> None:
    """Filesystem order is not stable, so the walk sorts and the report must too."""
    ds = make_dataset(tmp_path / "ds", name="sweep")
    for sequence in ("seq_c", "seq_a", "seq_b"):
        _ = _entry(ds, sequence=sequence, finished_days_ago=30.0)

    first = ds.sweep_tracking(apply=False, now=_NOW).payload()
    second = ds.sweep_tracking(apply=False, now=_NOW).payload()

    assert first == second


@pytest.mark.parametrize("verdict", ["inflight", "unrowed", "foreign"])
def test_every_refused_class_is_reported_to_the_operator(verdict: SweepClass) -> None:
    """A class nothing prints is a class nobody repairs."""
    from mosaic.core.pipeline.sweep import REFUSED_NOTES

    assert verdict in REFUSED_NOTES
    assert not deletable(verdict)


# --- 8.6's signal: promotion outranks age ------------------------------------


def test_a_promoted_run_is_reclaimable_before_its_window(tmp_path: Path) -> None:
    """Item 8.4's "promotion is the primary signal, age is the fallback".

    Once a corrected track set is in ``tracks_raw``, the tracker output it was
    corrected from has served its purpose -- so its retention window stops being
    the question. Without this the class existed with no producer and the design's
    ordering was aspirational.
    """
    ds = make_dataset(tmp_path / "ds", name="sweep")
    recent = _entry(ds, finished_days_ago=1.0)
    ds.set_display_name("", "seq_a", "")

    import numpy as np

    from mosaic.core.pipeline.promotion import promote_correction

    correction = tmp_path / "seq_a_fish0.npz"
    np.savez(correction, X=np.array([1.0]), Y=np.array([2.0]))
    _ = promote_correction(
        ds,
        "",
        "seq_a",
        correction,
        src_format="trex_npz",
        derived_from="trex.1.0-aaaa",
        apply=True,
        force=True,
    )

    report = ds.sweep_tracking(apply=False, now=_NOW)

    assert [e.verdict for e in report.entries] == ["promoted"]
    assert recent.exists(), "a dry run must not delete"


def test_an_applied_sweep_drops_the_row_of_a_grouped_entry(tmp_path: Path) -> None:
    """The row goes with the directory, whatever the entry's group.

    A working directory is named by the composite entry *key*
    (``make_entry_key(group, sequence)``), while the index matches a
    ``(group, sequence)`` pair. Passing the key as a bare sequence matched no
    row whenever the group was non-empty, so the sweep deleted the directory and
    left the row naming it -- the state "rows before files" exists to prevent,
    reached without any interruption.

    It agreed for an empty group, where the key *is* the sequence. That is every
    dataset the control plane creates, and it was every dataset in this file,
    which is why a green suite said nothing.
    """
    from mosaic.tracking.trex.dataset_runs import trex_index, trex_index_path

    ds = make_dataset(tmp_path / "ds", name="sweep")
    aged = _entry(ds, group="hex", sequence="hex_3", finished_days_ago=90.0)

    report = ds.sweep_tracking(
        apply=True, retention_overrides={"tracker": 0.0}, now=_NOW
    )

    assert not aged.exists(), "the aged directory was not reclaimed"
    assert report.rows_dropped == 1, "the directory went but its row stayed"
    assert len(trex_index(trex_index_path(ds)).read()) == 0
