"""The claim a feature fit holds on its run root.

Two executions of one fitting identity share one run root. The fit takes the root
through ``held_run_root`` after its state reads as missing, loads the state again
under the claim, and releases the claim before apply. A second execution arriving
mid-fit is refused with ``run_root_held``; one arriving after loads the saved
state. Apply runs unclaimed, so scope-free runs over different sequences still
apply at once.

Every concurrent case synchronizes on events with timeouts, and every thread is
joined. No ``claim-*`` refresh thread outlives the run that started it.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager
from pathlib import Path

import pytest

import mosaic.core.pipeline.entry_claim as entry_claim
import mosaic.core.pipeline.run as run_module
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.entry_claim import RunRootHeld, kept_claimed
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.markers import (
    InflightMarker,
    inflight_marker_path,
    new_inflight,
    read_inflight,
    write_inflight,
)
from mosaic.core.pipeline.run import resolve_feature_identity, run_feature
from mosaic.core.pipeline.storage_names import derive_storage_name
from mosaic.core.pipeline.types import DependencyLookup, Feature, InputStream, Result
from mosaic.core.scope import Scope
from mosaic.runlog import JsonlRunLog, parse_instant, run_log_path
from tests.helpers import (
    FitControl,
    FittingFeature,
    ScopedFittingFeature,
    add_tracks_variant,
    make_dataset,
    reconvert_entry,
    result_run_root,
)

_WAIT_SECONDS = 30.0

_PEER = "SOMEONE-ELSE"
"""The execution whose claim a test plants on the run root."""


@pytest.fixture(autouse=True)
def no_refresh_thread_outlives_its_run() -> Iterator[None]:
    yield
    leftover = [t.name for t in threading.enumerate() if t.name.startswith("claim-")]
    assert leftover == []


class _Run(threading.Thread):
    """One ``run_feature`` call on a thread of its own."""

    def __init__(
        self, ds: Dataset, feature: Feature, scope: Scope | None = None
    ) -> None:
        super().__init__(name="fit-claim-test-run")
        self._ds = ds
        self._feature = feature
        self._scope = scope
        self.result: Result | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = run_feature(self._ds, self._feature, scope=self._scope)
        except BaseException as exc:
            self.error = exc

    def finish(self) -> Result:
        self.join(_WAIT_SECONDS)
        assert not self.is_alive(), "the run never finished"
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _run_root(ds: Dataset, control: FitControl) -> Path:
    """The run root every :class:`FittingFeature` of *control* shares."""
    feature = FittingFeature(control)
    run_id, _resolved = resolve_feature_identity(
        ds, feature, scope=Scope(entries=[("", "seq_a"), ("", "seq_b")])
    )
    storage = derive_storage_name(feature.name, feature.inputs.storage_suffix())
    root = feature_run_root(ds, storage, run_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _claimed(root: Path) -> bool:
    return inflight_marker_path(root).exists()


def _peer_claim(*, expires_at: str | None = None) -> InflightMarker:
    marker = new_inflight(
        execution_id=_PEER,
        host=socket.gethostname(),
        pid=os.getpid(),
        phase=None,
        idle_seconds=3600.0,
    )
    if expires_at is None:
        return marker
    return marker.model_copy(update={"expires_at": expires_at})


def test_a_second_execution_mid_fit_is_refused_and_the_fit_runs_once(
    scenario_dataset: Dataset,
) -> None:
    control = FitControl(finish_fit=threading.Event())
    first = _Run(scenario_dataset, FittingFeature(control))
    first.start()
    try:
        assert control.fitting.wait(_WAIT_SECONDS)
        with pytest.raises(RunRootHeld) as refused:
            _ = run_feature(scenario_dataset, FittingFeature(control))
    finally:
        assert control.finish_fit is not None
        control.finish_fit.set()
        result = first.finish()

    assert refused.value.reason == "run_root_held"
    assert control.fits == 1
    assert result.failed_entries == ()
    assert not _claimed(_run_root(scenario_dataset, control))


def test_a_run_after_the_fit_released_loads_the_saved_state(
    scenario_dataset: Dataset,
) -> None:
    control = FitControl()

    _ = run_feature(scenario_dataset, FittingFeature(control))
    second = run_feature(scenario_dataset, FittingFeature(control))

    assert control.fits == 1
    assert second.failed_entries == ()
    assert not _claimed(_run_root(scenario_dataset, control))


def test_a_state_a_peer_saved_while_this_run_took_the_claim_is_loaded_not_refitted(
    scenario_dataset: Dataset,
) -> None:
    control = FitControl(peer_saves_first=True)

    result = run_feature(scenario_dataset, FittingFeature(control))

    assert control.fits == 0
    assert control.loads == 2, "the state is loaded again under the claim"
    assert result.failed_entries == ()


def test_a_ready_state_takes_no_claim(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = FitControl()
    _ = run_feature(scenario_dataset, FittingFeature(control))

    def held_run_root(
        *_args: object, **_kwargs: object
    ) -> AbstractContextManager[None]:
        raise AssertionError("a run whose state is ready claimed its run root")

    monkeypatch.setattr(run_module, "held_run_root", held_run_root)
    result = run_feature(scenario_dataset, FittingFeature(control))

    assert result.failed_entries == ()
    assert control.fits == 1


class _RootBinding(FittingFeature):
    """A fitting feature that records the engine's calls in order."""

    def __init__(self, control: FitControl) -> None:
        super().__init__(control)
        self.events: list[str] = []
        self.bound: list[Path] = []

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        loaded = super().load_state(run_root, artifact_paths, dependency_lookups)
        self.events.append(f"load_state {loaded}")
        return loaded

    def bind_run_root(self, run_root: Path) -> None:
        self.events.append("bind_run_root")
        self.bound.append(run_root)

    def fit(self, inputs: InputStream) -> None:
        self.events.append("fit")
        super().fit(inputs)


def test_a_fit_is_handed_its_run_root_once_after_the_load_misses(
    scenario_dataset: Dataset,
) -> None:
    control = FitControl()
    feature = _RootBinding(control)

    result = run_feature(scenario_dataset, feature)

    assert result.failed_entries == ()
    assert feature.events[-3:] == ["load_state False", "bind_run_root", "fit"]
    assert feature.events.count("bind_run_root") == 1
    assert feature.bound == [_run_root(scenario_dataset, control)]


def test_a_load_that_finds_a_state_is_not_followed_by_a_run_root(
    scenario_dataset: Dataset,
) -> None:
    control = FitControl()
    _ = run_feature(scenario_dataset, FittingFeature(control))
    feature = _RootBinding(control)

    _ = run_feature(scenario_dataset, feature, overwrite=True)

    assert feature.events == ["load_state True"]


def test_a_state_a_peer_saved_under_the_claim_is_not_followed_by_a_run_root(
    scenario_dataset: Dataset,
) -> None:
    feature = _RootBinding(FitControl(peer_saves_first=True))

    _ = run_feature(scenario_dataset, feature)

    assert feature.events == ["load_state False", "load_state True"]


def test_apply_is_unclaimed_so_runs_over_different_sequences_both_finish(
    scenario_dataset: Dataset,
) -> None:
    fitter = FitControl(finish_apply=threading.Event())
    applier = FitControl()
    first = _Run(scenario_dataset, FittingFeature(fitter), Scope(sequences=["seq_a"]))
    first.start()
    try:
        assert fitter.applying.wait(_WAIT_SECONDS)
        root = _run_root(scenario_dataset, fitter)
        assert not _claimed(root), "the claim is held while apply runs"
        second = run_feature(
            scenario_dataset, FittingFeature(applier), scope=Scope(sequences=["seq_b"])
        )
    finally:
        assert fitter.finish_apply is not None
        fitter.finish_apply.set()
        result = first.finish()

    assert (fitter.fits, applier.fits) == (1, 0)
    assert result.failed_entries == ()
    assert second.failed_entries == ()
    assert not _claimed(_run_root(scenario_dataset, fitter))


@pytest.mark.parametrize(
    "error", [RuntimeError("the fit failed"), Cancelled()], ids=["failed", "cancelled"]
)
def test_a_fit_that_raises_releases_its_claim(
    scenario_dataset: Dataset, error: Exception
) -> None:
    control = FitControl(fit_error=error)

    with pytest.raises(type(error)):
        _ = run_feature(scenario_dataset, FittingFeature(control))

    assert control.fits == 1
    assert not _claimed(_run_root(scenario_dataset, control))


def _end_peer_run_log(ds: Dataset) -> None:
    log = JsonlRunLog(run_log_path(ds.base_dir, _PEER), _PEER)
    log.started(kind="feature", target="prov-fitting__from__tracks")
    log.finished()
    log.close()


@pytest.mark.parametrize("held", ["expired", "orphaned"])
def test_an_abandoned_peer_claim_is_cleared_and_the_fit_runs(
    scenario_dataset: Dataset, held: str
) -> None:
    control = FitControl()
    root = _run_root(scenario_dataset, control)
    if held == "expired":
        write_inflight(root, _peer_claim(expires_at="2000-01-01T00:00:00+00:00"))
    else:
        write_inflight(root, _peer_claim())
        _end_peer_run_log(scenario_dataset)

    result = run_feature(scenario_dataset, FittingFeature(control))

    assert control.fits == 1
    assert result.failed_entries == ()
    assert not _claimed(root)


def test_a_live_peer_claim_refuses_the_fit_and_stays(
    scenario_dataset: Dataset,
) -> None:
    control = FitControl()
    root = _run_root(scenario_dataset, control)
    write_inflight(root, _peer_claim())

    with pytest.raises(RunRootHeld) as refused:
        _ = run_feature(scenario_dataset, FittingFeature(control))

    assert refused.value.held is not None
    assert refused.value.held.execution_id == _PEER
    assert control.fits == 0
    held = read_inflight(root)
    assert held is not None
    assert held.execution_id == _PEER


def test_a_timer_refreshes_the_claim_while_a_fit_runs(
    scenario_dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_claim, "INFLIGHT_REFRESH_SECONDS", 0.05)
    refreshed = threading.Event()
    stamps: list[tuple[str, str]] = []
    real_refresh = entry_claim.refresh_inflight

    def refresh_inflight(
        work_dir: Path, marker: InflightMarker, idle_seconds: float
    ) -> InflightMarker | None:
        before = read_inflight(work_dir)
        renewed = real_refresh(work_dir, marker, idle_seconds)
        if before is not None and renewed is not None and not refreshed.is_set():
            stamps.append((before.expires_at, renewed.expires_at))
            refreshed.set()
        return renewed

    monkeypatch.setattr(entry_claim, "refresh_inflight", refresh_inflight)
    control = FitControl(finish_fit=threading.Event())
    run = _Run(scenario_dataset, FittingFeature(control))
    run.start()
    try:
        assert control.fitting.wait(_WAIT_SECONDS)
        assert refreshed.wait(_WAIT_SECONDS), "the claim was never refreshed"
    finally:
        assert control.finish_fit is not None
        control.finish_fit.set()
        _ = run.finish()

    [(before, after)] = stamps
    earlier = parse_instant(before)
    later = parse_instant(after)
    assert earlier is not None
    assert later is not None
    assert later > earlier
    assert not _claimed(_run_root(scenario_dataset, control))


def _planted_claim(tmp_path: Path) -> tuple[Path, InflightMarker]:
    """A directory holding one execution's claim, whose expiry is already past."""
    work_dir = tmp_path / "entry"
    work_dir.mkdir()
    marker = _peer_claim(expires_at="2000-01-01T00:00:00+00:00")
    write_inflight(work_dir, marker)
    return work_dir, marker


def _wait_for_refresh(work_dir: Path, stale: str) -> None:
    """Return once the claim in *work_dir* expires later than *stale*."""
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        held = read_inflight(work_dir)
        assert held is not None
        if held.expires_at != stale:
            return
        time.sleep(0.01)
    raise AssertionError("the claim was never refreshed")


def test_kept_claimed_refreshes_the_claim_while_the_block_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_claim, "INFLIGHT_REFRESH_SECONDS", 0.05)
    work_dir, marker = _planted_claim(tmp_path)

    with kept_claimed(work_dir, marker, 3600.0):
        _wait_for_refresh(work_dir, marker.expires_at)
        held = read_inflight(work_dir)

    assert held is not None
    earlier = parse_instant(marker.expires_at)
    later = parse_instant(held.expires_at)
    assert earlier is not None
    assert later is not None
    assert later > earlier


def test_kept_claimed_stops_its_thread_on_leaving_the_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_claim, "INFLIGHT_REFRESH_SECONDS", 0.05)
    work_dir, marker = _planted_claim(tmp_path)

    with kept_claimed(work_dir, marker, 3600.0):
        _wait_for_refresh(work_dir, marker.expires_at)

    assert [t for t in threading.enumerate() if t.name.startswith("claim-")] == []
    settled = read_inflight(work_dir)
    time.sleep(0.2)
    assert read_inflight(work_dir) == settled


def test_kept_claimed_stops_its_thread_when_the_block_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(entry_claim, "INFLIGHT_REFRESH_SECONDS", 0.05)
    work_dir, marker = _planted_claim(tmp_path)

    with pytest.raises(RuntimeError, match="the block failed"):
        with kept_claimed(work_dir, marker, 3600.0):
            raise RuntimeError("the block failed")

    assert [t for t in threading.enumerate() if t.name.startswith("claim-")] == []


def test_an_untracked_fit_is_claimed_and_released(scenario_dataset: Dataset) -> None:
    control = FitControl()

    result = run_feature(scenario_dataset, FittingFeature(control), track=False)

    assert control.fits == 1
    assert result.failed_entries == ()
    assert not _claimed(_run_root(scenario_dataset, control))


_MOVED = ("", "seq_a")
"""The entry a test converts again after the first fit."""


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    """Two track tables of one variant, whose cells a fit over them records."""
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(
        dataset, "convert-trex.0.2-1111111111", "seq_a", "seq_b", std_format="mosaic_v1"
    )
    return dataset


def test_a_refit_whose_peer_fitted_again_meanwhile_loads_the_peers_state(
    tracked: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = FitControl()
    _ = run_feature(tracked, ScopedFittingFeature(control))
    reconvert_entry(tracked, _MOVED, composition="converted-again")
    claimed = run_module.held_run_root
    peers: list[Result] = []
    peer_started = threading.Event()

    def held_run_root(
        ds: Dataset,
        execution_id: str,
        run_root: Path,
        kind: str,
        idle_seconds: float,
        *,
        doing: str,
    ) -> AbstractContextManager[InflightMarker]:
        if not peer_started.is_set():
            peer_started.set()
            peers.append(run_feature(tracked, ScopedFittingFeature(control)))
        return claimed(ds, execution_id, run_root, kind, idle_seconds, doing=doing)

    monkeypatch.setattr(run_module, "held_run_root", held_run_root)

    result = run_feature(tracked, ScopedFittingFeature(control))

    assert len(peers) == 1
    assert control.fits == 2, "the first fit and the peer's refit"
    assert result.failed_entries == ()
    assert not _claimed(result_run_root(tracked, result))


def test_a_second_execution_mid_refit_is_refused(tracked: Dataset) -> None:
    control = FitControl()
    _ = run_feature(tracked, ScopedFittingFeature(control))
    reconvert_entry(tracked, _MOVED, composition="converted-again")
    control.fitting.clear()
    control.finish_fit = threading.Event()
    first = _Run(tracked, ScopedFittingFeature(control))
    first.start()
    try:
        assert control.fitting.wait(_WAIT_SECONDS)
        with pytest.raises(RunRootHeld) as refused:
            _ = run_feature(tracked, ScopedFittingFeature(control))
    finally:
        control.finish_fit.set()
        result = first.finish()

    assert refused.value.reason == "run_root_held"
    assert control.fits == 2
    assert result.failed_entries == ()
    assert not _claimed(result_run_root(tracked, result))
