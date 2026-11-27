"""A run that lost entities must not report itself finished.

A per-entity ``apply`` exception is caught, printed to stderr, and skipped, in
both the inline and the parallel-drain paths. Nothing propagates, so
``job_context`` takes its success branch and writes ``finished`` to the run-log,
``mosaic run --json`` emits a hard-coded ``"status": "finished"`` and exits 0, and
mosaic-queue maps that exit code to a ``finished`` ledger row.

The corruption is durable and silent in the specific way that matters: under the
queue, the child's stderr goes to DEVNULL, so the one line naming the failed entry
is destroyed before anyone can read it. What remains is a run reporting success
with N outputs missing, indistinguishable from a run whose scope was smaller. The
feature index does leave such a run unfinished -- ``mark_finished`` is gated on
every entry's parquet existing -- but that signal never reaches the ledger and
names no entity.

Both catch sites are pinned by one parametrized body: ``parallel_workers=1``
takes the inline path, ``2`` the drain path. A fix that repairs one and not the
other fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.manifest import ManifestEntry, iter_manifest
from mosaic.core.pipeline.run import AllEntriesFailed, run_feature
from mosaic.core.pipeline.types import (
    Inputs,
    InputStream,
    LoadSpec,
    ParquetLoadSpec,
    Result,
    TrackInput,
)
from mosaic.core.params import Params
from mosaic.core.scope import Scope
from mosaic.runlog import read_run, run_log_dir
from tests.helpers import (
    MockDataset,
    RunMode,
    add_track_sequences,
    assert_ran_in_mode,
    latest_events,
    latest_snapshot,
    make_dataset,
    result_outputs,
    run_modes,
    track_table,
)

_DOOMED = "s2"


class _FailsOnOneSequence:
    """Raises for ``_DOOMED`` and succeeds for every other entity.

    ``apply`` receives only a frame, so it finds the entity in the table's
    ``sequence`` column. Failing by name rather than by call order keeps the
    outcome independent of iteration order.
    """

    name = "test-fails-on-one"
    version = "0.1"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self, doomed: tuple[str, ...] = (_DOOMED,)) -> None:
        self._inputs = self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(None)
        self._doomed = doomed

    @property
    def inputs(self) -> _FailsOnOneSequence.Inputs:
        return self._inputs

    @property
    def params(self) -> _FailsOnOneSequence.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if str(df["sequence"].iloc[0]) in self._doomed:
            msg = "apply failed on purpose"
            raise RuntimeError(msg)
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


@pytest.mark.parametrize("parallel_workers", [1, 2], ids=["inline", "parallel"])
def test_one_failed_entity_is_not_a_finished_run(
    tmp_path: Path, parallel_workers: int
) -> None:
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g1", "s1"), ("g1", _DOOMED))
    execution_id = "TESTPARTIAL1"

    result = run_feature(
        ds,
        _FailsOnOneSequence(),
        parallel_workers=parallel_workers,
        execution_id=execution_id,
    )

    # The entity that succeeded is still on disk: a partial run keeps its work.
    assert result.run_id is not None
    run_root = (
        ds.get_root("features") / "test-fails-on-one__from__tracks" / result.run_id
    )
    assert (run_root / "g1__s1.parquet").exists()
    assert not (run_root / f"g1__{_DOOMED}.parquet").exists()

    # The result names what it lost, rather than looking like a clean run.
    assert result.failed_entries == (make_entry_key("g1", _DOOMED),)

    # And the run-log carries the count, so the queue can see it without stderr.
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["entries_failed"] == 1
    # Both halves of the partial outcome are on the record: one entry published,
    # one lost. A reader that saw only the failure could not tell this from a run
    # that lost everything.
    assert snapshot["entries_written"] == 1
    assert result.entries_written == 1


def test_every_entity_failing_is_a_failed_run(tmp_path: Path) -> None:
    """Losing every entity is a failure, not a finished run with no outputs."""
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g1", "s1"), ("g1", _DOOMED))
    execution_id = "TESTPARTIAL2"

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(
            ds,
            _FailsOnOneSequence(doomed=("s1", _DOOMED)),
            execution_id=execution_id,
        )

    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    assert snapshot["status"] == "failed"
    assert snapshot["entries_failed"] == 2
    # Recorded even though ``AllEntriesFailed`` was raised: the count is emitted
    # before that raise, so a run that lost everything still says what it holds.
    # Pair this with the partial case above, where the same code path reports a
    # non-zero -- a missing emission and a true zero look alike here alone.
    assert snapshot["entries_written"] == 0


def test_an_empty_scope_is_not_an_all_entries_failure(tmp_path: Path) -> None:
    """No entities is not the same as every entity failing.

    The all-failed check has to be guarded on having had work to do, or a run over
    an empty scope -- a legitimate outcome the suite already relies on -- would
    start raising.
    """
    ds = MockDataset(tmp_path)
    add_track_sequences(ds, ("g1", "s1"))

    result = run_feature(
        ds, _FailsOnOneSequence(), scope=Scope(sequences=["nosuchsequence"])
    )

    assert result.failed_entries == ()


# --- an input that loads no rows for the entry ---------------------------------

_EMPTY: Final = "empty"
_MODES: Final = run_modes(2)
_NO_ROWS_MESSAGE: Final = (
    "input 1 has no rows for this entry, and the other inputs join it by the "
    "entry or by its summary keys. Their rows cannot be written without its columns"
)
"""What the entry whose per-entry input holds no rows reports."""


class _Stub:
    """The boilerplate of a test feature; subclasses set the inputs and ``apply``."""

    name = "test-stub"
    version = "0.1"
    category = "per-frame"
    emits: EmitsLevel = "individual"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()

    class Params(Params):
        pass

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.params: _Stub.Params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class _OverTracks(_Stub):
    """A stub reading the tracks."""

    class Inputs(Inputs[TrackInput]):
        pass

    def __init__(self) -> None:
        super().__init__()
        self.inputs: _OverTracks.Inputs = self.Inputs(("tracks",))


class _Speeds(_OverTracks):
    """One row per frame and individual."""

    name = "test-speeds"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "id": df["id"], "speed": 1.0})


class _Totals(_OverTracks):
    """One row per entry that names no individual and no frame, none for ``_EMPTY``."""

    name = "test-totals"
    emits: EmitsLevel = "unidentified"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = 0 if (df["sequence"] == _EMPTY).all() else 1
        return pd.DataFrame({"total": [float(len(df))] * rows})


class _Echo(_Stub):
    """An ``as-input`` consumer that writes its merged inputs."""

    name = "test-echo"
    emits: EmitsLevel = "as-input"

    class Inputs(Inputs[Result]):
        pass

    def __init__(
        self, inputs: _Echo.Inputs, params: dict[str, object] | None = None
    ) -> None:
        super().__init__(params)
        self.inputs: _Echo.Inputs = inputs


class _FitsEcho(_Echo):
    """An ``as-input`` consumer that fits by reading every entry of its stream."""

    name = "test-fits-echo"

    def __init__(
        self, inputs: _Echo.Inputs, params: dict[str, object] | None = None
    ) -> None:
        super().__init__(inputs, params)
        self.fitted_entries: list[str] = []

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return False

    def fit(self, inputs: InputStream) -> None:
        self.fitted_entries = [entry_key for entry_key, _ in inputs()]


def _results(tmp_path: Path, *sequences: str) -> tuple[Dataset, Result, Result]:
    """The per-frame individual rows and the per-entry totals of *sequences*."""
    ds = make_dataset(tmp_path / "ds")
    add_track_sequences(ds, *(("g", s) for s in sequences), n_rows=6, n_ids=2)
    speeds = run_feature(ds, _Speeds())
    totals = run_feature(ds, _Totals())
    return (
        ds,
        Result(feature=speeds.feature, run_id=speeds.run_id),
        Result(feature=totals.feature, run_id=totals.run_id),
    )


def _echo(speeds: Result, totals: Result) -> _Echo:
    return _Echo(_Echo.Inputs((speeds, totals)))


@pytest.mark.parametrize("mode", _MODES, ids=lambda mode: mode.name)
def test_an_empty_per_entry_input_fails_the_entry_and_the_run_is_partial(
    tmp_path: Path, mode: RunMode
) -> None:
    ds, speeds, totals = _results(tmp_path, "s1", _EMPTY, "s2")

    result = run_feature(ds, _echo(speeds, totals), **mode.run_kwargs())

    assert_ran_in_mode(ds, mode)
    assert result.failed_entries == (make_entry_key("g", _EMPTY),)
    (event,) = latest_events(ds, "entry_error")
    assert event["key"] == make_entry_key("g", _EMPTY)
    error = json.loads(str(event["error"]))
    assert error["type"] == "BroadcastRowsMissing"
    assert error["message"] == _NO_ROWS_MESSAGE
    assert latest_snapshot(ds)["status"] == "finished"
    assert set(result_outputs(ds, result)) == {
        make_entry_key("g", "s1"),
        make_entry_key("g", "s2"),
    }


def test_every_entry_without_per_entry_rows_is_a_failed_run(tmp_path: Path) -> None:
    ds, speeds, totals = _results(tmp_path, _EMPTY)

    with pytest.raises(AllEntriesFailed):
        _ = run_feature(ds, _echo(speeds, totals))

    assert latest_snapshot(ds)["status"] == "failed"


def test_a_fit_skips_an_entry_without_per_entry_rows_and_apply_reports_it(
    tmp_path: Path,
) -> None:
    ds, speeds, totals = _results(tmp_path, "s1", _EMPTY)
    feature = _FitsEcho(_Echo.Inputs((speeds, totals)))

    result = run_feature(ds, feature)

    assert feature.fitted_entries == [make_entry_key("g", "s1")]
    assert result.failed_entries == (make_entry_key("g", _EMPTY),)


def test_a_neighbor_without_per_entry_rows_gives_no_context(tmp_path: Path) -> None:
    """The entry is reported, and its neighbor reads it as an absent neighbor."""
    names = ("s1", "s2", "s3")
    frames = 10
    tracks: dict[str, tuple[Path, LoadSpec]] = {}
    totals: dict[str, tuple[Path, LoadSpec]] = {}
    for index, name in enumerate(names):
        tracks_path = tmp_path / f"{name}.tracks.parquet"
        track_table(
            "g1", name, n_rows=frames, n_ids=1, start_frame=index * frames
        ).to_parquet(tracks_path)
        tracks[name] = (tracks_path, ParquetLoadSpec())
        totals_path = tmp_path / f"{name}.totals.parquet"
        rows = 0 if name == "s1" else 1
        pd.DataFrame({"total": [1.0] * rows}).to_parquet(totals_path)
        totals[name] = (totals_path, ParquetLoadSpec())

    def specs(name: str) -> list[tuple[Path, LoadSpec]]:
        return [tracks[name], totals[name]]

    manifest: dict[str, ManifestEntry] = {}
    for index, name in enumerate(names):
        before = names[index - 1] if index > 0 else None
        after = names[index + 1] if index < len(names) - 1 else None
        manifest[f"g1__{name}"] = ManifestEntry(
            file_specs=specs(name),
            prev_file_specs=None if before is None else specs(before),
            prev_entry_key=None if before is None else f"g1__{before}",
            next_file_specs=None if after is None else specs(after),
            next_entry_key=None if after is None else f"g1__{after}",
            entry_key=f"g1__{name}",
            group="g1",
            sequence=name,
            continuous=True,
            core_extent=(index * frames, (index + 1) * frames - 1),
            prev_extent=None
            if before is None
            else (index * frames - frames, index * frames - 1),
            next_extent=None
            if after is None
            else ((index + 1) * frames, (index + 2) * frames - 1),
        )
    missing: list[str] = []

    loaded = list(
        iter_manifest(
            manifest,
            overlap_frames=3,
            on_rows_missing=lambda key, exc: missing.append(
                f"{key}: {type(exc).__name__}"
            ),
        )
    )

    assert missing == ["g1__s1: BroadcastRowsMissing"]
    assert [key for key, _, _ in loaded] == ["g1__s2", "g1__s3"]
    _key, df, selector = loaded[0]
    assert not (df["frame"] < selector.first).any()
    assert (df["frame"] > selector.last).any()
    _key, df, selector = loaded[1]
    assert (df["frame"] < selector.first).any()
