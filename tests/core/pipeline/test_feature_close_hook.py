"""``run_feature`` calls a closable feature's ``close()`` exactly once per attempt.

A feature that keeps a resource open, such as a subprocess or a connection,
declares ``close()``. ``run_feature`` releases the resource whether the attempt
finished, failed, or was cancelled. ``run_feature`` is the only caller
for the instance it is handed. A defect here is a leaked resource that shows up
only under load.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.job import CancelToken, Cancelled
from mosaic.core.pipeline.run import (
    AllEntriesFailed,
    process_apply_worker,
    run_feature,
)
from mosaic.core.pipeline.types import (
    EmitsLevel,
    Inputs,
    InputStream,
    TrackInput,
)
from mosaic.core.params import Params
from tests.helpers import (
    add_track_sequences,
    entry_error_lines,
    latest_snapshot,
    make_dataset,
)


class _ClosingFeature:
    """Records ``fit``/``apply``/``close`` calls, and can fail any of them."""

    name = "test-close-hook"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        *,
        state_ready: bool = True,
        fit_raises: Exception | None = None,
        apply_raises: Exception | None = None,
        close_raises: Exception | None = None,
        on_apply: Callable[[], None] | None = None,
    ) -> None:
        self._inputs = self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(None)
        self._state_ready = state_ready
        self._fit_raises = fit_raises
        self._apply_raises = apply_raises
        self._close_raises = close_raises
        self._on_apply = on_apply
        self.close_calls = 0

    @property
    def inputs(self) -> _ClosingFeature.Inputs:
        return self._inputs

    @property
    def params(self) -> _ClosingFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return self._state_ready

    def fit(self, inputs: InputStream) -> None:
        if self._fit_raises is not None:
            raise self._fit_raises

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._on_apply is not None:
            self._on_apply()
        if self._apply_raises is not None:
            raise self._apply_raises
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})

    def close(self) -> None:
        self.close_calls += 1
        if self._close_raises is not None:
            raise self._close_raises


# --- close() runs exactly once -------------------------------------------


def test_close_runs_once_on_success(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))

    feature = _ClosingFeature()
    run_feature(ds, feature)

    assert feature.close_calls == 1


def test_close_runs_once_when_fit_raises(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    feature = _ClosingFeature(state_ready=False, fit_raises=RuntimeError("fit boom"))
    with pytest.raises(RuntimeError, match="fit boom"):
        run_feature(ds, feature)

    assert feature.close_calls == 1


def test_close_runs_once_when_every_entry_fails(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))

    feature = _ClosingFeature(apply_raises=RuntimeError("apply boom"))
    with pytest.raises(AllEntriesFailed):
        run_feature(ds, feature)

    assert feature.close_calls == 1


def test_close_runs_once_when_cancelled(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), ("g", "s3"))
    token = CancelToken()
    calls = {"n": 0}

    def on_apply() -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            token.cancel()

    feature = _ClosingFeature(on_apply=on_apply)
    with pytest.raises(Cancelled):
        run_feature(ds, feature, cancel_token=token)

    assert feature.close_calls == 1


def test_close_runs_once_on_a_cache_hit(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"))
    run_feature(ds, _ClosingFeature())

    cached = _ClosingFeature()
    result = run_feature(ds, cached)

    assert result.cache_hit is True
    assert cached.close_calls == 1


# --- a raising close() does not replace the original error ----------------


def test_a_raising_close_does_not_mask_the_fit_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"))

    feature = _ClosingFeature(
        state_ready=False,
        fit_raises=RuntimeError("fit boom"),
        close_raises=RuntimeError("close boom"),
    )
    with pytest.raises(RuntimeError, match="fit boom"):
        run_feature(ds, feature)

    assert feature.close_calls == 1
    assert "close boom" in capsys.readouterr().err


# --- a cancel raised through apply ends the attempt cancelled -------------


def test_apply_raising_cancelled_on_the_last_entry_with_the_token_set_is_not_an_entry_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tool an entry's ``apply`` drives can notice a cancel and raise it itself.

    Once the token is set, that raised ``Cancelled`` ends the attempt as
    cancelled. The run-log records the cancel, and it does not record the entry
    as lost.
    """
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, ("g", "s1"), ("g", "s2"), ("g", "s3"))
    token = CancelToken()
    calls = {"n": 0}

    def on_apply() -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            token.cancel()
            raise Cancelled()

    feature = _ClosingFeature(on_apply=on_apply)
    execution_id = "TESTCLOSEHOOKLASTCANCEL"
    with pytest.raises(Cancelled):
        run_feature(ds, feature, cancel_token=token, execution_id=execution_id)

    assert feature.close_calls == 1
    assert latest_snapshot(ds)["status"] == "cancelled"
    assert entry_error_lines(ds, execution_id) == []
    assert "apply failed" not in capsys.readouterr().err


# --- process_apply_worker closes the instance that it builds ---------------

# A process-pool worker reconstructs its feature from a module and class
# name plus pickled inputs/params dumps (see process_apply_worker), builds
# and discards an instance, and returns only the applied DataFrame. The test
# cannot read ``self.close_calls`` of that instance. Module-level globals are
# the one channel out, monkeypatched per test so pytest restores them.
_worker_events: list[str] = []
_worker_apply_raises: Exception | None = None
_worker_close_raises: Exception | None = None


class _WorkerClosingFeature(_ClosingFeature):
    """Reconstructed by ``process_apply_worker``; records calls via module state."""

    name = "test-worker-close-hook"

    def __init__(
        self,
        inputs: Inputs[TrackInput],
        params: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self._inputs = inputs
        self._params = self.Params.from_overrides(params)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        _worker_events.append("apply")
        if _worker_apply_raises is not None:
            raise _worker_apply_raises
        return super().apply(df)

    def close(self) -> None:
        _worker_events.append("close")
        if _worker_close_raises is not None:
            raise _worker_close_raises


def _call_worker(tmp_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    return process_apply_worker(
        __name__,
        "_WorkerClosingFeature",
        _WorkerClosingFeature.Inputs(("tracks",)).model_dump(),
        {},
        str(tmp_path),
        {},
        {},
        df,
    )


def test_process_apply_worker_closes_its_own_instance_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_worker_apply_raises", None)
    monkeypatch.setattr(sys.modules[__name__], "_worker_close_raises", None)
    _worker_events.clear()

    df = pd.DataFrame({"frame": [0, 1], "feat_a": [1.0, 2.0]})
    result = _call_worker(tmp_path, df)

    assert _worker_events == ["apply", "close"]
    assert list(result["value"]) == [2.0, 4.0]


def test_process_apply_worker_closes_once_when_apply_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys.modules[__name__], "_worker_apply_raises", RuntimeError("apply boom")
    )
    monkeypatch.setattr(sys.modules[__name__], "_worker_close_raises", None)
    _worker_events.clear()

    df = pd.DataFrame({"frame": [0], "feat_a": [1.0]})
    with pytest.raises(RuntimeError, match="apply boom"):
        _call_worker(tmp_path, df)

    assert _worker_events == ["apply", "close"]


def test_process_apply_worker_raising_close_does_not_mask_apply_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys.modules[__name__], "_worker_apply_raises", RuntimeError("apply boom")
    )
    monkeypatch.setattr(
        sys.modules[__name__], "_worker_close_raises", RuntimeError("close boom")
    )
    _worker_events.clear()

    df = pd.DataFrame({"frame": [0], "feat_a": [1.0]})
    with pytest.raises(RuntimeError, match="apply boom"):
        _call_worker(tmp_path, df)

    assert _worker_events == ["apply", "close"]
    assert "close boom" in capsys.readouterr().err
