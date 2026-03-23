"""Test InputsLike protocol for Feature contract."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library.registry import register_feature
from mosaic.core.pipeline.types import (
    Feature,
    InputRequire,
    Inputs,
    Params,
    Result,
    TrackInput,
)


class _TrackOnlyFeature:
    name = "track-only-test"
    version = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self, inputs: _TrackOnlyFeature.Inputs = Inputs(("tracks",))) -> None:
        self.inputs = inputs
        self._params = self.Params()

    @property
    def params(self) -> _TrackOnlyFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        return True

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None: ...
    def save_state(self, run_root: Path) -> None: ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class _MixedInputFeature:
    name = "mixed-test"
    version = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[TrackInput | Result]):
        pass

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: _MixedInputFeature.Inputs = Inputs(("tracks", Result(feature="nn"))),
    ) -> None:
        self.inputs = inputs
        self._params = self.Params()

    @property
    def params(self) -> _MixedInputFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        return True

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None: ...
    def save_state(self, run_root: Path) -> None: ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def test_track_only() -> None:
    f = _TrackOnlyFeature()
    assert f.inputs.is_single_tracks


def test_mixed_input() -> None:
    f = _MixedInputFeature()
    assert f.inputs.has_tracks
    assert f.inputs.is_multi


def test_register_feature_accepts_track_only() -> None:
    cls = register_feature(_TrackOnlyFeature)
    assert cls is _TrackOnlyFeature


def test_register_feature_accepts_mixed() -> None:
    cls = register_feature(_MixedInputFeature)
    assert cls is _MixedInputFeature


def test_is_empty() -> None:
    """Inputs.is_empty reflects whether the tuple is empty."""
    from typing import ClassVar

    class _AnyInputs(Inputs[TrackInput | Result]):
        _require: ClassVar[InputRequire] = "any"

    empty = _AnyInputs(())
    assert empty.is_empty
    non_empty = _AnyInputs(("tracks",))
    assert not non_empty.is_empty
    also_non_empty = Inputs(("tracks",))
    assert not also_non_empty.is_empty


class _TestFeature:
    """Minimal feature satisfying the new 4-method protocol."""

    name = "new-test"
    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self) -> None:
        self.inputs = self.Inputs(("tracks",))
        self._params = self.Params()

    @property
    def params(self) -> _TestFeature.Params:
        return self._params

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        return True

    def fit(self, inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"col": np.ones(len(df))})


def test_feature_protocol() -> None:
    f: Feature = _TestFeature()
    assert f.scope_dependent is False
    assert f.load_state(Path("/tmp"), {}, {}) is True


def test_satisfies_feature_protocol() -> None:
    """_TestFeature satisfies the new Feature protocol structurally."""
    f: Feature = _TestFeature()
    assert f.inputs.is_single_tracks
    assert f.name == "new-test"
