"""Test InputsLike protocol for Feature contract."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mosaic.behavior.feature_library.params import (
    Inputs,
    OutputType,
    Params,
    Result,
    TrackInput,
)
from mosaic.core.dataset import Feature, register_feature


class _TrackOnlyFeature:
    name = "track-only-test"
    version = "0.1"
    output_type: OutputType = "per_frame"
    parallelizable = False
    storage_feature_name = "track-only-test"
    storage_use_input_suffix = False

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

    def bind_dataset(self, ds: object) -> None: ...
    def set_scope_filter(self, scope: dict[str, object] | None) -> None: ...
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: ...
    def partial_fit(self, df: pd.DataFrame) -> None: ...
    def finalize_fit(self) -> None: ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def save_model(self, path: Path) -> None: ...
    def load_model(self, path: Path) -> None: ...


class _MixedInputFeature:
    name = "mixed-test"
    version = "0.1"
    output_type: OutputType = "per_frame"
    parallelizable = False
    storage_feature_name = "mixed-test"
    storage_use_input_suffix = True

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

    def bind_dataset(self, ds: object) -> None: ...
    def set_scope_filter(self, scope: dict[str, object] | None) -> None: ...
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None: ...
    def partial_fit(self, df: pd.DataFrame) -> None: ...
    def finalize_fit(self) -> None: ...
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def save_model(self, path: Path) -> None: ...
    def load_model(self, path: Path) -> None: ...


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


def test_satisfies_feature_protocol() -> None:
    """Both features satisfy the Feature protocol structurally."""
    f1: Feature = _TrackOnlyFeature()
    f2: Feature = _MixedInputFeature()
    assert f1.inputs.is_single_tracks
    assert f2.inputs.is_multi
