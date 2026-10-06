"""A feature is refused tracks in a space it does not compute in.

A planar feature handed a world-frame table does not fail on its own. It reads
``X`` and ``Y``, ignores ``Z``, and returns speeds and headings that are the
projection of the real ones onto one plane of the calibration -- plausible
numbers, recorded nowhere as wrong. ``run_feature`` therefore compares the space
of every track-shaped table a run resolves with the feature's ``track_spaces``
before anything is hashed or read, and a feature that declares nothing computes
in the two 2D spaces only.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.refusal import Refusal
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.track_space import (
    DEFAULT_TRACK_SPACES,
    TrackSpaceRefused,
    feature_track_spaces,
    refuse_unsupported_track_space,
)
from mosaic.core.pipeline.types import EmitsLevel, Inputs, InputStream, TrackInput
from mosaic.core.params import Params

from tests.helpers import add_tracks_variant


class _TrackFeature:
    """A stateless per-frame feature over ``"tracks"`` that declares no space."""

    name = "test-track-space"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self) -> None:
        self._inputs = self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(None)

    @property
    def inputs(self) -> _TrackFeature.Inputs:
        return self._inputs

    @property
    def params(self) -> _TrackFeature.Params:
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
        return pd.DataFrame({"frame": df["frame"], "id": df["id"], "value": 1.0})


class _WorldFeature(_TrackFeature):
    """The same feature, declaring that it computes in a world frame too."""

    name = "test-track-space-world"
    track_spaces: ClassVar[frozenset[str]] = frozenset({"image_px", "world_m"})


def _dataset(tmp_path: Path, *schemas: str) -> Dataset:
    """A dataset with one entry per schema, named ``seq_0``, ``seq_1``, ..."""
    manifest = new_dataset_manifest(name="spaces", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    for index, schema in enumerate(schemas):
        add_tracks_variant(
            dataset,
            f"convert-x{index}.0.1-{index:010d}",
            f"seq_{index}",
            std_format=schema,
        )
    return dataset


def test_an_undeclared_feature_computes_in_both_2d_spaces() -> None:
    assert feature_track_spaces(_TrackFeature()) == DEFAULT_TRACK_SPACES
    assert DEFAULT_TRACK_SPACES == {"image_px", "plane_cm"}


def test_a_declared_feature_computes_in_what_it_declares() -> None:
    assert feature_track_spaces(_WorldFeature()) == {"image_px", "world_m"}


def test_a_string_is_not_a_declaration() -> None:
    """``"world_m"`` iterates as letters; refuse it rather than read nonsense."""
    feature = _TrackFeature()
    feature.track_spaces = "world_m"  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TypeError, match="set of track spaces"):
        _ = feature_track_spaces(feature)


@pytest.mark.parametrize(
    ("schema", "space"),
    [
        ("mosaic_v1", "image_px"),
        ("trex_v2", "image_px"),
        ("mosaic_cm_v1", "plane_cm"),
        ("trex_v1", "plane_cm"),
        ("", "plane_cm"),
        ("mosaic_world_v1", "world_m"),
    ],
)
def test_the_resolved_scope_names_the_space_of_its_tables(
    tmp_path: Path, schema: str, space: str
) -> None:
    _, scope = build_manifest(_dataset(tmp_path, schema), Inputs(("tracks",)))
    assert scope.track_spaces == {space}
    assert scope.tracks_schemas == {("", "seq_0"): schema}


def test_an_unregistered_schema_contributes_no_space(tmp_path: Path) -> None:
    """Nothing can be said about it, so it is not judged either way."""
    _, scope = build_manifest(
        _dataset(tmp_path, "registered_elsewhere_v1"), Inputs(("tracks",))
    )
    assert scope.track_spaces == frozenset()


def test_a_planar_feature_is_refused_a_world_table(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "mosaic_world_v1")
    with pytest.raises(TrackSpaceRefused) as excinfo:
        _ = run_feature(dataset, _TrackFeature())
    refusal = excinfo.value
    assert isinstance(refusal, Refusal)
    assert refusal.reason == "schema_family_mismatch"
    assert refusal.detail["found"] == ["world_m"]
    assert refusal.detail["accepts"] == ["image_px", "plane_cm"]


def test_the_refusal_comes_before_any_output(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "mosaic_world_v1")
    with pytest.raises(TrackSpaceRefused):
        _ = run_feature(dataset, _TrackFeature())
    features_root = dataset.get_root("features")
    assert not any(features_root.rglob("*.parquet"))


def test_a_planar_feature_still_runs_on_pixel_tables(tmp_path: Path) -> None:
    result = run_feature(_dataset(tmp_path, "mosaic_v1"), _TrackFeature())
    assert result.run_id


def test_a_feature_declaring_the_world_space_runs_on_a_world_table(
    tmp_path: Path,
) -> None:
    result = run_feature(_dataset(tmp_path, "mosaic_world_v1"), _WorldFeature())
    assert result.run_id


def test_a_feature_declaring_only_the_world_space_is_refused_centimetres() -> None:
    feature = _TrackFeature()
    feature.track_spaces = frozenset({"world_m"})  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(TrackSpaceRefused):
        refuse_unsupported_track_space(feature, feature.name, {"plane_cm"})
