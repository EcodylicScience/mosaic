"""A feature is refused tracks in a space it does not compute in.

A planar feature handed a world-frame table does not fail on its own. It reads
``X`` and ``Y``, ignores ``Z``, and returns speeds and headings that are the
projection of the real ones onto one plane of the calibration: plausible
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

from mosaic.behavior.feature_library.frame_aggregate import FrameAggregate
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
from mosaic.core.pipeline.track_universe import carries_z
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.types import (
    EmitsLevel,
    Inputs,
    InputStream,
    Result,
    TrackInputs,
)
from mosaic.core.params import Params
from mosaic.core.schema import TRACK_SPACES, TrackSpace

from tests.helpers import add_tracks_table, track_table, world_track_table


class _TrackFeature:
    """A stateless per-frame feature over ``"tracks"`` that declares no space."""

    name = "test-track-space"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(TrackInputs):
        pass

    class Params(Params):
        pass

    def __init__(self, *inputs: Result[str]) -> None:
        self._inputs = self.Inputs(inputs or ("tracks",))
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
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset({"image_px", "world_m"})


class _PassThrough(_WorldFeature):
    """A track-producing feature: it returns the track frame it was handed."""

    name = "test-track-space-pass"
    track_spaces: ClassVar[frozenset[TrackSpace]] = frozenset(TRACK_SPACES)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


def _dataset(tmp_path: Path, *schemas: str) -> Dataset:
    """A dataset with one entry per schema, named ``seq_0``, ``seq_1``, ..."""
    manifest = new_dataset_manifest(name="spaces", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    for index, schema in enumerate(schemas):
        sequence = f"seq_{index}"
        table = (
            world_track_table("", sequence, n_rows=10)
            if schema == "mosaic_world_v1"
            else track_table("", sequence, n_rows=10)
        )
        add_tracks_table(
            dataset, f"convert-x{index}.0.1-{index:010d}", table, std_format=schema
        )
    return dataset


def _declaring(declared: object) -> object:
    """A track feature whose class declares *declared* as its track spaces."""
    return type("_Declared", (_TrackFeature,), {"track_spaces": declared})()


def _files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_an_undeclared_feature_computes_in_both_2d_spaces() -> None:
    assert feature_track_spaces(_TrackFeature()) == DEFAULT_TRACK_SPACES
    assert DEFAULT_TRACK_SPACES == {"image_px", "plane_cm"}


def test_a_declared_feature_computes_in_what_it_declares() -> None:
    assert feature_track_spaces(_WorldFeature()) == {"image_px", "world_m"}


@pytest.mark.parametrize(
    "declared",
    ["world_m", ["world_m"], ("world_m",), 3],
    ids=["a string", "a list", "a tuple", "a number"],
)
def test_only_a_set_is_a_declaration(declared: object) -> None:
    """``"world_m"`` iterates as letters; refuse it rather than read nonsense."""
    with pytest.raises(TypeError, match="set of track spaces"):
        _ = feature_track_spaces(_declaring(declared))


def test_a_value_outside_the_track_spaces_is_not_a_declaration() -> None:
    with pytest.raises(TypeError, match="world_cm"):
        _ = feature_track_spaces(_declaring(frozenset({"image_px", "world_cm"})))


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


def test_a_track_shaped_result_carrying_z_reads_as_a_world_table(
    tmp_path: Path,
) -> None:
    """A ``Result`` records no schema, so its footer says what it holds."""
    dataset = _dataset(tmp_path, "mosaic_world_v1")
    produced = run_feature(dataset, _PassThrough())
    reference = Result(feature=produced.feature, run_id=produced.run_id)

    _, scope = build_manifest(dataset, TrackInputs((reference,)))

    assert scope.track_spaces == {"world_m"}


def test_a_track_shaped_result_without_z_reads_no_space(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "mosaic_v1")
    produced = run_feature(dataset, _PassThrough())
    reference = Result(feature=produced.feature, run_id=produced.run_id)

    _, scope = build_manifest(dataset, TrackInputs((reference,)))

    assert scope.track_spaces == frozenset()


def test_a_planar_feature_is_refused_a_track_result_carrying_z(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "mosaic_world_v1")
    produced = run_feature(dataset, _PassThrough())
    downstream = _TrackFeature(Result(feature=produced.feature, run_id=produced.run_id))

    with pytest.raises(TrackSpaceRefused) as excinfo:
        _ = run_feature(dataset, downstream)

    assert excinfo.value.detail["found"] == ["world_m"]


def test_carries_z_reads_the_footer(tmp_path: Path) -> None:
    with_z = tmp_path / "with_z.parquet"
    without_z = tmp_path / "without_z.parquet"
    world_track_table(n_rows=3).to_parquet(with_z)
    track_table(n_rows=3).to_parquet(without_z)

    assert carries_z(with_z)
    assert not carries_z(without_z)


def test_carries_z_is_false_for_an_absent_file(tmp_path: Path) -> None:
    assert not carries_z(tmp_path / "absent.parquet")


def test_carries_z_is_false_for_a_torn_footer(tmp_path: Path) -> None:
    whole = tmp_path / "whole.parquet"
    world_track_table(n_rows=3).to_parquet(whole)
    torn = tmp_path / "torn.parquet"
    _ = torn.write_bytes(whole.read_bytes()[:-20])

    assert not carries_z(torn)


def test_a_planar_feature_is_refused_a_world_table(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "mosaic_world_v1")
    with pytest.raises(TrackSpaceRefused) as excinfo:
        _ = run_feature(dataset, _TrackFeature())
    refusal = excinfo.value
    assert isinstance(refusal, Refusal)
    assert isinstance(refusal, ValueError)
    assert refusal.reason == "schema_family_mismatch"
    assert refusal.detail == {
        "feature": "test-track-space",
        "accepts": ["image_px", "plane_cm"],
        "found": ["world_m"],
    }


def test_the_refusal_writes_no_run_directory_and_no_index_row(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "mosaic_world_v1")
    tracks_index_before = read_tracks_index(dataset).to_dict("records")

    with pytest.raises(TrackSpaceRefused):
        _ = run_feature(dataset, _TrackFeature())

    assert _files_under(dataset.get_root("features")) == []
    assert not any(path.is_dir() for path in dataset.get_root("features").iterdir())
    assert read_tracks_index(dataset).to_dict("records") == tracks_index_before


def test_a_planar_feature_still_runs_on_pixel_tables(tmp_path: Path) -> None:
    result = run_feature(_dataset(tmp_path, "mosaic_v1"), _TrackFeature())
    assert result.run_id


def test_a_planar_feature_runs_on_centimeter_tables(tmp_path: Path) -> None:
    """Centimeters are a planar space: the default accepts them."""
    result = run_feature(_dataset(tmp_path, "mosaic_cm_v1"), _TrackFeature())
    assert result.run_id


def test_a_feature_declaring_the_world_space_runs_on_a_world_table(
    tmp_path: Path,
) -> None:
    result = run_feature(_dataset(tmp_path, "mosaic_world_v1"), _WorldFeature())
    assert result.run_id


def test_a_space_independent_feature_runs_on_a_world_table(tmp_path: Path) -> None:
    feature = FrameAggregate(params={"columns": ["X", "Z"], "aggs": ["mean"]})
    result = run_feature(_dataset(tmp_path, "mosaic_world_v1"), feature)
    assert result.run_id


def test_a_feature_declaring_only_the_world_space_is_refused_centimeters() -> None:
    with pytest.raises(TrackSpaceRefused):
        refuse_unsupported_track_space(frozenset({"world_m"}), "test", {"plane_cm"})


def test_the_refusal_of_a_planar_feature_explains_the_dropped_axis() -> None:
    with pytest.raises(TrackSpaceRefused) as excinfo:
        refuse_unsupported_track_space(
            frozenset({"image_px", "plane_cm"}), "test", {"world_m"}
        )

    message = str(excinfo.value)
    assert "would compute on two of its three axes" in message
    assert "mosaic_world_v1" not in message


def test_the_refusal_of_a_world_feature_names_the_world_frame_table() -> None:
    with pytest.raises(TrackSpaceRefused) as excinfo:
        refuse_unsupported_track_space(frozenset({"world_m"}), "test", {"image_px"})

    message = str(excinfo.value)
    assert "three axes" not in message
    assert "computes in 3D needs a world-frame table (mosaic_world_v1)" in message
    assert "narrow the scope" in message


def test_a_feature_accepting_every_space_refuses_none() -> None:
    refuse_unsupported_track_space(frozenset(TRACK_SPACES), "test", set(TRACK_SPACES))
