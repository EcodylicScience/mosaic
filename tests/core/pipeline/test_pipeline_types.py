"""The typed pipeline surface in ``mosaic.core.pipeline.types``.

``COLUMNS``, ``Result`` and ``ResultColumn``, ``Inputs``, the load specs, and
``ArtifactSpec`` resolving a ``Result`` and reading a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline.types import (
    COLUMNS,
    Inputs,
    Result,
    TrackInput,
    resolve_order_col,
)


# --- COLUMNS global ---


def test_columns_defaults() -> None:
    assert COLUMNS.id_col == "id"
    assert COLUMNS.seq_col == "sequence"
    assert COLUMNS.group_col == "group"
    assert COLUMNS.frame_col == "frame"
    assert COLUMNS.time_col == "time"
    assert COLUMNS.order_by == "frames"
    assert COLUMNS.x_col == "X"
    assert COLUMNS.y_col == "Y"
    assert COLUMNS.orientation_col == "ANGLE"


def test_resolve_order_col_frames_first() -> None:
    import pandas as pd

    df = pd.DataFrame({"frame": [1, 2], "time": [0.0, 0.1]})
    assert resolve_order_col(df) == "frame"


def test_resolve_order_col_missing_both() -> None:
    import pandas as pd

    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Need"):
        resolve_order_col(df)


# --- Spec models ---


def test_npz_load_spec_requires_key() -> None:
    from mosaic.core.pipeline.types import NpzLoadSpec

    with pytest.raises(ValidationError):
        NpzLoadSpec()
    s = NpzLoadSpec(key="templates")
    assert s.kind == "npz"
    assert s.key == "templates"


def test_parquet_load_spec_defaults() -> None:
    from mosaic.core.pipeline.types import ParquetLoadSpec

    s = ParquetLoadSpec()
    assert s.kind == "parquet"
    assert s.columns is None
    assert s.numeric_only is True


def test_nested_spec_model_dump_roundtrip() -> None:
    from mosaic.core.pipeline.types import ArtifactSpec, NpzLoadSpec

    orig = ArtifactSpec(feature="test", load=NpzLoadSpec(key="X"))
    dumped = orig.model_dump()
    restored = ArtifactSpec.model_validate(dumped)
    assert restored == orig


def test_joblib_load_spec() -> None:
    from mosaic.core.pipeline.types import JoblibLoadSpec

    s = JoblibLoadSpec()
    assert s.kind == "joblib"
    assert s.key is None
    s2 = JoblibLoadSpec(key="scaler")
    assert s2.key == "scaler"


def test_result_use_latest() -> None:
    r = Result(feature="nn", run_id="v1")
    latest = r.use_latest()
    assert latest.run_id is None
    assert r.run_id == "v1"


def test_nn_result_defaults() -> None:
    from mosaic.core.pipeline.types import NNResult

    nn = NNResult()
    assert nn.feature == "nearest-neighbor"


def test_nn_result_accepts_variant_feature_names() -> None:
    from mosaic.core.pipeline.types import NNResult, Result

    # Auto-derived __from__tracks suffix
    nn = NNResult(feature="nearest-neighbor__from__tracks", run_id="r1")
    assert nn.feature == "nearest-neighbor__from__tracks"
    assert nn.run_id == "r1"

    # from_result copies feature + run_id from an existing Result
    src = Result[str](feature="nearest-neighbor__from__trajectory-smooth", run_id="r2")
    nn2 = NNResult().from_result(src)
    assert nn2.feature == "nearest-neighbor__from__trajectory-smooth"
    assert nn2.run_id == "r2"


def test_artifact_spec_auto_pattern() -> None:
    from mosaic.core.pipeline.types import (
        ArtifactSpec,
        JoblibLoadSpec,
        NpzLoadSpec,
    )

    a = ArtifactSpec(feature="x", load=NpzLoadSpec(key="y"))
    assert a.pattern == "*.npz"
    b = ArtifactSpec(feature="x", load=JoblibLoadSpec())
    assert b.pattern == "*.joblib"


def test_artifact_spec_pattern_kind_mismatch() -> None:
    from mosaic.core.pipeline.types import ArtifactSpec, NpzLoadSpec

    with pytest.raises(ValidationError):
        ArtifactSpec(feature="x", load=NpzLoadSpec(key="y"), pattern="foo.joblib")


def test_artifact_from_result() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    r = Result(feature="global-tsne", run_id="v1")
    art = GlobalTSNE.TSNECoordsArtifact.from_result(r)
    assert art.run_id == "v1"
    assert art.pattern == "global_tsne_templates.npz"


def test_artifact_from_result_wrong_feature() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    with pytest.raises(ValueError, match="expects feature"):
        GlobalTSNE.TSNECoordsArtifact.from_result(Result(feature="wrong"))


def test_artifact_from_result_chained() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    r = Result(feature="global-tsne", run_id="v1")
    art = GlobalTSNE.TSNECoordsArtifact.from_result(r.use_latest())
    assert art.run_id is None


def test_artifact_from_result_suffixed_feature() -> None:
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    r = Result(
        feature="global-tsne__from__pair-wavelet__from__pair-egocentric",
        run_id="v2",
    )
    art = GlobalTSNE.TSNECoordsArtifact.from_result(r)
    assert art.feature == r.feature
    assert art.run_id == "v2"
    assert art.pattern == "global_tsne_templates.npz"


# --- Inputs ---


def test_fi_tracks_only() -> None:
    i = Inputs(("tracks",))
    assert i.is_single_tracks
    assert not i.is_single_feature
    assert not i.is_multi
    assert i.has_tracks


def test_fi_single_feature() -> None:
    i = Inputs((Result(feature="speed-angvel"),))
    assert i.is_single_feature
    assert not i.is_single_tracks
    assert not i.is_multi
    assert not i.has_tracks


def test_result_str_delegates_to_repr() -> None:
    r = Result(feature="speed-angvel", run_id="0.1-abc")
    assert str(r) == repr(r)
    assert "Result(" in str(r)
    assert "speed-angvel" in str(r)


def test_fi_single_feature_with_run_id() -> None:
    i = Inputs((Result(feature="speed-angvel", run_id="v1"),))
    assert i.is_single_feature
    assert i.feature_inputs[0].run_id == "v1"


def test_fi_multi() -> None:
    i = Inputs(("tracks", Result(feature="nn")))
    assert i.is_multi
    assert not i.is_single_tracks
    assert not i.is_single_feature
    assert i.has_tracks
    assert len(i.feature_inputs) == 1


def test_fi_empty_raises() -> None:
    with pytest.raises(ValidationError):
        Inputs(())


def test_fi_empty_still_rejected_by_default() -> None:
    """Subclasses with default _require='nonempty' still reject empty inputs."""

    class StrictInputs(Inputs[Result]):
        pass

    with pytest.raises(ValidationError):
        StrictInputs(())


def test_fi_empty_allowed_with_opt_in() -> None:
    """Subclasses with _require='any' accept both empty and non-empty inputs."""
    from typing import ClassVar

    class EmptyOkInputs(Inputs[Result]):
        _require: ClassVar[str] = "any"

    i = EmptyOkInputs(())
    assert len(i.root) == 0
    assert not i.has_tracks
    assert i.feature_inputs == ()
    assert i.storage_suffix() is None


def test_fi_duplicate_raises() -> None:
    with pytest.raises(ValidationError):
        Inputs((Result(feature="nn"), Result(feature="nn")))


def test_fi_feature_inputs_property() -> None:
    i = Inputs(("tracks", Result(feature="a"), Result(feature="b")))
    feats = i.feature_inputs
    assert len(feats) == 2
    assert feats[0].feature == "a"
    assert feats[1].feature == "b"


def test_fi_storage_suffix_tracks_only() -> None:
    assert Inputs(("tracks",)).storage_suffix() == "tracks"


def test_fi_storage_suffix_single_feature() -> None:
    i = Inputs((Result(feature="speed-angvel"),))
    assert i.storage_suffix() == "speed-angvel"


def test_fi_storage_suffix_mixed() -> None:
    i = Inputs(("tracks", Result(feature="sa")))
    assert i.storage_suffix() == "tracks+sa"


def test_fi_storage_suffix_multi() -> None:
    i = Inputs((Result(feature="a"), Result(feature="b")))
    assert i.storage_suffix() == "a+b"


def test_fi_narrowed_accepts_correct() -> None:
    Narrowed = Inputs[Result[Literal["nn"]]]
    i = Narrowed((Result(feature="nn"),))
    assert i.is_single_feature


def test_fi_narrowed_rejects_wrong() -> None:
    Narrowed = Inputs[Result[Literal["nn"]]]
    with pytest.raises(ValidationError):
        Narrowed((Result(feature="wrong"),))


def test_fi_narrowed_tracks() -> None:
    Narrowed = Inputs[TrackInput]
    i = Narrowed(("tracks",))
    assert i.is_single_tracks


def test_fi_narrowed_with_run_id() -> None:
    Narrowed = Inputs[Result[Literal["nn"]]]
    i = Narrowed((Result(feature="nn", run_id="v1"),))
    assert i.feature_inputs[0].run_id == "v1"


def test_fi_roundtrip_tracks() -> None:
    orig = Inputs(("tracks",))
    dumped = orig.model_dump()
    restored = Inputs.model_validate(dumped)
    assert restored.root == orig.root


def test_fi_roundtrip_feature() -> None:
    orig = Inputs((Result(feature="nn", run_id="v1"),))
    dumped = orig.model_dump()
    restored = Inputs.model_validate(dumped)
    assert restored.root == orig.root


def test_result_column_direct():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(feature="global-ward", column="cluster")
    assert rc.feature == "global-ward"
    assert rc.column == "cluster"
    assert rc.run_id is None


def test_result_column_with_run_id():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(feature="global-kmeans", column="cluster", run_id="0.1-abc")
    assert rc.run_id == "0.1-abc"


def test_result_column_column_only():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(column="tsne_x")
    assert rc.feature == ""
    assert rc.column == "tsne_x"
    assert rc.run_id is None


def test_result_column_column_with_run_id():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(column="tsne_x", run_id="0.1-abc")
    assert rc.feature == ""
    assert rc.column == "tsne_x"
    assert rc.run_id == "0.1-abc"


def test_result_column_from_result():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(column="tsne_x").from_result(
        Result(feature="global-tsne", run_id="0.1-abc")
    )
    assert rc.feature == "global-tsne"
    assert rc.column == "tsne_x"
    assert rc.run_id == "0.1-abc"


def test_result_column_from_result_overwrites_run_id():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(column="tsne_x", run_id="old").from_result(
        Result(feature="global-tsne", run_id="new")
    )
    assert rc.feature == "global-tsne"
    assert rc.run_id == "new"


def test_result_column_from_result_use_latest():
    from mosaic.core.pipeline.types import ResultColumn

    rc = ResultColumn(column="cluster").from_result(
        Result(feature="global-ward", run_id="0.1-abc").use_latest()
    )
    assert rc.feature == "global-ward"
    assert rc.run_id is None


# --- ArtifactSpec.from_path ---


def test_from_path_npz(tmp_path: Path) -> None:
    import numpy as np

    from mosaic.core.pipeline.types import ArtifactSpec, NpzLoadSpec

    p = tmp_path / "data.npz"
    np.savez(p, templates=np.arange(12, dtype=np.float32).reshape(3, 4))
    spec = ArtifactSpec(feature="x", load=NpzLoadSpec(key="templates"))
    result = spec.from_path(p)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 4)
    assert result.dtype == np.float32


def test_from_path_npz_missing_key(tmp_path: Path) -> None:
    import numpy as np

    from mosaic.core.pipeline.types import ArtifactSpec, NpzLoadSpec

    p = tmp_path / "data.npz"
    np.savez(p, other=np.array([1.0]))
    spec = ArtifactSpec(feature="x", load=NpzLoadSpec(key="missing"))
    with pytest.raises(FileNotFoundError, match="missing"):
        spec.from_path(p)


def test_from_path_npz_1d_becomes_2d(tmp_path: Path) -> None:
    import numpy as np

    from mosaic.core.pipeline.types import ArtifactSpec, NpzLoadSpec

    p = tmp_path / "vec.npz"
    np.savez(p, vec=np.array([1.0, 2.0, 3.0]))
    spec = ArtifactSpec(feature="x", load=NpzLoadSpec(key="vec"))
    result = spec.from_path(p)
    assert result.ndim == 2
    assert result.shape == (1, 3)


def test_from_path_joblib(tmp_path: Path) -> None:
    import joblib

    from mosaic.core.pipeline.types import ArtifactSpec, JoblibLoadSpec

    p = tmp_path / "model.joblib"
    joblib.dump({"scaler": "mock_scaler", "model": "mock_model"}, p)
    spec = ArtifactSpec(feature="x", load=JoblibLoadSpec(key="scaler"))
    assert spec.from_path(p) == "mock_scaler"


def test_from_path_joblib_no_key(tmp_path: Path) -> None:
    import joblib

    from mosaic.core.pipeline.types import ArtifactSpec, JoblibLoadSpec

    p = tmp_path / "model.joblib"
    bundle = {"a": 1, "b": 2}
    joblib.dump(bundle, p)
    spec = ArtifactSpec(feature="x", load=JoblibLoadSpec())
    assert spec.from_path(p) == bundle


def test_from_path_parquet(tmp_path: Path) -> None:
    import pandas as pd

    from mosaic.core.pipeline.types import ArtifactSpec, ParquetLoadSpec

    p = tmp_path / "sizes.parquet"
    df = pd.DataFrame({"cluster": [0, 1], "count": [50, 30]})
    df.to_parquet(p, index=False)
    spec = ArtifactSpec(feature="x", load=ParquetLoadSpec(numeric_only=False))
    result = spec.from_path(p)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["cluster", "count"]
    assert len(result) == 2


def test_from_path_parquet_numeric_only(tmp_path: Path) -> None:
    import pandas as pd

    from mosaic.core.pipeline.types import ArtifactSpec, ParquetLoadSpec

    p = tmp_path / "mixed.parquet"
    df = pd.DataFrame({"name": ["a", "b"], "value": [1.0, 2.0], "score": [3, 4]})
    df.to_parquet(p, index=False)
    spec = ArtifactSpec(feature="x", load=ParquetLoadSpec())
    result = spec.from_path(p)
    assert isinstance(result, pd.DataFrame)
    assert "name" not in result.columns
    assert "value" in result.columns
    assert "score" in result.columns


def test_from_path_parquet_drop_columns(tmp_path: Path) -> None:
    import pandas as pd

    from mosaic.core.pipeline.types import ArtifactSpec, ParquetLoadSpec

    p = tmp_path / "drop.parquet"
    df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    df.to_parquet(p, index=False)
    spec = ArtifactSpec(
        feature="x",
        load=ParquetLoadSpec(drop_columns=["b"], numeric_only=False),
    )
    result = spec.from_path(p)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "c"]
