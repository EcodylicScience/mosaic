"""``pair-posedistance-pca`` applies a basis fitted elsewhere when ``model`` names one.

Without a model, ``fit`` derives the basis from the entries in scope, so the scope
is part of the run's identity. With one, ``apply`` reads the basis from the named
run and no entry of the scope is read by a fit, so the identity carries none.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from sklearn.decomposition import IncrementalPCA

from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.inventory import read_run_report
from mosaic.core.pipeline.run import compute_run_id
from mosaic.core.pipeline.types import InputStream, Result
from tests.helpers import world_track_table

N_FRAMES = 14
N_KEYPOINTS = 3
BASIS: dict[str, object] = {"n_components": 2, "pose": {"pose_n": N_KEYPOINTS}}
MODEL: dict[str, object] = {
    "feature": "pair-posedistance-pca",
    "run_id": "0.3-aaaaaaaaaa",
    "pattern": "model.joblib",
}


def _tracks(sequence: str, *, spread: float = 1.0) -> pd.DataFrame:
    """A two-individual pose table. *spread* changes how far apart the animals move."""
    rows: list[dict[str, object]] = []
    for individual in (0, 1):
        for frame in range(N_FRAMES):
            row: dict[str, object] = {
                "frame": frame,
                "time": frame / 30.0,
                "id": individual,
                "group": "g",
                "sequence": sequence,
                "X": spread * 10.0 * individual + frame,
                "Y": 5.0 * individual + 0.5 * frame,
            }
            for k in range(N_KEYPOINTS):
                row[f"poseX{k}"] = (
                    spread * 10.0 * individual + frame + k * (1 + frame % 3)
                )
                row[f"poseY{k}"] = 5.0 * individual + 0.5 * frame + 2.0 * k * spread
            rows.append(row)
    return pd.DataFrame(rows)


def _fitted(tmp_path: Path, **params: object) -> Path:
    """Fit on one sequence, save the state, and return the model file."""
    feature = PairPoseDistancePCA(params={**BASIS, **params})
    table = _tracks("fit")
    feature.fit(InputStream(lambda: iter([("g/fit", table)]), n_entries=1))
    feature.save_state(tmp_path / "fit")
    return tmp_path / "fit" / "model.joblib"


def _applying(tmp_path: Path, model: Path, **params: object) -> PairPoseDistancePCA:
    feature = PairPoseDistancePCA(params={**BASIS, "model": MODEL, **params})
    assert feature.load_state(tmp_path / "apply", {"model": model}, {})
    return feature


def _id(feature: PairPoseDistancePCA, *sequences: str) -> str:
    scope = ResolvedScope(entries={("g", sequence) for sequence in sequences})
    return compute_run_id(feature, None, None, scope)[0]


# --- identity ---------------------------------------------------------------------


def test_a_fit_stays_scope_dependent() -> None:
    feature = PairPoseDistancePCA(params=BASIS)

    assert feature.scope_dependent is True
    assert _id(feature, "a") != _id(feature, "a", "b")


def test_an_apply_of_a_model_does_not_depend_on_its_scope() -> None:
    feature = PairPoseDistancePCA(params={**BASIS, "model": MODEL})

    assert feature.scope_dependent is False
    assert _id(feature, "a") == _id(feature, "a", "b") == _id(feature, "c")


def test_the_run_id_of_an_apply_names_the_model() -> None:
    other = {**MODEL, "run_id": "0.3-bbbbbbbbbb"}

    one = PairPoseDistancePCA(params={**BASIS, "model": MODEL})
    two = PairPoseDistancePCA(params={**BASIS, "model": other})

    assert _id(one, "a") != _id(two, "a")


def test_a_fit_without_a_model_hashes_the_params_it_always_did() -> None:
    """The field is left out of the identity while unset, so no digest moves."""
    params = PairPoseDistancePCA(params=BASIS).params

    assert "model" in params.model_dump()
    assert "model" not in params.identity_dump()
    assert (
        "model"
        in PairPoseDistancePCA(params={**BASIS, "model": MODEL}).params.identity_dump()
    )


def test_the_class_still_declares_the_scope_dependence_of_its_fit() -> None:
    assert PairPoseDistancePCA.scope_dependent is True


# --- applying a model -------------------------------------------------------------


def test_an_apply_of_a_model_projects_onto_the_fitted_basis(tmp_path: Path) -> None:
    model = _fitted(tmp_path)
    feature = _applying(tmp_path, model)

    # The animals move differently here, so a basis refitted on this table would
    # not score it as the fit's basis does.
    other = _tracks("other", spread=3.0)
    scores = feature.apply(other)

    basis = joblib.load(model)["ipca"]
    refit = PairPoseDistancePCA(params=BASIS)
    refit.fit(InputStream(lambda: iter([("g/other", other)]), n_entries=1))
    refit_scores = refit.apply(other)
    components = ["PC0", "PC1"]
    assert basis.n_samples_seen_ == 2 * N_FRAMES
    assert not np.allclose(scores[components], refit_scores[components])

    fitting = PairPoseDistancePCA(params=BASIS)
    assert fitting.load_state(tmp_path / "fit", {}, {})
    assert np.allclose(scores[components], fitting.apply(other)[components])


def test_a_basis_fitted_before_the_model_param_is_a_model(tmp_path: Path) -> None:
    """A fit of version 0.2 wrote the same bundle, so an existing dataset can use it."""
    path = _fitted(tmp_path)
    bundle = joblib.load(path)
    bundle["version"] = "0.2"
    joblib.dump(bundle, path)

    scores = _applying(tmp_path, path).apply(_tracks("other"))

    assert list(scores.columns[:2]) == ["PC0", "PC1"]
    assert len(scores) == 2 * N_FRAMES


def test_a_named_model_that_was_not_found_is_refused_rather_than_refitted(
    tmp_path: Path,
) -> None:
    feature = PairPoseDistancePCA(params={**BASIS, "model": MODEL})

    with pytest.raises(FileNotFoundError, match="model"):
        _ = feature.load_state(tmp_path / "apply", {}, {})


def test_a_model_of_another_feature_is_refused_by_name(tmp_path: Path) -> None:
    """``global-kmeans`` also writes ``model.joblib``, so a wrong run loads."""
    path = tmp_path / "model.joblib"
    joblib.dump({"kmeans": object(), "feature_columns": [], "version": "0.1"}, path)
    feature = PairPoseDistancePCA(params={**BASIS, "model": MODEL})

    with pytest.raises(ValueError, match="pair-posedistance-pca"):
        _ = feature.load_state(tmp_path / "apply", {"model": path}, {})


def test_params_that_disagree_with_the_model_are_refused(tmp_path: Path) -> None:
    model = _fitted(tmp_path)

    with pytest.raises(ValueError, match="n_components"):
        _ = _applying(tmp_path, model, n_components=3)
    with pytest.raises(ValueError, match="pose"):
        _ = _applying(tmp_path, model, pose={"pose_n": 2})
    with pytest.raises(ValueError, match="include_inter"):
        _ = _applying(tmp_path, model, include_inter=False)


def test_the_model_artifact_names_the_result_of_a_fit() -> None:
    fitted = Result(
        feature="pair-posedistance-pca__from__tracks", run_id="0.3-1234567890"
    )

    artifact = PairPoseDistancePCA.ModelArtifact().from_result(fitted)
    feature = PairPoseDistancePCA(params={**BASIS, "model": artifact})

    assert (artifact.feature, artifact.run_id) == (fitted.feature, fitted.run_id)
    assert artifact.pattern == "model.joblib"
    assert feature.scope_dependent is False


# --- the explained variance ----------------------------------------------------------


class _Variance(BaseModel):
    """The report a fit writes."""

    n_components: int
    explained_variance_ratio: list[float]
    cumulative_explained_variance_ratio: list[float]
    n_samples_seen: int


def test_a_fit_reports_how_much_variance_its_components_keep(tmp_path: Path) -> None:
    model = _fitted(tmp_path)

    raw = read_run_report(model.parent, str(PairPoseDistancePCA.report_file))

    report = _Variance.model_validate(raw)
    ratio = report.explained_variance_ratio
    cumulative = report.cumulative_explained_variance_ratio
    assert report.n_components == len(ratio) == len(cumulative) == 2
    assert all(0.0 <= share <= 1.0 for share in ratio)
    assert ratio == sorted(ratio, reverse=True)
    assert cumulative[-1] == pytest.approx(sum(ratio))
    assert cumulative[-1] <= 1.0 + 1e-9
    assert report.n_samples_seen > 0
    fitted = joblib.load(model)["ipca"]
    assert ratio == pytest.approx(list(fitted.explained_variance_ratio_))


def test_the_report_is_not_part_of_the_run_identity() -> None:
    feature = PairPoseDistancePCA(params=BASIS)

    assert (
        _id(feature, "a")
        == compute_run_id(
            PairPoseDistancePCA(params=BASIS),
            None,
            None,
            ResolvedScope(entries={("g", "a")}),
        )[0]
    )


POINTS_A = np.array([[0.0, 0.0], [3.0, 4.0], [-1.5, 2.0], [6.0, -2.5]])
POINTS_B = np.array([[1.0, 1.0], [0.0, 5.0], [2.5, -3.0], [-4.0, 0.5]])


def test_the_inter_distances_are_every_pair_in_row_major_order() -> None:
    got = PairPoseDistancePCA.inter_all(POINTS_A, POINTS_B)

    expected = [float(np.hypot(*(a - b))) for a in POINTS_A for b in POINTS_B]
    assert got.shape == (POINTS_A.shape[0] * POINTS_B.shape[0],)
    assert got.tolist() == pytest.approx(expected, rel=1e-12)


def test_the_intra_distances_are_the_lower_triangle_in_row_major_order() -> None:
    n = POINTS_A.shape[0]
    tri_i, tri_j = np.tril_indices(n, k=-1)

    got = PairPoseDistancePCA.intra_lower_tri(POINTS_A, tri_i, tri_j)

    expected = [
        float(np.hypot(*(POINTS_A[i] - POINTS_A[j])))
        for i in range(n)
        for j in range(i)
    ]
    assert got.tolist() == pytest.approx(expected, rel=1e-12)


# --- the dimensions param -----------------------------------------------------------


def _lifted(sequence: str, *, depth: float = 1.0) -> pd.DataFrame:
    """The planar table with a ``Z`` column and a ``poseZ`` column per keypoint.

    Each height varies by keypoint, frame and individual, scaled by *depth*.
    """
    table = _tracks(sequence)
    heights = (table["frame"] % 4) + 0.5 * table["id"]
    table["Z"] = depth * heights
    for k in range(N_KEYPOINTS):
        table[f"poseZ{k}"] = depth * (heights + 1.5 * k * (1 + table["frame"] % 2))
    return table


def _norm(*components: float) -> float:
    return float(np.linalg.norm(components))


def _frame_vectors(
    table: pd.DataFrame, axes: tuple[str, ...], distance: Callable[..., float]
) -> np.ndarray:
    """One vector per frame of the pair (0, 1), laid out as the feature lays it out.

    A's lower triangle, B's lower triangle, then A to B in row-major order, with
    each distance computed by *distance* from the keypoint coordinates.
    """
    rows: list[list[float]] = []
    for frame in range(N_FRAMES):
        at_frame = table[table["frame"] == frame].set_index("id")
        points = {
            individual: np.array(
                [
                    [at_frame.loc[individual, f"pose{axis}{k}"] for axis in axes]
                    for k in range(N_KEYPOINTS)
                ],
                dtype=float,
            )
            for individual in (0, 1)
        }
        lower = [(i, j) for i in range(N_KEYPOINTS) for j in range(i)]
        rows.append(
            [float(distance(*(points[0][i] - points[0][j]))) for i, j in lower]
            + [float(distance(*(points[1][i] - points[1][j]))) for i, j in lower]
            + [
                float(distance(*(points[0][i] - points[1][j])))
                for i in range(N_KEYPOINTS)
                for j in range(N_KEYPOINTS)
            ]
        )
    return np.array(rows)


def _fitted_mean(tmp_path: Path, table: pd.DataFrame, **params: object) -> np.ndarray:
    """The mean vector of the basis fitted on one pair, one perspective."""
    feature = PairPoseDistancePCA(
        params={**BASIS, "duplicate_perspective": False, **params}
    )
    feature.fit(InputStream(lambda: iter([("g/fit", table)]), n_entries=1))
    feature.save_state(tmp_path)
    return np.asarray(joblib.load(tmp_path / "model.joblib")["ipca"].mean_)


def _scored(table: pd.DataFrame, **params: object) -> pd.DataFrame:
    """Fit on *table* and score it."""
    feature = PairPoseDistancePCA(params={**BASIS, **params})
    feature.fit(InputStream(lambda: iter([("g/fit", table)]), n_entries=1))
    return feature.apply(table)


def test_the_dimensions_leave_the_identity_while_they_are_two() -> None:
    default = PairPoseDistancePCA(params=BASIS).params
    planar = PairPoseDistancePCA(params={**BASIS, "dimensions": 2}).params
    spatial = PairPoseDistancePCA(params={**BASIS, "dimensions": 3}).params

    assert "dimensions" not in default.identity_dump()
    assert "dimensions" not in planar.identity_dump()
    assert spatial.identity_dump()["dimensions"] == 3
    assert all(
        params.model_dump()["dimensions"] == expected
        for params, expected in ((default, 2), (planar, 2), (spatial, 3))
    )


def test_the_planar_vectors_are_the_distances_in_the_plane(tmp_path: Path) -> None:
    table = _tracks("fit")

    mean = _fitted_mean(tmp_path, table)

    expected = _frame_vectors(table, ("X", "Y"), np.hypot).mean(axis=0)
    assert mean.shape == expected.shape
    assert mean.tolist() == pytest.approx(expected.tolist(), rel=1e-6)


def test_the_spatial_vectors_are_the_distances_in_three_axes(tmp_path: Path) -> None:
    table = _lifted("fit")

    mean = _fitted_mean(tmp_path, table, dimensions=3)

    expected = _frame_vectors(table, ("X", "Y", "Z"), _norm).mean(axis=0)
    planar = _frame_vectors(table, ("X", "Y"), np.hypot).mean(axis=0)
    assert mean.shape == expected.shape
    assert mean.tolist() == pytest.approx(expected.tolist(), rel=1e-6)
    assert not np.allclose(mean, planar)


def test_a_flat_spatial_table_scores_as_the_planar_table_does() -> None:
    flat = _lifted("fit", depth=0.0)

    spatial = _scored(flat, dimensions=3)
    planar = _scored(_tracks("fit"), dimensions=2)

    pd.testing.assert_frame_equal(spatial, planar, check_exact=True)


def test_two_dimensions_score_as_an_absent_dimensions_does() -> None:
    table = _tracks("fit")

    pd.testing.assert_frame_equal(
        _scored(table, dimensions=2), _scored(table), check_exact=True
    )


# --- a table whose axes disagree with the dimensions ----------------------------------


def _fitted_feature(table: pd.DataFrame, dimensions: int) -> PairPoseDistancePCA:
    feature = PairPoseDistancePCA(params={**BASIS, "dimensions": dimensions})
    feature.fit(InputStream(lambda: iter([("g/fit", table)]), n_entries=1))
    return feature


def test_a_planar_apply_refuses_a_world_frame_table() -> None:
    feature = _fitted_feature(_tracks("fit"), 2)

    with pytest.raises(ValueError, match="dimensions: 3"):
        feature.apply(world_track_table("g", "world", n_ids=2))


def test_a_spatial_apply_refuses_a_planar_table() -> None:
    feature = _fitted_feature(_lifted("fit"), 3)

    with pytest.raises(ValueError, match=r"(?s)\bZ\b.*dimensions: 2"):
        feature.apply(_tracks("apply"))


def test_a_spatial_apply_names_the_keypoint_column_a_world_table_lacks() -> None:
    feature = _fitted_feature(_lifted("fit"), 3)
    world = world_track_table("g", "world", n_ids=2).drop(columns="poseZ1")

    with pytest.raises(ValueError, match="poseZ1"):
        feature.apply(world)


def test_a_planar_fit_refuses_a_world_frame_table_before_it_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted: list[object] = []

    def record(batch: object, *_: object, **__: object) -> None:
        fitted.append(batch)

    monkeypatch.setattr(IncrementalPCA, "partial_fit", record)
    feature = PairPoseDistancePCA(params=BASIS)
    world = world_track_table("g", "world", n_ids=2)

    with pytest.raises(ValueError, match="dimensions: 3"):
        feature.fit(InputStream(lambda: iter([("g/world", world)]), n_entries=1))

    assert fitted == []


def test_a_spatial_fit_and_apply_score_a_world_frame_table() -> None:
    world = world_track_table("g", "world", n_ids=2)

    scored = _scored(world, dimensions=3)

    assert not scored.empty
    assert [c for c in scored.columns if c.startswith("PC")] == ["PC0", "PC1"]
