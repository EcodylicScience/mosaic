"""``pair-posedistance-pca`` applies a basis fitted elsewhere when ``model`` names one.

Without a model, ``fit`` derives the basis from the entries in scope, so the scope
is part of the run's identity. With one, ``apply`` reads the basis from the named
run and no entry of the scope is read by a fit, so the identity carries none.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.pairposedistancepca import PairPoseDistancePCA
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.run import compute_run_id
from mosaic.core.pipeline.types import InputStream, Result

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
