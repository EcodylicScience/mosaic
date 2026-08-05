"""The identity models' pre-fitted ``model`` artifact, and their fit scope.

Item 1.4: a feature whose ``fit()`` reads its input stream has the scope as its
training set, so the scope must be in its identifier (P2f). Flipping the flag
alone would make every new apply scope retrain a network, so each feature first
gains a params-level pre-fitted ``model`` reference -- after which fit and apply
are two runs with two identifiers, and only the training run carries a scope.

The exported weights are a torch ``.pth`` and an ``ArtifactSpec`` loads only
npz / parquet / joblib, so the referencable artifact is a joblib sidecar naming
the checkpoint beside it. These tests pin that indirection, which is the part
that can silently load the wrong file.

``torch`` is an optional extra and is not installed in CI, so the network
classes are replaced with a recording stand-in. Every feature imports its own
lazily inside ``load_state`` / ``save_state``, so patching the module attribute
is enough.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import joblib
import pytest

from mosaic.behavior.feature_library.dinov2_temporal_identity_model import (
    DinoV2TemporalIdentityArtifact,
)
from mosaic.behavior.feature_library.identity_embedding_model import (
    EmbeddingIdentityArtifact,
)
from mosaic.behavior.feature_library.identity_model import (
    ClassifierIdentityArtifact,
)
from mosaic.cli._features import build_feature
from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.run import compute_run_id

# The three identity models all consume egocentric crops, never tracks.
CROP_INPUTS: list[object] = [{"feature": "egocentric-crop"}]

IDENTITY_SLUGS = (
    "global-identity-model",
    "global-identity-embedding",
    "global-identity-dinov2-temporal",
)


@dataclass(frozen=True)
class ArtifactCase:
    """One feature's sidecar wiring.

    Attributes:
        slug: Registered feature slug.
        module: ``model_library`` module holding the network class.
        network_attr: Name of the network class within that module.
        bundle_name: Fixed filename of the joblib sidecar.
        weights_stem: Default ``weights_name`` param.
        pattern_of: Reads the artifact class's declared glob pattern. A
            callable rather than the class itself, because the artifact
            classes have no common annotation that survives strict variance
            checking.
    """

    slug: str
    module: str
    network_attr: str
    bundle_name: str
    weights_stem: str
    pattern_of: Callable[[], str]


ARTIFACT_CASES = (
    ArtifactCase(
        slug="global-identity-model",
        module="mosaic.behavior.model_library.identity_classifier",
        network_attr="ClassifierIdentityNetwork",
        bundle_name="identity_classifier_model.joblib",
        weights_stem="identity_classifier",
        pattern_of=lambda: ClassifierIdentityArtifact().pattern,
    ),
    ArtifactCase(
        slug="global-identity-embedding",
        module="mosaic.behavior.model_library.identity_embedding",
        network_attr="EmbeddingIdentityNetwork",
        bundle_name="identity_embedding_model.joblib",
        weights_stem="identity_embedding",
        pattern_of=lambda: EmbeddingIdentityArtifact().pattern,
    ),
    ArtifactCase(
        slug="global-identity-dinov2-temporal",
        module="mosaic.behavior.model_library.dinov2_temporal_identity",
        network_attr="DinoV2TemporalNetwork",
        bundle_name="dinov2_temporal_identity_model.joblib",
        weights_stem="dinov2_temporal_identity",
        pattern_of=lambda: DinoV2TemporalIdentityArtifact().pattern,
    ),
)


class RecordingNetwork:
    """Stand-in for the torch-backed identity networks.

    Records every checkpoint path it is asked to load, so a test can assert
    *which* file the sidecar resolved to without importing torch.

    ``export_checkpoint`` takes ``class_labels`` because the classifier passes
    it -- the head scores identities by index, so class order is the only link
    back to the animals. The embedding models carry their names in the sidecar
    instead and pass nothing, which is why it is optional here.
    """

    loaded: ClassVar[list[Path]] = []

    @classmethod
    def from_checkpoint(cls, path: Path) -> RecordingNetwork:
        cls.loaded.append(path)
        return cls()

    def export_checkpoint(
        self, path: Path, *, class_labels: list[str] | None = None
    ) -> Path:
        _ = class_labels
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"fake-checkpoint")
        return path


@pytest.fixture(autouse=True)
def clear_recorder() -> Iterator[None]:
    """The recorder is class-level state; no test may see another's loads."""
    RecordingNetwork.loaded.clear()
    yield
    RecordingNetwork.loaded.clear()


def patch_network(case: ArtifactCase, monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the case's network class for the recording stand-in."""
    module = importlib.import_module(case.module)
    monkeypatch.setattr(module, case.network_attr, RecordingNetwork)


def model_ref(case: ArtifactCase) -> dict[str, object]:
    """The params payload pinning a pre-fitted model, as a user would write it."""
    return {
        "feature": case.slug,
        "pattern": case.bundle_name,
        "load": {"kind": "joblib"},
    }


def write_bundle(run_root: Path, case: ArtifactCase, weights: str) -> Path:
    """Write a sidecar naming *weights*, plus the checkpoint it names."""
    run_root.mkdir(parents=True, exist_ok=True)
    _ = (run_root / weights).write_bytes(b"fake-checkpoint")
    bundle_path = run_root / case.bundle_name
    joblib.dump(
        {"weights": weights, "identity_names": ["a", "b"], "version": "0.1"},
        bundle_path,
    )
    return bundle_path


# --- Construction ---------------------------------------------------------


@pytest.mark.parametrize("slug", IDENTITY_SLUGS)
def test_identity_features_declare_scope_dependent(slug: str) -> None:
    """All three fit from the stream, so all three carry their scope (P2f)."""
    assert build_feature(slug, CROP_INPUTS, None).scope_dependent is True


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_pre_fitted_model_defaults_to_none(case: ArtifactCase) -> None:
    """Adding ``model`` must not make the params class refuse a bare default.

    ``GlobalModelParams`` enforces exactly-one-of ``templates``/``model`` and so
    raises on default construction. These features use plain ``Params`` for that
    reason, and the golden corpus builds every case with defaults.
    """
    assert build_feature(case.slug, CROP_INPUTS, None).params.model is None


# --- Identity -------------------------------------------------------------


@pytest.mark.parametrize("slug", IDENTITY_SLUGS)
def test_scope_moves_the_identifier(slug: str) -> None:
    """Two training sets must not share one identifier.

    The golden corpus pins the literals; this pins the relation, which is what
    actually matters and what would survive a deliberate regeneration.
    """
    feature = build_feature(slug, CROP_INPUTS, None)
    narrow, _ = compute_run_id(feature, None, None, Scope(entries={("", "seq_a")}))
    wide, _ = compute_run_id(
        feature, None, None, Scope(entries={("", "seq_a"), ("", "seq_b")})
    )
    assert narrow != wide


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_pinning_a_model_moves_the_identifier(case: ArtifactCase) -> None:
    """An inference run's training set is its ``model`` reference, so it is hashed."""
    scope = Scope(entries={("", "seq_a")})
    plain, _ = compute_run_id(
        build_feature(case.slug, CROP_INPUTS, None), None, None, scope
    )
    pinned, _ = compute_run_id(
        build_feature(case.slug, CROP_INPUTS, {"model": model_ref(case)}),
        None,
        None,
        scope,
    )
    assert plain != pinned


# --- The sidecar indirection ----------------------------------------------


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_pattern_resolves_past_the_sibling_joblibs(
    case: ArtifactCase, tmp_path: Path
) -> None:
    """Dependency resolution globs the pattern and takes ``files[0]``.

    A run root also holds ``identity_names.joblib`` and
    ``training_history.joblib``, both of which sort ahead of a derived
    ``*.joblib``. An auto-derived pattern would load the wrong object and the
    branch would look like it worked.
    """
    for name in ("identity_names.joblib", "training_history.joblib"):
        _ = (tmp_path / name).write_bytes(b"")
    _ = (tmp_path / case.bundle_name).write_bytes(b"")

    assert case.pattern_of() == case.bundle_name
    assert sorted(tmp_path.glob(case.pattern_of())) == [tmp_path / case.bundle_name]


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_load_state_resolves_the_weights_beside_the_bundle(
    case: ArtifactCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Branch 2 loads the checkpoint the sidecar names, from the sidecar's directory."""
    patch_network(case, monkeypatch)
    trained = tmp_path / "trained"
    bundle_path = write_bundle(trained, case, f"{case.weights_stem}.pth")

    feature = build_feature(case.slug, CROP_INPUTS, {"model": model_ref(case)})
    ready = feature.load_state(tmp_path / "empty-run", {"model": bundle_path}, {})

    assert ready is True
    assert RecordingNetwork.loaded == [trained / f"{case.weights_stem}.pth"]


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_load_state_ignores_this_runs_weights_name(
    case: ArtifactCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``weights_name`` is a params knob; the consuming run need not share it."""
    patch_network(case, monkeypatch)
    trained = tmp_path / "trained"
    bundle_path = write_bundle(trained, case, "named_by_the_training_run.pth")

    feature = build_feature(
        case.slug,
        CROP_INPUTS,
        {"model": model_ref(case), "weights_name": "something_else"},
    )
    ready = feature.load_state(tmp_path / "empty-run", {"model": bundle_path}, {})

    assert ready is True
    assert RecordingNetwork.loaded == [trained / "named_by_the_training_run.pth"]


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_load_state_falls_through_when_the_artifact_did_not_resolve(
    case: ArtifactCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolved reference must fit, not crash on a missing dict key."""
    patch_network(case, monkeypatch)
    feature = build_feature(case.slug, CROP_INPUTS, {"model": model_ref(case)})

    assert feature.load_state(tmp_path / "empty-run", {}, {}) is False
    assert RecordingNetwork.loaded == []


@pytest.mark.parametrize("case", ARTIFACT_CASES, ids=lambda c: c.slug)
def test_save_state_output_is_loadable_as_the_next_runs_model(
    case: ArtifactCase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The round trip the artifact exists for: run N's output pins run N+1."""
    patch_network(case, monkeypatch)
    run_root = tmp_path / "run-a"

    trainer = build_feature(case.slug, CROP_INPUTS, None)
    # The fitted network is private state with no public seam for injecting it,
    # and ``fit`` needs torch plus real crops. Assigned directly rather than
    # suppressed: the missing seam is a real design signal, worth seeing.
    trainer._network = RecordingNetwork()
    trainer.save_state(run_root)

    bundle_path = run_root / case.bundle_name
    assert bundle_path.exists(), "save_state must write the referencable sidecar"

    consumer = build_feature(case.slug, CROP_INPUTS, {"model": model_ref(case)})
    ready = consumer.load_state(tmp_path / "run-b", {"model": bundle_path}, {})

    assert ready is True
    assert RecordingNetwork.loaded == [run_root / f"{case.weights_stem}.pth"]
