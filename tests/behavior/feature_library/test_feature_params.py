"""Tests for Pydantic feature parameter models."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import ValidationError

from mosaic.core.pipeline.types import Result, TemplatesRef
from tests.helpers import run_id_digest


# --- FERAL's run identity ---


def test_feral_infer_batch_size_excluded_from_run_id() -> None:
    """The motivating case: FeralFeature.infer_batch_size is hash-excluded."""
    from mosaic.behavior.feature_library.feral_feature import FeralFeature

    base = {"feral_code_dir": "/tmp/feral", "model_dir": "/tmp/model"}
    p4 = FeralFeature.Params.from_overrides({**base, "infer_batch_size": 4})
    p8 = FeralFeature.Params.from_overrides({**base, "infer_batch_size": 8})
    pcs = FeralFeature.Params.from_overrides(
        {**base, "infer_batch_size": 4, "chunk_shift": 16}
    )

    # Persisted (provenance) but invisible to the run_id hash.
    assert p8.model_dump()["infer_batch_size"] == 8
    assert "infer_batch_size" not in p4.identity_dump()
    assert run_id_digest(p4) == run_id_digest(p8)
    # A real hyperparameter still moves the cache key.
    assert run_id_digest(p4) != run_id_digest(pcs)


def test_feral_device_excluded_from_run_id() -> None:
    """A model's weights do not change with the card that runs them.

    Keeping `device` in identity meant reloading finished predictions on a
    machine without a GPU minted a new run_id and asked for a full recompute --
    which is exactly the "just load the results and analyse them" path.
    """
    from mosaic.behavior.feature_library.feral_feature import FeralFeature

    base = {"model_dir": "models/feral/0.1-abc"}
    gpu = FeralFeature.Params.from_overrides({**base, "device": "cuda"})
    cpu = FeralFeature.Params.from_overrides({**base, "device": "cpu"})

    assert cpu.model_dump()["device"] == "cpu"
    assert "device" not in gpu.identity_dump()
    assert run_id_digest(gpu) == run_id_digest(cpu)
    # Precision, unlike hardware, does change the output -- and stays hashed.
    assert run_id_digest(gpu) != run_id_digest(
        FeralFeature.Params.from_overrides({**base, "inference_autocast": True})
    )


def test_feral_inference_identity_is_pinned() -> None:
    """The literal identifier the konstanz_trophallaxis analysis depends on.

    FERAL cannot go in tests/data/identity_golden.json: `build_feature("feral",
    ...)` raises typer.Exit because the feature does not read from tracks by
    default, and `FeralFeature.__init__` raises ImportError without the optional
    `feral` extra, which CI does not install. A golden case would be red or
    permanently skipped -- coverage in name only.

    So pin the same thing one layer down, where no optional dependency is
    involved: `Params.from_overrides` is a classmethod that never touches
    `__init__`. This is the exact parameter set of
    `troph_feral_infer_v2.ipynb` over the konstanz_trophallaxis dataset, whose
    FERAL run holds 95 sequences of V-JEPA2 inference. If this digest moves, that
    notebook stops finding its results and asks to recompute them; the notebook
    asserts the same value, so a mismatch should surface here first.

    **The version moved once, deliberately, and the digest did not.** `feral` went
    to `0.2` when its identity columns were renamed to `id1` / `id2` /
    `perspective`, and the crop feature it reads went to `0.3` for the same reason.
    So the run this pins is not reachable by name any more: the artifacts are still
    on disk under `0.1-a3cefdc108`, and a fresh run asks to recompute all 95
    sequences under `0.2-a3cefdc108`. The parameter set itself is untouched, which
    is what the unchanged digest says -- so if the digest ever moves, that is a
    different and unintended thing.
    """
    from mosaic.behavior.feature_library.feral_feature import FeralFeature
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.core.pipeline.types import Result

    crop = (
        "interaction-crop-pipeline__from__trajectory-smooth__from__tracks"
        "+pair-interaction-filter__from__trajectory-smooth__from__tracks"
    )
    inputs = FeralFeature.Inputs((Result(feature=crop, run_id="0.2-3fcc9dfab9"),))
    params = FeralFeature.Params.from_overrides(
        {
            "feral_code_dir": None,
            "model_name": "facebook/vjepa2-vitl-fpc32-256-diving48",
            "predict_per_item": 64,
            "chunk_length": 64,
            "chunk_shift": 16,
            "chunk_step": 1,
            "resize_to": 256,
            "device": "cuda",
            "model_dir": "models/feral/0.1-33340cc70f",
            "infer_batch_size": 16,
            "inference_autocast": False,
        }
    )

    # scope_dependent is False, so compute_run_id adds no _scope_entries term.
    assert FeralFeature.scope_dependent is False
    digest = hash_params(
        {
            "_params": params.identity_dump(),
            "_inputs": inputs.model_dump(),
            "_frame_range": [None, None],
        }
    )
    assert f"{FeralFeature.version}-{digest}" == "0.2-a3cefdc108"


def test_pair_filter_on_params() -> None:
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering

    inputs = GlobalWardClustering.Inputs((Result(feature="pair-wavelet"),))
    gw = GlobalWardClustering(
        inputs=inputs,
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
        },
    )
    assert gw.params.pair_filter is None
    gw2 = GlobalWardClustering(
        inputs=inputs,
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
            "pair_filter": {"feature": "nearest-neighbor"},
        },
    )
    assert gw2.params.pair_filter.feature == "nearest-neighbor"


# --- Validation behavior ---


def test_approach_avoidance_literal_validation() -> None:
    from mosaic.behavior.feature_library.approach_avoidance import ApproachAvoidance

    with pytest.raises(ValidationError):
        ApproachAvoidance.Params(velocity_units="invalid")


def test_pair_egocentric_none_override_rejected() -> None:
    from mosaic.behavior.feature_library.pair_egocentric import PairEgocentricFeatures

    with pytest.raises(ValidationError):
        PairEgocentricFeatures.Params.from_overrides({"neck_idx": None})


def test_temporal_stacking_pool_stats_normalization() -> None:
    from mosaic.behavior.feature_library.temporal_stacking import (
        TemporalStackingFeature,
    )

    p = TemporalStackingFeature.Params.from_overrides({"pool_stats": "MEAN"})
    assert p.pool_stats == ("mean",)
    p2 = TemporalStackingFeature.Params.from_overrides({"pool_stats": ["Mean", "STD"]})
    assert p2.pool_stats == ("mean", "std")


# --- Deep merge on nested spec models ---


def test_global_ward_partial_artifact_override() -> None:
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering

    p = GlobalWardClustering.Params.from_overrides({"templates": {"feature": "other"}})
    assert p.templates is not None
    assert p.templates.feature == "other"
    # The declared type names the file, so a partial override that does not
    # mention one still resolves an artifact rather than a glob.
    assert p.templates.pattern == "templates.parquet"


# --- The templates edge names its file ----------------------------------------


_TEMPLATES_CONSUMERS: tuple[tuple[str, bool], ...] = (
    ("global-scaler", True),
    ("global-tsne", True),
    ("global-kmeans", True),
    ("global-ward", True),
    ("xgboost", False),
    ("lightning-action", False),
)
"""Every feature whose training set arrives as a templates artifact, and whether
its matrix is filtered to numeric columns on the way in."""


def _recipe_templates_ref(consumer: str) -> TemplatesRef:
    """The ``templates`` reference a recipe builds for *consumer*.

    Spelled the way ``resolve_step_spec`` spells it -- feature and run only. The
    graph payload carries a pattern solely when the recipe wrote one, and never
    carries a load spec at all, so this is the shape the declared type has to
    complete on its own.
    """
    from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering
    from mosaic.behavior.feature_library.global_scaler import GlobalScaler
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering
    from mosaic.behavior.feature_library.lightning_action_feature import (
        LightningActionFeature,
    )
    from mosaic.behavior.feature_library.xgboost_feature import XgboostFeature

    ref: dict[str, object] = {"templates": {"feature": "up", "run_id": "r1"}}
    labeled: dict[str, object] = {**ref, "default_class": 0}
    builders: dict[str, Callable[[], TemplatesRef | None]] = {
        "global-scaler": lambda: GlobalScaler.Params.from_overrides(ref).templates,
        "global-tsne": lambda: GlobalTSNE.Params.from_overrides(ref).templates,
        "global-kmeans": lambda: (
            GlobalKMeansClustering.Params.from_overrides(ref).templates
        ),
        "global-ward": lambda: (
            GlobalWardClustering.Params.from_overrides(ref).templates
        ),
        "xgboost": lambda: XgboostFeature.Params.from_overrides(labeled).templates,
        "lightning-action": lambda: (
            LightningActionFeature.Params.from_overrides(labeled).templates
        ),
    }
    templates = builders[consumer]()
    assert templates is not None
    return templates


@pytest.mark.parametrize(
    ("consumer", "numeric_only"),
    _TEMPLATES_CONSUMERS,
    ids=[consumer for consumer, _ in _TEMPLATES_CONSUMERS],
)
def test_a_templates_reference_names_the_file_it_reads(
    consumer: str, numeric_only: bool
) -> None:
    """No consumer of a templates matrix resolves it by glob.

    A producer's run root holds one per-entry output parquet per sequence beside
    its named artifacts, so the derived ``*.parquet`` took whichever sorted first
    -- a per-entry table, read downstream as the training set with nothing
    raising.

    ``numeric_only`` rides along because it is the same declaration: a labeled
    matrix carries a string ``split`` column that both its consumers require by
    name, and the generic default filtered it out before ``fit`` ever saw it.
    """
    templates = _recipe_templates_ref(consumer)

    assert templates.pattern == "templates.parquet"
    assert templates.load.numeric_only is numeric_only


# --- Mutable default isolation ---


def test_nn_delta_bins_mutable_default_isolation() -> None:
    from mosaic.behavior.feature_library.nn_delta_bins import NearestNeighborDeltaBins

    p1 = NearestNeighborDeltaBins.Params()
    p2 = NearestNeighborDeltaBins.Params()
    assert p1.category_specs is not p2.category_specs


def test_orientation_relative_mutable_default_isolation() -> None:
    from mosaic.behavior.feature_library.orientation_relative import (
        OrientationRelativeFeature,
    )

    p1 = OrientationRelativeFeature.Params()
    p2 = OrientationRelativeFeature.Params()
    assert p1.quantiles is not p2.quantiles


# --- Result-based inputs (WardAssign, TemporalStacking, GlobalWard, GlobalKMeans) ---


def test_temporal_stacking_requires_inputs() -> None:
    """TemporalStacking constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.temporal_stacking import (
        TemporalStackingFeature,
    )

    with pytest.raises(TypeError):
        TemporalStackingFeature()


def test_global_ward_requires_inputs() -> None:
    """GlobalWard constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering

    with pytest.raises(TypeError):
        GlobalWardClustering(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                }
            }
        )


def test_global_ward_accepts_empty_and_result_inputs() -> None:
    """GlobalWard accepts both empty and Result-based inputs (_require='any')."""
    from mosaic.behavior.feature_library.global_ward import GlobalWardClustering

    gw = GlobalWardClustering(
        inputs=GlobalWardClustering.Inputs(()),
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            }
        },
    )
    assert len(gw.inputs.root) == 0
    assert gw.inputs.feature_inputs == ()

    gw2 = GlobalWardClustering(
        inputs=GlobalWardClustering.Inputs((Result(feature="pair-wavelet"),)),
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            }
        },
    )
    assert len(gw2.inputs.root) == 1


def test_global_kmeans_requires_inputs() -> None:
    """GlobalKMeans constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering

    with pytest.raises(TypeError):
        GlobalKMeansClustering(
            params={
                "templates": {
                    "feature": "extract-templates",
                    "pattern": "templates.parquet",
                }
            }
        )


def test_global_kmeans_accepts_empty_and_result_inputs() -> None:
    """GlobalKMeans accepts both empty and Result-based inputs (_require='any')."""
    from mosaic.behavior.feature_library.global_kmeans import GlobalKMeansClustering

    gk = GlobalKMeansClustering(
        inputs=GlobalKMeansClustering.Inputs(()),
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            }
        },
    )
    assert len(gk.inputs.root) == 0
    assert gk.inputs.feature_inputs == ()

    gk2 = GlobalKMeansClustering(
        inputs=GlobalKMeansClustering.Inputs((Result(feature="pair-wavelet"),)),
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            }
        },
    )
    assert len(gk2.inputs.root) == 1


# --- GlobalTSNE Result-based inputs ---


def test_global_tsne_requires_inputs() -> None:
    """GlobalTSNE constructor requires explicit inputs (no default)."""
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    with pytest.raises(TypeError):
        GlobalTSNE()


def test_global_tsne_result_inputs() -> None:
    """GlobalTSNE accepts Result-based inputs and computes correct storage_suffix."""
    from mosaic.behavior.feature_library.global_tsne import GlobalTSNE

    inputs = GlobalTSNE.Inputs(
        (
            Result(feature="pair-wavelet", run_id="0.1-abc"),
            Result(feature="pair-ego-wavelet"),
        )
    )
    gt = GlobalTSNE(
        inputs=inputs,
        params={
            "templates": {
                "feature": "extract-templates",
                "pattern": "templates.parquet",
            },
        },
    )
    assert gt.inputs.feature_inputs[0].feature == "pair-wavelet"
    assert gt.inputs.feature_inputs[0].run_id == "0.1-abc"
    assert gt.inputs.storage_suffix() == "pair-wavelet+pair-ego-wavelet"


# --- GlobalModelParams: the exclusive source, across a round trip -------------


@pytest.mark.parametrize("source", ["templates", "model"])
def test_a_global_model_params_file_reads_back(source: str) -> None:
    """What ``run_feature`` writes, ``reconcile`` has to be able to rebuild from.

    ``model_dump`` emits both ``templates`` and ``model``, the unused one as an
    explicit null, so a validator counting key *presence* rejected every params
    file any global model feature ever wrote. ``reconcile`` rebuilds a run's
    feature from exactly that file, so it could confirm none of them.
    """
    from mosaic.behavior.feature_library import GlobalTSNE

    params = GlobalTSNE.Params.from_overrides({source: {"feature": "upstream"}})
    dumped = params.model_dump()
    assert "templates" in dumped and "model" in dumped, "both keys are written"

    restored = GlobalTSNE.Params.from_overrides(dumped)

    assert restored.identity_dump() == params.identity_dump()


def test_a_global_model_params_still_needs_exactly_one_source() -> None:
    """The rule itself is unchanged: neither and both are still refused."""
    from mosaic.behavior.feature_library import GlobalTSNE

    with pytest.raises(ValidationError):
        _ = GlobalTSNE.Params.from_overrides({})
    with pytest.raises(ValidationError):
        _ = GlobalTSNE.Params.from_overrides(
            {"templates": {"feature": "t"}, "model": {"feature": "m"}}
        )
