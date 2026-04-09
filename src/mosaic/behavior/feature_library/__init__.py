"""
Feature library for behavior datasets.

This module provides a collection of features for behavioral analysis.
Features are automatically registered on import via the @register_feature decorator.

All features are automatically loaded when the feature_library is imported,
making them available in the global FEATURES registry.

Usage
-----
>>> from mosaic.behavior.feature_library import Inputs, Result
>>> from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
>>>
>>> # Track-only feature (default inputs)
>>> feat = SpeedAngvel()
>>> dataset.run_feature(feat)
>>>
>>> # Feature consuming another feature's output
>>> feat = SpeedAngvel(inputs=Inputs((Result(feature="nn"),)))
>>> dataset.run_feature(feat)
>>>
>>> # List all registered features
>>> from mosaic.behavior.feature_library.registry import FEATURES
>>> print(list(FEATURES.keys()))
"""

# Import all submodules to trigger @register_feature auto-registration
from mosaic.core.pipeline.types import (
    COLUMNS,
    ArtifactSpec,
    Feature,
    GlobalModelParams,
    GroundTruthLabelsSource,
    Inputs,
    InputsLike,
    Result,
    ResultColumn,
    TrackInput,
)

from . import (
    arhmm,
    approach_avoidance,
    body_scale,
    extract_labeled_templates,
    extract_templates,
    ffgroups,
    ffgroups_metrics,
    global_kmeans,
    global_scaler,
    global_tsne,
    global_ward,
    helpers,
    id_tag_columns,
    kpms,
    nearestneighbor,
    nn_delta_bins,
    nn_delta_response,
    orientation_relative,
    pair_egocentric,
    pair_interaction_filter,
    pair_position,
    pair_wavelet,
    pairposedistancepca,
    speed_angvel,
    temporal_stacking,
    trajectory_smooth,
    xgboost_feature,
)
from .arhmm import ArHmmFeature
from .approach_avoidance import ApproachAvoidance
from .body_scale import BodyScaleFeature
from .extract_labeled_templates import ExtractLabeledTemplates
from .extract_templates import ExtractTemplates
from .ffgroups import FFGroups
from .ffgroups_metrics import FFGroupsMetrics
from .global_kmeans import GlobalKMeansClustering
from .global_scaler import GlobalScaler
from .global_tsne import GlobalTSNE
from .global_ward import GlobalWardClustering
from .id_tag_columns import IdTagColumns
from .kpms import KpmsFeature
from .nearestneighbor import NearestNeighbor
from .nn_delta_bins import NearestNeighborDeltaBins
from .nn_delta_response import NearestNeighborDelta
from .orientation_relative import OrientationRelativeFeature
from .pair_egocentric import PairEgocentricFeatures
from .pair_interaction_filter import PairInteractionFilter
from .pair_position import PairPositionFeatures
from .pair_wavelet import PairWavelet
from .pairposedistancepca import PairPoseDistancePCA
from .registry import FEATURES, register_feature
from .speed_angvel import SpeedAngvel
from .temporal_stacking import TemporalStackingFeature
from .trajectory_smooth import TrajectorySmooth
from .xgboost_feature import XgboostFeature

# Optional: movement library integration (requires `movement` package)
try:
    from .movement import MovementFilterInterpolate, MovementSmooth
except ImportError:
    pass

# Lightning-action temporal classifier (requires `lightning-action` package)
try:
    from . import lightning_action_feature
    from .lightning_action_feature import LightningActionFeature
except ImportError:
    pass

# FERAL video behavior classifier (requires feral_code_dir at runtime)
from . import feral_feature
from .feral_feature import FeralFeature, FeralTrainingConfig

# Note: Templates are not imported (they're just examples)
# from . import feature_template__per_sequence
# from . import feature_template__global


__all__ = [
    # Registry
    "Feature",
    "FEATURES",
    "register_feature",
    # Types and params
    "ArtifactSpec",
    "COLUMNS",
    "GroundTruthLabelsSource",
    "Inputs",
    "InputsLike",
    "Result",
    "ResultColumn",
    "TrackInput",
    # Helpers
    "helpers",
    # Feature classes
    "ArHmmFeature",
    "ApproachAvoidance",
    "BodyScaleFeature",
    "ExtractLabeledTemplates",
    "ExtractTemplates",
    "FFGroups",
    "FFGroupsMetrics",
    "GlobalKMeansClustering",
    "GlobalModelParams",
    "GlobalScaler",
    "GlobalTSNE",
    "GlobalWardClustering",
    "IdTagColumns",
    "KpmsFeature",
    "NearestNeighbor",
    "NearestNeighborDelta",
    "NearestNeighborDeltaBins",
    "OrientationRelativeFeature",
    "PairEgocentricFeatures",
    "PairInteractionFilter",
    "PairPositionFeatures",
    "PairPoseDistancePCA",
    "PairWavelet",
    "SpeedAngvel",
    "TemporalStackingFeature",
    "TrajectorySmooth",
    "XgboostFeature",
    "FeralFeature",
    "FeralTrainingConfig",
    "LightningActionFeature",
    "lightning_action_feature",
    # Submodules
    "arhmm",
    "approach_avoidance",
    "body_scale",
    "extract_labeled_templates",
    "extract_templates",
    "ffgroups",
    "ffgroups_metrics",
    "global_kmeans",
    "global_scaler",
    "global_tsne",
    "global_ward",
    "id_tag_columns",
    "kpms",
    "nearestneighbor",
    "nn_delta_bins",
    "nn_delta_response",
    "orientation_relative",
    "pair_egocentric",
    "pair_interaction_filter",
    "pair_position",
    "pair_wavelet",
    "pairposedistancepca",
    "speed_angvel",
    "temporal_stacking",
    "trajectory_smooth",
    "xgboost_feature",
]
