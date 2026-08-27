"""Every parameter field declares itself, and every declared fact reaches the schema.

Two states are separable at the schema and each has its own ratchet. A field with
no ``Declared`` publishes no description key and sits in
``ALLOWED_MISSING_DESCRIPTION``, bounded by ``ALLOWLIST_CEILING``. A field
declared ``Declared(NEEDS_DESCRIPTION)`` publishes an empty one. Declaring a
field and describing it are separate pieces of work, so ``UNDOCUMENTED_CEILING``
bounds the two states together: moving a field between them leaves it flat and
only writing prose brings it down. Both ceilings only ever come down, which is
what stops either state from being reached by adding a line. A client drawing a
form from the schema has no other source for a control's label.

``ALLOWED_FIELD_DESCRIPTION`` is the second ratchet, over the forbidden spelling.
``Field(description=...)`` overrides a ``Declared`` silently, in either
declaration order, and the resulting property still carries a description -- so
the prose guard passes while the ``Declared`` is dead. ``FieldInfo.description is
None`` is the signal that separates the two.

The prose guard reads ``model_json_schema()`` rather than ``FieldInfo.metadata``.
A constraint reached through a reusable ``Annotated`` alias lives inside that
alias's own annotation, so the outer field reports an empty ``metadata`` while
its ``minimum`` and ``maximum`` do appear in the schema.
``test_hash_exclude_agrees_between_metadata_and_schema`` reads both on purpose:
their disagreement is what would let ``identity_dump()`` and the published schema
describe different sets of fields.

**Coverage is the import list.** ``Params.__subclasses__()`` sees a class only
once its module has been imported, so the models enumerated here are the two
registries plus ``UNREGISTERED_PARAMS_MODULES``. Together those reach every
``Params`` subclass under ``src/mosaic`` that imports in this environment; the
two that do not import here, ``feature_library.external.kpms_server`` (jax) and
``pose_training.localizer_model`` (torch), declare none.
"""

from __future__ import annotations

import importlib
from typing import Annotated

import pytest
from pydantic import Field

from mosaic.core.params import (
    HASH_EXCLUDE,
    NEEDS_DESCRIPTION,
    Declared,
    HashExclude,
    Params,
)
from mosaic.tracking import register_ops

UNREGISTERED_PARAMS_MODULES: tuple[str, ...] = (
    "mosaic.behavior.feature_library.feature_template__global",
    "mosaic.behavior.feature_library.feature_template__per_sequence",
    "mosaic.behavior.label_library",
    "mosaic.behavior.label_library.custom_label_template",
    "mosaic.behavior.label_library.label_converter_template",
    "mosaic.core.annotations.bbox",
    "mosaic.core.track_library.track_converter_template",
)
"""Modules holding a ``Params`` subclass that neither registry reaches.

The label and track converters reach users through the converter CLIs and hash
through the same ``identity_dump()``. The ``*_template`` modules are the classes
a person copies to add a converter or a feature, which is where an undescribed
field propagates from.
"""

# The op registry fills on an explicit call rather than on importing the
# package, so the op params models are unreachable without it.
register_ops()
for module_name in UNREGISTERED_PARAMS_MODULES:
    _ = importlib.import_module(module_name)


def params_models() -> dict[str, type[Params]]:
    """Return every shipped ``Params`` subclass, keyed by qualified name.

    Walks ``__subclasses__()`` transitively rather than reading the op and
    feature registries, which hold 62 between them and reach neither the
    converters nor the ``*_template`` modules.

    Two kinds of subclass are skipped. A model declared by a test module is a
    fixture rather than a shipped parameter model, so the walk keeps only
    ``mosaic.*``. Pydantic mints a concrete class for each parametrization of a
    generic model, named ``GlobalModelParams[KMeansModelArtifact]`` and carrying
    the generic's own fields, so the walk descends through those and records
    them under the generic's name instead.
    """
    models: dict[str, type[Params]] = {}
    seen: set[type[Params]] = set()
    pending: list[type[Params]] = list(Params.__subclasses__())
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        pending.extend(model.__subclasses__())
        if not model.__module__.startswith("mosaic."):
            continue
        if "[" in model.__qualname__:
            continue
        held = models.get(model.__qualname__, model)
        clash = (
            f"{model.__module__}.{model.__qualname__} and "
            f"{held.__module__}.{held.__qualname__} share one key; one of them "
            f"would go unchecked"
        )
        assert held is model, clash
        models[model.__qualname__] = model
    return models


PARAMS_MODELS = params_models()
MODEL_NAMES = sorted(PARAMS_MODELS)

ALLOWLIST_CEILING = 430
"""Only ever lowered. Raising it admits a field that reaches a client unlabelled."""

# Raised once, from 488: the label strict_schema and the two template verbose
# fields carried descriptions no source supported, and an honest placeholder
# counts here where a false description did not.
UNDOCUMENTED_CEILING = 440
"""Only ever lowered. Counts every field a client cannot draw a label for.

The sum of both states rather than the placeholder count alone, which is what
keeps it monotone through the sweep: declaring an undeclared field with
:data:`NEEDS_DESCRIPTION` moves it between the two and leaves the sum flat, and
only writing prose brings it down.
"""

_PENDING_PROSE_NAMED = 10
"""How many pending fields a failure names, so hundreds cannot bury the fix."""

ALLOWED_FIELD_DESCRIPTION: frozenset[tuple[str, str]] = frozenset(
    (
        # Converting this one means deleting the ``description=`` argument from
        # its ``Field`` and passing the prose to a ``Declared`` beside it, not
        # adding a marker next to the ``Field`` that already wins.
        ("SpeedAngvel.Params", "fps"),
    )
)

ALLOWED_MISSING_DESCRIPTION: frozenset[tuple[str, str]] = frozenset(
    (
        ("ApproachAvoidance.Params", "angle_units"),
        ("ApproachAvoidance.Params", "approacher_velocity_threshold"),
        ("ApproachAvoidance.Params", "avoider_velocity_threshold"),
        ("ApproachAvoidance.Params", "consecutive_frame_delta"),
        ("ApproachAvoidance.Params", "cos_approacher_threshold"),
        ("ApproachAvoidance.Params", "cos_avoider_threshold"),
        ("ApproachAvoidance.Params", "distance_threshold"),
        ("ApproachAvoidance.Params", "interpolation"),
        ("ApproachAvoidance.Params", "min_event_count"),
        ("ApproachAvoidance.Params", "min_event_length"),
        ("ApproachAvoidance.Params", "orientation_gate_cos"),
        ("ApproachAvoidance.Params", "sampling"),
        ("ApproachAvoidance.Params", "smooth_window_sec"),
        ("ApproachAvoidance.Params", "velocity_units"),
        ("ArHmmFeature.Params", "backend"),
        ("ArHmmFeature.Params", "downsample_rate"),
        ("ArHmmFeature.Params", "model"),
        ("ArHmmFeature.Params", "n_iter"),
        ("ArHmmFeature.Params", "n_lags"),
        ("ArHmmFeature.Params", "n_restarts"),
        ("ArHmmFeature.Params", "n_states"),
        ("ArHmmFeature.Params", "pca_dim"),
        ("ArHmmFeature.Params", "prune_threshold"),
        ("ArHmmFeature.Params", "random_state"),
        ("ArHmmFeature.Params", "standardize"),
        ("ArHmmFeature.Params", "sticky_weight"),
        ("ArHmmFeature.Params", "tol"),
        ("AttentionTarget.Params", "id_group_map"),
        ("BboxPolicy", "head_index"),
        ("BboxPolicy", "length_pad_frac"),
        ("BboxPolicy", "margin"),
        ("BboxPolicy", "method"),
        ("BboxPolicy", "min_pad_px"),
        ("BboxPolicy", "pad_frac_of_body"),
        ("BboxPolicy", "side_pad_frac"),
        ("BboxPolicy", "tail_index"),
        ("CollectiveMotionMetrics.Params", "alpha"),
        ("CollectiveMotionMetrics.Params", "area_method"),
        ("CollectiveMotionMetrics.Params", "exclude_cols"),
        ("CollectiveMotionMetrics.Params", "filter_expr"),
        ("CollectiveMotionMetrics.Params", "fps"),
        ("CollectiveMotionMetrics.Params", "heading_source"),
        ("CollectiveMotionMetrics.Params", "max_frame_gap"),
        ("CollectiveMotionMetrics.Params", "min_group_speed"),
        ("CollectiveMotionMetrics.Params", "min_individuals"),
        ("CollectiveMotionMetrics.Params", "speed_col"),
        ("CollectiveMotionMetrics.Params", "subgroup_col"),
        ("ConvertPointsParams", "class_attribute"),
        ("ConvertPointsParams", "class_names"),
        ("ConvertPointsParams", "cvat_xml"),
        ("ConvertPointsParams", "images_dir"),
        ("ConvertPointsParams", "overwrite"),
        ("ConvertPointsParams", "radii"),
        ("ConvertPointsParams", "seed"),
        ("ConvertPointsParams", "source_format"),
        ("ConvertPointsParams", "split"),
        ("ConvertPointsParams", "split_by"),
        ("ConvertPointsParams", "symlink_images"),
        ("ExtractLabeledTemplates.Params", "labels"),
        ("ExtractLabeledTemplates.Params", "n_per_class"),
        ("ExtractLabeledTemplates.Params", "n_total"),
        ("ExtractLabeledTemplates.Params", "pool"),
        ("ExtractLabeledTemplates.Params", "random_state"),
        ("ExtractLabeledTemplates.Params", "strategy"),
        ("ExtractLabeledTemplates.Params", "test_fraction"),
        ("ExtractTemplates.Params", "n_templates"),
        ("ExtractTemplates.Params", "pair_filter"),
        ("ExtractTemplates.Params", "pool"),
        ("ExtractTemplates.Params", "random_state"),
        ("ExtractTemplates.Params", "strategy"),
        ("FFGroups.Params", "distance_cutoff"),
        ("FFGroups.Params", "min_event_duration"),
        ("FFGroups.Params", "window_size"),
        ("FFGroupsMetrics.Params", "centroid_heading_col"),
        ("FFGroupsMetrics.Params", "exclude_cols"),
        ("FFGroupsMetrics.Params", "frame_chunk"),
        ("FFGroupsMetrics.Params", "group_col"),
        ("FFGroupsMetrics.Params", "speed_col"),
        ("FFGroupsMetrics.Params", "time_chunk_sec"),
        ("FeralFeature.Params", "chunk_length"),
        ("FeralFeature.Params", "chunk_shift"),
        ("FeralFeature.Params", "chunk_step"),
        ("FeralFeature.Params", "class_names"),
        ("FeralFeature.Params", "decision_threshold"),
        ("FeralFeature.Params", "default_class"),
        ("FeralFeature.Params", "device"),
        ("FeralFeature.Params", "feral_code_dir"),
        ("FeralFeature.Params", "infer_batch_size"),
        ("FeralFeature.Params", "inference_autocast"),
        ("FeralFeature.Params", "label_json"),
        ("FeralFeature.Params", "model_dir"),
        ("FeralFeature.Params", "model_name"),
        ("FeralFeature.Params", "predict_per_item"),
        ("FeralFeature.Params", "resize_to"),
        ("FeralFeature.Params", "training"),
        ("FeralFeature.Params", "video_dir"),
        ("FrameAggregate.Params", "agg"),
        ("FrameAggregate.Params", "column"),
        ("FrameAggregate.Params", "filter_expr"),
        ("FrameAggregate.Params", "output_column"),
        ("FrameAggregate.Params", "threshold"),
        ("FrameAggregate.Params", "transform"),
        ("GlobalIdentityDinoV2Temporal.Params", "backbone"),
        ("GlobalIdentityDinoV2Temporal.Params", "batch_size"),
        ("GlobalIdentityDinoV2Temporal.Params", "channels"),
        ("GlobalIdentityDinoV2Temporal.Params", "clip_len"),
        ("GlobalIdentityDinoV2Temporal.Params", "clip_stride"),
        ("GlobalIdentityDinoV2Temporal.Params", "crop_root"),
        ("GlobalIdentityDinoV2Temporal.Params", "embedding_dim"),
        ("GlobalIdentityDinoV2Temporal.Params", "epochs"),
        ("GlobalIdentityDinoV2Temporal.Params", "group_as_identity"),
        ("GlobalIdentityDinoV2Temporal.Params", "identities"),
        ("GlobalIdentityDinoV2Temporal.Params", "image_size"),
        ("GlobalIdentityDinoV2Temporal.Params", "learning_rate"),
        ("GlobalIdentityDinoV2Temporal.Params", "max_clips_per_identity"),
        ("GlobalIdentityDinoV2Temporal.Params", "model"),
        ("GlobalIdentityDinoV2Temporal.Params", "temporal_head"),
        ("GlobalIdentityDinoV2Temporal.Params", "val_split"),
        ("GlobalIdentityDinoV2Temporal.Params", "weights_name"),
        ("GlobalIdentityEmbedding.Params", "batch_size"),
        ("GlobalIdentityEmbedding.Params", "channels"),
        ("GlobalIdentityEmbedding.Params", "crop_root"),
        ("GlobalIdentityEmbedding.Params", "group_as_identity"),
        ("GlobalIdentityEmbedding.Params", "identities"),
        ("GlobalIdentityEmbedding.Params", "image_size"),
        ("GlobalIdentityEmbedding.Params", "max_images_per_identity"),
        ("GlobalIdentityEmbedding.Params", "model"),
        ("GlobalIdentityEmbedding.Params", "model_name"),
        ("GlobalIdentityEmbedding.Params", "weights_name"),
        ("GlobalIdentityModel.Params", "batch_size"),
        ("GlobalIdentityModel.Params", "channels"),
        ("GlobalIdentityModel.Params", "crop_root"),
        ("GlobalIdentityModel.Params", "epochs"),
        ("GlobalIdentityModel.Params", "freeze_backbone"),
        ("GlobalIdentityModel.Params", "group_as_identity"),
        ("GlobalIdentityModel.Params", "identities"),
        ("GlobalIdentityModel.Params", "image_size"),
        ("GlobalIdentityModel.Params", "learning_rate"),
        ("GlobalIdentityModel.Params", "max_images_per_identity"),
        ("GlobalIdentityModel.Params", "model"),
        ("GlobalIdentityModel.Params", "model_name"),
        ("GlobalIdentityModel.Params", "val_split"),
        ("GlobalIdentityModel.Params", "weights_name"),
        ("GlobalKMeansClustering.Params", "device"),
        ("GlobalKMeansClustering.Params", "k"),
        ("GlobalKMeansClustering.Params", "label_artifact_points"),
        ("GlobalKMeansClustering.Params", "max_iter"),
        ("GlobalKMeansClustering.Params", "model"),
        ("GlobalKMeansClustering.Params", "n_init"),
        ("GlobalKMeansClustering.Params", "pair_filter"),
        ("GlobalKMeansClustering.Params", "random_state"),
        ("GlobalKMeansClustering.Params", "templates"),
        ("GlobalModelParams", "model"),
        ("GlobalModelParams", "templates"),
        ("GlobalScaler.Params", "model"),
        ("GlobalScaler.Params", "templates"),
        ("GlobalTSNE.Params", "fit"),
        ("GlobalTSNE.Params", "knn_method"),
        ("GlobalTSNE.Params", "mapping"),
        ("GlobalTSNE.Params", "model"),
        ("GlobalTSNE.Params", "n_jobs"),
        ("GlobalTSNE.Params", "perplexity"),
        ("GlobalTSNE.Params", "random_state"),
        ("GlobalTSNE.Params", "templates"),
        ("GlobalWardClustering.Params", "method"),
        ("GlobalWardClustering.Params", "model"),
        ("GlobalWardClustering.Params", "n_clusters"),
        ("GlobalWardClustering.Params", "pair_filter"),
        ("GlobalWardClustering.Params", "templates"),
        ("HeadingFeature.Params", "front_idx"),
        ("HeadingFeature.Params", "method"),
        ("HeadingFeature.Params", "output_col"),
        ("HeadingFeature.Params", "rear_idx"),
        ("IdTagColumns.Params", "field_renames"),
        ("IdTagColumns.Params", "fields"),
        ("IdTagColumns.Params", "label_kind"),
        ("IdTagColumns.Params", "labels"),
        ("KpmsFeature.Params", "anterior_bodyparts"),
        ("KpmsFeature.Params", "downsample_rate"),
        ("KpmsFeature.Params", "fps"),
        ("KpmsFeature.Params", "kappa_ar"),
        ("KpmsFeature.Params", "kappa_full"),
        ("KpmsFeature.Params", "kpms_python"),
        ("KpmsFeature.Params", "latent_dim"),
        ("KpmsFeature.Params", "location_aware"),
        ("KpmsFeature.Params", "mixed_map_iters"),
        ("KpmsFeature.Params", "model"),
        ("KpmsFeature.Params", "num_iters_apply"),
        ("KpmsFeature.Params", "num_iters_ar"),
        ("KpmsFeature.Params", "num_iters_full"),
        ("KpmsFeature.Params", "outlier_scale_factor"),
        ("KpmsFeature.Params", "parallel_message_passing"),
        ("KpmsFeature.Params", "pose"),
        ("KpmsFeature.Params", "posterior_bodyparts"),
        ("KpmsFeature.Params", "remove_outliers"),
        ("KpmsFeature.Params", "resume"),
        ("KpmsFeature.Params", "save_every_n_iters"),
        ("LightningActionFeature.Params", "activation"),
        ("LightningActionFeature.Params", "batch_size"),
        ("LightningActionFeature.Params", "decision_threshold"),
        ("LightningActionFeature.Params", "default_class"),
        ("LightningActionFeature.Params", "device"),
        ("LightningActionFeature.Params", "dropout_rate"),
        ("LightningActionFeature.Params", "head"),
        ("LightningActionFeature.Params", "learning_rate"),
        ("LightningActionFeature.Params", "model"),
        ("LightningActionFeature.Params", "num_epochs"),
        ("LightningActionFeature.Params", "num_hid_units"),
        ("LightningActionFeature.Params", "num_lags"),
        ("LightningActionFeature.Params", "num_layers"),
        ("LightningActionFeature.Params", "optimizer"),
        ("LightningActionFeature.Params", "random_state"),
        ("LightningActionFeature.Params", "sequence_length"),
        ("LightningActionFeature.Params", "templates"),
        ("LightningActionFeature.Params", "weight_classes"),
        ("LightningActionFeature.Params", "weight_decay"),
        ("LocalOrderMetrics.Params", "body_scale"),
        ("LocalOrderMetrics.Params", "exclude_cols"),
        ("LocalOrderMetrics.Params", "filter_expr"),
        ("LocalOrderMetrics.Params", "heading_source"),
        ("LocalOrderMetrics.Params", "min_neighbors"),
        ("LocalOrderMetrics.Params", "n_peripheral"),
        ("LocalOrderMetrics.Params", "n_shells"),
        ("LocalOrderMetrics.Params", "radius"),
        ("LocalOrderMetrics.Params", "radius_units"),
        ("LocalOrderMetrics.Params", "subgroup_col"),
        ("LocalizerInferParams", "initial_channels"),
        ("LocalizerInferParams", "num_classes"),
        ("LocalizerInferParams", "thresholds"),
        ("LocalizerTrainParams", "augment"),
        ("LocalizerTrainParams", "base_model"),
        ("LocalizerTrainParams", "batch_size"),
        ("LocalizerTrainParams", "dataset_dir"),
        ("LocalizerTrainParams", "device"),
        ("LocalizerTrainParams", "early_stopping_patience"),
        ("LocalizerTrainParams", "epochs"),
        ("LocalizerTrainParams", "freeze_encoder"),
        ("LocalizerTrainParams", "initial_channels"),
        ("LocalizerTrainParams", "lr"),
        ("LocalizerTrainParams", "num_classes"),
        ("LocalizerTrainParams", "overwrite"),
        ("LocalizerTrainParams", "seed"),
        ("MovementFilterInterpolate.Params", "confidence_threshold"),
        ("MovementFilterInterpolate.Params", "fps"),
        ("MovementFilterInterpolate.Params", "include_centroid"),
        ("MovementFilterInterpolate.Params", "interpolation_method"),
        ("MovementFilterInterpolate.Params", "keypoint_names"),
        ("MovementFilterInterpolate.Params", "max_gap"),
        ("MovementSmooth.Params", "fps"),
        ("MovementSmooth.Params", "include_centroid"),
        ("MovementSmooth.Params", "keypoint_names"),
        ("MovementSmooth.Params", "method"),
        ("MovementSmooth.Params", "min_periods"),
        ("MovementSmooth.Params", "polyorder"),
        ("MovementSmooth.Params", "savgol_mode"),
        ("MovementSmooth.Params", "statistic"),
        ("MovementSmooth.Params", "window"),
        ("NearestNeighborDelta.Params", "diff_numframes"),
        ("NearestNeighborDelta.Params", "diff_numframes_col"),
        ("NearestNeighborDelta.Params", "divide_dangle_by_frames"),
        ("NearestNeighborDelta.Params", "emit_backward"),
        ("NearestNeighborDelta.Params", "focal_col"),
        ("NearestNeighborDelta.Params", "nn_dx_ego_col"),
        ("NearestNeighborDelta.Params", "nn_dx_world_col"),
        ("NearestNeighborDelta.Params", "nn_dy_ego_col"),
        ("NearestNeighborDelta.Params", "nn_dy_world_col"),
        ("NearestNeighborDelta.Params", "nn_id_col"),
        ("NearestNeighborDelta.Params", "sampling"),
        ("NearestNeighborDelta.Params", "scale_dangle_by_fps"),
        ("NearestNeighborDelta.Params", "speed_col"),
        ("NearestNeighborDelta.Params", "tag_cols"),
        ("NearestNeighborDelta.Params", "wrap_angle"),
        ("NearestNeighborDeltaBins.Params", "antisymm"),
        ("NearestNeighborDeltaBins.Params", "binmax"),
        ("NearestNeighborDeltaBins.Params", "category_specs"),
        ("NearestNeighborDeltaBins.Params", "exclude_cols"),
        ("NearestNeighborDeltaBins.Params", "exp_col"),
        ("NearestNeighborDeltaBins.Params", "focal_category_col"),
        ("NearestNeighborDeltaBins.Params", "group_size_col"),
        ("NearestNeighborDeltaBins.Params", "max_for_avg"),
        ("NearestNeighborDeltaBins.Params", "nbins"),
        ("NearestNeighborDeltaBins.Params", "neighbor_category_col"),
        ("NearestNeighborDeltaBins.Params", "nonfocal_flag_col"),
        ("NearestNeighborDeltaBins.Params", "nonfocal_flag_value"),
        ("NearestNeighborDeltaBins.Params", "trial_col"),
        ("OrientationRelativeFeature.Params", "nearest_k"),
        ("OrientationRelativeFeature.Params", "quantiles"),
        ("OrientationRelativeFeature.Params", "scale"),
        ("PairEgocentricFeatures.Params", "center_mode"),
        ("PairEgocentricFeatures.Params", "interpolation"),
        ("PairEgocentricFeatures.Params", "neck_idx"),
        ("PairEgocentricFeatures.Params", "pose"),
        ("PairEgocentricFeatures.Params", "sampling"),
        ("PairEgocentricFeatures.Params", "tail_base_idx"),
        ("PairFacing.Params", "angle_thresh_deg"),
        ("PairFacing.Params", "body_axis_from"),
        ("PairFacing.Params", "cm_per_pixel"),
        ("PairFacing.Params", "dist_thresh"),
        ("PairFacing.Params", "pose_abdomen_index"),
        ("PairFacing.Params", "pose_head_index"),
        ("PairFacing.Params", "x_prefix"),
        ("PairFacing.Params", "y_prefix"),
        ("PairInteractionFilter.Params", "frame_padding"),
        ("PairInteractionFilter.Params", "max_dist"),
        ("PairInteractionFilter.Params", "max_inv_orientation_diff_deg"),
        ("PairInteractionFilter.Params", "min_run_frames"),
        ("PairInteractionFilter.Params", "morphological_structure_size"),
        ("PairInteractionFilter.Params", "pose_head_index"),
        ("PairInteractionFilter.Params", "px_scale"),
        ("PairInteractionFilter.Params", "require_facing"),
        ("PairInteractionFilter.Params", "shift_dist"),
        ("PairInteractionFilter.Params", "use_pixel_coords"),
        ("PairPoseDistancePCA.Params", "batch_size"),
        ("PairPoseDistancePCA.Params", "duplicate_perspective"),
        ("PairPoseDistancePCA.Params", "include_inter"),
        ("PairPoseDistancePCA.Params", "include_intra_A"),
        ("PairPoseDistancePCA.Params", "include_intra_B"),
        ("PairPoseDistancePCA.Params", "interpolation"),
        ("PairPoseDistancePCA.Params", "n_components"),
        ("PairPoseDistancePCA.Params", "pose"),
        ("PairPositionFeatures.Params", "interpolation"),
        ("PairPositionFeatures.Params", "sampling"),
        ("PairWavelet.Params", "cols"),
        ("PairWavelet.Params", "f_max"),
        ("PairWavelet.Params", "f_min"),
        ("PairWavelet.Params", "log_floor"),
        ("PairWavelet.Params", "n_freq"),
        ("PairWavelet.Params", "pc_prefix"),
        ("PairWavelet.Params", "sampling"),
        ("PairWavelet.Params", "wavelet"),
        ("PointInferParams", "dor"),
        ("PointTrainParams", "augmentation"),
        ("PointTrainParams", "backend"),
        ("PointTrainParams", "base_model"),
        ("PointTrainParams", "batch"),
        ("PointTrainParams", "data"),
        ("PointTrainParams", "device"),
        ("PointTrainParams", "dor"),
        ("PointTrainParams", "epochs"),
        ("PointTrainParams", "imgsz"),
        ("PointTrainParams", "loc"),
        ("PointTrainParams", "loc_loss"),
        ("PointTrainParams", "model"),
        ("PointTrainParams", "overwrite"),
        ("PointTrainParams", "patience"),
        ("PointTrainParams", "resume"),
        ("PointTrainParams", "train_overrides"),
        ("PoseTrainParams", "augmentation"),
        ("PoseTrainParams", "base_model"),
        ("PoseTrainParams", "batch"),
        ("PoseTrainParams", "data"),
        ("PoseTrainParams", "device"),
        ("PoseTrainParams", "epochs"),
        ("PoseTrainParams", "imgsz"),
        ("PoseTrainParams", "model"),
        ("PoseTrainParams", "overwrite"),
        ("PoseTrainParams", "patience"),
        ("PoseTrainParams", "resume"),
        ("PoseTrainParams", "train_overrides"),
        ("ScaleToCm.Params", "cm_per_pixel"),
        ("ScaleToCm.Params", "columns"),
        ("ScaleToCm.Params", "mode"),
        ("ScaleToCm.Params", "suffix"),
        ("SocialMotionSummary.Params", "compute_burst_coast"),
        ("SocialMotionSummary.Params", "fps"),
        ("SocialMotionSummary.Params", "social_min_group_size"),
        ("SocialMotionSummary.Params", "speed_col"),
        ("SocialMotionSummary.Params", "subgroup_col"),
        ("SpeedAngvel.Params", "smooth_window"),
        ("SpeedAngvel.Params", "step_size"),
        ("TemporalStackingFeature.Params", "add_pool"),
        ("TemporalStackingFeature.Params", "fps"),
        ("TemporalStackingFeature.Params", "half"),
        ("TemporalStackingFeature.Params", "pair_filter"),
        ("TemporalStackingFeature.Params", "pool_stats"),
        ("TemporalStackingFeature.Params", "sigma_pool"),
        ("TemporalStackingFeature.Params", "sigma_stack"),
        ("TemporalStackingFeature.Params", "skip"),
        ("TemporalStackingFeature.Params", "use_temporal_stack"),
        ("TemporalStackingFeature.Params", "win_sec"),
        ("TrackSubsample.Params", "clip_len"),
        ("TrackSubsample.Params", "drop_nan"),
        ("TrackSubsample.Params", "method"),
        ("TrackSubsample.Params", "pose"),
        ("TrackSubsample.Params", "seed"),
        ("TrackSubsample.Params", "target_frames"),
        ("TrainLitposeParams", "backbone"),
        ("TrainLitposeParams", "base_config"),
        ("TrainLitposeParams", "base_model"),
        ("TrainLitposeParams", "device"),
        ("TrainLitposeParams", "idle_timeout"),
        ("TrainLitposeParams", "litpose_overrides"),
        ("TrainLitposeParams", "max_epochs"),
        ("TrainLitposeParams", "max_runtime"),
        ("TrainLitposeParams", "model_type"),
        ("TrainLitposeParams", "overwrite"),
        ("TrainLitposeParams", "project"),
        ("TrainSleapParams", "backbone"),
        ("TrainSleapParams", "base_model"),
        ("TrainSleapParams", "device"),
        ("TrainSleapParams", "head"),
        ("TrainSleapParams", "idle_timeout"),
        ("TrainSleapParams", "labels"),
        ("TrainSleapParams", "max_epochs"),
        ("TrainSleapParams", "max_runtime"),
        ("TrainSleapParams", "overwrite"),
        ("TrainSleapParams", "seed"),
        ("TrainSleapParams", "sleap_overrides"),
        ("TrainSleapParams", "validation_fraction"),
        ("TrajectorySmooth.Params", "expand_frames"),
        ("TrajectorySmooth.Params", "fps"),
        ("TrajectorySmooth.Params", "interpolate_centroid"),
        ("TrajectorySmooth.Params", "interpolate_pose"),
        ("TrajectorySmooth.Params", "savgol_polyorder"),
        ("TrajectorySmooth.Params", "savgol_window"),
        ("TrajectorySmooth.Params", "speed_threshold"),
        ("XgboostFeature.Params", "class_weight"),
        ("XgboostFeature.Params", "colsample_bytree"),
        ("XgboostFeature.Params", "decision_threshold"),
        ("XgboostFeature.Params", "default_class"),
        ("XgboostFeature.Params", "learning_rate"),
        ("XgboostFeature.Params", "max_depth"),
        ("XgboostFeature.Params", "model"),
        ("XgboostFeature.Params", "n_estimators"),
        ("XgboostFeature.Params", "random_state"),
        ("XgboostFeature.Params", "strategy"),
        ("XgboostFeature.Params", "subsample"),
        ("XgboostFeature.Params", "templates"),
        ("XgboostFeature.Params", "undersample_ratio"),
        ("XgboostFeature.Params", "use_smote"),
    )
)


def field_schemas(model: type[Params]) -> dict[str, dict[str, object]]:
    """Return the property schema of each of *model*'s fields."""
    properties: dict[str, dict[str, object]] = model.model_json_schema()["properties"]
    return properties


def metadata_of(model: type[Params], field: str) -> list[object]:
    """Return the ``Annotated`` metadata pydantic retained for *field*."""
    metadata: list[object] = model.model_fields[field].metadata
    return metadata


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_field_declares_a_description(model_name: str) -> None:
    """A field carries a ``Declared``, or it is on the shrinking allowlist.

    Presence of the description key is the question, not whether the prose is
    written: ``Declared(NEEDS_DESCRIPTION)`` publishes an empty one, which
    declares the field and hands the debt to
    :func:`test_the_pending_prose_only_shrinks`. A field with no ``Declared`` at
    all publishes no key, and nothing else writes one --
    ``use_attribute_docstrings`` is off and
    :func:`test_no_field_states_its_prose_on_field` forbids the ``Field``
    spelling.
    """
    model = PARAMS_MODELS[model_name]
    undescribed = {
        field
        for field, spec in field_schemas(model).items()
        if "description" not in spec
    }
    unlisted = sorted(
        field
        for field in undescribed
        if (model_name, field) not in ALLOWED_MISSING_DESCRIPTION
    )
    undeclared = (
        f"{model_name} declares no description for {unlisted}. "
        f"Add a Declared(...) to each field's Annotated."
    )
    assert not unlisted, undeclared

    listed = {
        field
        for allowed_model, field in ALLOWED_MISSING_DESCRIPTION
        if allowed_model == model_name
    }
    stale = sorted(listed & set(field_schemas(model)) - undescribed)
    described = (
        f"{model_name} now describes {stale}. Remove those entries from "
        f"ALLOWED_MISSING_DESCRIPTION and lower ALLOWLIST_CEILING to match."
    )
    assert not stale, described


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_no_field_states_its_prose_on_field(model_name: str) -> None:
    """``Declared`` is the one home for a description.

    A ``description=`` on ``Field`` wins over a ``Declared`` in the same
    ``Annotated``, whichever order the two are written in, and the property still
    carries a description -- so the prose guard above cannot see it. A non-None
    ``FieldInfo.description`` is the precise signal.
    """
    model = PARAMS_MODELS[model_name]
    on_field = sorted(
        field
        for field, info in model.model_fields.items()
        if info.description is not None
        and (model_name, field) not in ALLOWED_FIELD_DESCRIPTION
    )
    misplaced = (
        f"{model_name} states prose on Field for {on_field}. Move it into a "
        f"Declared(...) and delete the description= argument."
    )
    assert not on_field, misplaced


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_no_field_declares_two_descriptions(model_name: str) -> None:
    """Two ``Declared`` in one ``Annotated``: the last one wins, silently."""
    model = PARAMS_MODELS[model_name]
    doubled = sorted(
        field
        for field in model.model_fields
        if sum(isinstance(m, Declared) for m in metadata_of(model, field)) > 1
    )
    repeated = (
        f"{model_name} declares more than one Declared on {doubled}. Only the "
        f"last takes effect; state the prose once."
    )
    assert not doubled, repeated


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_hash_exclude_agrees_between_metadata_and_schema(model_name: str) -> None:
    """The set ``identity_dump()`` strips is the set the schema publishes.

    ``identity_dump()`` reads ``FieldInfo.metadata`` and a client reads
    ``x-mosaic-hash-exclude``. A marker whose hook stopped being called would
    leave the two describing different fields, and only a client would notice.
    """
    model = PARAMS_MODELS[model_name]
    from_metadata = {
        field
        for field in model.model_fields
        if any(isinstance(m, HashExclude) for m in metadata_of(model, field))
    }
    from_schema = {
        field
        for field, spec in field_schemas(model).items()
        if spec.get("x-mosaic-hash-exclude") is True
    }
    disagreement = (
        f"{model_name} marks {sorted(from_metadata)} in metadata but publishes "
        f"{sorted(from_schema)} in the schema."
    )
    assert from_metadata == from_schema, disagreement


def test_a_placeholder_declares_the_field_and_leaves_the_prose_empty() -> None:
    """The two states the guards above tell apart, pinned on a throwaway model.

    Everything here rests on a field with no ``Declared`` publishing no
    description key at all, while a placeholder publishes an empty one. Were
    pydantic to start emitting ``description`` for an undeclared field, or to
    drop an empty one, the declaration guard would silently stop guarding and
    the prose ceiling would silently stop counting.
    """

    class Sampled(Params):
        declared: Annotated[int, Declared("How many samples to take.")] = 1
        pending: Annotated[int, Declared(NEEDS_DESCRIPTION)] = 2
        bare: int = 3

    schemas = field_schemas(Sampled)
    assert schemas["declared"]["description"] == "How many samples to take."
    assert schemas["pending"]["description"] == NEEDS_DESCRIPTION
    assert "description" not in schemas["bare"], (
        "an undeclared field must publish no key, or the declaration guard "
        "cannot tell it from a placeholder"
    )


def test_the_attributes_section_states_only_what_was_declared() -> None:
    """Three ways the rendered section could state something false.

    Appending to ``cls.__doc__`` rather than ``cls.__dict__``'s entry would copy
    a parent's prose onto a child that declared none. Rendering a field left at
    ``NEEDS_DESCRIPTION`` would publish an attribute entry that says nothing.
    Synthesizing a summary for a class whose author wrote none would put words in
    their mouth, which is the reason ``gen_docs_reference.summary_line`` renders
    an empty cell instead.
    """

    class Documented(Params):
        """A model whose author wrote a summary."""

        stated: Annotated[int, Declared("How many.")] = 1
        owed: Annotated[int, Declared(NEEDS_DESCRIPTION)] = 2

    class Undocumented(Params):
        stated: Annotated[int, Declared("How many.")] = 1

    documented = Documented.__doc__ or ""
    assert documented.startswith("A model whose author wrote a summary.")
    assert "stated: How many." in documented
    assert "owed" not in documented, "a placeholder states nothing, so it is left out"

    undocumented = Undocumented.__dict__.get("__doc__") or ""
    assert undocumented.startswith("Attributes:"), (
        "a class that declared no docstring gets the section and no invented summary"
    )
    assert "Base for all feature parameter models" not in undocumented, (
        "the parent's prose must not be copied onto a child that declared none"
    )


def test_the_attributes_section_carries_a_declared_unit() -> None:
    """``x-mosaic-unit`` reaches a schema client; this is the same fact for a reader."""

    class Timed(Params):
        idle_timeout: Annotated[float, Declared("How long to wait.", unit="s")] = 900

    assert "idle_timeout: How long to wait. [s]" in (Timed.__doc__ or "")


def test_the_undocumented_count_only_shrinks() -> None:
    """A placeholder is a debt with a bound, never a resting state.

    ``Declared(NEEDS_DESCRIPTION)`` passes the declaration guard, so without this
    the cheapest way to satisfy that guard would be to declare every field and
    describe none -- a schema full of controls a client can label with nothing.

    Undeclared and placeholder fields are counted together because the sweep
    moves fields from the first state to the second in bulk. Counting the second
    alone would make that ceiling climb, which is no ratchet at all.
    """
    undeclared: list[str] = []
    pending: list[str] = []
    for model_name, model in PARAMS_MODELS.items():
        for field, spec in field_schemas(model).items():
            if "description" not in spec:
                undeclared.append(f"{model_name}.{field}")
            elif spec["description"] == NEEDS_DESCRIPTION:
                pending.append(f"{model_name}.{field}")

    total = len(undeclared) + len(pending)
    named = ", ".join(sorted(pending)[:_PENDING_PROSE_NAMED])
    rest = len(pending) - _PENDING_PROSE_NAMED
    grown = (
        f"{len(undeclared)} fields declare no description and {len(pending)} "
        f"declare NEEDS_DESCRIPTION, {total} against a ceiling of "
        f"{UNDOCUMENTED_CEILING}. Placeholders: {named}"
        f"{f' and {rest} more' if rest > 0 else ''}. Write prose and lower "
        f"UNDOCUMENTED_CEILING to match."
    )
    assert total <= UNDOCUMENTED_CEILING, grown


def test_the_allowlist_only_shrinks() -> None:
    """Nothing may join the allowlist, and nothing may rot on it.

    The ceiling stops a new undescribed field from passing on one added line.
    Resolving each entry stops a renamed or deregistered field from sitting there
    forever, which the per-model staleness check cannot see: it intersects with
    the fields a model still has.
    """
    grown = (
        f"ALLOWED_MISSING_DESCRIPTION holds {len(ALLOWED_MISSING_DESCRIPTION)} "
        f"entries against a ceiling of {ALLOWLIST_CEILING}."
    )
    assert len(ALLOWED_MISSING_DESCRIPTION) <= ALLOWLIST_CEILING, grown

    allowed = ALLOWED_MISSING_DESCRIPTION | ALLOWED_FIELD_DESCRIPTION
    unknown_models = sorted({model for model, _ in allowed} - set(PARAMS_MODELS))
    vanished = (
        f"allowlisted models {unknown_models} resolve to no Params subclass. "
        f"Delete their entries."
    )
    assert not unknown_models, vanished

    unknown_fields = sorted(
        f"{model}.{field}"
        for model, field in allowed
        if field not in PARAMS_MODELS[model].model_fields
    )
    renamed = (
        f"allowlisted fields {unknown_fields} no longer exist. Delete their "
        f"entries and lower ALLOWLIST_CEILING to match."
    )
    assert not unknown_fields, renamed


class DeclarationSample(Params):
    """A field of each declared kind, for reading the emitted schema back."""

    speed: Annotated[
        float | None,
        Field(gt=0.0),
        Declared("the maximum plausible speed", unit="cm/s"),
    ] = None
    workers: Annotated[int, HASH_EXCLUDE, Declared("how many workers run")] = 1


def test_every_declared_fact_appears_in_the_schema() -> None:
    """Read the three keys a client reads, straight out of the schema.

    This is what fails when pydantic stops calling a marker's
    ``__get_pydantic_json_schema__``.
    """
    properties = field_schemas(DeclarationSample)

    assert properties["speed"]["description"] == "the maximum plausible speed"
    assert properties["speed"]["x-mosaic-unit"] == "cm/s"
    assert "x-mosaic-hash-exclude" not in properties["speed"]

    assert properties["workers"]["description"] == "how many workers run"
    assert properties["workers"]["x-mosaic-hash-exclude"] is True
    assert "x-mosaic-unit" not in properties["workers"]


def test_the_sample_model_stays_out_of_the_walk() -> None:
    """A ``Params`` subclass declared by a test is a fixture, not a shipped model.

    Test modules throughout the suite declare one. Were the walk to pick them
    up, this file's own sample would be the first to break the guard it
    supports.
    """
    assert DeclarationSample.__qualname__ not in PARAMS_MODELS


def test_the_constraint_stays_inside_the_optional_branch() -> None:
    """``Declared`` writes at the field level; a constraint stays in its branch.

    A client unwrapping the non-null branch of an optional field finds the
    constraint there and must read the description from the property itself.
    """
    speed = field_schemas(DeclarationSample)["speed"]
    branches = speed["anyOf"]
    assert branches == [{"exclusiveMinimum": 0.0, "type": "number"}, {"type": "null"}]
