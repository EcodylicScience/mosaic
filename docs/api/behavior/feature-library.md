# Feature Library

Mosaic's feature library provides 30+ registered feature implementations
organized by output type. Features are composable pipeline stages that
read from tracks or upstream feature outputs and produce per-sequence
parquet files.

## Feature categories

| Category | Features |
|----------|----------|
| Per-frame kinematic | SpeedAngvel, BodyScale, OrientationRelative |
| Per-frame spatial | PairEgocentric, PairPosition, PairInteractionFilter, ApproachAvoidance |
| Per-frame social | NearestNeighbor, FFGroups, FFGroupsMetrics, NNDeltaResponse, NNDeltaBins |
| Per-frame context | TemporalStacking, PairWavelet |
| Dimensionality reduction | PairPoseDistancePCA, GlobalScaler |
| Embedding & clustering | GlobalTSNE, GlobalKMeansClustering, GlobalWardClustering, WardAssign, ExtractTemplates, ExtractLabeledTemplates |
| Classification | XgboostFeature, FeralFeature, KpmsFeature |

## Registry

::: mosaic.behavior.feature_library
    options:
      show_source: false
      members_order: alphabetical
      show_submodules: true
