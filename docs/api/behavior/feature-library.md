# Feature Library

Mosaic's feature library provides 30+ registered feature implementations
organized by output type. Features are composable pipeline stages that
read from tracks or upstream feature outputs and produce per-sequence
parquet files.

## Feature categories

| Category | Features |
|----------|----------|
| Per-frame kinematic | SpeedAngvel, BodyScale, OrientationRelative |
| Per-frame spatial | PairEgocentric, PairPosition, ApproachAvoidance |
| Per-frame social | NearestNeighbor, FFGroups, FFGroupsMetrics, NNDeltaResponse, NNDeltaBins |
| Spectral | PairWavelet |
| Reduction | PairPoseDistancePCA |
| Context | TemporalStacking |
| Global embed/cluster | GlobalTSNE, GlobalKMeansClustering, GlobalWardClustering, WardAssign |
| Template extraction | ExtractTemplates, ExtractLabeledTemplates |
| Scaling | GlobalScaler |
| Classification | XgboostFeature |
| External | KpmsFeature (keypoint-moseq) |

## Registry

::: mosaic.behavior.feature_library
    options:
      show_source: false
      members_order: alphabetical
      show_submodules: true
