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
| Embedding & clustering | GlobalTSNE, GlobalKMeansClustering, GlobalWardClustering, ExtractTemplates, ExtractLabeledTemplates |
| Sequence models | ArHmmFeature, KpmsFeature |
| Classification | XgboostFeature, FeralFeature |

!!! warning "KpmsFeature is non-commercial only"

    keypoint-MoSeq is licensed by the Harvard University Office of Technology
    Development for **non-commercial research and academic use only**, and
    commercial use is prohibited. It runs in a separate environment you build
    yourself, and mosaic will not start it until you confirm the terms apply to
    your use. See [Licensing](../../licensing.md).

    `ArHmmFeature` fits a comparable autoregressive model in mosaic's own code
    and carries no such restriction.

## Registry

::: mosaic.behavior.feature_library
    options:
      show_source: false
      members_order: alphabetical
      show_submodules: true
