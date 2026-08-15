# Feature Library

Features are composable pipeline stages that read from tracks or upstream feature
outputs and produce per-sequence parquet files.

!!! tip "The list of features lives in [the reference](../../reference/features.md)"

    This page renders the library's classes from their docstrings. For the roster --
    every registered feature, its version, its category and its parameters -- see the
    generated [features reference](../../reference/features.md), which is derived
    from the registry itself and checked in continuous integration.

    A hand-written table used to sit here. It said "40+", listed 28, and disagreed
    with the two other hand-written lists elsewhere in this documentation; the
    registry held 44.

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
