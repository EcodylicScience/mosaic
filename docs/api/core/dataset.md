# Dataset

The central orchestrator for mosaic workflows. Manages named roots, provides
methods for media indexing, track conversion, label management, feature
extraction, and model training.

## Getting one

Two front doors, and the difference between them is whether the dataset exists
yet. Note that the `Dataset` constructor is neither: it is built around a
manifest *path* rather than around loaded content, so `Dataset(path)` alone
yields an object that has read nothing and whose roots are all empty.

::: mosaic.core.dataset.open_dataset
    options:
      show_source: true

::: mosaic.core.dataset.new_dataset_manifest
    options:
      show_source: true

## The dataset object

::: mosaic.core.dataset.Dataset
    options:
      show_source: true
      members_order: source
