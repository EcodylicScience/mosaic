# Manifest

`dataset.yaml` is what makes a directory a mosaic dataset. This module holds the
format -- the root table, the declared scan sources, the dataset's notes and
typed tags -- with no import from `Dataset`, which holds one.

Two rules run in opposite directions and are the reason the module exists:

- **Roots live inside the dataset.** A root carries that root's own `index.csv`,
  so a root pointing outward would put the index outside too, and the dataset
  would stop being the thing you can copy, archive or sync.
- **Sources are expected to point outside.** A source names storage elsewhere;
  its files are recorded by absolute `abs_path` into an index that stays inside.
  A source directory is never created, and never walked at load time.

::: mosaic.core.manifest
    options:
      show_source: true
      members_order: source
