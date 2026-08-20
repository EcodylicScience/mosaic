# Datasets, roots and sources

A mosaic dataset is a directory with a `dataset.yaml` manifest. The manifest names two
different kinds of location, and the difference is the point.

## Roots live inside the dataset; sources do not

**Roots** are where mosaic writes. They live inside the dataset, so an `index.csv`
travels with it when the dataset is copied, archived or synced to another machine.

```
dataset.yaml          what this dataset is, and where its files come from
media_raw/            the originals index -- rows may point outside the dataset
media/                transcodes, extracted frames
tracks_raw/           raw tracker output as uploaded
tracks/<variant>/     standardized <group>__<sequence>.parquet, one dir per recipe
labels/<kind>/        converted manual annotations
features/<name>/<run_id>/   one directory per feature run
models/<name>/<run_id>/     one directory per trained model
```

**Sources** are where mosaic reads from. A source may point anywhere — a NAS, another
volume, a colleague's export directory — and its files are recorded by absolute path
into an index that stays inside the dataset. Nothing is copied.

That asymmetry is what lets a dataset be moved without moving terabytes of video, and
what lets two datasets read the same footage without duplicating it.

## What a scan does, and what it leaves alone

A scan **replaces what it claims and preserves everything else.**

A row that no scanned source claims survives the scan — one written by an explicit
assignment, or one pointing at a file outside the dataset. A file removed from a
claimed directory does leave. `--prune-unsourced` opts into dropping unclaimed rows,
and is the only way to lose one.

This matters when you add a source to a dataset that already has an index: scanning
will not quietly repartition what is already there.

There are two source modes:

- A **directory** source globs — extensions, patterns, recursive or not.
- A **files** source claims exactly the paths it lists and nothing beside them.

The second exists because no glob can express an arbitrary subset. Importing eleven of
the thirty clips in a folder is a files source; two files sources may share a directory
as long as their lists are disjoint.

## Identity a scan will not overwrite

A scan refreshes the cells it measured — duration, frame count, resolution — and never
overwrites an identity that a caller **assigned**: which group and sequence a file
belongs to. Without that rule, declaring an existing media directory as a source would
silently repartition every project built on it.

## Groups are an optional namespace

`group` is a required column that may be empty. With `sequence` it forms the composite
key and the filename (`<group>__<sequence>`, or just `<sequence>` when empty). It is
**not** the canonical way to categorize sequences for analysis — flexible, redefinable
grouping is what tags are for.

`group` keeps one structural role: it is the temporal-contiguity key. Windowed features
pull neighboring frames only from within the same group.
