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

## Two rules under `labels_raw`

Everything a person authored about the data lives in `labels_raw`, and it follows one
of two rules depending on how it got there.

| | An uploaded label file | A saved series, such as `keypoints/` |
| --- | --- | --- |
| What it is | The current truth about a sequence | One saved state of an editor |
| When it changes | Whatever was computed from it is stale | A new revision is written beside the old |
| Who reads it | A converter reads the current file | A consumer names the revision it read |

The second rule exists because an annotation tool saves often, and because a trained
model has to be answerable for the exact annotations it saw. If saving replaced the
previous state, the model's answer would change underneath it. So a revision is never
rewritten, a save that changed nothing writes nothing, and revisions simply
accumulate.

Which rule applies is declared, never guessed from a folder name: a series directory
carries a marker file, and a folder you happen to call `keypoints` elsewhere is an
ordinary folder.

## Libraries: a dataset that holds models

A model trained on annotations from several datasets belongs to none of them. It lives
in a **library**, which is an ordinary dataset used for that purpose, and other
datasets **link** it:

```yaml
libraries:
  - id: lab
    path: ../libraries/lab
    uuid: 6f1c...            # the library's own id, recorded when the link was made
```

A link is to models what a source is to raw files: a declared place outside the
dataset that it reads and never writes. A model reference that the dataset itself
cannot resolve is looked up in each linked library, so a model is named by run id the
same way wherever it is used, and that name does not change when the library moves.

Training in a library **copies** the annotated images it uses. That costs disk and buys
independence: a dataset the library drew from can be archived without breaking any
model trained from it.

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
