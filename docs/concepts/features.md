# Features and composition

Everything mosaic computes is a **feature**: a registered plugin that reads
standardized tracks, or another feature's output, and writes parquet.

That uniformity is the design. There is no separate model API, no separate storage, no
separate cache. A t-SNE embedding, a trained classifier and a per-frame speed are the
same kind of object, which is what lets them compose.

## Two flavors

**Per-frame and per-sequence** features are stateless transforms. They read one
sequence and write one sequence: `speed-angvel`, `nearest-neighbor`, `pair-egocentric`,
`pair-wavelet`, `temporal-stack`.

**Global** features fit once over a collection and then apply per sequence:
`global-scaler`, `global-tsne`, `global-kmeans`, and every trained model. Fitted state
lands under `models/`, the per-sequence application under `features/`.

Two features exist purely to make a choice explicit rather than to compute something
new. `heading` derives an angle from keypoints under a *named* method; `scale-to-cm`
converts pixels to centimeters using a recorded per-video factor. Both put the choice
into a run identifier, which is the whole reason they are features and not columns a
converter writes silently.

## Inputs must align at one entity level

This is the rule that catches most composition mistakes.

A per-individual result has one row per individual per frame. A pair result has one row
per *ordered* pair per frame. They share only `frame`, so merging them would pair every
individual with every pair and fit downstream on rows that never existed.

mosaic refuses that merge and names both levels rather than performing it. Pick one
level — usually the pair features — or run two chains and compare them.

### What identifies a pair row

A pair result is keyed by `(frame, id1, id2, perspective)`, and every column of that
key matters.

Pairs are **ordered**: with two individuals A and B, a frame carries a row for A
looking at B and another for B looking at A. `id1` is the focal — whose frame of
reference the row is written in — and `id2` is the other. `perspective` says which
of the two orderings the row is, `0` for the first and `1` for the mirror.

That makes `(frame, id1, id2)` alone *not* a key, which is why two pair results merge
one-to-one only when both carry `perspective`. It is also why a feature that rewrites
its columns has to carry `perspective` through: an output that drops it holds two rows
per frame with nothing to tell them apart, and mosaic will then refuse to join those
predictions back to the embedding they came from — correctly, since the key was
destroyed upstream rather than at the join.

The same holds for a model that reads frames in order. An autoregressive model reads
each frame after the one before it, so `arhmm` takes each `(id1, id2, perspective)` as a
sequence of its own and writes those columns, with `group` and `sequence`, beside its
states. `lightning-action` and `sequence-classifier` do the same, and train and predict on
each series in frame order.

A symmetric quantity, such as the distance between the two, is written the same on
both rows rather than stored once. One shape with no special case is what lets any
two pair results join.

## Arenas

A tracks table that `assign-arenas` wrote names each row's `arena`, the tank of its
body center (see [Arenas](../dataset.md#arenas)). A feature that pairs individuals pairs
only those of one arena, so two fish of two tanks never form a pair row. A table
without `arena` pairs every two individuals, as before.

Each feature declares how it relates individuals:

- **`none`**: it measures each individual alone, such as `speed-angvel` or `heading`.
- **`arena-aware`**: it pairs within one arena, or reads pairs that its input formed
  so. The `pair-*` family, `approach-avoidance`, `orientation-rel`,
  `neighbor-delta-response`, `interaction-crop-pipeline` and the pair branch of
  `temporal-stack`.
- **`arena-blind`**: it measures across individuals without arenas, such as
  `nearest-neighbor`, the group metrics (`collective-motion-metrics`, `ffgroups`,
  `ffgroups-metrics`, `local-order-metrics`), `social-motion-summary`,
  `attention-target`, `nn-delta-response` and `nn-delta-bins`. A run fails such a
  feature on an entry whose input holds two or more arenas, or whose recording has a
  layout of several arenas while the input names none. `run_feature` serves no output
  it cached before such a layout was declared. A `Pipeline` or recipe step whose run is
  already complete is not run again and keeps that output. Recompute it after declaring
  the layout, with `mosaic run --feature ... --overwrite` or
  `Pipeline.run(force_from=...)`. Run such a feature on a recording of one arena.

## Which tracks a feature reads

A dataset can hold several **tracks variants**: the same entries converted by different
recipes, each in its own `tracks/<variant>/` directory. A TRex conversion and a
DeepLabCut import of the same recording are two variants.

When an entry carries exactly one, it resolves silently. When it carries two genuinely
different recipes, mosaic **raises rather than guessing** — because picking one would
be a silent wrong answer, not an error. Naming the variant answers the refusal.

Different entries carrying different variants is legal and expected: some sequences
tracked, some imported.

The resolved variant enters the feature's run identifier, so a result computed from one
recipe can never be mistaken for the same feature computed from another.

## Reading across a sequence boundary

A windowed feature needs frames from beyond its sequence's edge. `overlap_frames`
supplies them within a **continuous group**, a group whose sequences are declared to be
time divisions of one recording rather than independent trials. Only those sequences
have neighbors. Every other sequence, in the empty group or in a group used as a
namespace, reads only its own frames, and the run prints one note counting them. One
overlapped step can therefore run over a dataset that mixes both. Each sequence's result
records the neighbors it read. Declaring a group, adding a sequence to it or converting
a neighbor again recomputes the sequences whose neighbors changed, and the next run
serves the rest from the cache.

The declaration is checked, not trusted. mosaic verifies that the recorded frame
extents are disjoint and increasing, and names both sequences and both ranges when they
are not. Neither half is sufficient alone: no measurement can establish that two
sequences divide one recording rather than being two recordings numbered consecutively,
and no declaration can be trusted about an axis that exists to be read.

A 6-hour recording is either one sequence covering six hours, or a continuous group
whose sequences are its half-hour divisions. It is not a group of independent
half-hours, in which one frame number would name a different moment in each.

See the [features reference](../reference/features.md) for every registered feature and
its parameters.
