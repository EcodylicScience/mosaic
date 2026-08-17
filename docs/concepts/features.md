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
converts pixels to centimetres using a recorded per-video factor. Both put the choice
into a run identifier, which is the whole reason they are features and not columns a
converter writes silently.

## Inputs must align at one entity level

This is the rule that catches most composition mistakes.

A per-individual result has one row per individual per frame. A pair result has one row
per *pair* per frame. They share only `frame`, so merging them would pair every
individual with every pair and fit downstream on rows that never existed.

mosaic refuses that merge and names both levels rather than performing it. Pick one
level — usually the pair features — or run two chains and compare them.

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
supplies them — but only within a **continuous group**, a group whose sequences are
declared to be time divisions of one recording rather than independent trials.

The declaration is checked, not trusted. mosaic verifies that the recorded frame
extents are disjoint and increasing, and names both sequences and both ranges when they
are not. Neither half is sufficient alone: no measurement can establish that two
sequences divide one recording rather than being two recordings numbered consecutively,
and no declaration can be trusted about an axis that exists to be read.

A 6-hour session is either one sequence covering six hours, or a continuous group whose
sequences are its half-hour divisions — never a group of independent half-hours, which
would make one frame number name a different moment in each.

See the [features reference](../reference/features.md) for every registered feature and
its parameters.
