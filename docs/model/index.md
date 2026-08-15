# Model

A **global** feature fits once over a collection of sequences and then applies to each
of them. That is where every model in mosaic lives: there is no separate model API,
no separate storage, and no separate cache — a trained model is a feature run, tagged
with the same kind of `run_id` as everything else.

Fourteen features are global. [The features reference](../reference/features.md#global-fit-once-apply-everywhere)
lists them all with their parameters.

## Unsupervised: embed, then cluster

The usual chain, and the one the
[CalMS21 notebook](../analyze/examples.md#calms21-end-to-end) walks:

1. **`extract-templates`** subsamples per-sequence feature data into a representative
   matrix — fitting a global model on every frame of every sequence is neither
   affordable nor better.
2. **`global-scaler`** fits a standard scaler on those templates and applies it per
   sequence.
3. **`global-tsne`** fits an openTSNE embedding on the scaled templates and maps every
   sequence into it. On Linux with CUDA, installing `faiss-gpu` accelerates the kNN
   step.
4. **`global-kmeans`** or **`global-ward`** clusters the embedding, assigning each
   sequence's frames by nearest centroid or 1-NN.

`compute_cluster_label_agreement` scores the result against manual labels when you
have them.

## Sequence models: behavioral syllables

**`arhmm`** fits an autoregressive hidden Markov model over pose dynamics and emits a
syllable per frame. It is mosaic's own implementation and carries no usage
restriction.

**`kpms`** drives [keypoint-MoSeq](https://keypoint-moseq.readthedocs.io/), which is
the better-known method and is **licensed for non-commercial research and academic
use only**. It runs in a separate environment you build yourself, and mosaic refuses
to start it until `MOSAIC_KPMS_LICENSE_ACCEPTED=1` says the terms apply to your use.
See [Licensing](../licensing.md); `arhmm` is the unrestricted alternative.

## Supervised: classify from labels

Convert manual annotations first — BORIS and CalMS21 formats are registered — then
train:

- **`xgboost`** on labeled templates, multiclass or one-vs-rest, with thresholds,
  class weighting and resampling.
- **`lightning-action`** for temporal action segmentation, which sees the sequence
  rather than the frame.
- **`feral`** for the FERAL V-JEPA video classifier, which reads egocentric crops
  rather than pose.

Pair `temporal-stack` with any of them to give a frame its context window.

## Visual identity

Three models learn who an individual is from egocentric crops, which
`egocentric-crop` cuts from video using the tracks:

| Feature | What it trains |
| --- | --- |
| `global-identity-model` | A classification head on a pretrained image backbone |
| `global-identity-embedding` | Nothing — a frozen backbone plus prototype k-NN |
| `global-identity-dinov2-temporal` | A temporal head over clips, on frozen DINOv2 |

The first two take any timm architecture tag or Hugging Face hub id. **mosaic ships
no weights**, and each backbone carries its own license — the strongest option for
academic wildlife re-identification is non-commercial. See
[Licensing](../licensing.md).

## What a global run stores

Fitted state lands under `models/<name>/<run_id>/`, and the per-sequence application
under `features/<name>/<run_id>/`. Because the `run_id` covers parameters and inputs,
refitting with identical settings is a no-op and a sweep organizes itself — but a
changed upstream feature produces a different `run_id`, which is what stops a model
from being applied over data it was not fitted on.
