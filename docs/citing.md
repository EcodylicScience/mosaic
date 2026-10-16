# Citing mosaic and the work it builds on

Mosaic is a platform. Most results it produces were produced by somebody else's
method, running in somebody else's software, and mosaic's contribution is that
the data, the parameters, and the runs were organized reproducibly around it.

!!! important "The rule"

    **Cite the tools that produced the science first**, as their authors ask.
    **Cite mosaic second**, as the platform that organized the data and the runs.

    If you tracked with SLEAP through mosaic, your methods section cites SLEAP
    for the pose tracking and mosaic for the pipeline. Not the other way around,
    and not mosaic alone.

## What you owe, by kind

Every component mosaic draws on falls into one of four kinds, and what you owe
depends only on the kind.

| Kind | What it means | What to cite |
| --- | --- | --- |
| **runs** | Mosaic executes it, or downloads and runs its weights. | **Cite it.** You may not realize you used it, so this is the kind to check. |
| **imports** | You ran it yourself; mosaic reads its output. | **Cite it if your data came from it.** You already know whether it did. |
| **implements** | Mosaic implements the published method in its own code. There is no package to cite. | **Cite the method.** |
| **builds on** | Foundation mosaic is built from. | **No citation expected.** |

The distinction that matters is between the first two. Something you **import**
(DeepLabCut, BORIS, CVAT) you chose and ran yourself, so you already know it
belongs in your methods section. Something mosaic **runs** may have been a
default you never typed, which is why this page exists.

To see what a component is and what its authors ask,
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) records the
origin, license and requested citation of the tools mosaic drives, the third-party
code it incorporates and the model weights it can fetch. The
[feature reference](reference/features.md) describes what each feature computes.

## The short answer

For most projects:

1. **Your tracker or pose estimator.** Whatever produced the pose data.
2. **Anything mosaic runs that you turned on.** Behavior models, identity
   backbones, detectors.
3. **The method behind any analysis your paper names.** If your methods section
   says "wavelet spectrogram and t-SNE embedding", cite the method, not only the
   libraries.
4. **Mosaic.**

## Citing mosaic itself

The repository carries a `CITATION.cff`, which GitHub renders as a "Cite this
repository" entry. A software DOI will be added once the project is archived.
Until then, cite the repository and the mosaic version recorded in your run's
`params.json` as `_mosaic_version`.

Mosaic does not require citation. It asks for it on the same terms it asks you to
cite everything else.

## A methods paragraph

Adapt this, filling in from your run's `params.json`:

> Pose tracks were produced with **{tracker}** {version} [ref] and standardized
> and analyzed with mosaic {version} [ref]. **{Analysis}** followed {method}
> [ref], using {key parameters}. Each analysis step is addressed by a content
> hash over its parameters and inputs, recorded in the accompanying dataset.

## Two things this page is not

**It is not the licensing page.** Whether you *may* use a component, and under
what terms, is a different question with different answers. keypoint-MoSeq
prohibits commercial use, TRex requires a paid license for company use, and
MegaDescriptor's weights are non-commercial.
[NOTICE](https://github.com/EcodylicScience/mosaic/blob/main/NOTICE) states
those terms.

Two components make attribution a *license condition* rather than a courtesy, and
both are called out in NOTICE: **keypoint-MoSeq**, whose license requires
acknowledging Harvard as the provider, and **MegaDescriptor**, whose CC-BY-NC
terms require attribution.

**It is not a complete bibliography of the field.** It covers what mosaic
actually draws on. Methods mosaic does not implement are not here, however
relevant.
