# Examples

Three worked notebooks live in
[`notebooks/`](https://github.com/EcodylicScience/mosaic/tree/main/notebooks) and
render directly on GitHub, outputs included.

!!! note "They are illustrated results, not runnable tutorials"

    Each notebook reads a dataset that is not bundled with the repository, and the
    paths in the first cells point at the machines they were written on. Read them
    for the shape of an analysis and the code that produces each figure; expect to
    substitute your own dataset before anything runs.

## CalMS21, end to end

[**calms21-template.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb)

The canonical path from a manifest to a trained classifier, on the CalMS21 mouse
social-interaction dataset:

1. build a `Dataset`, index raw tracks, convert them to the standard schema;
2. `pair-egocentric` and `pair-posedistance-pca` features;
3. wavelet expansion, global scaling, and a t-SNE embedding;
4. k-means and Ward clustering, scored against ground-truth labels;
5. supervised classification with `extract-labeled-templates` and `xgboost`, with
   optional temporal-context stacking;
6. predictions drawn back onto the embedding.

If you read one, read this one.

## Collective motion in shiners

[**collective-motion-shiners.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb)

A track converter written inside the notebook — the shortest complete example of
[adding a converter](../adding-a-converter.md) — then the collective-motion features
across four group sizes, with polarization and rotation order parameters and the
discrete collective states they imply.

## Collective motion in zebrafish

[**collective-motion-zebrafish.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-zebrafish.ipynb)

A converter for a tracker with **no pose keypoints** (Ctrax/JAABA `trx`), which is
the case that motivated making keypoints optional in the schema. Then collective
motion, and the `nearest-neighbor` to `nn-delta-response` to `nn-delta-bins` chain
that turns neighbor positions into a social-force map.
