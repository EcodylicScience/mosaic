# Examples

Five worked notebooks live in
[`notebooks/`](https://github.com/EcodylicScience/mosaic/tree/main/notebooks) and
render directly on GitHub, outputs included.

Four of them fetch their own data from Hugging Face, so they run as written --
set nothing, change nothing, execute from the top. Each says how large the
download is in its first markdown cell. The two tracking notebooks additionally
drive external tools ([TREx](https://trex.run), and Ultralytics or POLO for
training) which mosaic runs as separate programs and does not install; both say
so where it matters and skip those sections cleanly when the tool is absent.

## Start here

### CalMS21, end to end

[**calms21-template.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-template.ipynb)
&mdash; data: [mosaic-example-calms21](https://huggingface.co/datasets/EcodylicScience/mosaic-example-calms21)

The fullest path through mosaic in one file, on the CalMS21 mouse
social-interaction dataset:

1. declare sources, scan them, convert one file into both tracks and labels;
2. `heading`, `pair-egocentric` and `pair-posedistance-pca` features;
3. wavelet expansion, global scaling, and a t-SNE embedding;
4. k-means and Ward clustering, scored against ground-truth labels;
5. supervised classification with `extract-labeled-templates` and `xgboost`,
   over temporally stacked features;
6. predictions drawn back onto the embedding.

The idea it exists to show is that all of those are the same kind of thing -- a
feature run, addressed by a hash of its parameters and inputs -- so each one
caches and a parameter sweep organises itself.

## Writing a converter for your own tracker

Both of these define a converter inside the notebook and run it there, which is
the shortest complete version of
[writing a converter](guides/tracking/write-a-converter.md).

### Collective motion in shiners

[**collective-motion-shiners.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb)
&mdash; data: [mosaic-example-shiners](https://huggingface.co/datasets/EcodylicScience/mosaic-example-shiners)

A converter for SchoolTracker's `_fov.h5`, a format where several files make one
recording and neither identity nor frame numbering survives the joins -- so it
shows the repair as well as the conversion. Then collective motion across four
group sizes: polarization and rotation order parameters and the discrete states
they imply.

It is also the example of a tracker whose `X`/`Y` is **not** what mosaic's schema
normally promises, and of what to do about that honestly.

### Collective motion in zebrafish

[**collective-motion-zebrafish.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-zebrafish.ipynb)
&mdash; data: [mosaic-example-zebrafish](https://huggingface.co/datasets/EcodylicScience/mosaic-example-zebrafish)

A converter for a tracker with **no pose keypoints** (Ctrax/JAABA `trx`), which is
the case that motivated making keypoints optional in the schema. Then collective
motion, and the `nearest-neighbor` to `nn-delta-response` to `nn-delta-bins` chain
that turns neighbour positions into a social-force map.

Its other lesson is about trusting a tracker's metadata: six fields in these files
are stale, wrong, or a factor of two out, and each looks perfectly usable on its
own.

## Training a model and tracking with it

Both close the loop -- published tracks become pseudo-annotations, those train a
detector, and a tracker then runs the model the dataset produced. Neither ships
model weights; the section on each says why.

### CalMS21: a pose model, then TREx

[**calms21-pose-training-and-tracking.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/calms21-pose-training-and-tracking.ipynb)
&mdash; data: [mosaic-example-calms21-pose](https://huggingface.co/datasets/EcodylicScience/mosaic-example-calms21-pose)

Frame sampling, tracks turned into YOLO pose annotations through mosaic's
`AnnotationSet`, `train-pose`, then TREx tracking with that model as its detector
and visual identification on. Ends on an annotated video.

It is deliberately honest that the resulting tracks are not good tracks: the point
is the shape of the workflow and that every artifact along it is addressed and
reproducible.

### Shiners: a point model, and two ways to give TREx a body

[**shiners-polo-tracking.ipynb**](https://github.com/EcodylicScience/mosaic/blob/main/notebooks/shiners-polo-tracking.ipynb)
&mdash; data: [mosaic-example-shiners-polo](https://huggingface.co/datasets/EcodylicScience/mosaic-example-shiners-polo)

Published tracks become POLO point annotations, and TREx then tracks the same
footage two ways: with the trained point detector, and with no model at all using
its own background subtraction. The comparison is the payload -- the two routes
disagree about what `X`/`Y` means, and the notebook measures the difference rather
than asserting it.

Route B needs no model and no training environment, so it is what runs by default.
