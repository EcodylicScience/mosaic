# Train a FERAL video classifier

`feral` fine-tunes a V-JEPA video model to classify behavior from video, in place of
per-frame feature columns. Start from
[Train a behavior classifier](train-a-classifier.md), and install FERAL in an
environment of its own ([installation](../../installation.md#feral)).

## The recipe

This recipe converts CalMS21 tracks and labels, crops a video around each animal, and
trains `feral` on the crops. The `egocentric-crop` step decodes each entry's own media,
so the dataset also needs its videos indexed (`mosaic sources add --kind media`, then
`mosaic scan`, as in [Add sources and scan](../../dataset.md#add-sources-and-scan)).
Save it as `feral-crops.json`:

```json
{
  "schema_version": 1,
  "name": "feral-crops",
  "steps": [
    {"id": "tracks", "type": "op", "kind": "convert-tracks",
     "params": {"source_format": "calms21_npy"}},
    {"id": "labels", "type": "op", "kind": "convert-labels",
     "params": {"source_format": "calms21_npy"}},
    {"id": "crops", "type": "feature", "feature": "egocentric-crop",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "speed", "type": "feature", "feature": "speed-angvel",
     "inputs": ["tracks"], "tracks": {"step": "tracks"}},
    {"id": "labeled", "type": "feature", "feature": "extract-labeled-templates",
     "inputs": [{"step": "speed"}],
     "params": {"labels": {"step": "labels"}, "n_per_class": 500}},
    {"id": "classifier", "type": "feature", "feature": "feral",
     "inputs": [{"step": "crops"}],
     "params": {"templates": {"step": "labeled"}, "labels": {"step": "labels"},
                "unlabeled_as": 3, "training": {"epochs": 10}}}
  ]
}
```

```bash
mosaic pipeline run -m dataset.yaml --recipe @feral-crops.json
```

## What the params do

- **The templates run names the entries, and the videos come from the input.** The
  `labeled` step reads a numeric per-frame feature of the same entries, and only its
  `train` and `test` entries matter to `feral`. The model trains on the videos of the
  `train` entries, evaluates on the `test` entries, and ignores the sampled templates.
- **The input decides the videos.** An `egocentric-crop` or
  `interaction-crop-pipeline` input supplies its crop videos. A whole-sequence model
  takes the tracks as its input and reads each entry's own media, which must be one
  plain video file.
- **`labels` gives every frame its class.** `unlabeled_as` is the label value, in the
  original label space, that a frame no label covers trains as. Unset, a video that
  holds such a frame is left out. `training.val_fraction` of the train entries
  validates each epoch and chooses the best checkpoint.
- **The output is per frame.** `prob_<value>` and `predicted_label` use the original
  label values. The run also writes `feral_model.joblib`, `feral_labels.json`,
  `config.json` and `reports.json`.
- **The scope is not part of the run id.** The templates run names the training entries,
  so widening the scope applies the stored model to the new entries. Ask the first
  request over a scope that holds every train and test entry of the templates run, and
  the step refuses one that does not.

## Apply a model

`model` applies a trained run, and `model_dir` a checkpoint trained outside mosaic
(`model_best.pt` and, optionally, `config.json`). Name one of the three in a step, never
two:

```json
{"id": "predict", "type": "feature", "feature": "feral",
 "inputs": [{"step": "crops"}], "params": {"model": {"step": "classifier"}}}
```
