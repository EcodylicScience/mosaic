# Classify mouse behavior with HiDRA

[HiDRA](https://github.com/neuroethology/HiDRA) is a pretrained classifier of mouse
social behavior from pose. It needs no labels of your own. The `hidra` feature runs it
on two to four mice per arena, and writes per frame and per ordered pair the probability
that `id1` performs a behavior toward `id2`.

HiDRA's 82 heads cover 37 behaviors. Each head is one lab's definition of one behavior,
and labs define behaviors differently: two labs' `attack` heads disagree. Run
`hidra-predict --list-heads` to see them, then pick the heads whose definitions fit
your annotation.

## Install HiDRA

HiDRA is not installed with mosaic. It runs on Linux, in an environment of its own:

```bash
python3.12 -m venv ~/hidra-env
~/hidra-env/bin/pip install torch --index-url https://download.pytorch.org/whl/cu126
~/hidra-env/bin/pip install 'hidra[torch] @ git+https://github.com/neuroethology/HiDRA'
export MOSAIC_HIDRA_BIN=~/hidra-env/bin/hidra-predict
```

The CUDA 12.6 wheels still run on Pascal GPUs (GTX 10-series), which newer CUDA builds
of torch drop. Without a GPU, use the `cpu` index; HiDRA then runs much slower.

Then fetch the weights of the revision the feature uses (about 700 MB):

```bash
HIDRA_MODELS_DIR=~/.cache/mosaic/hidra/b5b56d316117de5900832de2dea6f2b7fea6f75f \
  ~/hidra-env/bin/hidra-download-models --revision b5b56d316117de5900832de2dea6f2b7fea6f75f
```

`MOSAIC_HIDRA_MODELS_ROOT` moves `~/.cache/mosaic/hidra` elsewhere. The feature
refuses a run whose weights are missing, naming this command.

## Run it

```python
from mosaic.behavior.feature_library.hidra_feature import HidraFeature

hidra = HidraFeature(params={
    "pose": {"pose_n": 6, "keypoint_names": [
        "head_midpoint", "left_ear", "right_ear", "neck", "mouse_center", "tail_base",
    ]},
    "keypoint_map": {
        "head_midpoint": "nose",
        "left_ear": "ear_left",
        "right_ear": "ear_right",
        "neck": "neck",
        "mouse_center": "body_center",
        "tail_base": "tail_base",
    },
    "heads": [
        {"lab": "ReflectiveManatee", "action": "sniff"},
        {"lab": "UppityFerret", "action": "huddle"},
        {"lab": "GroovyShrew", "action": "selfgroom"},
    ],
    "pix_per_cm": 16.0,
})
result = ds.run_feature(hidra)
```

- **`keypoint_map`** names which of HiDRA's bodyparts each of your keypoints is.
  HiDRA reads `tail_base`, `ear_right`, `ear_left`, `nose`, `neck`, `body_center` and
  `tail_tip`. A bodypart you cannot map is treated as unobserved, which HiDRA handles.
  Its smallest ensemble member uses only `tail_base`, both ears and `nose`, so map
  these when you can.
- **`pix_per_cm`** matters. HiDRA computes in centimeters, and a wrong scale lowers
  every head's accuracy. Check it against a body length: a mouse is about 8 to 9 cm
  from nose to tail base.
- **`heads`** sets the cost. Each lab costs one pass of the five-model ensemble,
  however many of its heads you choose.

## The output

One row per frame and ordered pair, keyed `(frame, id1, id2, perspective)`, with:

- `prob_<action>__<lab>`: the probability that `id1` performs the action toward `id2`.
  A self-directed head, such as `selfgroom`, has one value per animal, repeated on each
  of its rows.
- `call_<action>__<lab>`: whether the probability reaches the head's threshold.
- `predicted_label`: the called head with the highest probability, or `none`.

## Which HiDRA produced a run

HiDRA is installed from git, so its code can change while the weights revision stays
pinned. Each run records the install that predicted its entries under
`hidra_provenance/` in its run folder: the HiDRA version, the commit it came from, the
bundled thresholds and the head table, with `entries/<entry>.txt` naming the record of
each entry. An update of HiDRA between two runs then shows as two records.

## Limits

- Each arena must hold two to four individuals; without an `arena` column, the entry
  is one arena. HiDRA sees each arena's mice apart from the others', and an arena of
  one mouse is left out.
- An entry must have one frame rate. A sequence joined from clips recorded at
  different rates is refused.
- The weights revision is part of the run id, because probabilities change between
  revisions.
- Fine-tuning a head on your own labels is not supported yet.
