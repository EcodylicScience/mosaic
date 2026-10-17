# Import behavior annotations

Manual behavior annotations — who did what, when — come in as **labels**. Supervised
classifiers train on them, `overlay` draws them, and clustering is scored against them.

Three steps, the same shape as importing tracks: declare where the files are, scan,
convert.

Keypoint annotations for training a pose model are a different kind of label and
arrive a different way, as saved revisions: see [train a pose
model](../tracking/train-a-pose-model.md#5-train-one-model-from-several-datasets).

```bash
mosaic sources add -m dataset.yaml --kind labels \
    --path /data/boris_exports --patterns '*.csv' --src-format boris_aggregated_csv
mosaic scan -m dataset.yaml --kind labels
mosaic convert-labels -m dataset.yaml --kind behavior
```

From Python:

```python
from mosaic.core.manifest import LabelsScanSource

ds.add_scan_source(LabelsScanSource(
    id="boris",
    path="/data/boris_exports",
    patterns=("*.csv",),
    src_format="boris_aggregated_csv",
))
ds.scan_labels()
ds.convert_all_labels(kind="behavior")
```

That writes `labels/behavior/<group>__<sequence>.npz`, one per entry, with a row per
frame per individual. Every feature that reads labels reads them from there.

`convert-labels --json` reports the labels variant it wrote into, and how many files
it wrote and found already present.

## Convert as a recipe step

The conversion can also be the first step of a recipe, so one request converts the
labels and computes on them. The scan stays a separate command: a conversion step
whose format has no scanned file is refused when the recipe is planned.

```json
{"id": "labels", "type": "op", "kind": "convert-labels",
 "params": {"kind": "behavior", "source_format": "boris_aggregated_csv",
            "params": {"fps": 30.0}}}
```

`params` takes the converter's settings below, and refuses one the converter does not
declare. A feature's `labels` field then takes `{"step": "labels"}`, which reads the
labels that step writes, even where the dataset holds another conversion of them.

## The formats mosaic reads

| `--src-format` | Reads |
| --- | --- |
| `boris_aggregated_csv` | BORIS Aggregated Events export, CSV or TSV |
| `boris_pandas_pickle` | BORIS pandas DataFrame pickle |
| `calms21_npy` | CalMS21 `.npy` / `.json` behavior annotations |

All three produce the `behavior` label kind. If your annotation tool is not here, the
label converter registry works the same way the track converter registry does.

## Settings worth knowing

Pass converter parameters as JSON:

```bash
mosaic convert-labels -m dataset.yaml --kind behavior \
    --params '{"fps": 30.0, "subject_id_map": {"resident": 0, "intruder": 1}}'
```

| Parameter | Does |
| --- | --- |
| `fps` | Converts BORIS's second-based start and stop times into frames. Set it when the recording's frame rate is not on the media index |
| `subject_id_map` | Maps BORIS subject names onto the `id` values in your tracks. Get this wrong and the labels attach to the wrong animal |
| `background_label` | What an unannotated frame is called. `"none"` by default, and always label id 0 |
| `pair_behaviors` | Behaviors that belong to a pair rather than to one individual |
| `include_point_events` | Whether zero-duration events become single-frame labels |

`subject_id_map` is the one to check first: BORIS records subject names, tracks record
integer ids, and nothing can align them for you.

## Labeled, background and unlabeled frames

Each converted file records its `background_id`: the label a frame with no annotated
event takes. A BORIS file's background is its `background_label`, at id 0. CalMS21 has
no background class, because all four of its classes are behaviors, so its files
record `-1`.

A label of `-1` means **unlabeled**: nobody annotated that row, and it is not any
class. Templates leave unlabeled rows out, and scoring drops and counts them.

Labels attach to rows, not just to frames. A row of a pair feature takes the label of
its own direction, so a CalMS21 label, which describes the resident's behavior toward
the intruder, labels the resident-focal row and leaves the intruder-focal row
unlabeled. An individual label applies to that individual's rows and to the pair rows
focal on it, and a label with no subject applies to every row of the frame.

Files converted before mosaic recorded `background_id` still load. A BORIS file takes
its background from the conversion's recorded parameters. Any other such file warns
(`LegacyLabelFileWarning`) and reads unannotated frames as `-1`, or as a class named
`none` if it has one. Convert the labels again to record the background and silence
the warning. The dataset then holds two label variants, so pass `labels_run_id=` to
choose one, or remove the old one.

## Check the result

```python
from mosaic.core.helpers import load_labels_auto

labels = load_labels_auto(
    ds.get_root("labels") / "behavior" / "<run_id>" / "day1__trial01.npz"
)
```

Labels are stored per conversion run, in `labels/behavior/<run_id>/<entry>.npz`.
`load_labels_auto` returns one label per frame, with the file's background for a frame
no event covers.

`mosaic inventory` reports label coverage beside everything else. The surest check is
visual: render an overlay and watch whether the labels track the right animal.

## Next

- [Train a behavior classifier](train-a-classifier.md) — the labels' main consumer.
- [Render an annotated video](../media/render-a-video.md) — draw them back onto the
  recording to check they landed on the right animal.
