# Import behavior annotations

Manual behavior annotations — who did what, when — come in as **labels**. Supervised
classifiers train on them, `overlay` draws them, and clustering is scored against them.

Three steps, the same shape as importing tracks: declare where the files are, scan,
convert.

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
| `background_label` | What an unannotated frame is called. `"none"` by default |
| `pair_behaviors` | Behaviors that belong to a pair rather than to one individual |
| `include_point_events` | Whether zero-duration events become single-frame labels |

`subject_id_map` is the one to check first: BORIS records subject names, tracks record
integer ids, and nothing can align them for you.

## Check the result

```python
from mosaic.core.helpers import load_labels_auto

labels = load_labels_auto(ds.get_root("labels") / "behavior" / "day1__trial01.npz")
```

`mosaic inventory` reports label coverage beside everything else. The surest check is
visual: render an overlay and watch whether the labels track the right animal.

## Next

- [Train a behavior classifier](train-a-classifier.md) — the labels' main consumer.
- [Render an annotated video](../media/render-a-video.md) — draw them back onto the
  recording to check they landed on the right animal.
