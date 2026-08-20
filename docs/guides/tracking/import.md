# Import tracks you already have

If tracking was done somewhere else, mosaic can read and convert the files for further analysis.  Three steps: declare where the files are, scan, convert.

```bash
mosaic sources add -m dataset.yaml --kind tracks \
    --path /data/trex_out --patterns '*.npz' --src-format trex_npz
mosaic scan -m dataset.yaml --kind tracks
mosaic convert-tracks -m dataset.yaml
```

That writes `tracks/<variant>/<group>__<sequence>.parquet`, one per entry, and a row
per entry in the tracks index. From here every feature in the library reads them.

The same thing from Python:

```python
from mosaic.core.dataset import open_dataset

ds = open_dataset("dataset.yaml")
ds.scan_tracks()
ds.convert_all_tracks()
```

## The formats mosaic reads

Eight converters are registered. `--src-format` names one, and the choice also fixes
which schema the output carries.

| `--src-format` | Reads | Emits |
| --- | --- | --- |
| `trex_npz` | TRex per-individual `.npz` that records `cm_per_pixel` (TRex 2.0.0 and later) | `trex_v2` |
| `trex_npz_scaled` | The same export from before TRex recorded that factor, with the factor supplied | `trex_v2` |
| `trex_npz_cm` | The same export where the factor is not recoverable, kept in centimeters | `mosaic_cm_v1` |
| `sleap_analysis_h5` | SLEAP HDF5 from `sleap-convert --format analysis` | `mosaic_v1` |
| `deeplabcut` | DeepLabCut `.csv` / `.h5`, single- and multi-animal — and Lightning Pose, which writes the same shape | `mosaic_v1` |
| `ultralytics_tracks` | Ultralytics tracker predictions parquet | `mosaic_v1` |
| `calms21_npy` | CalMS21 `.npy` arrays | `mosaic_v1` |
| `calms21_json` | The `.json` spelling of CalMS21 | `mosaic_v1` |

[The track formats reference](../../reference/track-formats.md) lists each with its
parameters, and [the schema families](../../concepts/tracks.md#the-schema-families)
explain why the three TRex rows differ.

## Several recipes over one dataset

A dataset can hold more than one tracks variant: the same entries converted by
different recipes, each in its own `tracks/<variant>/` directory. Different entries
carrying different variants enables comparison of tracking results.

When a tracks entry carries exactly one variant, features resolve it automatically. When it carries more than one, you need to name the variant: `--tracks-run-id` on `mosaic run`, or `tracks_run_id=` on `run_feature` -- if not, mosaic raises an error saying that specification is needed.

## If your format is not on the list

Write a converter — it is one class and one method. See [Write a
converter](write-a-converter.md).
