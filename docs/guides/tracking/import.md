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

A variant is named `<kind>.<version>-<digest>`, such as
`convert-trex_npz.0.3-6bb5efbf05`, and that name is its directory under `tracks/`.
When an entry carries exactly one variant, features resolve it automatically. When it
carries more than one recipe, mosaic raises an error naming them, and you name the one
to read. Rows converted before mosaic recorded variants do not count, because a named
variant supersedes them.

```python
from mosaic.behavior.feature_library import SpeedAngvel

ds.run_feature(SpeedAngvel(), tracks_run_id="convert-trex_npz.0.3-6bb5efbf05")
ds.load_tracks("day1", "trial01", run_id="convert-trex_npz.0.3-6bb5efbf05")
```

```bash
mosaic run -m dataset.yaml --feature speed-angvel \
    --tracks-run-id convert-trex_npz.0.3-6bb5efbf05
```

`build_manifest` and `load_values` in `mosaic.core.pipeline` take `tracks_run_id=`
too. `load_tracks` and `drop_entries` take the same value as `run_id=`. An empty
string, `""`, names the tables converted before mosaic recorded variants, whose index
rows have no `run_id`.

To stop an entry resolving to a variant, drop that variant's index rows. The parquet
stays on disk unless you also pass `delete_files=True`:

```python
from mosaic.core.scope import Scope

ds.drop_entries(
    scope=Scope(entries=[("day1", "trial01")]),
    run_id="convert-trex_npz.0.3-6bb5efbf05",
)
```

## A corrected track set

A track set you corrected by hand, such as TRex output with its identity swaps fixed,
cannot be recomputed. It becomes source data. `promote_correction` copies it into
`tracks_raw/` as the entry's next revision, `corrected.rev1`, `corrected.rev2`, and
indexes it:

```python
from mosaic.core.pipeline.promotion import promote_correction

report = promote_correction(
    ds, "day1", "trial01", "/data/fixed/trial01_fish0.npz",
    src_format="trex_npz", derived_from="trex.0.2-abc123def0",
)
report.blocked        # what the promotion would make stale
```

That call only reports. Pass `apply=True` to copy and index the file, then run
`ds.convert_all_tracks()`. A promotion is blocked while anything built from the entry's
raw tracks exists, its converted table included. `force=True` proceeds and deletes
nothing, leaving those results stale until you re-run them. Pass a list of paths for a
correction spread over several files. `derived_from` names the tracker run you
corrected, which lets `mosaic sweep-tracking` reclaim that run's working directory
without waiting out its retention window. `promote_correction` raises `ValueError`
when `derived_from` names a run that tracked a [media variant](../media/preprocess.md),
whatever `force` says, because that run's output is in the variant's pixels and frames.

Only the newest revision converts, and it converts into a variant of its own beside
the uncorrected one. Name it as above, or drop the uncorrected variant's rows. In the
tracks index, which `read_tracks_index(ds)` from `mosaic.core.pipeline.tracks_index`
returns, the correction's row has a `source_abs_path` whose file name begins
`corrected.rev<N>`, such as `corrected.rev1.npz`.

## If your format is not on the list

Write a converter — it is one class and one method. See [Write a
converter](write-a-converter.md).
