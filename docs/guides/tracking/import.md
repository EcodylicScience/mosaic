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

`--source-format` converts the raw files of one format only, and `--json` reports the
tracks variants the conversion wrote into.

A run in which some entries fail reports `partial` and exits 0. A run in which every
entry fails reports `failed` and exits 1; `converted` counts only the entries written
or already current. `--json` also gives `written`, the tables this call wrote, and
`current`, the entries it found current and left as they were, so converting a second
day into a dataset that holds the first reports the first day as `current`. A
`convert-tracks` step that wrote nothing is a cache hit.

## Convert as a recipe step

The conversion can be the first step of a recipe, so one request converts the tracks
and computes on them. Scanning stays a separate command: a conversion step whose
format has no scanned file is refused when the recipe is planned.

```json
{"id": "tracks", "type": "op", "kind": "convert-tracks",
 "params": {"source_format": "trex_npz"}},
{"id": "speed", "type": "feature", "feature": "speed-angvel",
 "inputs": ["tracks"], "tracks": {"step": "tracks"}}
```

`params` takes the converter's settings, and refuses one the converter does not
declare. One step converts one format. The `tracks` reference pins the variant the
step writes, which the plan names before the conversion runs. It is the variant of the
uploaded files: a promoted correction converts into a variant of its own, which a
recipe names by `tracks_run_id` like any other.

On a dataset with no tracks yet, the plan cannot know which entries the conversion
will write, so a step whose identity covers its entries, such as `extract-templates`,
shows as unresolved until the conversion has run. Each step names its run when it
starts.

## The formats mosaic reads

Seventeen converters are registered. `--src-format` names one, and the choice also fixes
which schema the output carries.

| `--src-format` | Reads | Emits |
| --- | --- | --- |
| `trex_npz` | TRex per-individual `.npz` that records `cm_per_pixel` (TRex 2.0.0 and later) | `trex_v2` |
| `trex_npz_scaled` | The same export from before TRex recorded that factor, with the factor supplied | `trex_v2` |
| `trex_npz_cm` | The same export where the factor is not recoverable, kept in centimeters | `mosaic_cm_v1` |
| `sleap_analysis_h5` | SLEAP HDF5 from `sleap-convert --format analysis` | `mosaic_v1` |
| `deeplabcut` | DeepLabCut `.csv` / `.h5`, single- and multi-animal — and Lightning Pose, which writes the same shape | `mosaic_v1` |
| `ultralytics_tracks` | Ultralytics tracker predictions parquet | `mosaic_v1` |
| `calms21_npy` | CalMS21 `.npy` arrays, when [you trust the files](#files-mosaic-will-not-unpickle) | `mosaic_v1` |
| `calms21_json` | The `.json` spelling of CalMS21 | `mosaic_v1` |
| `zebrafish_3d_workflow_csv` | The per-experiment `.csv` of a zebrafish 3D tracking workflow: `frame_index` and `fish{n}_{part}_{axis}` columns in centimeters, empty fields as gaps. It records no frame rate. The workflow's default rate, and the published release's, is 140 fps: set `fps_default: 140` or pass `fps=140` for those files | `mosaic_world_v1` |
| `tracktor_corrected_tsv` | The corrected `*_tracked_fixed_final.tsv` of a Tracktor run, as the DGRP looming release writes it: one arena per file, `Fly<n>` ids that follow the animals. Time is its `seconds_total`. Frames before 62 are dropped (`first_frame`). Pixels, with `stimuli` and the recording's labels carried as columns | `mosaic_v1` |
| `tracktor_raw_csv` | The raw `*_tracked.csv` of the same run, whose ids swap between animals. It has no time column, so the corrected `.tsv` beside it times the table, and a missing `.tsv` fails the entry | `mosaic_v1` |
| `synthetic` | A [synthetic track file](#synthetic-tracks) describing a 2D simulation, in pixels | `mosaic_v1` |
| `synthetic_3d` | A synthetic track file describing a 3D simulation, in meters | `mosaic_world_v1` |
| `window_mosaic_v1` | A [window file](#windows-of-another-dataset) that cuts frames out of the `mosaic_v1` tracks of another dataset | `mosaic_v1` |
| `window_trex_v2` | A window file over `trex_v2` tracks | `trex_v2` |
| `window_mosaic_cm_v1` | A window file over `mosaic_cm_v1` tracks | `mosaic_cm_v1` |
| `window_mosaic_world_v1` | A window file over `mosaic_world_v1` tracks | `mosaic_world_v1` |

[The track formats reference](../../reference/track-formats.md) lists each with its
parameters, and [the schema families](../../concepts/tracks.md#the-schema-families)
explain why the three TRex rows differ.

## The frame rate of an imported table

A SLEAP, DeepLabCut, Ultralytics or zebrafish 3D workflow file records frame numbers and
no frame rate, so mosaic times each entry (`time` is `frame` divided by the rate) at the
first of these that it has:

1. The `fps` parameter, `--params '{"fps": 25}'`.
2. The frame rate that the entry's indexed media measured.
3. The dataset's `meta.fps_default`.

An entry with none of the three fails, the run reports `partial`, and the table is
not written at a guessed rate. Scan the entry's media, set `fps_default` in the
manifest, or pass `fps`.

An earlier mosaic never read the media's rate. It timed a table at `fps`, else
`fps_default`, else 30 fps, whether or not the media was indexed. Convert again any
table that was converted without `fps` or `fps_default`.

## Tracktor tables carry their own time

A Tracktor recording drops frames, so `frame` over a rate gives a wrong time, and the
two Tracktor formats take no frame rate. `time` is the file's `seconds_total`, and the
entry's media rate and `fps_default` never reach the table. Every feature that turns a
step into seconds reads `time`: `speed-angvel`, `collective-motion-metrics`,
`trajectory-smooth`, the pair and response features and `social-motion-summary` divide
by the seconds between rows, and `label-statistics` sums them over a bout. A method that
reads its rows as evenly spaced (the Savitzky-Golay features, the wavelets,
`temporal-stack` and `hidra`) measures one rate from `time` and warns that the steps
are uneven, naming the largest gap. To give those a uniform grid, run `resample-tracks`
first. It interpolates on `time`. Positions are pixels. For millimeters, run `scale-to-cm` with
`cm_per_pixel` set, such as `0.019` for a 30 mm arena that is 160 pixels across.

```bash
mosaic sources add -m dataset.yaml --kind tracks --path /data/tracktor \
    --patterns '*_tracked_fixed_final.tsv' --src-format tracktor_corrected_tsv
```

## Synthetic tracks

A synthetic track file describes a simulated world, one file per sequence, and
mosaic converts it into a tracks table. Use it to try a feature or a pipeline on data
whose truth you know, or to size a job before the real tracks exist. The same file and
parameters give the same table.

```toml
format = "mosaic-synthetic/1"
seed = 7
dimensions = 2          # 2 for synthetic, 3 for synthetic_3d
frames = 300000
fps = 30.0
individuals = 30        # 1 to 300
burn_in_frames = 0      # simulated and not written

[arena]
shape = "box"           # "box" or "disk"
width = 1000.0          # box: x spans [0, width]
height = 1000.0         # box: y spans [0, height]
# radius = 500.0        # disk: centered at (radius, radius)
# depth = 0.5           # 3D only: z spans [0, depth]
# start_radius = 100.0  # start within this distance of the center. Default: anywhere

[motion]
mean_speed = 100.0      # per second
speed_sd = 30.0         # default 0
speed_tau_s = 1.0       # speed autocorrelation time in seconds, default 1
turning_sd = 0.5        # heading noise in radians per sqrt(second), default 0.5

[zones]                 # optional
repulsion = 10.0
orientation = 60.0
attraction = 150.0
field_of_view_deg = 270.0   # default 270
max_turn_rate = 6.0         # radians per second, default 6

[body]                  # optional: without it the table has the center alone
keypoints = 5           # 0 to 32, default 0
length = 30.0           # required when keypoints is above 0
keypoint_sd = 1.0       # position noise per keypoint, default 0
confidence_sd = 0.05    # default 0.05

[dropouts]              # optional: without it every row is written
id_rate = 0.001         # per-frame chance that a present id goes missing
id_mean_frames = 10.0   # mean length of a missing run, default 10
gap_rate = 0.0005       # per-frame chance that a frame gap starts
gap_mean_frames = 5.0   # mean length of a frame gap, default 5
```

`format`, `seed`, `dimensions`, `frames`, `fps` and `individuals` are required, with an
`[arena]` sized for its shape and a `[motion]` that gives `mean_speed`. The other keys
take the defaults in the comments. Without `[zones]` the individuals walk independently.
With it they follow the repulsion, orientation and attraction zones of Couzin, Krause,
James, Ruxton and Franks (2002). A key mosaic does not know is refused, and the error
names the file and the key.

```bash
mosaic sources add -m dataset.yaml --kind tracks \
    --path /data/synthetic --patterns '*.toml' --src-format synthetic
mosaic scan -m dataset.yaml --kind tracks
mosaic convert-tracks -m dataset.yaml
```

The file name without `.toml` is the sequence. Every length in the file is in the unit
of the table it produces, and speeds are per second:

- `synthetic` reads `dimensions = 2` and writes `mosaic_v1`, in pixels.
- `synthetic_3d` reads `dimensions = 3` and writes `mosaic_world_v1`, in meters with
  `Z` up. The arena needs a `depth`.

A file of the other dimension is refused, and the message names the format that reads
it. `time` is `frame` divided by `fps`, and the table writes `poseX{k}`, `poseY{k}` and
a confidence `poseP{k}` for each keypoint. A file is refused when its table would contain
more than 600 million values (lower `frames`, `individuals` or `keypoints`) or when a
step at `mean_speed` plus four `speed_sd` is longer than the arena (lower `mean_speed`
or raise `fps`).

### Ids after a gap

By default an individual keeps one id, and the table records that ids follow animals.
With `relabel_after_gap` on a file with `[dropouts]`, an individual missing in the
previous frame, from a dropout or a frame gap, takes the next unused id when it returns,
as a tracker that loses and re-finds an animal does. The table then records ids as not
following animals, and a step that needs ids following an animal refuses it.

```bash
mosaic convert-tracks -m dataset.yaml --params '{"relabel_after_gap": true}'
```

## Windows of another dataset

A window file cuts spans of frames out of the tracks of another dataset and writes each
as an entry of the dataset you convert into. Use it to test on a stretch of a long
recording, or to make a smaller dataset from a large one. The source dataset is read and
not written.

```toml
format = "mosaic-window/1"
source = "../guppies/dataset.yaml"   # a manifest or its directory, from this file
tracks_run_id = "convert-trex_npz.0.3-6bb5efbf05"   # the source's tracks variant

[[windows]]
group = ""                # the source entry
sequence = "trial_01"
start = 0                 # first frame, inclusive. Default: the entry's first
stop = 200000             # last frame, exclusive. Default: past the entry's last
as_group = ""             # the written entry's group. Default: the source group
as_sequence = "trial_01_first_part"
# Default: the source sequence for a window without bounds, else
# "<sequence>_f<start>-<stop>", with "first" or "end" for a bound left out.
```

A leading `~` of `source` expands to your home directory. `format`, `source`,
`tracks_run_id` and at least one window are required, and a window needs its
`sequence`. Two windows that write the same entry are refused, as is a window with
`start` at or above `stop`.

```bash
mosaic sources add -m analysis/dataset.yaml --kind tracks \
    --path /data/windows --patterns '*.toml' \
    --src-format window_trex_v2 --multi-sequences-per-file
mosaic scan -m analysis/dataset.yaml --kind tracks
mosaic convert-tracks -m analysis/dataset.yaml --params '{"carries_tracklets": true}'
```

`--multi-sequences-per-file` is required, because one window file writes several
entries. A window that cannot be cut, such as one with no rows in its frames, fails as a
separate entry and the others are written.

### Choosing the source variant and format

`tracks_run_id` names the source's tracks variant, the directory under its `tracks/`.
List the variants of the source without changing it, and read each `run_id`:

```bash
mosaic inventory -m /data/guppies/dataset.yaml --kind tracks-variant --json
```

To see the schema and tracklets answer of each entry as well, read the tracks index.
Pass `ensure_roots=False` so that opening the source creates nothing in it:

```python
from mosaic.core.dataset import open_dataset
from mosaic.core.pipeline.tracks_index import read_tracks_index

source = open_dataset("/data/guppies/dataset.yaml", ensure_roots=False)
tracks = read_tracks_index(source)
tracks[["group", "sequence", "run_id", "std_format", "carries_tracklets"]]
```

The format follows the source's schema, `std_format`: `window_mosaic_v1` for
`mosaic_v1`, `window_trex_v2` for `trex_v2`, `window_mosaic_cm_v1` for `mosaic_cm_v1`
and `window_mosaic_world_v1` for `mosaic_world_v1`. A window of another schema is
refused, and the message names the right format. A source whose tables record a blank
schema or `trex_v1` is refused as well. Run `mosaic upgrade-tracks --apply` on the
source dataset first, then name the variant it writes.

### What a window keeps

A window keeps the source's frame numbers and times. A window of frames 1000 to 2000
starts at frame 1000 and keeps the source's `time`. The rows are the source's, in
their stored order, with every column it has.

The window also records whether ids follow animals, and it must record what the source
does. A `synthetic` source converted without `relabel_after_gap`, or a TRex source,
records `true`, and the command above passes it. A conversion with another answer is
refused, and the message names the answer to pass. Leave `carries_tracklets` unset for a
source that records none.

A window is not rewritten when its source is converted again under the same variant. Run
`mosaic convert-tracks -m analysis/dataset.yaml --overwrite` to rewrite the windows.
A source converted into another variant needs that variant's `tracks_run_id` in the file.

## Files mosaic will not unpickle

A raw file can come from someone else, or arrive in an upload. mosaic reads TRex `.npz`,
SLEAP `.h5` and DeepLabCut `.h5` files without running anything stored in them, and
refuses a file that is malformed or declares more array data than the limit allows. A
refused file fails its own entry, and the message names the file and the reason. `mosaic
convert-tracks` warns, converts the rest and reports `partial`. As a recipe step,
`convert-tracks` records an entry error for each refused file, writes the rest, and fails
the step; rerun it once the file is fixed or removed. A refused file that contains
several sequences, such as a CalMS21 file, fails as one file. A sequence of a readable
file that fails to convert is a failed entry named by its group and sequence, and the
file's other sequences are written.

Two formats can refuse a file you expected to load:

- **CalMS21 `.npy`** stores a pickled dict, which can run code when it loads. mosaic
  loads it only with `MOSAIC_TRUST_PICKLED_FILES=1`. The original CalMS21 `.json` loads
  without the flag, so use `calms21_json` where you have it.
- **DeepLabCut `.h5`** from DeepLabCut or Lightning Pose loads without the flag. A file
  in a layout mosaic does not recognize is refused. Export it as DeepLabCut CSV, which
  always loads, or set the flag to read the file with pandas.

```bash
export MOSAIC_TRUST_PICKLED_FILES=1
mosaic convert-tracks -m dataset.yaml
```

Set the flag only for files you made or got from a source you trust. Exactly `1` turns
it on. A deployment that accepts uploads must never set it. The BORIS pickle label
converter reads the same flag.

`MOSAIC_RAW_ARRAY_LIMIT_BYTES` sets the most array data one raw file may declare, 2 GiB
(`2147483648`) by default. Raise it for a larger export. A value that is not an integer
raises an error naming the variable.

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
