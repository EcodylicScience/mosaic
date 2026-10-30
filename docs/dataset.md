# The mosaic dataset

A mosaic dataset is a directory holding a `dataset.yaml` manifest, and a fixed set of
named folders declared in it. This enables **consistent naming** for the built-in
caching and run organization, along with **portability** if you want to move the
dataset or analyze certain results separately.

## How a dataset is organized

**Named folders.** The manifest declares one folder per kind of content — media,
tracks, labels, features, models. Everything inside mosaic refers to a location by
root name rather than by path, so nothing depends on where the dataset sits.

**One index row per raw file.** Your recordings may live outside the dataset, on a NAS
or an external volume. Rather than move or copy them, a scan records each one as a row
in an `index.csv` that lives inside the dataset: where the file is, and what was
measured from it. That row is what mosaic references afterwards.

**One folder per result, named by what produced it.** Every derived output lands in a
directory whose name is a hash of the parameters and inputs behind it, with those
parameters written beside it:

```
tracks/
├── convert-trex_npz.0.2-6bb5efbf05/     hash of the conversion recipe
│   ├── day1__trial01.parquet
│   └── day1__trial07.parquet
├── sleap.1.6-d490e1bd63/                a tracker run with an embedding model
│   ├── day1__trial01.parquet
│   ├── day1__trial07.parquet
│   └── appearance/                      one vector per tracked detection
│       ├── day1__trial01.parquet
│       └── day1__trial07.parquet
└── index.csv

features/
└── speed-angvel__from__tracks/          what the feature read
    ├── 0.1-a3f2b1c8e9/                  hash of parameters, inputs, frame range
    │   ├── day1__trial01.parquet
    │   ├── day1__trial07.parquet
    │   └── params.json                  the settings in full
    └── index.csv
```

Each `index.csv` carries one row per computed entry — `run_id`, `group`, `sequence`,
`abs_path`, `n_rows` and what the run consumed — so what exists can be read without
opening a parquet. `params.json` means a directory name you cannot decode by eye is
still explainable from its contents. A lineage too long for a directory name is
abbreviated to its start and a `~` digest, and `_inputs` in `params.json` lists the
full one.

The structure enables efficient organization and running:

- **Automatic caching.** Every artifact has one address, derived the same way every
  time, so identical parameters and inputs produce a directory name that already
  exists and the work is skipped.
- **A parameter sweep organizes itself**, and a stale result cannot be mistaken for a
  current one: change something upstream and everything downstream of it is named
  differently.
- **The dataset is portable.** The indexes addressing its contents live inside it, so
  copying the directory to another machine, a cluster volume or an archive carries the
  organization along with the data.

## Create a dataset

```python
from mosaic.core.dataset import new_dataset_manifest, open_dataset

manifest = new_dataset_manifest("Experiment 1", "~/dataset")
ds = open_dataset(manifest)
```

```bash
mosaic init ~/dataset --name "Experiment 1"
```

Both write `~/dataset/dataset.yaml` and create every folder it declares.

`open_dataset` is also how you reopen an existing one — it is
`Dataset(path).load()` as a single call. Reach for it rather than the
constructor. `Dataset(path)` **reads nothing**: it takes a manifest *path*, so a
caller can point at a dataset that does not exist yet and create it. That makes
the bare constructor a working expression whose roots are all empty, and every
accessor on it then fails against a manifest file that is perfectly correct.

### Manifest versions

`dataset.yaml` records the format it was written in as `manifest_version`. A manifest
in an older format opens in a newer mosaic. It is migrated in memory, and the file on
disk changes only when something saves the dataset. A dataset on a read-only mount
therefore opens too. A manifest in a newer format than your mosaic understands raises
`ManifestVersionError` rather than being read under the wrong rules. Upgrade mosaic
instead of editing the version.

A top-level key that mosaic does not use survives a load and a save. A key you add to
`dataset.yaml` by hand is kept.

A dataset is saved at version 3, and at version 4 once it divides an imgstore into
sequences ([Divide a recording](#divide-a-recording)): a media source declares a
`division`, or its media index contains division rows. A mosaic older than version 4
refuses to open such a dataset, because it would read each division row as the
whole store. Every other dataset stays at version 3 and opens in an older mosaic.

## Dataset folder layout

```
~/dataset/
├── dataset.yaml
├── media_raw/
├── tracks_raw/
├── labels_raw/
│   └── keypoints/          saved revisions of annotation sets
├── media/
│   ├── frames/
│   └── preprocess/         media variants, one directory per variant
├── labels/
├── tracks/
├── frame_predictions/
│   ├── predict-frames/     model predictions on extracted frames
│   └── score-predictions/  those predictions scored against annotations
├── selections/
│   └── select-frames/      frame lists chosen from tracks
├── arenas/                 the arena layout of each recording and camera
├── _tracking/
│   ├── trex/  trex-convert/  sleap/  litpose/  ultralytics/
│   └── infer-pose/  infer-points/  infer-localizer/
├── features/
├── models/
└── .mosaic/
    ├── runs/               one log per attempt
    ├── pipelines/          recipes applied here, and requests/ for each submission
    └── claims/             failures counted per entry, across resubmissions
```

Directory assignments:

- **Raw files**, what you supplied — `media_raw/` recordings index, `tracks_raw/` raw
  tracks, `labels_raw/` raw annotations.
- **Declarations**, what you stated about the recordings — `arenas/` the
  [arena layouts](#arenas).
- **Derived files**, computed by mosaic and recomputable — `media/` transcode
  derivatives, `media/frames/` extracted PNGs for annotation, `media/preprocess/`
  media variants, `tracks/` standardized parquet (with `tracks/<variant>/appearance/`,
  the vectors of a run that embeds detections), `labels/` converted labels,
  `frame_predictions/` model predictions on extracted frames and their scores,
  `selections/` frame lists chosen from tracks, `features/`, `models/`.
- **Temporary files** — `_tracking/` raw tracker output before conversion, reclaimed
  by `mosaic sweep-tracking` once a run is finished.
- **Run records**, in `.mosaic/`: what was attempted and submitted here, as
  [Run work and see what ran](guides/pipelines/run-and-see-what-ran.md#where-what-ran-is-recorded)
  describes.

Deleting a derived or temporary folder costs time, never data. Keep `.mosaic/claims/`.
It counts how often each entry has failed in a pipeline, across resubmissions, and
nothing else records that. Deleting it makes entries known to fail look new, and they
are retried. [Failed entries](concepts/pipelines.md#failed-entries-waiting-and-quarantined)
covers how to retry one entry instead.

Each file mosaic writes under a lock keeps a zero-byte `<name>.lock` beside it:
`index.csv.lock`, a tracks variant's `params.json.lock`, and `dataset.yaml.lock`. They
are not data, and mosaic never removes them.
Leave them in place. Deleting one while a process writes the index lets two writers
erase each other's rows. A script that lists a dataset folder should expect them and
skip them.

**`labels_raw/keypoints/` is versioned, and the rest of `labels_raw/` is not.** An
uploaded label file is the current truth about its sequence: change it, and whatever
was computed from it is stale. A keypoint annotation set is saved whenever a training
run or an export needs a fixed copy of it, so each changed save becomes a new folder,
`<set>/rev1`, `<set>/rev2`, and nothing is ever rewritten. Each folder holds the
set's full saved state, `annotations.mosaic.json`, and a COCO Keypoints export of it,
`annotations.coco.json`, for other tools. A trained model names the revision it read,
which is how it stays tied to exactly those annotations. Treat a revision folder as
read-only: one that was edited afterwards is refused.
`labels_raw/behavior/` is reserved for saved behavior scorings, which will work the
same way.

## Where a dataset can live

A dataset needs a filesystem that supports file locks, because every index write takes
one. A local disk does. On a mount that refuses locks, the first index write raises
`IndexLockUnsupported`, and mosaic does not fall back to writing unlocked. Keep the
dataset on a disk that locks, and declare the recordings on the share as a
[source](#add-sources-and-scan).

A lock excludes the processes of one machine. It excludes other machines only when the
filesystem enforces locks across them. Two machines writing one synced folder, such as
a Dropbox folder, each lock their own copy, and the sync service then makes conflicted
copies of the index without an error. An NFS volume mounted with `nolock` keeps each
machine's locks local in the same way. Write to such a dataset from one machine at a
time.

## Index files placed in the dataset

If you copy or move your files into `media_raw/` or `tracks_raw/`, index them in one
call:

```python
ds.index_media([ds.get_root("media_raw")], extensions=(".mp4", ".h264"))
ds.index_tracks_raw([ds.get_root("tracks_raw")], patterns="*.npz", src_format="trex_npz")
```

Each writes one row per file into that root's `index.csv` — for media, where the file
is and what `ffprobe` measured from it. Nothing is recorded in the manifest, so
picking up new files means calling it again.

## Add sources and scan

When the files stay where they are — a NAS, an external volume — or when one dataset
draws on several places at once, declare each as a **source**. The manifest then
records where everything comes from, and one `scan` refreshes all of it:

```python
from mosaic.core.manifest import MediaScanSource, TracksScanSource

ds.add_scan_source(MediaScanSource(
    id="day1",
    path="/Volumes/behavior-nas/day1",
    extensions=(".mp4", ".h264"),
))
ds.add_scan_source(TracksScanSource(
    id="trex-out",
    path="/Volumes/behavior-nas/trex_out",
    patterns=("*.npz",),
    src_format="trex_npz",
))
```

```bash
mosaic sources add -m ~/dataset/dataset.yaml --kind media \
    --path /Volumes/behavior-nas/day1 --extensions .mp4,.h264
mosaic sources add -m ~/dataset/dataset.yaml --kind tracks \
    --path /Volumes/behavior-nas/trex_out --patterns '*.npz' --src-format trex_npz
```

Each source carries its whole recipe — extensions or globs, the converter that reads
them, how identity is derived from a path — so one dataset can draw on several kinds
of input at once.

The above shows specifying directories. Alternatively a **files** source claims
exactly the paths:

```bash
mosaic sources add -m ~/dataset/dataset.yaml --kind media --id pilot \
    --path /Volumes/behavior-nas/pilot \
    --file trial_03/cam0.mp4 --file trial_07/cam0.mp4
```

Then scan:

```python
ds.scan_media()
ds.scan_tracks()
```

```bash
mosaic scan -m ~/dataset/dataset.yaml
```

A scan writes the same index rows as the calls above, for every declared source at
once. One call per kind, and each raises if that kind declares no source; the CLI form
dispatches over whatever the manifest holds. Rescanning is safe to repeat, because a
scan replaces only what its own sources claim.

A source path may also be relative to the dataset, so `--path media_raw` declares the
files you placed there as a source and brings them under `mosaic scan` too.

## Entry names

Each sequence is an **entry**, named by a `group` and a `sequence`, and `group` may be
empty ([why](concepts/datasets.md#groups-are-an-optional-namespace)). An entry is a
recording of its own, unless its group divides one recording into sequences
([Divide a recording](#divide-a-recording)). Neither name may contain `/`, `\` or a
NUL character, because each has to work as a single directory name, and mosaic raises
`ValueError` rather than write one into an index. Join the levels of a hierarchy with
`__` instead, which `parse_hierarchy` splits back into named levels:

```python
from mosaic.core.helpers import parse_hierarchy

parse_hierarchy("day1__arena2", "trial07", ["day", "arena", "trial"])
# {'day': 'day1', 'arena': 'arena2', 'trial': 'trial07'}
```

An entry is stored under the key `<group>__<sequence>`, or `<sequence>` when the group
is empty, so `("", "day1__0900")` and `("day1", "0900")` would share one. mosaic
refuses to write an entry whose key another entry already has.

## Divide a recording

A recording split into several clips is one entry, whose frames are numbered across
its clips. Divide a recording only when you need its parts as separate sequences, such
as the hours of a day-long recording, and still want windowed features to read across
the boundaries between them
([why both are needed](concepts/features.md#reading-across-a-sequence-boundary)). The
recording is then a **continuous group**: the group is the recording's name, and each
part is one of its sequences. Every other group is a namespace, in which each entry is
one recording.

### Divide an imgstore when you scan it

A media source with a `division` divides each imgstore it finds into sequences of
whole chunks:

```python
from mosaic.core.manifest import MediaDivision, MediaScanSource

ds.add_scan_source(MediaScanSource(
    id="motif",
    path="/Volumes/motif/23036146",
    division=MediaDivision(frames=60000),
))
ds.scan_media()
```

```bash
mosaic sources add -m dataset.yaml --kind media --path /Volumes/motif/23036146 \
    --divide-frames 60000
mosaic scan -m dataset.yaml
```

- **The rule.** `frames` is a whole multiple of the store's `chunksize`. A store of
  10000-frame chunks divides every 60000 frames, and a scan refuses 65000. An hour is
  not a rule: at 20 fps it is 72000 frames, which is not a multiple of 10000. Chunk
  ids alone decide which frames a sequence contains.
- **The recording and its sequences.** By store (the default), each store is a
  recording named `<camera_serial>_<directory name>`, or the directory name when the
  store does not record a serial. With `--recording-by camera` (`identity="camera"`)
  every store of one camera serial is one recording, named by the serial, its stores in
  start order on one frame axis. `--recording-name` (`name=`) names the recording
  instead. A sequence is named by the UTC time of its first frame, `YYYYMMDDTHHMMSSZ`,
  from the store's `created_utc` or its frame times. A name, once written, is kept.
- **Partial sequences.** Only closed chunks count. A store whose last chunk is short
  has ended, and its last sequence is written. A store whose last chunk is full may
  still be recording, and its last, partial sequence waits until it closes. The scan
  lists it on stderr, and under `held_back` with `--json`.
  `mosaic scan --include-partial` (`scan_media(include_partial=True)`) writes it. A
  rescan measures each store again, keeps every sequence written before, and adds the
  sequences that have closed since. A chunk's index is written when the chunk closes.
- **Gaps.** A step between frames wider than ten frame intervals ends a sequence early,
  and the next starts after it. A store boundary ends a sequence too. The sequence
  before a gap or a boundary is not the neighbor of the one after it.
- **Refusals.** A scan refuses, before it writes anything: a rule that is not whole
  chunks, a store of a synchronized camera set, a store whose identity
  `write_media_index` assigned (scan with `--reassign` to divide it), a name another
  recording or an empty-group sequence has, and a group declared by hand with other
  sequences. A division whose chunk lacks an index is refused alone, and a store with
  a short interior chunk or a gap inside a chunk is refused whole. Each is listed under
  `refused` with its reason.
- **Dividing again.** A changed rule or name, a sequence that no longer reproduces, and
  the first division of a store indexed whole each divide the recording again, which is
  refused while the entries it replaces have outputs (tracks, labels, features, media
  variants and tracking runs). A store indexed whole replaces only its own entry, also
  when the other stores of its camera are divided. The refusal lists the outputs.
  Without outputs, `mosaic scan` and `write_media_index` divide the recording again and
  say so on stderr.
- **Undividing.** Indexing a divided store whole also divides its recording again: a
  scan without a division, a source whose division was removed, `--reassign` over the
  divisions a `write_media_index` scope assigned, and a scope without a division. It is
  refused while the entries of the divisions it replaces have outputs. Once it is
  written, a recording without a divided store left leaves `continuous_groups`.

Each sequence reads only its chunks, and its `first_frame` on the recording's frame axis
is the first frame of its chunks. A tracker, an `infer-*` op and a converter number a
sequence's table from its first frame on the recording's axis. Its `time` continues the
sequence before it at that sequence's frame rate, and after a gap or a store boundary
starts at the time the store's clock gives, which counts the gap. A sequence whose
neighbor before it is refused also starts at the clock's time. A table written then
keeps that time once the neighbor is written, which differs from the continued time
by the clock's drift, from microseconds to milliseconds, and the table does not read
as out of date. A converter reads the sequence's frames, numbered from 0, and a table
numbered on the recording's axis already is refused. The scan declares the recording continuous. `set_continuous_groups` refuses
to leave it out.

### Declare a recording divided by hand

1. **Number frames across the whole recording** when you convert. Each sequence starts
   at the sum of the frame counts of the files before it. The second half-hour of a
   30 fps recording starts at frame 54000. No built-in converter applies such an offset,
   and [your own converter](guides/tracking/write-a-converter.md) has to.
   Sequences that each restart at frame 0 cannot form a continuous group.
2. **Declare the group** once its sequences are indexed. Its name must not be empty,
   contain `__` or `:`, or be the name of a sequence in the empty group. The list
   replaces any earlier one, and is saved as `continuous_groups:` in `dataset.yaml`:

    ```python
    ds.set_continuous_groups(["rec1"])
    ```

3. **Measure the frame extents** if the tracks were converted before mosaic recorded
   them. Their extent reads blank, and a blank extent is refused:

    ```python
    ds.measure_frame_extents()
    ```

    ```bash
    mosaic measure-tracks -m dataset.yaml --apply
    ```

Index each sequence's media under the group, one entry per sequence. A sequence reads
its own media, and starts at the summed frame counts of the sequences before it in a
natural sort on their names. Name the sequences so that they sort in recording order. A
tracks table that numbers them in another order cannot be drawn or cropped. A sequence
without media cannot be drawn, cropped or tracked, and neither can a sequence after it.
Every sequence of the group must also be recorded at one frame rate. When a sequence's
media changes length, the tracker tables of the sequences after it read as out of date,
and the next tracker run renumbers them.

### Overlap

A feature run with `overlap_frames` reads across the boundaries of a continuous group.
A division of a store reads only across a boundary whose frames continue. A gap, a
store boundary or a refused division ends the overlap. Sequences outside a continuous
group read only their frames, and the run prints one note to stderr counting them.

## Arenas

An arena is one tank of a camera's view. When each tank holds one animal, name the
tanks once per recording, and pair animals only within one tank. Declare a layout as
rectangles in the source video's pixels. Rectangles are half-open (`x0 <= x < x1`), lie
inside the frame and do not overlap. Arena names are entry-name tokens: no `/`, `__` or
`:`.

```python
from mosaic.core.arenas import ArenaLayout, Rect

ys = [0, 170, 346, 524, 702, 878, 1068]
strip = {  # each left tank shows a reflection of its neighbor at x 95 to 105
    f"A{row}": [Rect(x0=95, y0=y0, x1=105, y1=y1)]
    for row, (y0, y1) in enumerate(zip(ys, ys[1:]), start=1)
}
layout = ArenaLayout.grid(xs=[0, 506, 1012], ys=ys, exclude=strip)
ds.set_arena_layout(layout, group="23036146_20210219_112253")  # a divided recording
ds.set_arena_layout(layout, group="", sequence="pilot", camera="top")
```

```bash
mosaic arenas set -m dataset.yaml -r 23036146_20210219_112253 \
    --xs 0,506,1012 --ys 0,170,346,524,702,878,1068 --exclude-x 95,105
mosaic arenas show -m dataset.yaml
```

`--exclude-x x0,x1` excludes a vertical strip from every grid arena it crosses, and
`--exclude-y y0,y1` a horizontal one. Both repeat, and a strip whose lower edge is not
first or that crosses no arena is refused. Any other exclusion is declared in a layout
file, `mosaic arenas set --layout layout.json`, which holds an `ArenaLayout` as
JSON.

The grid names columns by letter and rows by number (`A1` to `A6`, `B1` to `B6`), each
arena with capacity 1. A divided recording takes one layout for all its sequences,
named by its group with an empty sequence. An undivided recording is named by its
entry. A layout for one camera takes precedence over a layout for every camera. Setting
a layout again replaces it.

The layouts are stored in `arenas/`, one JSON file per recording and camera, and
`arenas/index.csv` records each file's digest. A file edited by hand is refused when it
is read. Set it again instead.

**Assign the arenas.** The `assign-arenas` op reads a tracks variant and writes a new
one, as `resample-tracks` does. Each row gains `arena`, the tank that contains its body
center (`X`, `Y`). A row in an excluded rectangle, in no tank or without a position is
dropped, and above a tank's capacity in one frame the rows with the highest
`confidence` are kept. The run log counts the dropped rows of each entry. At capacity 1
each `id` becomes the tank's position in the layout, so one id names one tank in every
sequence of the recording, and an overlapped feature reads the same animal across a
division boundary. Frames are unchanged.

```bash
mosaic run -m dataset.yaml --kind assign-arenas --params '{}'
```

Run the features on the assigned variant. A feature that pairs animals pairs only
within one arena, and a feature that measures across animals without arenas is refused
on a recording with several (see [Features](concepts/features.md#arenas)). Changing a
layout recomputes that recording's assigned tables at the next run, and the features
that read them.

## Hold groups out of training

Give a group a role when some of the data must never train a model, so that a score on
it means something. A role is one of:

- `train`: what a model may learn from.
- `test`: held out. A fitted model may be applied to it and it may be scored, and no
  step may fit on it.
- `sealed`: held out, and scored only by you, once. Keep its labels outside the dataset
  until then.

```bash
mosaic splits set -m dataset.yaml train day1 day2
mosaic splits set -m dataset.yaml test day3
mosaic splits list -m dataset.yaml
mosaic splits remove -m dataset.yaml day3
```

The roles are saved as `splits:` in `dataset.yaml`, and `mosaic describe` shows them.
`ds.set_splits("test", ["day3"])` and `ds.remove_splits(["day3"])` do the same in
Python. A group names one role, setting it again replaces the first, and the empty
group cannot have one.

Mosaic checks a role when it plans a recipe. A step that fits on the entries it covers,
such as `extract-labeled-templates`, is refused when they include a `test` or `sealed`
group: scope it to the `train` groups with a step scope, and apply what it fitted to the
other groups
([how](guides/pipelines/chain-steps.md#score-on-held-out-data)). A group with no role is
never refused.

`splits` is a top-level key that moved no `manifest_version`, so an older mosaic opens the
dataset, keeps the key through a save, and does not enforce it. Do not rely on a role
with a version that predates it. An agent's tools read the roles and cannot write them
([why](agents.md#held-out-groups)).

## Link a model library

A trained model is named by its run id. To use one that was trained in another
dataset, link that dataset as a library:

```bash
mosaic libraries add -m dataset.yaml --id lab --path ../libraries/lab
mosaic libraries list -m dataset.yaml
```

A model the dataset does not hold is then looked up in each linked library, so
`infer-pose --params '{"model": "<run_id>"}'` and a tracker's `detect_model=<run_id>`
work as they do for a local model. Like a source, a library may sit outside the
dataset and nothing is written through the link. The link records the library's own
id, so if a different dataset later appears at that path it is refused instead of
being used by mistake. See [train one model from several
datasets](guides/tracking/train-a-pose-model.md#5-train-one-model-from-several-datasets).

## Describe a dataset

```python
from mosaic.core.dataset_summary import dataset_summary
from mosaic.tracking import dataset_inventory

dataset_summary(ds, found=dataset_inventory(ds))  # what `mosaic describe --json` prints
```

```bash
mosaic describe -m ~/dataset/dataset.yaml
```

The summary gives the manifest's name, `uuid`, notes, tags, libraries and `splits`, and counts
the sequences, recordings, videos, tracks variants, feature runs and trained models the
dataset holds. `sequences` counts the entries with a tracks table, and `recordings`
counts over the entries of the media and tracks indexes. It lists each recording with
its kind, and a divided one with its number of sequences and its first and last sequence
in recording order: by their first frames on the recording's axis, then by name. A
recording divided from its stores also gives its `division`: the frames of a sequence,
the chunksize, whether it is one store or one camera, the camera serial, the UTC starts
of its first and last sequences, and `n_breaks`, the boundaries where a gap, a store
boundary or a missing sequence stops the frames from continuing. `mosaic describe`
prints that as one `division` line per recording. The summary also lists the track
schemas of the tables and the frame rates of the videos.
`mosaic sequences` lists each entry of the media and tracks indexes with its label and,
under `--json`, the recording it belongs to, whether it has a tracks table, and for a
sequence divided from a store its `first_frame` on the recording's axis, its
`frame_count`, its `start_utc` and whether it `follows_gap`. These four are `null` for
every other entry. The MCP `list_sequences` returns the same four fields. A division is
labeled by its start on the store's local clock, such as `2021-02-19 11:22:53 PST`, or
in UTC (`2021-02-19 19:22:53Z`) when the store lacks a `timezone_local`. A write labels
only a sequence without a label. A label you set is kept, and a later scan labels the
sequences written before it. A label you clear with an empty `set_display_name` is
written again by the next scan or division write.
To find the datasets in a directory and up to three levels below it, call
`discover_datasets(root)` from `mosaic.core.dataset_discovery`.

## Dataset inventory

```python
from mosaic.core.pipeline.inventory import inventory_wire
from mosaic.tracking import dataset_inventory

found = dataset_inventory(ds)
inventory_wire(ds, found)  # the JSON that `mosaic inventory --json` prints
```

```bash
mosaic inventory -m ~/dataset/dataset.yaml
```

`dataset_inventory` reports every computed artifact, its identity, and its
**coverage** — which entries exist, not merely whether something does.
`inventory` from `mosaic.core.pipeline.inventory` reads the same way but reports
tracker runs, frame runs, trained models and prepared datasets only after
`register_inventory_contributors` or `register_ops` from `mosaic.tracking` has
run, and lists them as unavailable otherwise.

[Keep a dataset organized](guides/pipelines/keep-organized.md) covers the statuses it
reports and the commands that repair what it finds.
