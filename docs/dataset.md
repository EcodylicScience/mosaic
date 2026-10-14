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
still explainable from its contents.

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
- **Derived files**, computed by mosaic and recomputable — `media/` transcode
  derivatives, `media/frames/` extracted PNGs for annotation, `media/preprocess/`
  media variants, `tracks/` standardized parquet, `labels/` converted labels,
  `features/`, `models/`.
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

Each recording is an **entry**, named by a `group` and a `sequence`, and `group` may be
empty ([why](concepts/datasets.md#groups-are-an-optional-namespace)). Neither name may
contain `/`, `\` or a NUL character, because each has to work as a single directory
name, and mosaic raises `ValueError` rather than write one into an index. Join the
levels of a hierarchy with `__` instead, which `parse_hierarchy` splits back into
named levels:

```python
from mosaic.core.helpers import parse_hierarchy

parse_hierarchy("day1__arena2", "trial07", ["day", "arena", "trial"])
# {'day': 'day1', 'arena': 'arena2', 'trial': 'trial07'}
```

## Declare a continuous recording

A recording split into several clips is one entry, whose frames are numbered across
its clips. Declare a **continuous group** only when you need the parts of one
recording as separate sequences, such as the half-hour divisions of a 6-hour session,
and still want windowed features to read across the boundaries between them
([why both are needed](concepts/features.md#reading-across-a-sequence-boundary)).
mosaic makes no continuous group by itself. Build one in three steps:

1. **Number frames across the whole recording** when you convert. Each sequence starts
   at the sum of the frame counts of the files before it. The second half-hour of a
   30 fps session starts at frame 54000. No built-in converter applies such an offset,
   and [your own converter](guides/tracking/write-a-converter.md) has to.
   Sequences that each restart at frame 0 cannot form a continuous group.
2. **Declare the group**, which must have a non-empty name. The list replaces any
   earlier one, and is saved as `continuous_groups:` in `dataset.yaml`:

    ```python
    ds.set_continuous_groups(["session1"])
    ```

3. **Measure the frame extents** if the tracks were converted before mosaic recorded
   them. Their extent reads blank, and a blank extent is refused:

    ```python
    ds.measure_frame_extents()
    ```

    ```bash
    mosaic measure-tracks -m dataset.yaml --apply
    ```

A feature run with `overlap_frames` then reads across the boundaries.

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

## Dataset inventory

```python
from mosaic.core.pipeline.inventory import inventory

inventory(ds)
```

```bash
mosaic inventory -m ~/dataset/dataset.yaml
```

`inventory` reports every computed artifact, its identity, and its **coverage** —
which entries exist, not merely whether something does.

[Keep a dataset organized](guides/pipelines/keep-organized.md) covers the statuses it
reports and the commands that repair what it finds.
