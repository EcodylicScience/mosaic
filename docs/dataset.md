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

## Dataset folder layout

```
~/dataset/
├── dataset.yaml
├── media_raw/
├── tracks_raw/
├── labels_raw/
├── media/
│   └── frames/
├── labels/
├── tracks/
├── _tracking/
│   ├── trex/  sleap/  litpose/  ultralytics/
│   └── infer-pose/  infer-points/  infer-localizer/
├── features/
└── models/
```

Directory assignments:

- **Raw files**, what you supplied — `media_raw/` recordings index, `tracks_raw/` raw
  tracks, `labels_raw/` raw annotations.
- **Derived files**, computed by mosaic and recomputable — `media/` transcode
  derivatives, `media/frames/` extracted PNGs for annotation, `tracks/` standardized
  parquet, `labels/` converted labels, `features/`, `models/`.
- **Temporary files** — `_tracking/` raw tracker output before conversion, reclaimed
  by `mosaic sweep-tracking` once a run is finished.

Deleting a derived or temporary folder costs time, never data.

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

## Dataset inventory

```python
from mosaic.core.pipeline.inventory import inventory

inventory(ds)
```

```bash
mosaic inventory -m ~/dataset/dataset.yaml
```

`inventory` reports every computed artifact, its identity, and its **coverage** —
which entries exist, not merely whether something does. That distinction matters:
"the feature ran" and "the feature covers all forty sequences" are different facts,
and only the second lets you trust a result.

Every answer is computed from disk at read time. There is no database in the dataset
and no cached status file, so a view can be out of date but never wrong.
