# The mosaic dataset

A mosaic dataset is a directory holding a `dataset.yaml` manifest, and a
fixed set of named folders declared in it. This enables **consistent naming** for the built-in caching and run organizing, along with **portability** if you want to move the dataset or analyze certain results separately.

## Create a dataset

```python
from mosaic.core.dataset import new_dataset_manifest, open_dataset

manifest = new_dataset_manifest("Case A 2026", "study")
ds = open_dataset(manifest)
```

```bash
mosaic init study --name "Cage A 2026"
```

Both write `study/dataset.yaml` and create every folder it declares.

## The manifest

`dataset.yaml` is YAML with a generated header. Its identity fields are minted once
and never rewritten, so they identify the dataset rather than its most recent edit:

```yaml
manifest_version: 2
name: Cage A 2026
version: 0.1.0
uuid: b0c8073d-41fc-4eff-aa93-765736bf182a
created_at: '2026-08-16T18:42:52.101323+00:00'
roots:
  media_raw: media_raw
  tracks_raw: tracks_raw
  labels_raw: labels_raw
  labels: labels
  media: media
  tracks: tracks
  _tracking: _tracking
  trex: _tracking/trex
  # ... one root per integrated tracker and inference kind
  features: features
  models: models
  frames: media/frames
```

Beyond `roots` it carries `sources`, `notes`, `tags`, `continuous_groups` and `meta`.
Unknown top-level keys are preserved verbatim through a load-and-save round trip, and
an older `manifest_version` is migrated in memory on read and left alone on disk — so
a read-only mount works and a newer manifest raises rather than being read under the
wrong rules.

Comments are not preserved: the header is regenerated on save. Durable prose belongs
in `notes`.

## The folders

```
study/
├── dataset.yaml
├── media_raw/          originals index; ffprobe metadata
├── tracks_raw/         raw tracks you uploaded
├── labels_raw/         raw annotations you uploaded
├── media/              transcode derivatives + their own index
│   └── frames/         extracted PNGs for annotation
├── labels/             converted labels, one directory per kind
├── tracks/             standardized parquet, one directory per recipe
├── _tracking/          raw tracker output, before conversion
│   ├── trex/  sleap/  litpose/  ultralytics/
│   └── infer-pose/  infer-points/  infer-localizer/
├── features/           feature outputs, one directory per run
└── models/             trained model artifacts, one directory per run
```

The split that matters is **raw against derived**. `media_raw`, `tracks_raw` and
`labels_raw` hold what you supplied; `media`, `tracks`, `labels`, `features` and
`models` hold what mosaic computed and can compute again. Deleting a derived folder
costs time, never data.

`_tracking/` is separate from `tracks_raw/` for that reason: it holds a tracker's own
working output, which is regenerable, while `tracks_raw/` holds only user content. It
is excluded by name from every scan that walks the dataset for user files, and
`mosaic sweep-tracking` reclaims it once a run is finished.

## Sources: where the data actually is

Roots are always inside the dataset. Recordings usually are not — they sit on a NAS,
an external volume, a shared mount. A **source** is a declared recipe pointing at
wherever they are; its files are recorded by absolute path into an index that stays
inside.

```python
from mosaic.core.manifest import MediaScanSource, TracksScanSource

ds.add_scan_source(MediaScanSource(
    id="cage-a",
    path="/Volumes/behavior-nas/cage_a",
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
mosaic sources add -m study/dataset.yaml --kind media \
    --path /Volumes/behavior-nas/cage_a --extensions .mp4,.h264
mosaic sources add -m study/dataset.yaml --kind tracks \
    --path /Volumes/behavior-nas/trex_out --patterns '*.npz' --src-format trex_npz
```

Each source carries its whole recipe — extensions or globs, the converter that reads
them, how identity is derived from a path — so one dataset can draw from a folder of
`.mp4` and a folder of CalMS21 arrays at once.

Two modes. A **directory** source globs. A **files** source claims exactly the paths
it lists, which is what importing part of a folder needs, since no glob expresses an
arbitrary subset:

```bash
mosaic sources add -m study/dataset.yaml --kind media --id pilot \
    --path /Volumes/behavior-nas/pilot \
    --file trial_03/cam0.mp4 --file trial_07/cam0.mp4
```

A source directory is never created and never walked at load time.

!!! warning "A files source currently blocks later manifest edits"

    Once a dataset declares one, `notes`, `tags` and `continuous_groups` can no
    longer be written to it: the save re-validates the manifest through a full dump,
    which materializes the defaults a files source is not allowed to carry, and the
    write is refused. Scanning and conversion are unaffected. Set the notes and tags
    you want before declaring a files source.

## Scanning writes the indexes

```python
ds.scan_media()
ds.scan_tracks()
```

```bash
mosaic scan -m study/dataset.yaml
```

One call per kind, and each raises if that kind declares no source — `scan_labels()`
here would, since only media and tracks were declared above. The CLI form dispatches
over whatever the manifest holds.

For media a scan means running `ffprobe` over each file and recording what it
measured.

What lands is an `index.csv` in each root — the concrete artifact the whole scheme
rests on. `media_raw/index.csv` carries one row per recording, with the identity
columns (`group`, `sequence`, `camera`, `video_uuid`) beside the measured ones
(`width`, `height`, `fps`, `frame_count`, `duration`, `codec_name`, `rotation_degrees`
and roughly thirty more from the probe).

A scan **replaces what its sources claim and preserves everything else**. A row under
no scanned source survives — one you assigned by hand, or one pointing outside the
dataset. A file removed from a claimed directory does leave. And a scan never
overwrites an identity a caller assigned: it refreshes the cells it measured and
keeps `group` and `sequence` as set.

Beside every `index.csv` is a zero-byte `index.csv.lock`, created on the first locked
write and never removed. It is not data, nothing reads it, and deleting it while a
writer holds it reintroduces the lost update the lock prevents.

## Derived folders are addressed by content

This is the part that makes the layout work rather than merely tidy.

A **tracks variant** is named `<kind>.<version>-<10 hex digits>`, where the digits
hash the conversion recipe:

```
tracks/
├── convert-trex_npz.0.2-6bb5efbf05/
│   ├── cage-a__trial01.parquet
│   └── cage-a__trial07.parquet
└── index.csv
```

A **feature run** is named `<version>-<10 hex digits>`, where the digits hash the
feature's parameters, its inputs and its frame range:

```
features/
└── speed-angvel__from__tracks/
    ├── 0.1-a3f2b1c8e9/
    │   ├── cage-a__trial01.parquet
    │   └── cage-a__trial07.parquet
    └── index.csv
```

The `__from__tracks` suffix records what the feature read. The hash under it records
how it was configured.

Three consequences follow directly, and they are why the structure is worth having:

- **Re-running costs nothing.** Same parameters and inputs give the same directory
  name, which already exists, so the work is skipped.
- **Parameter sweeps organize themselves.** Twelve settings produce twelve sibling
  directories under one feature, each self-describing.
- **A stale result cannot masquerade as a current one.** Change an upstream feature
  and everything downstream of it gets a different name, so a model can never be
  applied over data it was not fitted on.

Each run directory also has a `params.json` sidecar recording the settings in full,
and each feature's `index.csv` carries one row per computed entry: `run_id`,
`version`, `params_hash`, `group`, `sequence`, `abs_path`, `finished_at`, `n_rows`,
and the composition of what it consumed.

`tracks/index.csv` is the same idea one level up, with one row per
`(run_id, group, sequence)` carrying `producer` (which tracker or converter made it),
`producer_run_id`, `n_rows`, `n_keypoints` and the frame extent.

## Reading what is there

```python
from mosaic.core.pipeline.inventory import inventory

inventory(ds)
```

```bash
mosaic inventory -m study/dataset.yaml
```

Every answer is computed from disk at read time. There is no database in the dataset
and no cached status file, so a view can be stale but never wrong.

## Notes and tags

```python
from mosaic.core.manifest import DatasetTag

ds.set_notes("Cage A pilot, Feb-Apr 2026.")
ds.define_tag(DatasetTag(
    name="cohort",
    type="categorical",
    type_constraints={"options": ["2026-spring", "2026-fall"]},
))
ds.set_tag_value("cohort", "2026-spring")
```

```bash
mosaic notes set -m study/dataset.yaml "Cage A pilot, Feb-Apr 2026."
mosaic tags define -m study/dataset.yaml cohort --type categorical \
    --options 2026-spring,2026-fall
mosaic tags set -m study/dataset.yaml cohort 2026-spring
```

Both live in the manifest, so they travel with the data. Tags are typed — `label`,
`text`, `int`, `float`, `bool`, `categorical` — and a value outside the declared
constraints is refused when you set it rather than discovered later.

These describe the dataset. The per-sequence tags that group sequences for analysis
are a different thing, owned by the API that manages a project.
