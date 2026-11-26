# Write a converter for your format

A track converter reads one raw file and returns one schema-valid table: per frame,
per individual, with `frame, time, id, group, sequence, X, Y`. Keypoints
(`poseX*` / `poseY*`) are optional. mosaic writes the result to
`tracks/<variant>/<group>__<sequence>.parquet` and indexes it.

Check [the track formats reference](../../reference/track-formats.md) first. Nineteen
converters ship, and one may already read your files.

## Two worked examples

Both notebooks define a converter, run it, and check the result. Each covers a
different hard case, and together they are the fastest way in.

- [**collective-motion-shiners.ipynb**][shiners] — several files make one
  recording, and neither identity nor frame numbering survives the joins. Shows
  `merges_per_sequence`, a custom schema lifting one forbidden column, and
  coordinates converted back to pixels.
- [**collective-motion-zebrafish.ipynb**][zebrafish] — a tracker that locates **no
  keypoints at all**, and metadata that disagrees with its own trajectories.

[`track_converter_template.py`][template] is a skeleton to copy;
[`deeplabcut.py`][deeplabcut] is the smallest shipped converter.

## Three questions

1. **How do files and sequences correspond?** One file, one sequence is the default. For
   one file holding several sequences, set `enumerable = True` and implement
   `enumerate_sequences`, *and* pass `--multi-sequences-per-file` on the source —
   without the flag the file becomes a single entry with no warning. A sequence that
   fails to convert is reported as a failed entry, and the file's other sequences still
   convert. For several files making one sequence, set `merges_per_sequence = True` and
   override `sequence_from_stem`. The two are mutually exclusive. Merged files must
   number their frames across the sequence, because a repeated (frame, id) row fails the
   entry.
2. **Does your tracker measure something `mosaic_v1` forbids?** Velocity, speed,
   heading and the weighted centroid belong to features. If your tracker genuinely
   measures one, declare a schema that `extends="mosaic_v1"` and `allows` exactly
   that column; otherwise emit `mosaic_v1` and let a feature derive it.
3. **Are your coordinates already video pixels?** `mosaic_v1` says they are, and
   nothing checks it — a scaled table validates cleanly and every downstream
   distance is wrong by a factor nobody recorded. Convert in the converter, reading
   the factor from the file rather than from a parameter. A 3D source is not in pixels
   at all: declare `mosaic_world_v1` (meters, right-handed, +Z up, with a `Z` column)
   and convert into that frame. When the file records no factor, make the factor and any
   axis change parameters, as `zebrafish_3d_workflow_csv` and `sdannce_mat` do, so that
   they enter the variant's identifier. `track_library/world_frame.py` holds the
   validator of the unit and axes parameters, the transform and the body center.

[What a tracker reports, and in what units](../../concepts/tracks.md) has the rules
behind all three.

## Declare `output_schema`

Set it on the class. It defaults to the legacy `trex_v1`, and forgetting it fails
quietly: a `mosaic_v1`-shaped table is recorded as `trex_v1`, and the error surfaces
much later, at the first feature run mixing those entries with real `mosaic_v1`
ones. Say `mosaic_v1`.

## Declare `carries_tracklets`

Set it on the class: `True` when the `id` column follows one animal over the video,
`False` when ids restart or number detections each frame, and `None` when the file
cannot say. The tracks index records the answer, and a step that needs ids following
an animal refuses tracks recorded `False`. `None` records no answer and refuses
nothing.

## Load the raw file safely

A converter may read a file someone else made, so loading it must not run anything the
file stores. Never use `allow_pickle=True`, `pd.read_pickle`, `pd.read_hdf` or `pickle`
on a raw file. The helpers in `mosaic.core.track_library.safe_load` cover the usual
formats:

- `check_npz` checks an `.npz` before `np.load(path, allow_pickle=False)` reads it.
- `check_hdf5_file` checks an HDF5 file before h5py reads it, and
  `read_pandas_hdf_frame` reads a pandas DataFrame from one.
- `require_within_limit` refuses arrays that declare more bytes than
  `MOSAIC_RAW_ARRAY_LIMIT_BYTES` allows.
- A format only unpickling can read goes through `load_pickled_npy_dict`, which refuses
  the file unless `MOSAIC_TRUST_PICKLED_FILES=1`. Pass the safe form of the same data,
  such as JSON, so the refusal tells the user what to convert to.

A refusal raises `UnsafeRawFileError`, so a batch conversion fails that entry and goes
on. `tests/meta/test_converters_do_not_unpickle.py` fails a converter that unpickles
outside `safe_load`.

## Shipping it

Three routes reach a converter, and where it lives decides which processes find
it.

**An installed package** is found by every process that has the package: the
command line, a queue's workers and mosaic-api. Declare the converter in the
package's `pyproject.toml`:

```toml
[project.entry-points."mosaic.track_converters"]
acme__idtracker_mat = "acme_tracks.converters:IdTrackerMatConverter"
```

The entry point's name is the converter's `src_format`, and its object is the
`TrackConverter` subclass. The `src_format` is a namespace and a name joined by
two underscores, such as `acme__idtracker_mat`. The namespace is lowercase
letters and digits with single underscores, and the name is lowercase letters,
digits and underscores. No built-in format contains `__`, so mosaic can add a
format later without taking the name of one you already use. Name the namespace
after your package. mosaic does not check that it matches. A plugin module may
register its classes with `@register_track_converter` or leave that to the
entry points.

The conversion and the plan that names the format run only where the package is
installed. Install it in the environment of the command line, in every queue
worker image and in the environment of mosaic-api, which validates recipes. A
recipe that names a format a process has not installed fails planning with a
message that names the format. Nothing falls back to another format. The tracks
index records the package and its version in the `converter_distribution`
column of each table it converted. That record is not part of the variant's
identifier, so a new release of the package that leaves the output unchanged
keeps the variant. Raise the class's `version` when its output changes.

mosaic refuses an entry point rather than guess, and the built-ins keep
working. It refuses one whose name is a built-in's, repeats another entry
point's name or breaks the namespace rule. It refuses one whose class is not a
`TrackConverter`, whose `src_format` differs from the entry point's name, whose
`version` is not digits separated by dots or which fails to load. The built-in
keeps a name that an entry point repeats. Asking for any other refused format
raises an error that names the entry point and its package.

**A notebook** is the Python-only route. Register the class with
`@register_track_converter` before the first call that names the format. A
fresh `mosaic` process never runs the cell, so the command line refuses the
format by name. A cell run again replaces its class, because a class of the
same module and name takes over the registered one. A class of another module
or name cannot take a format that is registered: give it a `src_format` of its
own.

**In the tree**, add the module to [`core/track_library/`][library] and make
two edits to its `__init__.py`: `from . import myformat`, **and** `"myformat"`
in `__all__`. The import line alone fails lint with `F401`. A built-in format
has no `__` in its name (`tests/meta/test_track_format_names.py`). The CLI then
works:

```bash
mosaic sources add -m dataset.yaml --kind tracks --path /data/myformat_out \
    --patterns '*.npz' --src-format myformat_npz
mosaic scan -m dataset.yaml --kind tracks
mosaic convert-tracks -m dataset.yaml
```

`--patterns` must cover your extension. It defaults to `*.npy`, `*.h5`, `*.csv`, so
a `.npz` format matches nothing and converts zero rows without failing.

### What the tests then check

`converter_problems` in `mosaic.core.track_library.conformance` returns one
sentence for each way a converter class breaks the contract, and an empty list
when it conforms. It checks that the class declares `version`,
`carries_tracklets` and `output_schema` itself or in a parent below
`TrackConverter`, that `version` is a run identifier's version segment, that
the registry, when it holds a class under its `src_format`, holds this one,
that a merging format overrides `sequence_from_stem`, that no format is both
`enumerable` and `merges_per_sequence`, and that `Params` does not hash a
`group`, a `sequence` or a `group_from`. Given a sample file it also converts
it and validates the table against `output_schema`. A package tests its own
converter with one call:

```python
from pathlib import Path

from acme_tracks.converters import IdTrackerMatConverter
from mosaic.core.track_library.conformance import converter_problems

SAMPLE = Path(__file__).parent / "data" / "sample.mat"


def test_the_converter_conforms() -> None:
    assert converter_problems(IdTrackerMatConverter, SAMPLE) == []
```

Pass `params=` when the sample needs a converter setting. The registry test
[`test_track_converters.py`][tests] calls the same helper for every built-in,
so an in-tree converter meets it the moment its import line lands. Add a
per-format test beside it, converting a synthetic fixture written into
`tmp_path`.

[deeplabcut]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/deeplabcut.py
[template]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/track_converter_template.py
[library]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/__init__.py
[tests]: https://github.com/EcodylicScience/mosaic/blob/main/tests/core/track_library/test_track_converters.py
[shiners]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb
[zebrafish]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-zebrafish.ipynb
