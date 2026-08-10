# Adding a track converter

Reading tracker output you already have on disk: raw files in, one schema-valid
`tracks/<variant>/<group>__<seq>.parquet` per entry out, under a recipe identity
that names what produced it. This page is for **files**;
[Adding a tracker](adding-a-tracker.md) is for a tool mosaic *runs*. The two meet
here, since a tracker integration bridges its raw output through a converter, so
what you write here is the same object `tracking/<tool>/dataset_runs.py` hands to
`publish_tracks_table`.

Read [`deeplabcut.py`][deeplabcut] first, the smallest complete converter, then
[`calms21.py`][calms21] for one file holding many sequences and [`trex.py`][trex]
for many files holding one. [`track_converter_template.py`][template] is a
skeleton to copy.

## Answer these first

They decide how much of the rest applies.

1. **Can you reuse a shipped format?** Six are registered, in five rows:

   | `src_format`                    | Reads                                            | Schema      |
   | ------------------------------- | ------------------------------------------------ | ----------- |
   | `deeplabcut`                    | DLC `.csv` / `.h5`, single- and multi-animal; also Lightning Pose | `mosaic_v1` |
   | `sleap_analysis_h5`             | `sleap-convert --format analysis` HDF5           | `mosaic_v1` |
   | `trex_npz`                      | TRex per-individual `.npz`                       | `trex_v2`   |
   | `ultralytics_tracks`            | Ultralytics MOT output                           | `mosaic_v1` |
   | `calms21_npy` / `calms21_json`  | CalMS21 arrays, task1/2/3                        | `mosaic_v1` |

2. **How do files and sequences correspond?** One file, one sequence is the
   default, and needs no declaration.

   *One file holding several sequences is **two** declarations.* `enumerable =
   True` on the converter, **and** `multi_sequences_per_file=True` on the
   `TracksScanSource` (`--multi-sequences-per-file`). Only that flag leaves
   `sequence` blank, and only a blank `sequence` triggers the expansion, so
   `enumerable` alone does nothing: `enumerate_sequences` is never called, the
   whole file collapses into one entry named after the stem, and nothing fails.

   *Several files making one sequence is one declaration.* `merges_per_sequence
   = True` on the converter, plus an override of `sequence_from_stem` stripping
   the per-file suffix. `convert_all_tracks` reads this off the converter alone,
   so the source declares nothing. Get it wrong and every per-individual file
   becomes its own entry.

3. **Where do `group` and `sequence` come from?** The scan, before your converter
   runs. `sequence` is `sequence_from_stem(path.stem)`; `group` is what the
   source's `group_pattern` regex captures, or under `multi_sequences_per_file`
   its `group_from` (`"filename"` or `"parent"`). Declaring both is refused, they
   apply on opposite sides of that flag. Your converter writes the hints through
   rather than inventing them, and any `or path.stem` fallback in it covers only
   a direct `convert()` call. `group` may be empty and usually is: a namespace
   disambiguating repeated sequence names, not the grouping used for analysis.

4. **Are the coordinates already video pixels?** If not, see [Units](#units).
   Nothing validates units, so a centimeter table passes `mosaic_v1` cleanly and
   every downstream distance is wrong by a factor nobody recorded.

## A minimal converter

A whole converter for an invented format, a `.npz` holding `x` and `y` arrays of
shape `(frames, individuals, keypoints)`. The decorator is the registration.

```python
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.track_converter import (
    EntryHints, TrackConverter, TrackConvertParams, register_track_converter,
)
from mosaic.core.track_library.helpers import norm_hint


class MyFormatParams(TrackConvertParams):
    fps: float = 30.0  # everything that determines the output, and nothing else


@register_track_converter
class MyFormatConverter(TrackConverter[MyFormatParams]):
    """A ``.npz`` of positions -> a ``mosaic_v1`` table.

    ``poseX0..poseX3`` are nose, left ear, right ear, tail base. Keypoint
    identity is positional and recorded nowhere else, so this is the mapping.
    """

    src_format = "myformat_npz"
    version = "0.1"
    output_schema = "mosaic_v1"
    Params = MyFormatParams

    def convert(
        self, path: Path, params: MyFormatParams, hints: EntryHints
    ) -> pd.DataFrame:
        data = np.load(path)
        x, y = data["x"], data["y"]  # each (frames, individuals, keypoints)
        n_frames, n_ids, n_keypoints = x.shape
        frames = np.arange(n_frames)
        per_id: list[pd.DataFrame] = []
        for i in range(n_ids):
            xi, yi = x[:, i], y[:, i]
            columns: dict[str, object] = {
                "frame": frames,
                "time": frames / params.fps,
                "id": np.full(n_frames, i),
                "X": np.nanmean(xi, axis=1),  # body center: the keypoint mean
                "Y": np.nanmean(yi, axis=1),
                "group": norm_hint(hints.group) or "",
                "sequence": norm_hint(hints.sequence) or path.stem,
            }
            for k in range(n_keypoints):
                columns[f"poseX{k}"], columns[f"poseY{k}"] = xi[:, k], yi[:, k]
            per_id.append(pd.DataFrame(columns))
        return pd.concat(per_id, ignore_index=True)
```

## Registration: where it has to live

A decorator only runs when its module is imported, and **mosaic discovers
nothing on its own**: no entry-point scan, no import hook, so a fresh `mosaic`
process never imports a third-party module.

**In the tree**, two edits to [`track_library/__init__.py`][library]:
`from . import myformat` beside the others, **and** `"myformat"` in `__all__`.
The import line alone fails lint with `F401`, and CI lints every changed file.

**Out of the tree**, your converter is Python API only: import your module before
the first call naming the format, in that process. Every `mosaic` CLI verb that
resolves the format refuses it, `get_track_converter` raising with the registered
formats listed and `index_tracks_raw` refusing at indexing rather than writing an
index nothing can convert. If you want the CLI, the converter goes in
`track_library/`.

A **custom schema** splits finer, because `ensure_track_schema` runs on write
paths only (the conversion loops in `dataset.py`, the tracker bridge, the
inference op, `upgrade_tracks`) and nothing on the read path calls it. Registered
in a notebook, it converts fine **in that process**, and feature runs in a fresh
process work too because `schema_family()` is total and never raises on an
unknown name. But `mosaic convert-tracks` from the command line **fails** with
`UnknownTrackSchemaError`, and any feature run whose scope mixes those entries
with `mosaic_v1` ones **is refused**, an unregistered name being its own family.

So for anything reused, register in a module every consuming process imports. A
notebook is the right home only when it is the whole story: a self-contained
tutorial, or an analysis converted and read in one session.

## Parameters versus hints

`params` is the **recipe**: it, plus `src_format` and `version`, identifies a
tracks variant, as does a promoted correction's revision, so a corrected table is
a different variant rather than the same one with different contents. `hints` is
**which entry** is being produced, and is never hashed.

Each direction has a concrete failure. A sequence name in `Params` reaches the
digest, so one recipe over two hundred sequences mints two hundred variants and
as many directories under `tracks/`. A genuine knob passed as a hint reaches no
digest, so two recipes collide on one identity and the second run reads the
first's tables as its own. `EntryHints` is a frozen dataclass and deliberately
not a `Params`, so the first failure is structurally impossible. Two more
properties of `Params`:

- **Extras are forbidden.** `MyFormatParams.from_overrides({"fpss": 30})` raises;
  a typo aimed at the wrong converter used to be dropped silently, which is how a
  misspelled `neck_idx` produced a quietly wrong heading. On a mixed-format
  dataset pass `params_by_format={"myformat_npz": {...}}`.
- **`strict_schema` is inherited and `HASH_EXCLUDE`d.** It changes what is
  checked, never what is written, so flipping it must not mint a second variant.
  Mark your own diagnostic knobs the same way: `Annotated[bool, HASH_EXCLUDE]`.

## The schema you declare

`output_schema` is the single place your tables' schema is named: the caller
validates the frame you return against it and records that name on the index row,
so a converter cannot claim one schema while its rows are recorded under another.
What `mosaic_v1` requires, what it forbids and why, and what to do when your tool
genuinely measures a forbidden quantity are stated once under `output_schema` in
[Adding a tracker](adding-a-tracker.md). Below is what only a converter hits.

**The default is `trex_v1`, and the consequence of forgetting it is silent.**
`trex_v1` describes what a converter written before schemas existed emits:
centimeters, `X` at the head, nothing forbidden. Over a `mosaic_v1`-shaped table
it prints a missing-recommended report that reads as harmless and records
`trex_v1` on the index row; the failure arrives at the first feature run mixing
those entries with `mosaic_v1` ones, where `schema_family()` puts the two in
different families and `_refuse_mixed_schemas` refuses. Say `mosaic_v1`.

**A forbidden column raises `ForbiddenTrackColumnError` whatever `strict` says.**
The check sits above the strict gate: a missing required column leaves a merely
incomplete table and raises only under `strict=True`; a forbidden column leaves a
confidently wrong one. The set is fifteen names: `VX`, `VY`, `AX`, `AY`, `SPEED`
and its three `#`-suffixed spellings, `ANGLE`, plus `ANGULAR_V` and `ANGULAR_A`
and their two `#centroid` spellings, and `X#wcentroid` / `Y#wcentroid`. The lift
for a genuinely measured one is surgical: `allows={"ANGLE"}` on a schema that
`extends="mosaic_v1"` frees that column, leaves the other fourteen forbidden, and
still resolves to family `mosaic_v1`, so those tables mix freely with every other
`mosaic_v1` table. `trex_v2` is the same construction over the whole set.

**`allows` is silently inert without `extends`.** The subtraction happens inside
the `extends` branch of `register_track_schema`, so a `TrackSchema` declaring
`forbidden=DERIVED_COLUMNS, allows={"ANGLE"}` and no base registers without error
and still raises on `ANGLE`. Extend the base; never restate its forbidden set.

### Units

Spatial columns are video pixels, and nothing checks it, so a scaled or
normalized table passes validation and every downstream distance is wrong by a
factor nobody recorded. Divide back to pixels in the converter, reading the
factor **from the file** and never from mosaic's settings or a default.
[`trex.py`][trex] is the worked pattern: `calibration_from_frame` reads
`cm_per_pixel` off the table, `unscale_to_pixels` divides every column it can
classify as a length, a column it cannot classify raises rather than being scaled
on a guess, and a file recording no factor raises `MissingTrexCalibrationError`.
A factor that can only come from the caller is a `Params` field, and so part of
the recipe: two calibrations, two variants. An unknown factor raises. `1.0` is a
scale, not an absence.

Physical units belong downstream, in `scale-to-cm`, which takes the scale from
the media index beside the video it describes (`Dataset.set_media_calibration`),
records it in a run identifier, and refuses on an uncalibrated dataset. Never
write centimeters into `X`.

## Wiring it up

Three files in the tree (converter module, import line, test module), or two out
of it. Then declare where the raw files are, scan and convert. In the tree, where
the converter is registered in every process, the CLI works:

```bash
mosaic sources add -m dataset.yaml --kind tracks --path /data/myformat_out \
    --patterns '*.npz' --src-format myformat_npz
mosaic scan -m dataset.yaml --kind tracks
mosaic convert-tracks -m dataset.yaml
```

`--patterns` must cover your extension: it defaults to `*.npy`, `*.h5`, `*.csv`,
matching no `.npz` at all, and a source matching nothing scans and converts zero
rows without failing.

Out of the tree, the same three steps from Python, in a process that imported
your module (for `ds` itself see [Getting started](getting-started.md)):

```python
from mosaic.core.manifest import TracksScanSource

ds.add_scan_source(TracksScanSource(
    id="myformat", path="/data/myformat_out", patterns=("*.npz",),
    src_format="myformat_npz",
    group_pattern=r"^(session\d+)",  # optional: capture a group from the stem
))
ds.scan_tracks()
outcome = ds.convert_all_tracks()
assert outcome.ok, f"{outcome.failed} file(s) failed to convert"
assert outcome.converted, "nothing matched: check patterns and src_format"
```

**Check both counts.** `convert_all_tracks` warns on stderr and keeps going when
a file fails, so a batch that converted nothing looks like one that converted
everything, and a batch that matched nothing reports a clean zero. It re-raises
`TrackSchemaError` and its subclasses, so a forbidden column or an unregistered
schema name aborts instead. `mosaic convert-tracks --json` reports
`"status": "partial"` when `failed` is nonzero; without `--json` it prints the
two counts. The exit code is 0 either way.

## Invariants that must hold

- **Do not validate your own output.** The caller validates every table against
  `output_schema`; a second check under another name is how the two drifted.
- **Version declared, never detected**, bumped by hand when output semantics
  change. A visible identity segment, not a hash term, so a bump renames nothing.
- **`enumerable` and `merges_per_sequence` are opposite claims** about one
  file-to-sequence relationship; declaring both is rejected by test.
- **Emit `poseX0..N` / `poseY0..N` contiguously from zero**, in the source file's
  own keypoint order, `poseP<k>` only where a confidence was reported. A `poseX`
  without its `poseY` is skipped rather than half-reported.
- **Keypoint identity is positional and is recorded nowhere on disk**, not on the
  tracks index row and not in the variant's `params.json`. Your docstring is
  currently the only home for the index-to-part mapping, so write it there.
- **Entry names are one path component.** `EntryHints` refuses `/`, `\` and NUL
  on construction. Join levels with `__`, the `parse_hierarchy` default:
  CalMS21's slash-shaped ids become `task1__test__mouse075`.
- **Import an optional dependency lazily**, inside the reader, raising
  `ImportError` naming the install. Importing mosaic must not require it.

## What the tests will tell you

[`tests/test_track_converters.py`][tests] is parametrized over
`TRACK_CONVERTERS`, so an in-tree converter inherits five assertions the moment
its import line lands: no entry-identity field reaches `identity_dump()`, a
non-empty `version` is declared, the registry key equals the declared
`src_format`, a merging format overrides `sequence_from_stem`, and no format
claims both directions of the file-to-sequence relationship at once.

Add a per-format test beside them: a synthetic fixture written into `tmp_path` by
a small local writer (`_write_calms21`, `_write_trex_npz`,
`_write_sleap_analysis_h5`), one conversion, then assertions on what came back.
One `id` per individual, the frames each is present on, the required columns and
pose prefixes, `group` / `sequence` from the hints and `time` from the params.
Guard an optional dependency with `pytest.importorskip`, and assert your failure
modes: a file holding several sequences with no hint raises rather than silently
picking one.

## Where to look next

[`notebooks/collective-motion-shiners.ipynb`][shiners] is the end-to-end worked
example: a SchoolTracker and Fovea HDF5 converter defined in the notebook itself,
on the CC0 golden-shiners dataset (Davidson et al. 2021, *J R Soc Interface*
18:20210142). It reconstructs a body-local midline into pixels, registers
`schooltracker_v1` for a genuinely measured body-axis heading, and runs the
collective-motion features on the result.

[deeplabcut]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/deeplabcut.py
[calms21]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/calms21.py
[trex]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/trex.py
[template]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/track_converter_template.py
[library]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/__init__.py
[tests]: https://github.com/EcodylicScience/mosaic/blob/main/tests/test_track_converters.py
[shiners]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb
