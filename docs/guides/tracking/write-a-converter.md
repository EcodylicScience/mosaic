# Write a converter for your format

A track converter reads one raw file and returns one schema-valid table: per frame,
per individual, with `frame, time, id, group, sequence, X, Y`. Keypoints
(`poseX*` / `poseY*`) are optional. mosaic writes the result to
`tracks/<variant>/<group>__<sequence>.parquet` and indexes it.

Check [the track formats reference](../../reference/track-formats.md) first — eight
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

1. **How do files and sequences correspond?** One file, one sequence is the
   default. For one file holding several sequences, set `enumerable = True` and
   implement `enumerate_sequences`, *and* pass `--multi-sequences-per-file` on the
   source — without the flag the file collapses into a single entry, silently. For
   several files making one sequence, set `merges_per_sequence = True` and override
   `sequence_from_stem`. The two are mutually exclusive.
2. **Does your tracker measure something `mosaic_v1` forbids?** Velocity, speed,
   heading and the weighted centroid belong to features. If your tracker genuinely
   measures one, declare a schema that `extends="mosaic_v1"` and `allows` exactly
   that column; otherwise emit `mosaic_v1` and let a feature derive it.
3. **Are your coordinates already video pixels?** `mosaic_v1` says they are, and
   nothing checks it — a scaled table validates cleanly and every downstream
   distance is wrong by a factor nobody recorded. Convert in the converter, reading
   the factor from the file rather than from a parameter.

[What a tracker reports, and in what units](../../concepts/tracks.md) has the rules
behind all three.

## Declare `output_schema`

Set it on the class. It defaults to the legacy `trex_v1`, and forgetting it fails
quietly: a `mosaic_v1`-shaped table is recorded as `trex_v1`, and the error surfaces
much later, at the first feature run mixing those entries with real `mosaic_v1`
ones. Say `mosaic_v1`.

## Shipping it inside mosaic

mosaic discovers nothing on its own — no entry-point scan, no import hook — so where
your module lives decides what can reach it.

**In a notebook or your own package**, import it before the first call naming the
format. This is Python API only: a fresh `mosaic` process never imports it, so the
CLI refuses the format by name.

**In the tree**, add the module to [`core/track_library/`][library] and make two
edits to its `__init__.py` — `from . import myformat`, **and** `"myformat"` in
`__all__`. The import line alone fails lint with `F401`. The CLI then works:

```bash
mosaic sources add -m dataset.yaml --kind tracks --path /data/myformat_out \
    --patterns '*.npz' --src-format myformat_npz
mosaic scan -m dataset.yaml --kind tracks
mosaic convert-tracks -m dataset.yaml
```

`--patterns` must cover your extension. It defaults to `*.npy`, `*.h5`, `*.csv`, so
a `.npz` format matches nothing and converts zero rows without failing.

### What the tests then check

[`tests/test_track_converters.py`][tests] is parametrized over the registry, so an
in-tree converter inherits five assertions the moment its import line lands: no
entry-identity field reaches the parameter digest, `version` is non-empty, the
registry key equals the declared `src_format`, a merging format overrides
`sequence_from_stem`, and no format claims both `enumerable` and
`merges_per_sequence`. Add a per-format test beside them, converting a synthetic
fixture written into `tmp_path`.

[deeplabcut]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/deeplabcut.py
[template]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/track_converter_template.py
[library]: https://github.com/EcodylicScience/mosaic/blob/main/src/mosaic/core/track_library/__init__.py
[tests]: https://github.com/EcodylicScience/mosaic/blob/main/tests/test_track_converters.py
[shiners]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-shiners.ipynb
[zebrafish]: https://github.com/EcodylicScience/mosaic/blob/main/notebooks/collective-motion-zebrafish.ipynb
