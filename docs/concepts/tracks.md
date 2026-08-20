# What a tracker reports, and in what units

Every tracker mosaic drives, and every format it imports, produces the same thing: one
parquet table per sequence, validated against a registered schema. Four rules govern
what is in it. Knowing them is what keeps you from comparing quantities that are not
comparable.

## Spatial columns are video pixels

Every position, every distance, every axis length in `tracks/` is in pixels of the
source video, whichever tracker produced it.

Physical units come from the `scale-to-cm` feature, which reads a per-video
`cm_per_pixel` from the media index. They are never stored in the table. Set the
factor with `Dataset.set_media_calibration`, or pass one to the feature directly.

The one exception announces itself in its name: the `mosaic_cm_v1` schema is
centimeters, for archived data whose factor cannot be recovered. It is its own schema
family, and a feature run whose scope resolves both families is refused rather than
mixed.

## `X` and `Y` are the body center

On every tracker. For a pose-only tracker that is the mean of the frame's keypoints;
for one that measures a centroid, its own.

Where a tracker reports other landmarks, they arrive under their own names — `trex_v2`
carries TRex's head position as `X#head` / `Y#head`, for instance. `X` and `Y` stay the
body center so that a feature reading them means the same thing across trackers.

## A tracker reports; a feature derives

The standard schema **forbids** `VX`, `VY`, `AX`, `AY`, `SPEED*`, `ANGLE`, `ANGULAR_*`
and the weighted-centroid columns. A feature computes these; a tracker does not measure
them, and a converter that wrote one would present a derivation as an observation.
Writing a forbidden column raises regardless of strictness.

A tracker that genuinely measures one declares a schema that allows it — `trex_v2` does,
because TRex really does report `SPEED` and a midline.

Heading is the case worth knowing. Run the `heading` feature and name the method you
want; the method then enters the run identifier, so two analyses using different
definitions cannot be confused for each other.

## Keypoints are optional

A centroid-only tracker — TRex without posture, a box model, an export carrying a center
and nothing else — emits no `poseX*` / `poseY*` columns at all.

Features defined on keypoints (`heading`, `body-scale`, `orientation-rel`, the `pair-*`
family) refuse a table without them and say so by name. Features that merely prefer them
(`egocentric-crop`, `overlay`) fall back to the body center.

## The schema families

Four schemas are registered. Which one a table carries is declared by the converter or
tracker that produced it, and recorded in the tracks index.

| Schema | Units | `X`/`Y` | Use it for |
|---|---|---|---|
| `mosaic_v1` | pixels | body center | The tracker-neutral standard. Keypoints optional |
| `trex_v2` | pixels | body center | `mosaic_v1` plus what TRex measures: `SPEED`, `ANGLE`, the midline family, and `X#head`/`Y#head` |
| `mosaic_cm_v1` | centimeters | body center | Archived data whose pixel factor is lost. Its own family |
| `trex_v1` | centimeters | head | Legacy. Registered so archived datasets keep resolving |

The validator never rejects a column merely for being unknown, so additive columns —
an optional `camera` axis, say — stay back-compatible. Only the named forbidden set is
refused.

See the [track formats reference](../reference/track-formats.md) for the full column
lists and every registered converter.
