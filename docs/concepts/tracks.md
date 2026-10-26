# What a tracker reports, and in what units

Every tracker mosaic drives, and every format it imports, produces the same thing: one
parquet table per sequence, validated against a registered schema. Five rules govern
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

Each schema family declares its coordinate space, pixels or centimeters, and the overlay
and the crop features refuse a table in centimeters, because a marker placed at those
coordinates lands somewhere else in the frame. A table whose schema declares no space is
drawn as pixels, with a warning. Convert a `trex_v1` table with `mosaic upgrade-tracks`.

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

## `frame` counts the media's frames

`frame` is the 0-based position of a frame in the entry's media. For an entry recorded
as several clips it counts across the clips in order, as every tracker mosaic drives
reads them. In a [continuous group](../dataset.md#divide-a-recording) it
counts across the whole recording. A later sequence starts where the one before it
ended. A converter takes `frame` from the file it reads. Tracks imported as one
file per clip therefore have to be numbered across the recording already.

Each sequence of a continuous group has media of its own, which contains its frames
only. Its first frame on the recording's axis is the sum of the media frame counts of
the sequences before it. A tracker reads the sequence's media and numbers its table from
there. When a sequence before it changes length, the table reads as out of date, and the
next run renumbers it. Playback and the crops find tracks frame `f` at media frame
`f - first_frame`, and refuse a frame outside the sequence. Extract-frames indices and
annotation frame indices count from 0 in each sequence's media.

For a Motif or Loopbio imgstore, `frame` is the store's `frame_index`, its contiguous
running count. It is not the `frame_number` the camera stamped on each frame, which
skips where the camera dropped frames. A camera log keyed by `frame_number` therefore
has to be mapped through the store's own index before it is joined on `frame`.

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
