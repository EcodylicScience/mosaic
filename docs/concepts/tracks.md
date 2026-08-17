# What a tracker reports, and in what units

Every tracker mosaic drives, and every format it imports, produces the same thing: one
parquet table per sequence, validated against a registered schema. Three rules govern
what is in it. Not knowing them is how you end up comparing quantities that are not
comparable, with nothing on disk recording that you did.

## Spatial columns are video pixels

Every position, every distance, every axis length in `tracks/` is in pixels of the
source video.

This was not always true, and the exception is instructive. TRex reports centimetres,
scaled by a `cm_per_pixel` factor it writes into the export. mosaic's TRex converter
divides that back out, reading the factor **from the file** rather than from whatever
parameter mosaic passed — because TRex substitutes its own value when the parameter is
unset, so the parameter records an intention and the file records the fact.

A physical unit is obtained downstream, by the `scale-to-cm` feature, from a per-video
factor on the media index. It is never stored in the table. That is deliberate: the
factor is per video and is sometimes not recoverable at all — TRex scaled its output
for years before it began recording by how much, and nothing can divide back out a
number nobody wrote down.

There is one exception, and it announces itself: the `mosaic_cm_v1` schema family is
centimetres and says so in its name. It exists for archived data whose factor is
genuinely lost. A scope that resolves both families at once is refused rather than
mixed.

## `X` and `Y` are the body centre

On every tracker. This was also not always true — TRex puts the *head* in its bare
`X`/`Y`, which mosaic preserves as `X#head`/`Y#head` and replaces with the centroid.

A feature reading `X` across two trackers was otherwise comparing a head position to a
body centre, in two unit systems, with nothing recording either fact.

## A tracker reports; a feature derives

The standard schema **forbids** `VX`, `VY`, `AX`, `AY`, `SPEED*`, `ANGLE`, `ANGULAR_*`
and the weighted-centroid columns. These are quantities a feature computes, not ones a
tracker measures, and a converter that wrote one would be presenting a derivation as an
observation.

Violating the forbidden set raises regardless of strictness, because every tracker
write path validates permissively and a wrong table must not reach disk. A tracker that
genuinely measures one of these declares a schema that allows it — `trex_v2` does,
because TRex really does measure `SPEED` and the midline.

Heading is the sharpest case. The principal-component fit that converters once used has
an arbitrary sign, and its flips read downstream as real turns. Anything wanting a
heading runs the `heading` feature and chooses the method, which then enters the run
identifier — so two analyses using different definitions cannot be confused for each
other.

## Keypoints are optional

A centroid-only tracker — TRex without posture, a box model, an export carrying a
centre and nothing else — emits no `poseX*`/`poseY*` columns at all.

Requiring a pair made those trackers fabricate one: a verbatim copy of `X`/`Y` under a
name promising a detected landmark. Features defined on keypoints (`heading`,
`body-scale`, `orientation-rel`, the `pair-*` family) refuse a table without them and
say so by name; features that merely prefer them fall back to the body centre.

## The schema families

Four schemas are registered. Which one a table carries is declared once, by the
converter or tracker that produced it, and recorded in the tracks index.

| Schema | Units | `X`/`Y` | Notes |
|---|---|---|---|
| `mosaic_v1` | pixels | body centre | The tracker-neutral standard. Keypoints optional |
| `trex_v2` | pixels | body centre | `mosaic_v1` plus what TRex genuinely measures |
| `mosaic_cm_v1` | centimetres | body centre | Its own family; for data whose factor is lost |
| `trex_v1` | centimetres | head | Legacy, kept registered because archived data is in it |

The validator never rejects a column merely for being unknown, so additive columns stay
back-compatible. Only the named forbidden set is refused.

See the [track formats reference](../reference/track-formats.md) for the full column
lists and every registered converter.
