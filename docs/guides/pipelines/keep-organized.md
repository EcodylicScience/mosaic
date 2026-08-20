# Keep a dataset organized

A dataset accumulates: tracker working directories, transcode derivatives, feature
runs from parameter sweeps, index rows for files that have since moved. These are the
commands for seeing what is there and reclaiming what is not needed.

## What does this dataset hold?

```bash
mosaic inventory -m dataset.yaml
mosaic inventory -m dataset.yaml --kind feature --json
```

`mosaic inventory` reports every computed artifact, its identity, and its
**coverage** — which entries exist, not merely whether something does. That
distinction is the point: "the t-SNE ran" and "the t-SNE covers all forty sequences"
are different facts, and only the second one lets you trust a figure.

Status is **derived at read time, never stored**:

| Status | Meaning |
| --- | --- |
| `absent` | nothing of this is on disk |
| `partial` | some keys covered, some missing, nothing damaged |
| `complete` | coverage answers for everything wanted |
| `complete-but-drifted` | complete, but a recorded source has moved |
| `inconsistent` | the index and the files disagree |

Truth is on disk, so a stale view is impossible rather than merely unlikely.
`inconsistent` is judged only on a *finished* run: outputs are written before their
index rows, so files ahead of rows is what a run in progress looks like, not damage.

Two narrower listings: `mosaic sequences` lists what has been converted, and
`mosaic features list` / `mosaic tracking list` enumerate the registries.

```python
from mosaic.core.pipeline.inventory import inventory

inventory(ds)
```

## Reclaiming and repairing

| Command | When |
| --- | --- |
| `mosaic reindex` | Files were deleted underneath the index; drops rows whose files are gone |
| `mosaic reprobe-media` | Media metadata is wrong or missing; re-probes in place |
| `mosaic reconcile` | The identity scheme moved in a mosaic upgrade; re-addresses artifacts |
| `mosaic sweep-tracking` | Reclaim tracker working directories that are finished and past their window |
| `mosaic prune-media` | Delete transcode derivatives no forward link reaches |
| `mosaic upgrade-tracks` | Rescale centimeter-era TRex tables to pixels |

`sweep-tracking` is the one that matters for disk: raw tracker output under
`_tracking/<tool>/<run_id>/` is often far larger than the parquet it produced, and it
is only needed until the bridge has run.

`upgrade-tracks` **refuses** a table that does not record its conversion factor,
rather than guessing one. Nothing can divide back out a number nobody wrote down.

## Describing the dataset

`mosaic notes` holds free text. `mosaic tags` holds typed attributes — `label`,
`text`, `int`, `float`, `bool`, `categorical` — validated against declared
constraints.

These describe the *dataset*. The per-sequence tags that group sequences for analysis
are a different thing, owned by the API that manages a project.
