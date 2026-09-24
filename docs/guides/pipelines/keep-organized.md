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
| `mosaic prune-joined` | Delete joined exports an earlier `export-joined` version made |
| `mosaic upgrade-tracks` | Rescale centimeter-era TRex tables to pixels |

`sweep-tracking` is the one that matters for disk: raw tracker output under
`_tracking/<tool>/<run_id>/` is often far larger than the parquet it produced, and it
is only needed until the bridge has run.

`upgrade-tracks` **refuses** a table that does not record its conversion factor,
rather than guessing one. Nothing can divide back out a number nobody wrote down.

### After an upgrade that moves an identifier

`reconcile` is what a mosaic upgrade calls for, and 0.13.0 is one. Twelve run
identifiers move in it, because which media a run covered used to decide what its
outputs were called: six tracker run roots, the four `resample-tracks` variant
directories and their tracks-index rows, and the `transcode` and `export-store`
identifiers, which name no file. Run it without `--apply` first; it reports and
writes nothing.

```bash
mosaic reconcile -m dataset.yaml
mosaic reconcile -m dataset.yaml --apply
```

It **re-addresses** an artifact rather than recomputing it, so this is a rename pass
and not a re-run. Nothing is decoded again and no derivative is re-encoded.

One thing it cannot repair: a pipeline request still in flight. `Request.entries`
became `Request.scope`, request models forbid unknown fields, and there is no
migration, so a `.mosaic/pipelines/requests/*.json` written before the upgrade fails
to load. A submission whose steps have finished keeps its results, which are
addressed by `run_id`. Resubmit one that is still running.

## Describing the dataset

`mosaic notes` holds free text. `mosaic tags` holds typed attributes — `label`,
`text`, `int`, `float`, `bool`, `categorical` — validated against declared
constraints.

These describe the *dataset*. The per-sequence tags that group sequences for analysis
are a different thing, owned by the API that manages a project.
