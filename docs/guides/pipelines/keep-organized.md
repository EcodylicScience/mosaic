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

A run is measured against the sequences it was asked for. A template sampler run on the
training sequences, or a classifier applied to the test sequences through a step scope,
reads `complete` when it has all of them, and `partial` only when it lost some. A run
started without a scope was asked for every sequence. It reads `partial` once the
dataset gains one, and running it again fills it in. A step that depends on its scope
makes a different run over more sequences, and its run does not read short of them.

Truth is on disk, so a stale view is impossible rather than merely unlikely.
`inconsistent` is judged only on a *finished* run: outputs are written before their
index rows, so files ahead of rows is what a run in progress looks like, not damage.

`--entry`, `--status-not` and `--tracks-run-id` narrow the listing, and `--limit`
pages it: the last line names the `--cursor` that lists the next page. Each artifact
has a `ref`, such as `feature/speed-angvel__from__tracks/0.2-6ce34fc21e`, and
`mosaic inventory show <ref>` reports that artifact with its recorded params and the
entries behind its counts:

```bash
mosaic inventory -m dataset.yaml --status-not complete --limit 20
mosaic inventory show feature/speed-angvel__from__tracks/0.2-6ce34fc21e -m dataset.yaml
```

Two narrower listings: `mosaic sequences` lists what has been converted, and
`mosaic features list` / `mosaic tracking list` enumerate the registries.

```python
from mosaic.core.pipeline.inventory import inventory_wire
from mosaic.tracking import dataset_inventory

found = dataset_inventory(ds)
inventory_wire(ds, found, status_not=["complete"], limit=20)
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
| `mosaic pipeline plan --cleanup superseded` | List the feature runs a recipe's new params superseded, and which may go |

`sweep-tracking` is the one that matters for disk: raw tracker output under
`_tracking/<tool>/<run_id>/` is often far larger than the parquet it produced, and it
is only needed until the bridge has run.

### Superseded feature runs

Changing a step's params gives it a new run in the same storage, and the old run stays
on disk. Plan the recipe with `--cleanup superseded` to list those other runs, each
removable or kept with its reason, and submit with the same flags and the plan's
fingerprint to remove the removable ones after every step has finished:

```bash
mosaic pipeline plan   -m dataset.yaml --recipe @recipe.json --cleanup superseded
mosaic pipeline submit -m dataset.yaml --recipe @recipe.json --cleanup superseded \
    --expect-plan <fingerprint>
mosaic pipeline run    -m dataset.yaml --request <request id>
```

`mosaic pipeline run --recipe` takes the same flags. Planning writes nothing. A run is
kept when its feature is costly to fit again (a GPU model, kpms, VAME, an overlay), when
its `params.json` is missing, when an attempt that may still run names it or reads it,
when another request holds it, when it covers entries or a fit its replacement does
not, when it is younger than `--min-age-hours` (24 by default) or among the
`--keep-newest` newest (1 by default), and when a kept run reads it. Only the storages
the recipe names are touched, never tracks, models or media. A dataset that a service
manages refuses a cleanup.

Each removed run's `params.json` is recorded in the cleanup's run-log, which
`mosaic status --execution-id <cleanup id>` reads, so the run can be computed again.
Removing the newest run of a storage changes what a reference that names no run id
reads, which `--keep-newest 1` avoids.

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
