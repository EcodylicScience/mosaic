# Inventory

What a dataset holds: every computed artifact, its identity, and its coverage.

Nothing else in the toolkit answers that question. `mosaic sequences` lists the
sequences the tracks index names, `mosaic runs` and `mosaic status` report
*attempts* from the run-log, and `mosaic features list` is the registry — what
the installation knows how to compute, not what this dataset has.

```bash
mosaic inventory --manifest dataset.yaml
mosaic inventory --manifest dataset.yaml --kind feature --json
```

```python
from mosaic.core.pipeline.inventory import inventory

found = inventory(ds)
for record in found.records:
    print(record.ref.kind, record.name, record.run_id, record.status)
```

## Coverage is not a boolean, and its key is not one type

A run covering 50 of 90 sequences is not done, and it becomes less done the
moment the scope widens to 120 — so what is reported is *which* keys exist,
never a flag. What is covered differs by kind:

| Kind | Key |
|---|---|
| feature run | `(group, sequence)` |
| tracks variant | `(group, sequence)` |
| labels variant | `(group, sequence)` |
| tracker run | `(group, sequence)` |
| frame run | `(group, sequence, camera)` |
| trained model | the run identifier — one artifact |
| media derivative | the media row's `video_uuid` |

The frame run carries a camera because the cameras of one recording share an
entry: without it, a run that extracted one camera reads as covering the entry
and the other is never seen as missing.

The media derivative is why the interface takes a per-kind reference rather than
`(storage_name, run_id)`. Transcode has **no run-addressed directory at all** —
its output is named by recipe and reuse is gated by that filename plus the
forward link on the source row. Asked for a directory that was never supposed to
exist, a directory-shaped check reports zero of N, so an already-clean corpus
reads as permanently incomplete and anything acting on that resubmits the same
work forever.

## Status is derived, never stored

Recomputed from the artifact record each time it is asked for. Nothing here
writes a status cell, because a stored one goes stale and forks from the
artifacts it describes.

| Status | Meaning |
|---|---|
| `absent` | nothing of this is on disk |
| `partial` | some keys covered, some missing, nothing damaged |
| `complete` | coverage answers for everything wanted |
| `complete-but-drifted` | complete, but a recorded source has moved |
| `inconsistent` | the index and the files disagree |

Files ahead of index rows is damage only on a *finished* run: outputs are written
before their rows, so on a live one it is an ordinary run in progress.

## Truth is on disk; this is a view

The `index.csv` files plus the files themselves are the record. Everything here
is a cache, is never authoritative, and is never written anywhere it could be
mistaken for the record — there is deliberately no `.mosaic/inventory.json`.

A stale view causes redundant or delayed work, never wrong work, which is why
polling is enough and there is no filesystem watcher (inotify does not work over
NFS). Two holders keep independent views and coordinate through nothing.

::: mosaic.core.pipeline.inventory.scan
    options:
      show_source: false
      members_order: source

## The record and its vocabulary

::: mosaic.core.pipeline.inventory.model
    options:
      show_source: false
      members_order: source

## The run sidecar

The one reader of a feature run's `params.json`. Absent and unreadable are
different answers: the sidecar write is best-effort, so a run root can
legitimately exist with none.

::: mosaic.core.pipeline.inventory.params
    options:
      show_source: false
      members_order: source

## Holding a view across calls

::: mosaic.core.pipeline.inventory.cache
    options:
      show_source: false
      members_order: source

## The ops half

`core` does not import `tracking`, so tracker runs, frame runs and trained
models reach an inventory by registration. A kind with no contributor is
reported in `unavailable_kinds` rather than shown as zero artifacts — a caller
that imported only `mosaic.core` has not imported the producers, and "no tracker
runs" would be a wrong answer where "nobody can tell you" is a true one.

::: mosaic.core.pipeline.inventory.contributors
    options:
      show_source: false
      members_order: source
