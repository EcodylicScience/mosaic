# Operate

Running work, seeing what ran, and keeping a dataset honest as it grows.
[The CLI reference](../reference/cli.md) documents every command and flag; this page
is about which one to reach for.

## Running work

`mosaic run` executes one feature or one op under the **job contract**: a run is
identified, its parameters are recorded, its progress is reported, and its outcome is
appended to a run log.

```bash
mosaic run -m dataset.yaml --feature speed-angvel
mosaic run -m dataset.yaml --kind transcode --params '{"entry": ["", "trial01"]}'
mosaic run -m dataset.yaml --feature global-tsne --json    # machine-readable outcome
```

Parameters come inline as JSON, from `@file.json`, or from `@-` on stdin. Scope with
`--entries`, `--groups` or `--sequences`; `--entries` takes explicit
`group:sequence` pairs and is the only one that can express an arbitrary set, since
the other two combine as a cross-product.

`mosaic runs` lists attempts, `mosaic status <execution_id>` reports one, and
`mosaic cancel` sends a signal to a running attempt's recorded process.

**A run reports what it lost.** A failure on one entry is recorded and surfaced as a
`partial` status with the failed entries named — it is not silently dropped, and it
is not promoted to a total failure. Losing every entry does raise.

## Knowing what a dataset holds

```bash
mosaic inventory -m dataset.yaml
mosaic inventory -m dataset.yaml --kind feature --json
```

`mosaic inventory` reports every computed artifact, its identity, and its coverage —
which entries exist, not merely whether something does. Status is **derived, never
stored**: `absent`, `partial`, `complete`, `complete-but-drifted`, `inconsistent`.

Truth is on disk. There is no inventory database and no filesystem watcher; every
view is a cache that is thrown away rather than reconciled, so a stale answer is
impossible rather than merely unlikely.

Two narrower listings: `mosaic sequences` lists what has been converted, and
`mosaic features list` / `mosaic tracking list` enumerate the registries.

## Declaring where data comes from

`sources:` in the manifest says which directories and files each scan reads. A source
may point anywhere — a NAS, another volume — and its files are recorded by absolute
path into an index that stays inside the dataset.

```bash
mosaic sources add -m dataset.yaml --kind media --path /mnt/nas/cage-a --extensions .mp4
mosaic sources list -m dataset.yaml
mosaic scan -m dataset.yaml
```

**A scan replaces what it claims and preserves everything else.** A row under no
scanned source survives — one written by an assignment, or one pointing at a file
outside the dataset. A file removed from a claimed directory does leave.
`--prune-unsourced` opts into dropping unclaimed rows.

Two source modes: a **directory** source globs, and a **files** source claims exactly
the paths it lists — which is what importing part of a folder needs, since no glob
expresses an arbitrary subset.

## Keeping it honest

| Command | When |
| --- | --- |
| `mosaic reindex` | Files were deleted underneath the index; drops rows whose files are gone |
| `mosaic reprobe-media` | Media metadata is wrong or missing; re-probes in place |
| `mosaic reconcile` | The identity scheme moved in a mosaic upgrade; re-addresses artifacts |
| `mosaic sweep-tracking` | Reclaim tracker working directories that are finished and past their window |
| `mosaic prune-media` | Delete transcode derivatives no forward link reaches |
| `mosaic upgrade-tracks` | Rescale centimetre-era TRex tables to pixels |

`upgrade-tracks` **refuses** a table that does not record its conversion factor,
rather than guessing one. Nothing can divide back out a number nobody wrote down.

## Describing the dataset

`mosaic notes` holds free text. `mosaic tags` holds typed attributes — `label`,
`text`, `int`, `float`, `bool`, `categorical` — validated against declared
constraints. These describe the *dataset*; the per-sequence tags that group sequences
for analysis are a different thing, owned by the API that manages a project.

## Index files

Every `index.csv` has a zero-byte `index.csv.lock` beside it. It is a lock sidecar,
created on the first locked write and never removed — not data, nothing reads it, and
deleting it while a writer holds it reintroduces exactly the lost update the lock
prevents. Anything walking a root should expect it.
