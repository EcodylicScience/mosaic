# Run work and see what ran

`mosaic run` executes one feature or one op under the **job contract**: the run is
identified, its parameters are recorded, its progress is reported, and its outcome is
appended to a run log.

```bash
mosaic run -m dataset.yaml --feature speed-angvel
mosaic run -m dataset.yaml --kind transcode --params '{"entries": [["", "trial01"]]}'
mosaic run -m dataset.yaml --feature global-tsne --json    # machine-readable outcome
```

Features and ops go through the same entry point, which is why a pipeline can hold
both.

## Scoping

Parameters come inline as JSON, from `@file.json`, or from `@-` on stdin.

A feature run narrows with `--entries`, which takes explicit `group:sequence` pairs.
An op narrows inside `--params`, where `entries` takes the same pairs and a `groups`
/ `sequences` pair given beside it is enumerated against the dataset into them.
**Only an explicit pair list can express an arbitrary set** — groups and sequences
combine as a cross-product, so they cannot name three specific recordings out of a
grid.

The same holds in Python:

```python
ds.run_feature(SpeedAngvel(), entries=[("day1", "trial01"), ("day2", "trial07")])
```

## Reading the outcome

```bash
mosaic runs   -m dataset.yaml --kind feature --json
mosaic status -m dataset.yaml --execution-id <ULID> --progress --json
mosaic cancel -m dataset.yaml --execution-id <ULID>
```

`mosaic runs` lists attempts, `mosaic status` reports one, and `mosaic cancel` sends a
signal to a running attempt's recorded process.

**A run reports what it lost.** A failure on one entry is recorded and surfaced as a
`partial` status with the failed entries named. It is not silently dropped, and it is
not promoted to a total failure. Losing *every* entry does raise.

That matters most before a global fit: a scaler fitted over eleven sequences when you
meant twelve is a wrong result, not a smaller one. Check for shortfalls before
fitting.

## Where what-ran is recorded

There is no database on the dataset filesystem. Everything is plain files, which works
on NFS, HPC scratch and external drives alike.

**Results** live in each feature's `features/<name>/index.csv` — one row per computed
entry with `run_id`, `version`, `params_hash`, `group`, `sequence`, `abs_path` and
`finished_at` — beside the parquet outputs themselves. Cache-hit and completeness are
decided by globbing those parquet files, never by a status flag.

```python
from mosaic.core.pipeline import list_feature_runs

list_feature_runs(dataset, "speed-angvel__from__tracks")
```

**Attempts** live in one append-only JSONL run-log per attempt, under
`<dataset_root>/.mosaic/runs/<execution_id>.jsonl`. Each records its lifecycle
(`started` → `finished` / `failed` / `cancelled`), a liveness heartbeat, and coarse
per-entry progress. One writer, append-only, so it is NFS-safe. These files are
ephemeral: bounded by the work and safe to age out.

```python
from mosaic.runlog import read_runs, read_run, run_log_dir

run_dir = run_log_dir(dataset.base_dir)
read_runs(run_dir, kind="feature")      # newest-first attempt snapshots
read_run(run_dir, "<execution_id>")     # one attempt's status and progress counters
```

The distinction is worth holding on to: the index says what **exists**, the run log
says what was **attempted**. A run that failed leaves a log entry and no index row.
