# Run work and see what ran

`mosaic run` executes one feature or one op under the **job contract**: the run is
identified, its parameters are recorded, its progress is reported, and its outcome is
appended to a run log.

```bash
mosaic run -m dataset.yaml --feature speed-angvel
mosaic run -m dataset.yaml --kind transcode --entries trial01 --params '{"target": "analysis"}'
mosaic run -m dataset.yaml --feature global-tsne --json    # machine-readable outcome
```

Features and ops go through the same entry point, which is why a pipeline can hold
both.

## Scoping

Parameters come inline as JSON, from `@file.json`, or from `@-` on stdin. `--scope`
takes the same three forms.

Both a feature run and an op run narrow with the same four flags. `--entries` takes
explicit `group:sequence` pairs, and a bare token is a sequence in the empty group.
A token splits on its first colon. Everything before that colon is the group. The
token `one:two` names the group `one`. The sequence `one:two` in the empty group is
written `:one:two`, and a group whose own name contains a colon has no token spelling.
`--groups` and `--sequences` name a cross product the dataset enumerates, and either
may be given without the other. `--entries` cannot be combined with them.
**Only an explicit pair list can express an arbitrary set** — groups and sequences
combine as a cross-product, and three specific recordings out of a grid are not one.

`--scope` names the whole selector as JSON, in the shape of the `Scope` model the
Python example below constructs:

```bash
mosaic run -m dataset.yaml --kind transcode --scope '{"entries": [["day1", "trial01"]]}'
```

A program submitting work uses it, having formatted a selector rather than typed one.
A pair arrives as a two-element array. Nothing splits it on the way in. That is the
difference from `--entries`, and every group name reaches the run as it was written.

`--scope` excludes `--entries`, `--groups` and `--sequences`. Name the scope one way
or the other. `--params` and `--scope` cannot both read stdin, because the first `@-`
consumes it.

A scope key inside `--params` is refused. `--params` names the settings a feature's
or an op's model validates, and a selector is not one of them.

The same holds in Python:

```python
from mosaic.core.scope import Scope

ds.run_feature(
    SpeedAngvel(),
    scope=Scope(entries=[("day1", "trial01"), ("day2", "trial07")]),
)
```

## Reading the outcome

```bash
mosaic runs   -m dataset.yaml --kind feature --json
mosaic status -m dataset.yaml --execution-id <ULID> --progress --json
mosaic cancel -m dataset.yaml --execution-id <ULID>
mosaic cancel -m dataset.yaml --request <request id>
```

`mosaic runs` lists attempts, `mosaic status` reports one, and `mosaic cancel` sends a
signal to a running attempt's recorded process. With `--request` it signals the
request's driver, which stops the running step and records the others `cancelled`.
A request without a running driver has its running steps signalled and its other
steps recorded `cancelled`. A process on another machine is not signalled.

`mosaic pipeline status -m dataset.yaml --request <request id>` reports a submitted
request step by step. `--wait-seconds 45` holds the report until a step of the request
ends, the request does, or the time is up, so a script need not poll or watch the
driver's process. `--brief` lists only the steps that are running or did not finish
cleanly, and counts the rest.

### When a process is gone

Each attempt reports a **liveness**: `alive`, `dead`, `ended`, or `unknown`.
`mosaic pipeline status` shows it beside each step's phase, and the agent tools report
it too.

- A process on this machine is `alive` while its pid runs and `dead` once it has
  exited. A machine that rebooted since the attempt started also makes it `dead`.
- A process on another machine writes a line to its run-log at least once a minute.
  It is `dead` after ten minutes without one.
- A run-log written by mosaic 0.17 or earlier on another machine declares no
  heartbeat, so this machine cannot tell. Such an attempt is `unknown`.

A request reads `orphaned` when nothing drives it any more: its driver stopped and no
step process may still run, or every step that started is `dead`. Resume it with
`mosaic pipeline run -m dataset.yaml --request <request id>`. Its finished steps are
skipped. `mosaic cancel` records a `dead` attempt `cancelled` without signalling
anything.

A failed step ends only the branch below it. The steps below it read `failed` with
the error type `ParentFailed`, which names the failed step, and the other steps still
run. A request that has ended is not run again, and `mosaic pipeline run --request`
reports its status and starts nothing. Answer the failed step, then submit the recipe
again. Its finished steps are cache hits.

For an `unknown` attempt, first make sure its process has exited. Then record it and
submit the recipe again, and the steps that finished are cache hits:

```bash
mosaic cancel -m dataset.yaml --request <request id> --assume-exited
```

`--assume-exited` is refused for an attempt whose heartbeat is current.

A process is matched to its machine by host name, so every machine that shares a
dataset needs a host name of its own. A Docker container gets a new host name each
time it starts. Give it a stable one with `docker run --hostname <name>`, and run one
container with that name at a time.

A request submitted with `--cleanup superseded` removes its approved superseded runs
after every step has finished, and stays `running` until that is done.
`mosaic pipeline status` prints the cleanup below the steps: its status, how many runs
it removed of those approved, and each run it kept with the reason. `skipped` means a
step did not finish, so nothing was removed. A run removed because its recipe was
retired names that recipe (`removed[].retired` under `--json`). A cleanup that fails or
is cancelled leaves the request's verdict to its steps. A queue that runs the steps
itself runs no cleanup; `mosaic pipeline run --request` does.

**Compute several entries at once.** A feature that computes each entry on its own
(`parallelizable` in `mosaic pipeline catalog <name>`) can use `--workers`:

```bash
mosaic pipeline run -m dataset.yaml --recipe @recipe.json --workers 4
mosaic pipeline run -m dataset.yaml --request <request id> --workers 4
```

The count changes neither a run's identity nor its outputs, so a run with four workers
is a cache hit for one with one, and a recipe never carries it. `--parallel-mode thread`
is the default. `process` uses every core, but each worker is a separate interpreter
that imports the feature libraries and receives a pickled copy of each entry's table,
so it uses much more memory. A worker that the host stops ends the step with
`WorkerProcessDied`. Run the request again with fewer workers or in thread mode. The
entries already written are kept. `--min-free-memory-fraction 0.1` makes a run wait for
its entries in flight while less than that share of the host's memory is available, and
fail in a resumable state when it stays lower. The commands that take the options are
`mosaic run --feature`, `mosaic run --graph-request` and `mosaic pipeline run`.

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
(`started` → `finished` / `failed` / `cancelled`), a liveness heartbeat, the phase
the attempt is in (`fit`, then `apply`, for a fitted feature), and coarse per-entry
progress. While an attempt runs, an `alive` line is added after each minute in which
nothing else was written. One writer, append-only, so it is NFS-safe. These files are
ephemeral: bounded by the work, at most one `alive` line a minute, and safe to age
out.

```python
from mosaic.runlog import read_runs, read_run, run_log_dir

run_dir = run_log_dir(dataset.base_dir)
read_runs(run_dir, kind="feature")      # newest-first attempt snapshots
read_run(run_dir, "<execution_id>")     # one attempt's status and progress counters
```

The distinction is worth holding on to: the index says what **exists**, the run log
says what was **attempted**. A run that failed leaves a log entry and no index row.

The record says that a run happened, not that its answer repeats. To check that, run
`mosaic pipeline verify -m dataset.yaml --recipe @recipe.json`, with the `--step-scope`
options the recipe ran with. It fits each fitting step again in a scratch directory,
compares the result with the stored run and writes nothing to the dataset. It exits 1 on a
difference, and when it compared nothing.
[Reproducibility](../../concepts/reproducibility.md#checking-that-a-result-reproduces)
says what counts as a match.
