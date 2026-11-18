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

**By sequence tag.** `--scope` also selects the entries that have given
[sequence tags](../../dataset.md#sequence-tags). Other keys are refused beside
`sequence_tags`. The values of one tag are alternatives, different tags must all match,
and an empty list names a `label` tag and matches the entries it is assigned to. A value
is read by its tag's type, so `03` selects the number 3 of an `int` tag and only the
text `03` of a `categorical` tag.

```bash
mosaic run -m dataset.yaml --feature speed-angvel \
    --scope '{"sequence_tags": {"genotype": ["wt", "het"], "treated": []}}'
```

`mosaic prerequisites --scope` takes the same object. The selection reads the latest
revision of the tags. It is refused when it names a tag the dataset does not define or
when its values match zero entries. The run has the identity of the same entries named
with `--entries`.

The pipeline verbs take the selection as `--tag NAME=VALUE`, repeatable, and
`--tag NAME` for a label, beside `--group`:

```bash
mosaic pipeline plan -m dataset.yaml --recipe @recipe.json \
    --tag genotype=wt --tag genotype=het --tag treated
```

`pipeline plan`, `verify`, `run` and `submit` take `--tag`, and so does
`mosaic summarize`
([Summaries across entries](../analysis/features.md#summaries-across-entries)). `--tag`
excludes `--entry`. Beside `--group` it keeps the selected entries of those groups.
`--step-scope` takes `{"sequence_tags": ...}` as a step's selector. The scope and every
step scope of a command select from the latest revision of the tags. A submitted request
records its entries and that revision. The tag values are not recorded. A tag imported
later changes neither the request nor its runs. Its runs have the ids of the same
entries named with `--entry`.

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

### Keep the entries when the dataset grows

A run, a plan or a read that names no scope covers the entries the dataset holds at that
moment. Add a second day of recordings, and the command that covered the first day
covers both. A **saved selection** names an entry set that does not grow. Save it before
the dataset grows, and name it wherever you would name entries ([how to save
one](../../dataset.md#save-a-selection)):

```bash
mosaic selection save -m dataset.yaml day1 --group day1
mosaic run -m dataset.yaml --feature speed-angvel --entries @day1
mosaic run -m dataset.yaml --feature sequence-aggregate \
  --inputs '[{"feature": "speed-angvel__from__tracks"}]' \
  --params '{"columns": ["speed"]}' --entries @day1
mosaic inventory -m dataset.yaml --entry @day1
mosaic summarize -m dataset.yaml feature/sequence-aggregate__from__speed-angvel__from__tracks/<run_id> --entry @day1
```

`@name` is a token for `--entries` of `mosaic run` and `mosaic prerequisites`, and for
`--entry` of `mosaic summarize`, `mosaic inventory` and `mosaic pipeline
plan|verify|run|submit`. It stands alone: it excludes every other entry token, and
`--groups`, `--sequences` and `--group`. A selection has a JSON form for a flag that
takes a JSON selector, `{"selection": "day1"}`, alone in its object. That is `--scope`
of `mosaic run` and `mosaic prerequisites`, and `--step-scope` of `mosaic pipeline
plan`, `verify`, `run`, `submit` and `score`:

```bash
mosaic run -m dataset.yaml --kind transcode --scope '{"selection": "day1"}'
mosaic pipeline run -m dataset.yaml --recipe @recipe.json --step-scope 'fit={"selection": "train"}'
```

`mosaic track` and `mosaic preview` take no selection. They still read `@name` as the
name of a sequence in the empty group, as every command did before selections. To name
such a sequence where `@name` means a selection, write it `:@name`. A selection that
does not exist is refused, and the refusal lists the dataset's selections.

A selection resolves to its entries before anything else reads the scope, so a recipe
run through one has the run ids, the plan and the fingerprint of the same entries typed
out. A submitted request records the name beside the entries in its request file. An
entry list in an op's `--params` is not resolved.

When you name no scope, mosaic cautions where the dataset has grown past the run it
reads. A plan says which steps read a run over fewer entries than the plan covers, and
how many entries the plan adds to it, and advises passing a saved selection as the scope
to keep the earlier set. A read of a run recorded for every entry, such as `mosaic
summarize` or the MCP `preview_output` in `summary` mode, says how many of the dataset's
entries the run covers, and `mosaic summarize` prints it on stderr. A read that names a
scope does not caution, and a read of a run that was asked for a narrower set does not.
A caution refuses nothing and changes no run id.

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
(`parallelizable` in `mosaic pipeline catalog <name>`) can use `--workers`, and so can
the `convert-tracks` and `resample-tracks` steps, which convert one file or table at a
time:

```bash
mosaic pipeline run -m dataset.yaml --recipe @recipe.json --workers 4
mosaic pipeline run -m dataset.yaml --request <request id> --workers 4
```

The count changes neither a run's identity nor its outputs, so a run with four workers
is a cache hit for one with one, and a recipe never carries it. `--parallel-mode thread`
is the default. Choose `process` for a feature whose per-entry work is a Python loop,
which holds the interpreter lock in threads. Each worker is a separate interpreter that
imports the feature libraries and receives a pickled copy of each entry's table, so it
uses much more memory. A worker that the host stops ends the step with
`WorkerProcessDied`. Run the request again with fewer workers or in thread mode. The
entries already written are kept.

Inside a container, memory means the container's limit, not the host's.
`--min-free-memory-fraction 0.1` makes a run wait for its entries in flight before it
loads another while less than that share of the limit is available, and fail in a
resumable state when it stays lower.

`--admission-margin 1.5` starts an entry beside the ones in flight only when its
predicted memory, times 1.5, fits what the run may still grow to under the limit. The
run learns what an input row costs from its own entries, so the first entry runs alone,
and entries start largest first. It counts the memory the kernel cannot reclaim, which
is what a limit kills a job for: anonymous and shared memory, and under a cgroup v2
limit the kernel memory too. It does not count the page cache or GPU memory. Where
`/proc` is missing, as on macOS, it counts the resident memory of the process tree
instead. An entry that does not fit beside the others waits and then runs alone. One
predicted over the limit even alone fails the step with `EntryOverMemoryLimit`. Run it
again with a larger limit; the entries already written are kept. The margin is 0 (off)
by default, or from 1 to 4, and like `--workers` it changes neither a run's identity nor
its outputs. It applies to feature steps. The `convert-tracks` and `resample-tracks`
steps compute their entries in parallel without it and keep the memory floor alone.

The commands that take these options are `mosaic run --feature`,
`mosaic run --graph-request`, `mosaic pipeline run` and `mosaic pipeline score`.
Without `--workers` a run computes entries in turn. `--parallel-mode`,
`--min-free-memory-fraction` or `--admission-margin` alone runs one worker per physical
core, and `--workers 1` with one of them is refused.

### What a step will cost, and a probe of it

```bash
mosaic pipeline input-size -m dataset.yaml --request <request id> --step <step> --json
mosaic run -m dataset.yaml --graph-request <request id> --step <step> \
    --probe 5 --execution-id <new ULID> --json
```

`input-size` reports how many entries a step would compute and the input rows of each,
largest first, without running anything. Each entry also carries what its work scales
with, read from the tracks and media indexes: `ids_per_frame`, `pair_frames`, `frames`,
`keypoints` and `fps`, each `null` where the index does not hold it. A `pair_frames` that
the index never recorded is read as a lower bound from the rows and frames, and
`ids_per_frame` then reads as the rows over the frames, both flagged by
`pair_frames_bound`. `mosaic measure-tracks --apply` records the measured values (see
[`mosaic measure-tracks`](../../dataset.md#declare-a-recording-divided-by-hand)). The
report also names the feature's version, the level it emits, and a digest of the step's
params with whether they equal the feature's defaults.

`--probe N` computes the N largest of those entries (1 to 5) under an `--execution-id` of
its own, the first alone and then two at once in threads, and keeps what it writes. Three
give one entry alone and one pair, which `cost.levels` reads as levels 1 and 2. A feature
that can run in worker processes and is probed with 5 computes its three largest entries
in threads as above, then its two smallest as a pair of worker processes, so the cost
holds both modes. A feature that cannot run in processes computes at most 3 entries
whatever `--probe` asks. The process pair starts together, so it reports a short level 1
beside level 2. `--step-entries group:sequence`, repeated, computes the entries named
instead. Neither flag may carry the step's own execution id, which is the step's status.
`--probe` sets its own workers, so it refuses `--workers`, `--parallel-mode`,
`--min-free-memory-fraction`, `--admission-margin` and `--overwrite`. A probe needs at
least 2 usable CPUs: on one, its entries run in turn in threads and `cost.levels` has no
level 2.

The step is run as usual afterwards. It computes the entries the probe did not and reads
the probe's from its cache, and `mosaic pipeline status` reads only the step's own
attempt, so a probe never marks the step finished. A step whose result covers all its
entries at once (its run id is its set of entries, as for a template or a global fit), an
op, and a step the request binds or pins to a run refuse a subset with the reason
`subset_refused`, and its `cause` says which.

A scheduler that drives steps itself can skip the job for a step with nothing to
compute:

```bash
mosaic pipeline settle -m dataset.yaml --request <request id> --step <step> --json
```

`settle` measures the step as `input-size` does and prints the same report with a
`settle` object. Its `outcome` is one of three. `settled` means the step had nothing to
compute. Its attempt is recorded under the step's execution id as a cache hit or a bound
run (`state`), and its children pin that run. `declined` means the step passed the size
checks and was left unwritten, because an attempt would compute or refuse it or the step
already has a run-log. `not_attempted` means the step has entries to compute, its inputs
are not ready or it is an op. For either outcome, `reason` names the cause and the
scheduler runs the step. `settle` does not compute and writes nothing unless it settles.
Do not call it for a step that runs with `--overwrite`. `mosaic --version` prints the
installed release. A scheduler can check it before calling `settle`.

Every feature attempt records what it cost. `mosaic.run_cost.read_run_cost` reads it:

```python
from mosaic.run_cost import read_run_cost
from mosaic.runlog import run_log_dir

cost = read_run_cost(run_log_dir(dataset.base_dir), "<execution_id>")
cost.memory_limit_bytes, cost.memory_limit_source   # "cgroup_v2", "cgroup_v1" or "host"
cost.peak_rss_bytes, cost.peak_nonreclaimable_bytes, cost.seconds_per_row
cost.levels      # per mode and count of entries in flight: seconds, rows, cpu_seconds
```

A level's throughput is its `rows` over its `seconds`, so the speedup of a second thread
is the throughput at level 2 over the throughput at level 1. A level counts only entries
whose output was written, and spreads an entry's rows evenly over its flight, from the
start of its load to its output written. A log written before levels were recorded has
them rebuilt from its entries on the same rule, and `rebuilt` says so. `cost.entries`
lists each measured entry with its mode, times, CPU and covariates, and
`cost.skipped_lines` counts the lines of the log that could not be read.
`cost.peak_points` holds the largest memory sample of each mode, with the rows and pair
frames in flight (`pair_frames_bound` is true when one of those entries carried a lower
bound). `cost.params_digest`, `cost.params_default` and `cost.emits` identify the
feature's params and the level it emits.

`cost.machine` names the machine the job ran on: its CPU model, logical CPUs, threads per
core and memory. Set the environment variable `MOSAIC_MACHINE_CLASS` to a name of your own
for each kind of machine, such as a node type, and `cost.machine.machine_class` carries
it. It reads `""` when unset.

`peak_rss_bytes` is the largest resident memory of the process tree, and it counts a
page that several workers map once for each. `peak_nonreclaimable_bytes` is the figure
that the admission gate uses, `None` where it was not measured. It leaves out the page
cache and GPU memory. It counts kernel memory only under a cgroup v2 limit. It is the
largest of the samples taken every half second and at the run's events, in any mode, so
it can miss a shorter spike, which the high-water mark that `peak_rss_bytes` includes in
thread mode does not. `peak_bytes_per_row` is in whichever of the two the run used. In
process mode its per-row figure also comes from each worker's samples of its anonymous
and shared memory, which leave out the worker's kernel memory and also miss a spike
shorter than half a second.

An attempt that ran in a container reports the container's limit and where it was read
from. A refused subset leaves a run-log with no `probe` event and no measurements, and
`cost.refusal_reason` and `cost.refusal_cause` name the refusal.

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
difference, and when it compared nothing. `--step <id>`, repeatable, verifies only the
steps named, and `--n-threads` bounds the threads of every BLAS and OpenMP library while a
step is fitted (1 by default).
[Reproducibility](../../concepts/reproducibility.md#checking-that-a-result-reproduces)
says what counts as a match.
