# Step-cost bench

Measures the fixed costs of a pipeline step: the parent process's cost per entry
of a cheap feature, the start-up of one cached step in its own process, and a
recipe of many cached steps in one process. It runs outside the test suite.

```bash
benchmarks/step-cost/run.sh WORK OUT [SECTION...]
```

`WORK` holds the datasets, about 15 GB at the default 160 entries, and `OUT` the
results. Run it on an idle machine, under `heavy` where installed. The
sections and the environment variables (`PY`, `ENTRIES`, `WORKERS`) are listed
at the top of `run.sh`.

| File | Role |
| --- | --- |
| `run.sh` | builds the dataset and runs each section |
| `make_raw.py` | writes the synthetic track files: 9,000 frames, 4 individuals, 5 keypoints, 36,000 rows per entry |
| `recipes/` | convert; convert and `speed-angvel`; six steps; 61 steps (`speed-angvel` at `step_size` 1 to 60) |
| `timed.py` | runs one command, samples its process tree's CPU, appends its wall to `OUT/results.tsv` |
| `summary.py` | prints `results.tsv`, the target figures, the CPU of the process workers and the pipe bytes per entry |
| `pipes.py` | runs one `mosaic` command and counts the bytes it sends to its worker processes and receives from them, in `OUT/pipes.tsv` |
| `micro.py` | per-entry operations timed alone |
| `stacksample.py` | wall-clock stack samples of every thread of one `mosaic` command |
| `cli_run.py` | runs the `mosaic` CLI in-process for `pipes.py` and `stacksample.py`, with its exit status and wall |
| `pstat.py` | the costliest functions of a cProfile file |
| `fingerprint.py` | hashes of a dataset's outputs and index rows |

A change that must keep run ids and outputs runs the bench before and after over
the same `WORK` path and compares the `fingerprint-*.txt` files.

## Results

### Mac, mosaic d73a57f0

macOS, 10 cores, 32 GB, about 800 processes on the host, 160 entries.

| Run | Wall |
| --- | --- |
| `speed-angvel` cold, serial / thread 4 / thread 8 / process 8 | 17.0 / 15.4 / 16.9 / 23.1 s |
| the same recipe all cached, one process | 5.1 s |
| one cached step as its own `mosaic run --graph-request` process | 4.2 to 4.5 s |
| one cached step through `pipeline settle` | 4.2 to 4.5 s |
| 61 cached steps in one `pipeline run` process | 36.3 s |
| `mosaic --version` | 0.79 s |
| `import mosaic.behavior.feature_library` | 3.2 s |

The parent spends (16.97 - 5.1) / 160 = 74 ms per entry of the cold step at every
worker count. Measured alone, an entry costs about 20 ms to read, apply and write.

### Linux task machine, mosaic fcff1e14

Linux 6.8, 20 cores, about 545 processes on the host, 160 entries, 8 workers.

| Run | Wall |
| --- | --- |
| `speed-angvel` cold, serial / thread 4 / thread 8 / process 8 | 29.2 / 37.5 / 40.0 / 34.3 s |
| the same recipe all cached, one process | 5.5 s |
| one cached step as its own `mosaic run --graph-request` process | 4.3 to 5.1 s |
| one cached step through `pipeline settle` | 4.3 to 5.3 s |
| six cached steps through `pipeline run --request`, one process per step | 27.3 s |
| 61 cached steps in one `pipeline run` process | 91.6 s |
| `mosaic --version` | 1.11 s |
| `import mosaic.behavior.feature_library` | 2.9 s |

The parent spends (39.96 - 5.48) / 160 = 216 ms per entry of the cold step in
thread mode, and thread 8 is slower than serial. In that run
`psutil.Process.children`, which reads every process under `/proc`, takes 70% of
the main thread's samples, and `MemoryWatch.cpu_seconds` alone 65%. Measured
alone, `MemoryWatch.cpu_seconds` costs 11.7 ms and `MemoryWatch.sample` 21.5 ms.
The footer reads stay under 1%. In the 61-step cached run, `DataFrame.iterrows`
in `rows_in_scope` and `rows_without_tracklets` takes 67% of the main thread.
Every worker setting wrote byte-identical outputs and index rows.

### Footer memo, before and after

The footer memo and the single dataset-base resolution of a feature run,
measured before (`fcff1e14`) and after the change over one `WORK` path per
machine, 160 entries, 8 workers. The `fingerprint-*.txt` files are
byte-identical before and after on both machines.

| Figure                                      | Mac before | Mac after | Task machine before | Task machine after |
| ------------------------------------------- | ---------- | --------- | ------------------- | ------------------ |
| processes on the host                       | 739        | 741       | 547                 | 551                |
| `speed-angvel` cold, serial (s)             | 15.63      | 15.94     | 29.20               | 28.06              |
| `speed-angvel` cold, thread 4 (s)           | 14.27      | 14.03     | 37.45               | 37.65              |
| `speed-angvel` cold, thread 8 (s)           | 14.49      | 14.30     | 39.96               | 38.21              |
| `speed-angvel` cold, process 8 (s)          | 19.46      | 18.84     | 34.25               | 34.31              |
| the same recipe all cached, one process (s) | 4.59       | 4.55      | 5.48                | 5.53               |
| one cached step process, median of 16 (s)   | 4.25       | 4.20      | 4.98                | 5.03               |
| 61 cached steps in one process (s)          | 34.48      | 34.26     | 91.58               | 92.40              |
| parent cost per entry, cold thread 8 (ms)   | 61.9       | 60.9      | 215.5               | 204.2              |

The cold thread-8 parent cost per entry moves from 61.9 to 60.9 ms on the Mac
and from 215.5 to 204.2 ms on the task machine. Both are within the spread of
repeated runs: the Mac's two back-to-back cached runs took 5.11 and 4.59 s
before the change.

A cProfile of the cold serial run after the change shows the footer reads at
about 1% of the run, 0.40 s of 38.6 s on the task machine and 0.25 s of 20.6 s
on the Mac. In the cold serial run, 640 of the 800 `read_parquet_column_names`
calls come from `tracks_table_columns` while the graph is planned, outside the
memo. In a cached step process, every one of the 160 `read_footer` calls comes
from `tracks_table_columns` too. The inventory's whole-file checks in that
process open 800 footers more through `parquet_is_readable`, as before the
change. The count of footer parses in the cold run falls from 960 to 800. A
stack sample of the cold thread-8 run on the Mac after the change puts every
footer frame below 4.2% of the main thread, the least share that it reports. A
sample at the earlier `d73a57f0` put them at 10.7%. The memo removes 160 of the
960 parses, so it accounts for only part of that drop.

### Process-tree walks, before and after

The memory watch's process-tree walks, measured before (`53b03cdd`) and after
the change (`c662aff9`), each pair run back to back on one machine over one
`WORK` path, 160 entries, 8 workers. The `fingerprint-*.txt` files of a full
before and after run are byte-identical on both machines. The task machine
carried other work during the pair that the table gives: a load average of 12
to 13 with eight dolt servers and seven pytest processes at both starts. The Mac
ran at a load average of 5.6 to 6.5 with one dolt server, after a stretch above
30.

| Figure                                                    | Mac before | Mac after  | Task machine before | Task machine after |
| --------------------------------------------------------- | ---------- | ---------- | ------------------- | ------------------ |
| processes on the host                                     | 829        | 865        | 597                 | 584                |
| `speed-angvel` cold, serial (s)                           | 18.78      | 9.76       | 41.68               | 22.00              |
| `speed-angvel` cold, thread 4 (s)                         | 16.80      | 8.43       | 52.27               | 20.04              |
| `speed-angvel` cold, thread 8 (s)                         | 17.64      | 8.63       | 51.67               | 21.62              |
| `speed-angvel` cold, process 8 (s)                        | 22.94      | 12.79      | 46.44               | 24.60              |
| the same recipe all cached, one process (s)               | 4.87       | 4.65       | 7.56                | 7.36               |
| one cached step process, median of 16 (s)                 | 4.49       | 4.28       | 6.80                | 5.12               |
| 61 cached steps in one process (s)                        | 36.67      | 35.22      | 126.07              | 93.94              |
| parent cost per entry, cold thread 8 (ms)                 | 79.8       | 24.9       | 276.4               | 89.1               |
| `MemoryWatch.cpu_seconds`, median alone (ms)              | 8.1        | under 0.05 | 17.6                | 0.1                |
| `MemoryWatch.sample`, median alone (ms)                   | 15.9       | 0.2        | 34.9                | 0.9                |
| cold thread-8 main thread under `psutil.Process.children` | 15.6%      | below 3.8% | 70.3%               | 13.7%              |

The targets for the parent's per-entry cost of the cold thread-8 step are met:
it falls by 54.9 ms (68.8%) on the Mac, where at least 20 ms and 25% were
wanted, and by 187.3 ms (67.8%) on the task machine, where at least 25% was
wanted.

The targets for one cached step process and for 61 cached steps in one process
are not met on either machine. A cached step process moves from 4.49 to 4.28 s
on the Mac (under 1.5 s wanted) and from 6.80 to 5.12 s on the task machine, 75%
of the before figure (at most 35% wanted). The 61-step run moves from 36.67 to
35.22 s on the Mac (under 10 s wanted) and from 126.07 to 93.94 s on the task
machine, 75% of the before figure (at most 28% wanted).

An earlier task-machine pair ran its before run at a load average of 2.3 and its
after run at 14.5. Its parent cost per entry moved from 222.2 to 93.5 ms, 57.9%
lower, and its cached step process from 5.28 to 6.67 s. Both pairs meet the 25%
target.

On the task machine the main thread still spends 13.7% under
`psutil.Process.children` after the change, down from 70.3%. The watch repeats
its walk at most once an interval and after a pool's submission. The micro run
measures `psutil children(recursive=True)` alone at 11.8 ms on the task machine
and 8.2 ms on the Mac after the change. On the Mac the main-thread samples show
no `children` frame above 3.8%, the least share that the sampler reports.

### Workers write, before and after

Each entry's worker writes its output and sends the parent a report in place of
the output frame. Measured before (at `2c40cfa6`) and after (at the commit that
adds this section), back to back on one machine over one `WORK` path, 160
entries, 8 workers: a Linux 6.17 workstation with an Intel i7-14700 (8
performance cores with two threads each and 12 efficiency cores, 20 cores in
all) and 31 GB, about 535 processes on the host. Each run used a container of
the mosaic agent image built from `2c40cfa6`, with the tree under test first on
the import path, pinned to 10 efficiency cores, which share no hyperthreads
(`--cpuset-cpus 16-25`). The host was otherwise idle. Its load average was 0.02
at the start of the before run and 1.98 at the start of the after run, which
followed the before run directly. The table gives that pair. A second pair of
the `cold` and `pipes` sections over the same two commits, started at load
averages of 0.46 and 4.33 (the latter again from the run before it), gave the
figures in brackets.

| Figure                                                | Before        | After         |
| ----------------------------------------------------- | ------------- | ------------- |
| `speed-angvel` cold, serial (s)                       | 11.87 (11.40) | 11.60 (11.54) |
| `speed-angvel` cold, thread 4 (s)                     | 9.80 (9.97)   | 8.58 (8.63)   |
| `speed-angvel` cold, thread 8 (s)                     | 10.08 (10.09) | 9.19 (8.99)   |
| `speed-angvel` cold, process 8 (s)                    | 14.27 (14.11) | 11.79 (11.89) |
| the same recipe cached, thread 8, shorter of two (s)  | 5.55 (5.56)   | 5.55 (5.54)   |
| the same recipe cached, process 8, shorter of two (s) | 5.64 (5.54)   | 5.57 (5.56)   |
| parent cost per entry, cold thread 8 (ms)             | 28.3 (28.3)   | 22.8 (21.6)   |
| parent cost per entry, cold process 8 (ms)            | 53.9 (53.6)   | 38.9 (39.6)   |
| bytes per entry to the workers, process 8             | 2,161,738     | 2,162,061     |
| bytes per entry from the workers, process 8           | 2,449,668     | 334           |
| process 8, mean CPU of a worker                       | 53.6%         | 92.5%         |
| process 8, CPU seconds of the parent / the workers    | 14.2 / 35.7   | 10.7 / 41.5   |

The bytes from the workers fall from the output frame to the report. Before the
change each entry's result carried 2,449,668 bytes, 274 bytes more than the
mean pickled size of an entry's `speed-angvel` output, 2,449,394 bytes. After
it, a result carries 334 bytes. The bytes to the workers, which carry each
entry's input table, stay at 2.16 MB per entry. The counts come from
`Connection._send_bytes` and `_recv_bytes` in the parent: the call queue's
feeder thread sent 168 messages, one per entry and one stop per worker. The
executor's manager thread received the results, one exit message from each
worker as it takes its stop, and the executor's empty wake-ups.

The parent's per-entry cost falls by 5.5 ms (19%) in thread mode and by 15.0 ms
(28%) in process mode. The write moves into the workers: in the process run the
parent spends 3.5 s less CPU and the workers 5.8 s more. The stack samples of
the parent's main thread show it. Before the change `write_output` holds 30% of
that thread in the cold thread-8 run and 16% in the process-8 run. After it,
`write_output` is absent from the main thread's list in both runs, whose least
shares are 5.4% and 4.1%. In the thread-8 run it holds 17.8% of the pool
threads' samples. In the process-8 run the main thread waits on the pool for
33% of its samples.

The parent's per-entry cost that remains is its loads, which stay in the parent
in this change. After it, in the cold thread-8 run, `iter_manifest` holds 27.3%
of the main thread's samples (1.83 s), `load_entry_data` 14.9%, `read_footer`
10.0% and `_apply_entries` 39.7%. None of the four appears in the cached
thread-8 run, whose main thread spends 69.6% of its samples in `plan_pipeline`
and lists no frame below 4.4%.

Against the bench's targets: process workers now average above 60% CPU. The
parent stays above 15 ms per entry, at 22.8 ms in thread mode and 38.9 ms in
process mode. Thread 8 runs 1.26 times as fast as serial, short of twice.
Outputs and index rows are unchanged against `2c40cfa6`.

The `fingerprint-*.txt` files of the nine bench datasets are byte-identical
before and after. So are the fingerprints of the 18 datasets that
`tests/core/pipeline/test_output_byte_identity.py` leaves under a fixed
`--basetemp` (six cases, each in serial, thread 3 and process 3), run at
`2c40cfa6` and at the commit that adds this section back to back on a second
Linux machine: 237 lines, every one equal.

### Workers load, before and after

Each entry's worker reads, merges and filters its own input. The parent sends
the worker the file paths, the columns, the frame and time bounds and the
footers it read to size the entry. `speed-angvel` takes one individual-level
input and no overlap, so every bench entry loads in its worker. Measured before
(at `b313f809`) and after (at the commit that adds this section), back to back
on the host and in the containers of the workers-write section, over one `WORK`
path, 160 entries, 8 workers, with 526 to 543 processes on the host. The host
was otherwise idle. Its load average was 0.00 at the start of the before run
and 2.65 at the start of the after run, which followed it directly. The table
gives that pair. A second pair of the `cold` and `pipes` sections, started at
load averages of 4.32 and 5.34 from the runs before them, gave the figures in
brackets. Each after run ran later than its before run, on a host that the
bench's own earlier runs had warmed. The load averages trail those runs and
show no other work.

The first before serial wall, 14.82 s, is out of line with every other serial
cold wall of these runs and of the workers-write section, 10.98 to 11.87 s, and
the first pair's thread-8 speed over serial of 1.58 rests on it. The second
pair compares like with like: thread 8 runs 1.31 times as fast as serial before
the change and 1.17 times after it.

| Figure                                                | Before                    | After                   |
| ----------------------------------------------------- | ------------------------- | ----------------------- |
| `speed-angvel` cold, serial (s)                       | 14.82 (11.63)             | 11.36 (10.98)           |
| `speed-angvel` cold, thread 4 (s)                     | 9.25 (8.60)               | 8.47 (8.52)             |
| `speed-angvel` cold, thread 8 (s)                     | 9.37 (8.91)               | 9.67 (9.40)             |
| `speed-angvel` cold, process 8 (s)                    | 12.81 (11.73)             | 10.96 (11.21)           |
| the same recipe cached, thread 8, shorter of two (s)  | 5.71 (5.57)               | 5.59 (5.60)             |
| the same recipe cached, process 8, shorter of two (s) | 5.58 (5.57)               | 5.58 (5.59)             |
| parent cost per entry, cold thread 8 (ms)             | 22.9 (20.9)               | 25.5 (23.8)             |
| parent cost per entry, cold process 8 (ms)            | 45.2 (38.5)               | 33.6 (35.1)             |
| cold thread 8, speed over serial                      | 1.58 (1.31)               | 1.17 (1.17)             |
| bytes per entry to the workers, process 8             | 2,162,061                 | 2,294                   |
| bytes per entry from the workers, process 8           | 334                       | 364                     |
| process 8, mean CPU of a worker                       | 91.5% (91.9%)             | 101.0% (105.2%)         |
| process 8, CPU seconds of the parent / the workers    | 11.4 / 47.4 (11.3 / 41.6) | 8.7 / 38.6 (9.1 / 44.0) |

The bytes to the workers fall from the input table to the load's description:
2,162,061 bytes per entry before the change and 2,294 after it, over the same
168 messages of the call queue's feeder thread. The result grows by 30 bytes,
to 364, with the input's bytes and the written flag. In the process run the
parent spends 2.7 s less CPU. A worker's mean CPU above 100% counts more than
one of its threads at once.

In the cold thread-8 run before the change, `iter_manifest` holds 28.8% of the
main thread's samples, `load_entry_data` 19.6% and `read_footer` 6.7%. After
it, `iter_manifest` and `load_entry_data` are absent from the main thread,
whose least listed share is 3.1%, and `read_footer` holds 7.1%, of which 5.1%
sizes each entry in `entry_input_size`. The main thread waits for a free worker
in 37.6% of its samples. The loads move to the pool threads, where
`load_entry_data` holds 1.04 s (3.8%) of their sampled seconds. The pool
threads' samples grow from 2,067 to 2,767, and `speed-angvel`'s `apply` among
them from 10.96 s (53.0%) to 16.96 s (61.3%), so the pool's growth lies mostly
in `apply` and not in the loads. Contention for the interpreter lock among the
eight threads may explain it; the samples do not show it. Thread 8 is slower
than before, by 0.30 s and 0.49 s in the two pairs, and its speed over serial
falls from 1.31 to 1.17 in the second pair. In the process-8 run the main
thread's shares of `iter_manifest`, `load_entry_data` and `read_footer` fall
from 15.4%, 7.8% and 4.7% before to below its least listed share, 2.5%, and it
waits for a worker in 48.7% of its samples.

The `fingerprint-*.txt` files of the nine bench datasets are byte-identical
before and after. So are the fingerprints of the 21 datasets that
`tests/core/pipeline/test_output_byte_identity.py` leaves under a fixed
`--basetemp` (seven cases, each in serial, thread 3 and process 3), run at
`b313f809` and at the commit that adds this section on a second Linux machine:
270 lines, every one equal.
