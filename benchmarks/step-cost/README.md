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
| `summary.py` | prints `results.tsv` and the three target figures |
| `micro.py` | per-entry operations timed alone |
| `stacksample.py` | wall-clock stack samples of every thread of one `mosaic` command |
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
the change (`b819e312`), each pair run back to back on one machine over one
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
