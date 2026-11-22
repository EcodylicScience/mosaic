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
