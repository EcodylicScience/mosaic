---
name: mosaic-testing
description: "Run and read the mosaic toolkit's tests. Load before running pytest in this repository, before running the full suite, when a test or meta test fails, when a golden file mismatches, and before reporting work done. Targeted runs go local through locate-tests; the full suite (pytest -m \"slow or not slow\") runs on the task machine through `remote` and `.claude/skills/tools/mosaic-suite`, and its failures are compared with that machine's baseline. mosaic-suite-hook.py denies a full suite run on this machine. Covers markers and CI jobs, the `-m` pitfall, -n0, goldens and MOSAIC_UPDATE_GOLDEN, mosaic-check before a commit, and the task machine's untracked settings. Triggers: pytest, run the tests, full suite, slow tests, test on the task machine, remote, mosaic-suite, baseline failures, golden, MOSAIC_UPDATE_GOLDEN, meta test, tests/meta, flaky, xdist, -n0, definition of done, mosaic-check."
---

# Testing mosaic

Placement and helper rules are in `tests/CLAUDE.md` and `CONTRIBUTING.md` (section
"Running the tests"). This skill is the run procedure.

## Targeted runs, here

```bash
.claude/skills/tools/locate-tests <changed files>        # the tests to run, plus tests/meta
.venv/bin/python -m pytest -q -p no:cacheprovider <paths or node ids>
```

- `LOCATE_TESTS_ALWAYS` in `.claude/settings.json` adds `tests/meta` to every
  selection. The meta tests catch most half-finished changes, and `tests/CLAUDE.md`
  names what fails each one.
- `addopts` in `pyproject.toml` runs `-n auto` and `-m "not slow"`. Pass `-n0` for a
  debugger, `-s`, or to tell an interference failure from a real one.
- A `-m` on the command line replaces the default `not slow`: `-m "not media"` also
  runs every slow test.
- Markers (`[tool.pytest.ini_options]` `markers`): `slow`, `media`, and `tracker`,
  `identity`, `feral`, `lightning_action`, which each select one CI job in
  `.github/workflows/ci.yml`. `tests/meta/test_pytest_config.py` checks that each
  test runs in exactly one job.
- Run a batch past about 2 minutes under `heavy` (`.claude/skills/tools/heavy`), which
  queues it behind the other heavy runs on this machine.

## The full suite, on the task machine

```bash
git add -N <each new file>
remote --log=<fresh FILE.log> .claude/skills/tools/mosaic-suite [extra pytest args]
remote --wait <FILE.log>
```

Run it in the background from the main session. `remote` ships the tracked files of
this checkout or worktree, uncommitted edits included. `remote --help` lists the
options.

`mosaic-suite` runs `pytest -m "slow or not slow" -q -rfE` there and ends with one
line: the failure count and how many fall outside the baseline. It exits 0 when
every failure is in the baseline. A run with new failures lists them, and a full run
also lists baseline failures that passed. Compare a branch against the baseline
instead of against zero.

`.claude/skills/tools/mosaic-suite-hook.py` denies a full suite on this machine,
with or without `heavy`. `MOSAIC_LOCAL_SUITE=1` on the command lets it run, and only
the user can grant that.

### Task machine settings

The task machine keeps its settings outside the repository, in
`~/.config/mosaic-suite/` (or `$MOSAIC_SUITE_CONFIG`):

- `env`: sourced before the run. It sets a `PATH` whose ffmpeg has
  libx264, and the `MOSAIC_CI_*`, `MOSAIC_ULTRALYTICS_BIN`, `MOSAIC_POLO_BIN` and
  `MOSAIC_VAME_PYTHON` variables that enable the external-tool tests.
- `baseline-failures.txt`: the node ids that fail on `main` there. Refresh it after
  `main` moves: `remote --log=<fresh FILE.log> .claude/skills/tools/mosaic-suite --update-baseline`
  from an up-to-date `main`.

Keep host names, user names and paths of the task machine out of tracked files. The
repository is public.

## Goldens

`tests/helpers/golden.py` (`read_golden`, `write_golden`) compares identifiers and
documents against files under `GOLDEN_DIR` (`tests/helpers/paths.py`). `MOSAIC_UPDATE_GOLDEN=1` rewrites them. Rewrite a
golden only for a change you intend, and check the diff: a run id or op identity
that moves needs a `version` bump on its feature or op and a CHANGELOG entry
(`/extending-mosaic`).

## Before a commit

```bash
heavy --log=<fresh FILE.log> .claude/skills/tools/mosaic-check
```

It runs ruff, basedpyright against the per-file baseline, the docs reference check
and the selected tests over the files that differ from `main`. The commit hook
(`mosaic-commit-hook.py`) repeats the ruff and docs checks and refuses private
names in added lines.

Before reporting work done, run `mosaic-check` and the full suite through
`mosaic-suite`. CI skips the type check and runs the unmarked slow tests only weekly
(`CLAUDE.md`, section "Commands and definition of done").

## When many tests fail at once

A crash in many unrelated tests, `BPT trap`, `AVFFrameReceiver is implemented in
both`, a missing `flock` or a full disk points at the machine instead of the change.
Check the machine before the code, and see `CLAUDE.md`, section "Development
environment", on the one ffmpeg build per environment.
