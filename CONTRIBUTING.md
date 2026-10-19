# Contributing to mosaic

Thanks for your interest in contributing.

## Scope

This repository contains backend tooling for behavioral analysis pipelines. Please open an issue before large changes so architecture, scope, and compatibility can be agreed first.

## Contribution Workflow

1. Fork the repository and create a focused branch.
2. Make your changes with tests and documentation updates where relevant.
3. Ensure the project installs and imports cleanly:
   - `pip install -e .`
   - `python -c "from mosaic.core.dataset import Dataset; print('OK')"`
4. Open a pull request with:
   - clear summary of the change
   - motivation and design notes
   - testing evidence (commands run and outcomes)

## Environment Notes

`pip install -e .` is enough: every dependency, `mosaic-media` included,
resolves from PyPI.

To work against an unreleased `mosaic-media`, clone it as a sibling and install
it editable *over* the released wheel:

```bash
pip install -e "../mosaic-media[io,cli]"
```

Order matters only in that the editable install wins whenever it is the later
one. CI does exactly this, so the suite runs against that repository's `main`
rather than its last release.

Prefer `uv pip install` over `uv sync`. `uv sync` installs the project without
extras and prunes anything it considers extraneous, silently undoing an extras
install.

**`uv lock` resolves, and it does so only because DALI's metadata is declared
for it.** `lightning-action` requires `nvidia-dali-cuda110` with no environment
marker, and NVIDIA serves that as a placeholder sdist that downloads the real
wheel from pypi.nvidia.com at build time. There is no macOS wheel to download,
so resolving it meant building it, and building it fails on any machine without
CUDA — which stopped `uv lock` on every platform and left the lock stale at
0.12.0. A `[[tool.uv.dependency-metadata]]` entry in `pyproject.toml` declares
the stub's requirements, so uv resolves without building it. Two things about
that entry: `requires-dist` must carry DALI's real dependencies, because
declaring `[]` also unblocks the lock but silently drops five distributions from
the resolved set and so shrinks the licence inventory; and it pins no version, so
if NVIDIA ever changes the stub's requirements it needs updating. (Ultralytics
used to be the blocker, for a different reason; it no longer appears in mosaic's
dependency graph at all.) Nothing in the repository consumes the lock: every CI
job installs with `uv pip install`, and `scripts/gen_third_party_inventory.py` is
the one reader.

**Regenerate `docs/reference/` under the `docs` dependency group.** The CLI page
is rendered by the Typer that is installed, and `[dependency-groups] docs` pins
the one the docs job's `--check` compares against. Install that group before
running `scripts/gen_docs_reference.py --write`:

```bash
uv pip install --group docs
python scripts/gen_docs_reference.py --write
```

Regenerating under any other Typer rewrites the page's metavars and CI rejects
it.

## Running the tests

A bare `pytest` runs every test not marked `slow`, in parallel over one worker per
core. Pytest is configured in `pyproject.toml`.

```bash
pytest                          # everything except the slow tests
pytest -m slow                  # the slow tests only
pytest -m "slow or not slow"    # everything
pytest tests/core/media         # one area
pytest -n0                      # one process, for a debugger or -s
```

Six markers are declared. `slow` takes seconds to minutes. `media` needs both
`ffmpeg` and `ffprobe` on `PATH`. `tracker`, `identity`, `feral` and
`lightning_action` are job markers.
Each selects the CI job whose environment its tests need, and `pytest -m identity`
runs what the `identity` job runs. A test in one of those areas is marked with it.

`-m` on the command line replaces the `-m "not slow"` in `addopts` instead of
combining with it. `pytest -m "not media"` therefore runs the slow tests too.

CI's `test` job skips the slow tests, and a weekly workflow runs them. Run
`pytest -m "slow or not slow"` before a change that touches slow code paths.

## Where a test goes

A test module belongs in the directory that mirrors the `src/mosaic` package of the
concept it exercises: `tests/core/pipeline/graph/` for the pipeline graph,
`tests/tracking/trex/` for TREx.

- **The concept decides, ahead of the implementing module.** The preprocess op is
  implemented in `core/pipeline/preprocess.py`, and its tests are in
  `tests/core/media/preprocess/` beside the media steps. A tracker op's tests are in
  `tests/tracking/<tool>/`.
- **A subpackage gets a directory once it has several test modules.** Until then
  its tests belong in the nearest enclosing one, as `test_inputs_subclass.py` for
  `core/pipeline/types/` is in `tests/core/pipeline/`.
- **Tests of a top-level module (`src/mosaic/*.py`) go to `tests/core/`,** as those
  of `user_paths` do, or to the `core` package of their concept, as those of
  `media_probe_config` go to `tests/core/media/`.
- **`tests/meta/` contains the tests of the repository, its configuration and the
  suite itself.** It is the one directory without a `src/mosaic` counterpart.
- **Every test directory contains an `__init__.py`, and test module basenames are
  unique across the suite.**
- **Only the root `conftest.py` implements hooks.** A nested conftest declares
  fixtures. A nested `pytest_collection_modifyitems` receives every item of the
  session, and a nested `pytest_configure` runs only once pytest collects its
  directory.
- **Shared test code is imported only as `from tests.helpers import X`.** A test
  does not import another test module or a helper submodule. It takes repository
  paths from `REPO_ROOT`, `TESTS_ROOT`, `SOURCE_ROOT` and `GOLDEN_DIR` in
  `tests.helpers` and derives none from its `__file__`.

`tests/meta/test_suite_layout.py` fails on a test directory without a `src/mosaic`
counterpart or an `__init__.py`, a module at the top of `tests/` other than
`conftest.py` and `__init__.py`, a duplicate basename, a nested hook, and an import or
`__file__` path outside these rules. The hook, import and `__file__` failures name the
file and line.
`tests/meta/test_repository_references.py` fails on a reference to a test module,
or a link into the repository on GitHub, whose target does not exist. A move or a
rename therefore rewrites its references in the same change.

## Code and Review Expectations

- Keep changes minimal and scoped to one concern per PR.
- Preserve backward compatibility unless a breaking change is explicitly discussed.
- Add or update tests for behavior changes, in the directory that mirrors the
  package they exercise (see [Where a test goes](#where-a-test-goes)).
- Update docs/notebooks if user-facing behavior changes, and update
  `src/mosaic/agent/SKILL.md` when the change alters what a user does.
- Guidance for coding agents is the root [CLAUDE.md](CLAUDE.md) plus the
  `CLAUDE.md` files in subdirectories, each loaded when an agent works in that
  directory.

## Releasing

A release is a commit on `main` followed by a tag. The tag starts
`.github/workflows/release.yml`, which publishes to PyPI.

**The release commit.** It changes three things and is titled `Release X.Y.Z: ...`:

- In `CHANGELOG.md`, the `## Unreleased: ...` heading becomes `## X.Y.Z: ...`.
- `version` in `pyproject.toml` becomes `X.Y.Z`.
- In `uv.lock`, the version of the `mosaic-behavior` entry becomes `X.Y.Z`. Leave
  every other entry as it is.

**Before tagging**, run these on the release commit:

```bash
basedpyright
pytest -m "slow or not slow"
python scripts/gen_docs_reference.py --check
mkdocs build --strict
```

Then push the commit to `main` and wait for the CI workflow to pass on it. The
tag is created only after that.

**Tag and push.**

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

`release.yml` runs four jobs in order:

- `verify` fails the run unless the tag equals the version in `pyproject.toml`,
  the tagged commit is an ancestor of `origin/main`, `CHANGELOG.md` has a
  `## X.Y.Z` heading with no `Unreleased` heading above it, and the CI workflow
  succeeded for that exact commit.
- `build` builds the sdist and wheel, runs `twine check --strict`, and checks that
  the wheel carries `SKILL.md`, the LitPose template, `LICENSE` and `NOTICE` and
  carries no `tests/` or `docs/`. It then installs the wheel into a clean
  environment, where `mosaic-media` resolves from PyPI, and checks the installed
  version, `mosaic --help` and the packaged `SKILL.md`.
- `publish` uploads to PyPI through trusted publishing, in the `pypi` environment.
- `release` creates the GitHub release for the tag from the `## X.Y.Z` section of
  `CHANGELOG.md`, with the wheel and sdist attached.

A failure in `verify` or `build` publishes nothing. Delete the tag, fix the cause
on `main`, and tag again. Once `publish` has run, the version is taken for good:
PyPI refuses to replace a file under a version number it has served. Yank a bad
release on pypi.org and publish the next version.

The workflow's header comment lists the one-time PyPI and GitHub setup it needs.

## Contributor License Agreement

Two things are true at once, and both are intended. Your contribution enters this repository under the repository's own license, and separately you grant the Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG), which holds the rights in this project, a license broad enough to also distribute the work under other terms. The second does not narrow the first: the AGPL grant to everyone who receives this repository is irrevocable, so nothing contributed here can later be withdrawn from the AGPL version, and a differently licensed distribution by MPG, or by those it licenses, is an additional channel rather than a replacement.

By submitting a contribution, you confirm that:
- you have the right to submit the contribution,
- the contribution is your original work (or you have required permissions),
- your contribution is offered to this project, and distributed by this project, under the GNU Affero General Public License v3.0 or any later version — the same license as the rest of the repository,
- you additionally grant MPG a perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to use, modify, sublicense, and distribute your contribution under any license terms, including terms other than the AGPL, so that MPG, and those it licenses, may offer this project under a commercial license alongside the AGPL one,
- you retain copyright in your contribution; the grant above is a license, not an assignment.

## License

Contributions are provided under the AGPL-3.0-or-later license in `LICENSE`. The additional grant above lets MPG, and those it licenses, also distribute this project under other terms; it changes neither the license under which you received this repository nor the license under which any contribution reaches a recipient of it.

Third-party components that mosaic invokes, optionally installs, or downloads weights from carry their own licenses, and some carry obligations the AGPL does not — keypoint-MoSeq prohibits commercial use outright, and TRex requires a paid license for company use. See [NOTICE](NOTICE).

## Security

Please do not open public issues for sensitive vulnerabilities. Report security concerns privately to the maintainers.
