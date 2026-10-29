---
name: mosaic-release
description: "Prepare a release of the mosaic toolkit (mosaic-behavior on PyPI). Load when asked to cut, prepare or tag a release, bump the version, or fold the changelog's Unreleased sections into a version. The procedure is CONTRIBUTING.md, section Releasing: one release commit (CHANGELOG heading, pyproject.toml version, the mosaic-behavior entry of uv.lock), the checks, a green CI run on main, then the tag. This skill adds what an agent must do differently: fold every Unreleased section, run the full suite on the task machine, and leave every push and tag to the user. Triggers: release, cut a release, new version, bump version, version X.Y.Z, tag, v0., PyPI, publish, changelog Unreleased, release branch, release.yml."
---

# Releasing mosaic

Follow `CONTRIBUTING.md`, section "Releasing". `.github/workflows/release.yml`
enforces it: its `verify` job refuses a tag that disagrees with `pyproject.toml`, a
commit that is not on `main` or not green in CI, and a `CHANGELOG.md` with an
`Unreleased` heading above the `## X.Y.Z` one.

## The release commit

Work on a release branch in a worktree (`/mosaic-worktree`).

- `CHANGELOG.md` usually has several `## Unreleased: <title>` sections, one per
  merged change. They all become the one `## X.Y.Z: <summary>` section. Keep each
  section's content, and make each former heading a subsection of the release.
  Show the user the new heading and the summary line before committing.
- `version` in `pyproject.toml`, and the version of the `mosaic-behavior` entry in
  `uv.lock`. Edit that one line by hand. A `uv lock` run can move other entries.
- Title the commit `Release X.Y.Z: <summary>`.

## Checks

Run the CONTRIBUTING checks with the full suite on the task machine:

```bash
.venv/bin/basedpyright
remote --log=<fresh FILE.log> .claude/skills/tools/mosaic-suite
.venv/bin/python scripts/gen_docs_reference.py --check
.venv/bin/mkdocs build --strict -d <scratch directory>
```

Report every new failure outside the baseline to the user.

## Push and tag

Pushing to `main`, creating the tag and pushing it publish the release. Once
`publish` has run, PyPI refuses that version number for good. Do not do any of
these. Hand the user the commands with the commit to tag, and stop.

Each `remote` run ships the current tree to the task machine. After the release is on `main`, refresh its baseline
(`/mosaic-testing`).
