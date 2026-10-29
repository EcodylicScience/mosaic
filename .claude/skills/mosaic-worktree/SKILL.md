---
name: mosaic-worktree
description: "Create, check and remove a git worktree of the mosaic toolkit (mosaic-behavior). Load before starting work on a branch, when a run in a worktree seems to import or test the main checkout instead, when a worktree has no .venv, and after a merge when the branch and worktree are cleaned up. mosaic uses a conda environment at ./.venv instead of uv. A worktree links the main checkout's .venv and imports the worktree's src/ through PYTHONPATH, which mosaic-worktree-hook.py adds to python and pytest runs. Covers the branch off main, the .venv symlink, the import check, a sibling mosaic-media checkout, and teardown (no merge or push without the user's word). Triggers: worktree, git worktree, new branch, branch off main, wrong mosaic imported, mosaic.__file__, PYTHONPATH, .venv missing, symlink, delete the worktree, delete the branch, clean up after merge."
---

# mosaic worktrees

Work on a branch in a worktree off `main`, under `.claude/worktrees/` (ignored by
`.gitignore`).

## Create

```bash
git worktree add -b <branch> .claude/worktrees/<branch> main
ln -sfn "$PWD/.venv" .claude/worktrees/<branch>/.venv
```

Run the second line from the main checkout. The conda environment is not copied: the
worktree shares the main checkout's `.venv`, whose editable install points at the
main checkout's `src/`.

## Import the worktree's source

`.claude/skills/tools/mosaic-worktree-hook.py` prefixes every Bash command that runs
python or pytest inside a worktree with `export PYTHONPATH=<worktree>/src`. The
working directory, or the target of a leading `cd`, decides the worktree. The hook
leaves a command alone when it sets `PYTHONPATH` itself or when `remote` launches
it. It denies a `.venv/` run in a worktree that has no `.venv`, naming the
`ln -sfn` to run.

Check it once per worktree:

```bash
.venv/bin/python -c "import mosaic; print(mosaic.__file__)"   # prints <worktree>/src/mosaic/__init__.py
```

Outside Claude Code's Bash tool (a subprocess you start, a script on another machine)
the hook does not apply. Set `PYTHONPATH="$PWD/src"` yourself there.

basedpyright resolves `src/` from `[tool.basedpyright]` in `pyproject.toml`, without
the prefix.

## Sibling repositories

To test against an unreleased `mosaic-media`, install it editable over the release
(`CONTRIBUTING.md`, section "Environment Notes"). That changes the shared `.venv` for
every worktree and the main checkout. Say so to the user before you do it.

## Tests and the task machine

`remote` ships the worktree's tracked files, so `git add -N` each new file first. The
full suite runs through `mosaic-suite` (see `/mosaic-testing`).

## Finish

Merge and push only on the user's word. After the user has merged:

```bash
rm .claude/worktrees/<branch>/.venv
git worktree remove .claude/worktrees/<branch>
git branch -d <branch>
```

Remove the symlink first. A removal step then cannot delete files of the shared
environment through it.
