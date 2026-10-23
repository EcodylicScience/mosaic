# Handoff: agent-interface slices S6, S4, S5 (continue on kn-cgpu02)

Written 2026-10-23 by the session that built S1 to S3. You are a fresh Claude Code session with none of
that context. This file, `PLAN.md`, `BRIEFING.md` and `REVIEW-DEFERRED.md` carry what you need.
Read the repository's own `CLAUDE.md` and the area guides it links as well; they load automatically.

## What exists

Branch `agent-interface/s1-s2` (pushed to origin). Four commits on top of mosaic `main` 575d9905
(rebased 2026-10-23; `main` keeps moving, so rebase again before you finish):

| Commit | Slice |
| --- | --- |
| `bbb2910f` | S1 held-out protocol: `splits:` roles in `dataset.yaml`, `fits` declaration, planner guard, `assume_added`, `mosaic pipeline score`, `score_sealed` confirmation |
| `e437b777` | S2 scope-free fits (`sequence-classifier`, `lightning-action`, fitted-weight `probability-ensemble`; apply-mode `arhmm`, `kpms`, `vame`, identity models), `evaluate-labels` stacks inputs, `mosaic pipeline verify` |
| `eed34e81` | S3 `tune-threshold`, block cross-validation (`cv_unit`, `cv_blocks_per_sequence`, `cv_gap_frames`) for `sequence-classifier` and `xgboost`, frame numbers in labeled templates, `temporal_reach` |
| `c0bc4dd9` | Fixes from a six-lens review of the whole branch |

Nothing is merged. The branch name is historical (it was cut for S1 and S2). All of S1 to S3 passed
the full suite (10.3k tests, 0 failed) before the last rebase, and ruff, docs build and basedpyright
(no new errors) were clean. The CHANGELOG "Unreleased" section, `SKILL.md`, the guides and the area
`CLAUDE.md` files already describe S1 to S3.

## Your job

Implement **S6, then S4, then S5** from `PLAN.md` (sections "S6", "S4", "S5"; the briefing's
sections are in `BRIEFING.md`). Work on a new branch off `agent-interface/s1-s2`:

```
git fetch origin
git checkout -b agent-interface/s6-s4-s5 origin/agent-interface/s1-s2
```

Do not push to `main`. Push your branch only when Jacob asks. One commit per slice (tests, docs,
CHANGELOG, `SKILL.md` and the regenerated reference in the same commit as the code), as the S1 to S3
history does. Follow root `CLAUDE.md`'s definition of done: `ruff check --fix && ruff format`,
`basedpyright` (new code clean), `pytest -m "slow or not slow"`, `python scripts/gen_docs_reference.py
--write` then `--check`, `mkdocs build --strict`. Never edit `docs/reference/` by hand.

Jacob's standing preferences (from memory): docs carry no implementation detail unless it guides the
user or prevents a mistake; propose before re-architecting; commit and push only when asked; use
they/them for people unless told otherwise.

## Decisions already taken (do not reopen)

- S5 is **one `manage_dataset` tool with an `action`** (create, add_source, remove_source, scan,
  set_tags). The MCP shared tool cap (`SHARED_TOOL_LIMIT` = 15, `tests/meta/test_shared_tool_cap.py`)
  is Jacob's own setting and **may be raised**; there are 14 tools, `patch_recipe` makes 15 and
  `manage_dataset` 16, so raise it deliberately (update the two tests and the doc sentence).
- S6 `patch_recipe` has ops `add_step`, `remove_step`, `set_param`, `replace_step`. **`set_scope` is
  dropped**: step scopes live on the request, not the recipe. Instead, accept the library `Scope` JSON
  (`{"entries"|"groups"|"sequences": ...}`) in MCP `entries`/`step_entries`, as the CLI's
  `--step-scope` does; keep the bare pair list working.
- S6 `repeat`: expanded at parse by a `mode="before"` validator on `Recipe`; the **expanded** form is
  hashed and saved; **no `schema_version` bump**.
- `patch_recipe` persists its result with `save_recipe` (content-addressed) so the digest resolves
  later; mark it non-read-only, idempotent, no confirmation.
- Sealed groups' labels stay out of the manifest. **No MCP tool may write `splits`.**
- S4: `mosaic agent serve --http HOST:PORT --token-file FILE` using the `mcp` SDK's
  `Server.streamable_http_app` (`mcp 2.3.0`, already installed with the `mcp` extra; it brings
  `uvicorn` and `starlette`). Spike the confirmation behavior over stateless HTTP first (plan, S4
  item 3): legacy clients fall back to the `confirm: true` path, which a model can set.

## Facts about the code as it stands (measured, not from the plan)

- MCP tools: 14. Compact schema bytes, input / output (limit 8192 each, `MAX_SCHEMA_BYTES`):
  `plan_recipe` 3919 / **8103**, `submit_pipeline` 5429 / 1146, `get_request_status` 909 / 4484,
  `get_inventory` 2054 / 3879, `describe_step_types` 1592 / 3634, `describe_dataset` 232 / 2828,
  `validate_recipe` 1080 / 558, `suggest_steps` 1547 / 2581. `INSTRUCTIONS` is 1337 characters (limit
  2048). **`plan_recipe`'s output schema has only about 89 bytes of headroom: add no field to
  `PlanWire` or `PlanPage`** (S1's `assume_added` result goes through existing cautions strings for
  that reason). A new `DatasetSummary.next_steps` field (S5) is additive on a different tool whose
  output has room, but it is shared with `mosaic describe --json` and mosaic-api, so it needs a default.
- `ConfirmationCondition` (`src/mosaic/agent/mcp/spec.py`) already includes `score_sealed` from S1.
  S5 adds `create_dataset` and `add_source`; `tests/agent/mcp/test_tool_spec.py` and
  `test_confirmation_gate.py` pin the sets and must be updated, and `cli/agent.py`'s `--approve`
  help lists them.
- **S5 trap:** `tests/meta/test_agent_never_writes_splits.py` fails on *any* call to `mutate_manifest`,
  `write_manifest`, `save`, `set_splits` or `remove_splits` inside `mosaic.agent` (its `ALLOWED_CALLS`
  is empty on purpose). `manage_dataset` must write the manifest through a function in `mosaic.core`
  (add one, e.g. next to `Dataset.add_scan_source`) and the adapter calls that function; do not loosen
  the test to allow raw manifest writes, and keep `splits` out of every tool.
- `LocalBackend` (`src/mosaic/agent/mcp/local.py`) holds a fixed manifest list and a `_served` map
  under a non-reentrant lock; `_register_contributors` takes the same lock (plan, S5 item 4).
- Saved recipes are `.mosaic/pipelines/<digest>.json`, written only by `submit_request` today.
  `Recipe.model_validate` is the single parse hook; problem locations are mapped back to raw
  `steps[i]` in `graph/validate.py` (`_structural_problem`), which shifts after expansion: report
  problems by step id.
- The held-out guard (`core/pipeline/graph/heldout.py`, `plan.py`) reads each step's target after step
  scopes narrow it; `repeat` expansion must happen before that (it does if done at parse), and the
  one-scope-format change must keep `Scope.selects` semantics.
- `Plan` has internal fields not in the wire (`request_scope`, `reach_cautions`, `bound`,
  `deferred_by`); the wire and fingerprint are unchanged for datasets without `splits`.
- Line numbers in `PLAN.md` come from an exploration before S1 to S3 landed; verify every anchor.

## Deferred, owed and open

- `REVIEW-DEFERRED.md` lists findings the review deferred. The most important: a bound run fitted on a
  held-out group passes the S1 guard (needs a decision on reading a bound run's recorded fit scope);
  new tests are slow (`tune-threshold` pipeline, `verify`, `score`), and main's CI test job has hit its
  20-minute timeout, so shrink fixtures if you can.
- `lightning-action` has no `cv_folds` or block cross-validation.
- Many temporal features declare no `temporal_reach` (each gives a caution upstream of a block-CV
  classifier); declaring 0 for the verified per-row ones would remove noise.
- Acceptance reruns that need the kn-cgpu02 CalMS21 sandbox (not CI), all still owed:
  S1 Task 1 rerun with `mosaic pipeline score`; S2 `mosaic pipeline verify` on both demo recipes
  (the TCN tolerance across thread counts and on GPU is documented, not exercised); S3 Task 3 rerun.
  The sandbox notes are in Jacob's memory (`project_calms21_agent_sandbox_kn_cgpu02`): demo folders
  are train-only and the test set is scored afterwards. Ask Jacob before using the sandbox or the GPU,
  and do not run anything inside the Docker-fenced sandbox's data.
- Not run in the previous session: the `lightning-action`, `feral` and `tracker` CI lanes, the real
  kpms and VAME environments.

## Environment on kn-cgpu02 (verify, it was not checked)

Follow root `CLAUDE.md` "Development environment": a conda env at `./.venv` with `ffmpeg`, `av`,
`py-opencv` from conda-forge first, then `pip install -e ".[all]"` plus `pip install -e ".[mcp]"` for
the MCP tests (and the `test` dependency group). One ffmpeg build per environment. Torch tests carry
the `identity` marker (`pytest -m identity`). The box was hard-rebooted on 2026-10-20: check it, the
GPU and disk before starting. Use loopback and an ephemeral port for S4's HTTP tests.

## Working method that worked

Large slices were built with multi-agent workflows whose stages end in a git commit and skip
themselves when that commit exists, so a lost connection costs little. Use that, or work directly,
as you prefer. Review each slice adversarially before committing: confirm each finding by running
code, drop taste-only items. When done, report to Jacob what was implemented, what deviated, and what
is deferred, then wait.
