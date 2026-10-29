---
name: extending-mosaic
description: "The checklist for adding or changing a plugin of the mosaic toolkit: a feature (@register_feature), an op (@register_op), a track or label converter, a media step (@register_media_step), a tracker, a CLI verb, or an MCP tool. Load before adding one, before changing a registered plugin's params, output or identity, and before finishing such a branch. Each entry names the files to touch, the guarding conformance and meta tests, and the follow-ups agents most often miss: the version bump, the goldens, docs/reference regeneration, the README counts, the CHANGELOG entry, and the users' agent guide. The rules themselves stay in CLAUDE.md and the area guides, and this skill cites them. Triggers: new feature, add a feature, register_feature, new op, register_op, converter, register_track_converter, register_label_converter, media step, register_media_step, MediaStepSpec, new tracker, TrackingRoot, new CLI command, Typer, new MCP tool, TOOLS, params change, bump version, gen_docs_reference, docs/reference, README counts, CHANGELOG."
---

# Extending mosaic

New functionality is almost always a plugin. The registrars and their registries are
the table in `CLAUDE.md`, section "Registries". Read the area guide of the package you
work in before you start. This skill lists what a change must also touch.

## Every plugin change

1. **Params.** A Pydantic model (`core/params.py`). A field declares its prose with
   `Declared(...)` (`tests/meta/test_params_declaration.py`). `HASH_EXCLUDE` marks
   throughput knobs only.
2. **Version.** A field, default or output that changes meaning bumps the plugin's
   `version` (`CLAUDE.md`, "Params are Pydantic").
3. **Goldens.** An identity that moves fails `tests/core/pipeline/test_feature_run_id_golden.py`
   or `tests/core/pipeline/test_op_identity_golden.py`. Rewrite with
   `MOSAIC_UPDATE_GOLDEN=1` only when the move is intended, and explain every moved
   line (`/mosaic-testing`).
4. **Generated reference.** `docs/reference/` is generated from the registries and the
   Typer app:

   ```bash
   .venv/bin/python scripts/gen_docs_reference.py --write
   ```

   Commit the result. The docs workflow runs `--check`, and the commit hook refuses a
   stale page when a committed module registers something or lives under
   `src/mosaic/cli/`.
5. **README counts.** A new or removed plugin can change a headline number of
   `README.md` (`test_readme_counts_match_the_registries` in
   `tests/meta/test_docs_assets.py`).
6. **CHANGELOG.** A change that moves an identifier or a name on disk gets an entry
   in `CHANGELOG.md`: a new `## Unreleased: <title>` section above the others, saying
   what moved in what another repository can observe. The file's opening
   paragraph states its scope.
7. **Users' agent guide.** A change to what a user does updates
   `src/mosaic/agent/SKILL.md`. Every URL there needs a page under `docs/`
   (`tests/meta/test_agent_guide.py`). Update the matching guide under `docs/guides/`
   and its `mkdocs.yml` nav entry, then run `mkdocs build --strict`.

## Feature (`@register_feature`)

- The protocol and its required attributes: `src/mosaic/behavior/CLAUDE.md`, "Feature
  protocol". The module joins the imports of `behavior/feature_library/__init__.py`, or
  `behavior/visualization_library/__init__.py` for media.
- `requires`, `provides` and `prerequisites`: `src/mosaic/core/pipeline/graph/CLAUDE.md`,
  "Declarations and the registries". An optional package loads through its
  prerequisite (`OptionalModule.load()`).
- Guards: `tests/behavior/feature_library/test_feature_conformance.py`,
  `test_column_contract_conformance.py`, and the step-detail size
  `MAX_STEP_TYPE_DETAIL_CHARS` (`agent/mcp/spec.py`).
- Output goes through `run_feature()` and `write_parquet_atomic` (`CLAUDE.md`,
  "Cross-cutting invariants"). Seed every random state.

## Op (`@register_op`)

- A tracking op's module is listed in `REGISTERING_MODULES`
  (`tracking/registration.py`, `tests/tracking/test_registration.py`). A core op's
  module is listed in `CORE_OP_MODULES` (`core/pipeline/ops.py`).
- Identity: `<kind>.<version>-<10 hex>` (`src/mosaic/core/pipeline/CLAUDE.md`,
  "Identifiers").
- A job that declares a progress total reports progress
  (`tests/meta/test_progress_reporting.py`). A failed entry is reported instead of dropped
  (`CLAUDE.md`, "A run reports its losses").

## Track converter (`register_track_converter`)

- The steps are in the docstring of `src/mosaic/core/track_library/__init__.py`
  ("Adding a New Track Converter"), starting from `track_converter_template.py`.
- Declare `output_schema` and its family's `space`: `src/mosaic/core/CLAUDE.md`, "Track
  converters and schemas".

## Label converter (`register_label_converter`)

- Add the call to `behavior/label_library/__init__.py` (`src/mosaic/behavior/CLAUDE.md`,
  "Label converter registration").

## Media step (`@register_media_step`)

- Register in `core/media/preprocess/registry.py`, and add the spec to the
  `MediaStepSpec` union in `core/media/preprocess/specs.py`
  (`tests/core/media/preprocess/test_media_steps.py`).
- A variant is consumed by name and published in source space (`CLAUDE.md`,
  "Cross-cutting invariants").

## Tracker

- `src/mosaic/tracking/CLAUDE.md`, "Wiring a tracker": a `TrackingRoot` row in
  `core/pipeline/tracking_roots.py` with `output_schema` set, publication through
  `publish_tracks_table`, and the tool's `ToolDecoder`.
- Guard: `tests/tracking/common/test_tracker_conformance.py`, which names the missing
  piece of a half-wired tracker.
- A tool that runs in a separate environment stays a separate program
  (`CLAUDE.md`, "Dependency invariants").

## CLI verb

- A Typer command under `src/mosaic/cli/`. A read-only verb opens its dataset with
  `load_dataset(..., read_only=True)` (`cli/_context.py`) and joins the cases of
  `tests/cli/test_cli_read_only.py`. A dry run reads without the lock
  (`index_lock_unless_dry_run`).
- A verb with `--json` prints the same wire model its MCP tool returns.
- Regenerate `docs/reference/` (step 4 above).

## MCP tool

- `src/mosaic/agent/CLAUDE.md`, "Tool contract": one entry in `TOOLS`
  (`agent/mcp/spec.py`), the `ToolBackend` method of the same name, a library function
  with a CLI `--json` verb beside its concept, and `narrowing` and `confirm_when` where
  they apply.
- At most 16 tools (`tests/meta/test_shared_tool_cap.py`). No module under `core`,
  `behavior` or `tracking` imports `mosaic.agent` (`tests/meta/test_agent_layering.py`).

## Finish

Run `mosaic-check`, then the full suite through `mosaic-suite` (`/mosaic-testing`).
