# tests

This directory owns the test suite, its shared helpers, and the meta tests that check
the repository's conventions.

The placement, import and conftest rules are "Where a test goes" in
[CONTRIBUTING.md](../CONTRIBUTING.md), and the commands are "Running the tests" there.

## Running

- Use `-n0` also to bisect a failure caused by another test in the same worker.
- Keep `--dist worksteal`. A case costs anywhere from microseconds to a video
  encode, and a static split leaves workers idle at the end of a run.
- Warnings are errors, every `xfail` is strict, and an empty parametrize fails
  collection (`[tool.pytest.ini_options]` in `pyproject.toml`).

## Shared code

- A helper imports the feature library only inside the function that needs it.
- A new `importorskip` (or module-level `find_spec` probe) names a module that a
  CI job installs, listed in the `CI_*_MODULES` tuples of
  `tests/helpers/environment.py`. A guard on a base dependency fails too
  (`test_optional_dependency_coverage.py`).
- A golden file under `tests/data/` is read and written through `read_golden` and
  `write_golden`, and `MOSAIC_UPDATE_GOLDEN=1` rewrites it. A golden test's failure
  message gives `regenerate_command(__name__)`.

## Meta test failure conditions

CONTRIBUTING.md gives the conditions of `test_suite_layout.py` and
`test_repository_references.py`.

- `test_pytest_config.py`: drift in `addopts`, the markers, or the CI workflows'
  selection by marker.
- `test_optional_dependency_messages.py`: an install hint or `require` call naming
  an undeclared extra, or a self-referential extra that does not resolve.
- `test_helpers_imports.py`: `import tests.helpers` importing the feature library.
- `test_graph_imports.py`: `core.pipeline.graph` importing the feature library or a
  submodule at import, an exported name that does not resolve, or a module other
  than `resolve` importing `FEATURES`.
- `test_status_imports.py`: the import or a call of a status reader, a request
  lister or the entry-error report, or the import of the cancel module, loading
  pandas, cv2, av or the feature library, or `import mosaic.runlog` loading
  pydantic.
- `test_external_tool_separation.py`: an import across a separated tool's
  boundary, or a declaration of the tool in mosaic's dependencies.
- `test_behavior_import_is_torch_free.py`: `import mosaic.behavior` importing torch.
- `test_core_lazy_exports.py`: `mosaic.core` binding a name at import, or a leaf
  import loading pandas or h5py.
- `test_pipeline_lazy_exports.py`: `mosaic.core.pipeline` importing a submodule at
  import, or importing it loading cv2, av or pandas.
- `test_tracking_lazy_exports.py`: `mosaic.tracking`, `tracking.common`,
  `tracking.frame_extraction` or `tracking.ops` importing a submodule at import, an
  exported name that does not resolve, or an inventory read loading a tracker
  subpackage, a trainer, frame extraction, cv2, av or the feature library.
- `test_media_lazy_exports.py`: `mosaic.core.media` or `core.media.preprocess`
  importing a frame reader or a step at import, an exported name that does not
  resolve, or `import mosaic.core.dataset` loading cv2 or av.
- `test_lazy_export_identity.py`: a public name of a package that binds names
  through `lazy_export` that equals a submodule basename, or that resolves to an
  object other than the one its table names.
- `test_core_params_is_a_leaf.py`: `core.entry`, `core.individuals` or
  `core.strict_model` importing mosaic, or `core.params` beyond `strict_model`.
- `test_params_declaration.py`: a parameter field without a description
  declaration, a declared fact missing from the schema, or a ratchet that grew.
- `test_unwired_fields.py`: a field declared `unwired=` that is now wired.
- `test_progress_reporting.py`: a job that declares a progress total and reports none.
- `test_docs_assets.py`: the two hero diagrams differing in words, geometry or
  palette rules, or a README count that no longer matches its registry.
- `test_agent_guide.py`: a site URL in `src/mosaic/agent/SKILL.md` with no page
  under `docs/`, a wheel without the guide, or a wheel that ships a `CLAUDE.md`.
- `test_tools_import.py`: a script under `tools/` that fails to import.
- `test_agent_layering.py`: a module under `core`, `behavior` or `tracking` that
  imports `mosaic.agent`.
- `test_agent_never_writes_splits.py`: a module under `mosaic.agent` whose syntax tree
  stores an attribute or item named `splits`, calls `set_splits`, `remove_splits`,
  `mutate_manifest`, `write_manifest` or `save`, or updates `splits` through
  `model_copy` or `setattr`.
- `test_shared_tool_cap.py`: more than 16 tools in `mosaic.agent.mcp.spec.TOOLS`.
- `test_storage_name_marker.py`: a feature slug or a literal input name that contains
  `~`, the shortened storage-name marker.
