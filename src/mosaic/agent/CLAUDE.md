# mosaic.agent

This package serves the agents of mosaic's users. `SKILL.md` is the guide that
`mosaic agent guide` prints, and `__init__.py` installs it into a project. `mcp/` is
the MCP server behind `mosaic agent serve`. The user documentation is
`docs/agents.md`.

## Layering

- `mosaic.agent` is an adapter beside `mosaic.cli`. It imports `core`, `behavior` and
  `tracking`. A module under those three does not import it
  (`tests/meta/test_agent_layering.py`).
- `mcp/spec.py` declares the tool contract. It imports neither the MCP SDK, pandas,
  cv2, av, `mosaic.behavior` nor `mosaic.tracking`. A server outside this repository
  can then import it. An output model from a module beyond that budget is named by a
  `ModelRef` and imported on use (`tests/agent/mcp/test_tool_spec.py`).
- `mcp/_binding.py` is the one module that imports the SDK. `mcp/server.py` loads
  `MCP_SDK` before importing it, which names the `mcp` extra when the SDK is absent
  (`tests/agent/mcp/test_mcp_extra.py`). The modules beside the server leave the SDK
  unloaded.
- `mcp/local.py` imports the library inside the methods that use it. Building a
  `LocalBackend` loads neither pandas nor the feature library.

## Tool contract

- `TOOLS` in `spec.py` declares each tool once, and a backend answers it through the
  `ToolBackend` method of the same name. The set contains at most 15 tools
  (`tests/meta/test_shared_tool_cap.py`), and each schema stays under
  `MAX_SCHEMA_BYTES`.
- A tool wraps a library function and its wire model, which live beside their concept
  with a CLI verb printing the same model under `--json`. Compute in the library and
  keep the adapter to addressing, paging and the confirmation gate.
- A failure returns the error envelope (`core/pipeline/error_envelope.py`). Dataset
  content (notes, tags, names, error text) is returned in typed fields only.
- `call_tool` refuses a read tool's result above `MAX_RESULT_CHARS` with
  `result_too_large`, naming the `ToolSpec.narrowing` arguments. A write tool's
  result is returned whatever its size. A new argument that shrinks a result joins
  its tool's `narrowing`, and a new `limit` must
  (`test_every_tool_with_a_limit_names_it_among_its_narrowing`).
- `ToolSpec.always_load` becomes the tool's `anthropic/alwaysLoad` meta in
  `_binding.py`. A success returns the summary, the output as JSON text, and the
  structured content.
- A step type's projection (`step_type_detail`, `param_rows`) lives in `graph/`, and
  `mosaic features describe` and `mosaic pipeline catalog <name>` print it. Each step
  type's detail stays under `MAX_STEP_TYPE_DETAIL_CHARS`, which a test measures over
  the catalog, so a new feature or op with a large params model must fit it.
- `submit_pipeline` and `cancel_request` are the only tools that change a dataset, and
  both declare `confirm_when`. No tool takes `overwrite` (`test_tool_spec.py`).
- The confirmation conditions are the ones `test_tool_spec.py` pins. A whole-dataset
  or large plan is a `PlanWire.cautions` entry, not a condition. In `_binding.py` an
  elicitation's `cancel` is `confirmation_unanswered` and a decline or an unticked
  accept is `confirmation_declined`.
- A confirmation question ends in the line
  `[mosaic-confirm tool=<tool> conditions=<a>,<b>]` (`spec.confirmation_tag`), which
  `read_confirmation_tag` and operators' Claude Code hooks (`mosaic agent
  confirm-hook`) parse. Changing its grammar is a contract change. Finding text is
  put on one line, and a finding quotes dataset content with `repr`, so nothing in a
  dataset can forge the tag. `confirm_hook` answers a confirmation whose tag it cannot
  read with its fallback, as it answers an unapproved one (`test_confirm_hook.py`).
- A run's report is a resource, not a tool: a feature declaring `report_file` has it
  served at `.../artifacts/feature/{name}/{run_id}/report` through
  `read_run_report`.
- `docs/agents.md` lists the operations the server leaves to the user, such as
  deletion, `--apply` on a sweep or prune verb, and manifest writes. Do not add a tool
  for one. The one deletion the tools make is removing superseded feature runs: the
  `cleanup` argument of `submit_pipeline`, refused (`cleanup_not_allowed`) unless the
  server runs with `--allow-cleanup feature-runs`, and asked about through the
  `cleanup` condition. The policy comes from the server's configuration, never from a
  call (`test_confirmation_gate.py`). `plan_recipe` takes `cleanup` as a read. `retire`
  widens the same deletion to the runs of retired recipes in any storage. It needs
  `cleanup`, applies to one call, and shares the flag and the condition, so its
  finding names the retired recipes.
- Parallelism limits (`max_workers`, `allow_process_mode`, `min_free_memory_fraction`)
  are the operator's. `submit_pipeline` takes `workers` and `parallel_mode` within them
  and `ParallelLimits.settings_for` refuses a request over a limit
  (`parallel_limit`, `test_confirmation_gate.py`). `plan_recipe` takes neither. The
  settings reach the driver through `DriverStarter` and then argv flags, and are not
  part of a recipe, a request or a run id.
- An accepted confirmation confirms only the conditions it named. `_binding.py` runs
  the call again without `confirm` first and asks again when it meets another
  (`test_mcp_server.py`).

## Standard output

Standard output is reserved for JSON-RPC. `serve` enters the SDK's `stdio_server`
before it imports the binding and the library. A submitted request runs in a detached
driver whose streams go to a log file. `tests/agent/mcp/test_mcp_stdio.py` plans,
submits, polls and previews over a real stdio pipe and checks the stream.

## Tests

Tests are in `tests/agent/mcp/`. The CI `test` job installs `.[mcp]` and sets
`MOSAIC_CI_MCP=1`. `tests/conftest.py` then fails a run that cannot import the SDK
instead of skipping its tests.
