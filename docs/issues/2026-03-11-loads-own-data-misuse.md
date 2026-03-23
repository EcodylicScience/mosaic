# `loads_own_data()` misused by features with declared inputs

## Problem

`loads_own_data()` was designed for features with no pipeline-provided inputs
(e.g. GlobalWard, `InputRequire = "empty"`) that load everything from artifact
specs. Four features misuse it as a workaround to bypass the pipeline's
per-sequence iterator:

| Feature         | `loads_own_data()` | `InputRequire` | Actual data source              |
|-----------------|--------------------|----------------|---------------------------------|
| GlobalWard      | True               | `"empty"`      | Artifact specs in Params        |
| GlobalKMeans    | True               | `"any"`        | Artifacts + optional Result inputs |
| **GlobalTSNE**  | True               | `"nonempty"`   | Result inputs via StreamingFeatureHelper |
| **WardAssign**  | True               | `"nonempty"`   | Result inputs via StreamingFeatureHelper |
| **KpmsFit**     | True               | `"nonempty"`   | Dataset index (TrackInput)      |
| **KpmsApply**   | True               | `"nonempty"`   | Dataset index (TrackInput)      |

Bold rows are the misuses: they *require* inputs yet claim to load their own
data.

## Why it matters

When `loads_own_data() = True`, `run_feature` (in `core/pipeline/run.py`):

1. Passes an **empty list** to `fit()` -- the feature re-discovers inputs
   internally, duplicating resolution logic.
2. **Rejects group/sequence scope filters** -- treats feature as having no
   pairs, even though it does.
3. **Rejects frame/time filters** -- raises RuntimeError. This is the main
   user-facing bug: GlobalTSNE, WardAssign, KpmsFit, and KpmsApply *could*
   support `filter_start_frame` / `filter_end_frame` since they load data
   themselves, but the pipeline blocks it preemptively.

## Why they do it

The pipeline's standard path for `needs_fit() = True` collects all DataFrames
into a list before passing to `fit()`:

```python
all_dfs = [df for item in iter_inputs()]
feature.fit(all_dfs)
```

Global features need cross-sequence access with controlled memory (streaming
one sequence at a time via `StreamingFeatureHelper`). Setting
`loads_own_data() = True` sidesteps the eager collection, letting `fit()` load
data on its own terms.

## Correct fix

`loads_own_data() = True` should only be valid for `InputRequire = "empty"`.
Features that have inputs but need custom loading should use a different
mechanism. Options:

**Option A -- Lazy iterator**: Change the pipeline to pass a lazy iterator
(not a pre-collected list) to `fit()`. Features that want streaming control
just iterate it themselves. `loads_own_data()` becomes unnecessary for
Result-based features.

**Option B -- `fit_mode` enum**: Replace the boolean with a three-way:
`"pipeline"` (standard iterator), `"self"` (no inputs, loads from artifacts),
`"streaming"` (has inputs, loads them in fit via helper). Pipeline adjusts
behavior per mode.

**Option C -- Accept iterator in fit**: Pass `X_iter` as a generator that
yields per-sequence DataFrames. Features can consume it or ignore it. This is
close to option A but requires no protocol change since `fit()` already accepts
`Iterable[pd.DataFrame]`.

Option A/C is simplest. The signature `fit(X_iter: Iterable[pd.DataFrame])`
already exists; the pipeline just needs to pass a generator instead of a
materialized list.

## Scope of fix

- Change GlobalTSNE, WardAssign, KpmsFit, KpmsApply to
  `loads_own_data() = False`
- Change GlobalKMeans to return `loads_own_data()` based on whether inputs are
  empty (True only in fit-only mode)
- Update `run_feature` to pass a lazy generator to `fit()` instead of a
  collected list
- Remove the frame/time filter rejection for features that have inputs but
  previously claimed `loads_own_data()`
- Wire frame filters through `StreamingFeatureHelper` / direct loading paths

## Resolution

Resolved by dropping `loads_own_data()` from the Feature protocol entirely.
The pipeline now uses `inputs.is_empty` to determine whether a feature has
pipeline-provided data. `fit()` receives a lazy generator instead of a
materialized list, so features that bypass the pipeline iterator
(GlobalTSNE, WardAssign, etc.) pay zero memory cost. Frame/time filters
are no longer blocked for features with non-empty inputs.

## Related

- `docs/plans/2026-03-11-unified-scope.md` -- scope unification plan
- `docs/plans/2026-03-12-drop-loads-own-data.md` -- implementation plan
