# StreamingFeatureHelper issues

Three related issues in the global feature data loading and output path.

## 1. Alignment bug in `load_key_data()`

**File:** `src/mosaic/behavior/feature_library/helpers.py`, line ~555

`StreamingFeatureHelper.load_key_data()` concatenates arrays from multiple
inputs using min-trim + `np.hstack`:

```python
T_min = min(m.shape[0] for m in mats)
mats_trim = [m[:T_min] for m in mats]
X_full = np.hstack(mats_trim)
```

This assumes all inputs have rows in the same order and silently drops excess
rows. No frame-level alignment is performed.

### When it breaks

- **Mixing frame-scoped and unscoped runs.** A feature run with
  `filter_start_frame=1000` produces fewer rows per sequence than an unfiltered
  run. If one input comes from each, the row counts differ and the min-trim
  silently discards trailing rows from the longer input. Worse, the remaining
  rows are not aligned -- frame 1000 from the scoped run is hstacked with
  frame 0 from the unscoped run.

- **Partial/scoped runs with different sequence coverage.** If one input was
  computed for a subset of sequences/groups (e.g. a single-group scope) and
  another was computed for all, the per-sequence parquet files may have
  different row counts for the same key. Again, min-trim silently misaligns.

### Why it hasn't been caught

All current global features consume inputs that were produced by the same
pipeline run with the same scope, so row counts match in practice. The bug is
latent -- it will surface when users mix results from different scoped runs
or when frame filtering is applied to global feature inputs.

### Fix

Replace min-trim + hstack with `pd.merge` on frame columns
(`frame`/`time`/`id`/`sequence`/`group`), following the same approach as
`yield_input_data` in `iteration.py:489`. This is a breaking change to
`load_key_data()`'s return type (numpy array -> DataFrame), so all callers
need updating.

## 2. `load_key_data` vs `load_key_data_with_identity` duplication

**File:** `src/mosaic/behavior/feature_library/helpers.py`, lines ~511 and ~609

Two methods with nearly identical logic:

| Method | Returns | Callers |
|--------|---------|---------|
| `load_key_data()` | `(X, frames)` | GlobalTSNE, GlobalWard, feature_template__global |
| `load_key_data_with_identity()` | `(X, frames, id1, id2, entity_level)` | WardAssign, GlobalKMeans |

The identity variant duplicates the entire loading loop, min-trim, and memory
cleanup. The only addition is a call to `_load_identity_from_spec()` inside
the loop and identity-array trimming at the end.

### Why both exist

WardAssign and GlobalKMeans need per-frame identity columns (`id1`, `id2`,
`entity_level`) to write them into their output parquet. GlobalTSNE and
GlobalWard don't need identity, so they use the simpler 2-tuple variant. The
identity variant was added alongside the parquet output without refactoring
the original.

### Fix

Merge into a single method that always loads identity when available and
returns a single struct or DataFrame. Callers that don't need identity simply
ignore those fields. This cleanup pairs naturally with fix #1 (switching to
DataFrame-based alignment), since a merged DataFrame already contains all
columns including identity.

## 3. Global features bypass the standard transform/output path

**Files:**
- `src/mosaic/behavior/feature_library/global_tsne.py` (`_persist_mapped_coords`)
- `src/mosaic/behavior/feature_library/ward_assign.py` (`_write_sequence_outputs`)
- `src/mosaic/behavior/feature_library/global_kmeans.py` (`save_model`)

GlobalTSNE, WardAssign, and GlobalKMeans all write per-sequence parquet
directly during `fit()` or `save_model()`, bypassing the standard pipeline
output path (`run_feature` -> `transform()` -> `_collect_completed` ->
`write_output`). Their `transform()` methods return stubs (e.g.
`{"global_tsne_done": [True]}`), and they register outputs via
`get_additional_index_rows()` instead.

### Why this exists

These features process all sequences during `fit()` via
`StreamingFeatureHelper` (one sequence at a time for memory control). The
standard transform loop expects to call `transform(df)` per sequence with a
DataFrame provided by the pipeline's input iterator. Global features that do
cross-sequence work (fitting on all data, then mapping back per sequence)
can't use this pattern -- they need to control the iteration themselves.

This is the same underlying issue as the `loads_own_data()` misuse documented
in `docs/issues/2026-03-11-loads-own-data-misuse.md`. The `loads_own_data`
flag was removed, but the bypass pattern remains: features still write their
own outputs during fit and provide index rows after the fact.

### Why it matters

- Output writing logic is duplicated across three features instead of living
  in one place (`write_output` in `run.py`).
- Index row construction (`_build_index_row`, `_append_index_row`) is
  reimplemented in each feature rather than handled by the pipeline.
- The standard output path's guarantees (overlap trimming, overwrite checks,
  consistent naming) don't apply to these bypass writes.

### Fix

When `fit()` receives a lazy iterator (the fix from the `loads_own_data`
issue), features can consume it during fit for cross-sequence work, then
yield per-sequence results back through `transform()` for the pipeline to
write normally. This eliminates the bypass pattern entirely. Until then, the
current approach works but duplicates output-writing concerns.

## 4. Global features drop identity columns from per-sequence parquet output

**Files:**
- `src/mosaic/behavior/feature_library/global_tsne.py` (`_persist_mapped_coords`, `save_model`)

GlobalTSNE writes per-sequence parquet with columns `tsne_x`, `tsne_y`,
`frame`, `sequence`, `group` -- but no `id1`/`id2`. When the input is a pair
feature (one row per frame x id-pair), the `frame` column contains
duplicates and the output is lossy: there is no way to distinguish which row
belongs to which identity pair.

GlobalKMeans and WardAssign already preserve identity via
`load_key_data_with_identity()` and write `id1`/`id2`/`entity_level` columns.
GlobalTSNE uses `load_key_data()` which discards identity (see issue #2).

### Consequences

- Downstream consumers cannot merge t-SNE coords with identity-aware outputs
  (e.g. KMeans clusters) on a unique key. Merging on `frame` + `sequence` +
  `group` produces a cross product when multiple rows share the same frame.
- VizGlobalColored currently works around this by assuming same-length arrays
  are aligned by construction (same `StreamingFeatureHelper` iteration order).
  This is fragile -- any difference in scope or filtering between the coord
  and label features would silently misalign data.

### Fix

GlobalTSNE (and any future global feature operating on pair inputs) should
use `load_key_data_with_identity()` and write `id1`/`id2`/`entity_level`
columns, matching the pattern in GlobalKMeans and WardAssign. This pairs
with fix #2 (merging the two load methods) -- once there is a single method
that always returns identity, all global features get it for free.

Once all global features write identity columns, `VizGlobalColored` can
merge coords and labels on the full key (`frame`, `sequence`, `group`,
`id1`, `id2`) -- the clean path. The current workaround (assuming
same-length arrays are aligned by construction) should be removed at that
point. Merging on the full key should become the only alignment strategy
for `ResultColumn`-based label sources.
