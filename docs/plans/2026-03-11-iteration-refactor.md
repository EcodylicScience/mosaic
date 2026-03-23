# iteration.py Refactor Plan

**Goal:** Reduce complexity, remove dead code, eliminate redundant index reads, and fix unsound logic in `iteration.py`.

**Baseline:** 50 tests in `test_iteration.py` all passing. All changes must keep tests green.

---

### 1. Extract shared filtering helper

`yield_sequences`, `yield_feature_frames`, and `resolve_tracks_pairs` all repeat the same groups/sequences/allowed_pairs mask-building pattern. Extract into a single `_filter_index(df, groups, sequences, allowed_pairs)` helper.

### 2. Deduplicate `resolve_tracks_pairs` vs `yield_sequences`

Both read the tracks index, apply the same filters, and iterate rows. `resolve_tracks_pairs` just collects `(pairs, safe_map)` instead of loading parquets. Consider having `resolve_tracks_pairs` call the shared filter helper, or merging the pair-collection into the filter step.

### 3. Deduplicate `pair_safe_map` construction

Lines 346-349 and 393-397 build `pair_safe_map` identically. Extract `_build_safe_map(df_scope)` or similar.

### 4. Eliminate triple index read in `yield_sequences_with_overlap`

Currently reads tracks index 3+ times: once for `path_lookup`, once via `yield_sequences`, and once per sequence via `get_adjacent_sequences` -> `get_sequences_sorted_by_time`. Read once, pass the DataFrame through.

### 5. Inline `get_sequences_sorted_by_time` / `get_adjacent_sequences`

Only used by `yield_sequences_with_overlap`. No external callers. Inline or restructure so the overlap function owns adjacency logic directly with a single index read.

### 6. Remove dead `df_feat is None` check

Line 675: `if df_feat is None or df_feat.empty` -- the `None` branch is unreachable. Both code paths either assign a DataFrame or `continue`. Remove the `None` check.

### 7. Remove or flag the fallback path in `yield_inputset_frames`

Lines 648-674 re-read the feature index CSV per `(group, sequence)` inside the inner loop. All callers (`resolve_inputset_scope`, `inputset_from_inputs`) always populate `path_map`. Verify this path is truly dead, then remove it. If kept, it should at least read the index once outside the loop.

### 8. Stop re-wrapping exceptions in `_resolve_input_specs`

Line 452-453: `raise type(exc)(str(exc)) from exc` recreates the same exception with the same message, losing traceback detail. Just `raise` instead.

### 9. Remove `resolve_inputset_scope` if dead

Zero callers outside its own definition. Verify it's unused, then remove along with its `_load_inputset` import.

### 10. Replace cartesian concat fallback with error

Lines 689-693: if no merge columns overlap, it silently concatenates side-by-side, producing wrong results when row counts differ. Raise a clear error instead.
