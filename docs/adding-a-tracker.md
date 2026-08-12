# Adding a tracker

Wiring a new external tracker or pose tool into mosaic: called via subprocess in
its own environment, its output bridged into standardized
`tracks/<variant>/*.parquet` with full run identity and provenance.

Everything a run does *around* the tool lives in
[`mosaic/tracking/common/`](https://github.com/EcodylicScience/mosaic/tree/main/src/mosaic/tracking/common).
You supply what is genuinely your tool's: how to launch it, what defines a
result, which phases it runs, and how to read what it wrote.

The worked references are the three in the tree. Read `litpose/` first — it is
the smallest complete integration — then `sleap/` for an ungated follow-up step,
then `trex/` for two gated phases.

## Answer these first

They decide how much of the rest applies.

1. **Does it need its own environment?** If yes, point mosaic at it with env
   vars and never add it to mosaic's own dependencies. If no -- it is already an
   optional mosaic dependency and runs in this process -- everything below still
   applies except files 1 and 2: there is no `ToolEnv`, no location ladder and no
   `run_supervised`, so what tests patch is whatever module-level function drives
   the tool, and the invocation suite is replaced by a *preflight* suite checking
   what mosaic declares against what is installed. `ultralytics_track/` is the
   worked reference.
2. **Does it produce identity, or pose only?** A pose-only tool has no tracker
   knobs and emits one `id` per instance. Otherwise identical.
3. **What *shape* is its model?** One weights file, a directory holding a config
   beside a checkpoint, an ordered set of directories, or a path stem whose real
   file has an extension. You declare the shape; you do not write a resolver.
   `train-<tool>` inherits `<tool>`'s shape, so a mosaic training run and an
   external directory resolve the same way.
4. **Can you reuse a converter?** If it exports DeepLabCut-format CSV or HDF5,
   use `src_format="deeplabcut"` and write no converter at all — that is what
   Lightning Pose does. Writing one is
   [Adding a converter](adding-a-converter.md).
5. **Does reading its output need a Python library in the mosaic env?** If so it
   is an *optional* dependency: lazy-import it with a clear message and guard the
   tests with `pytest.importorskip`.

## The seven files

### 1. `tracking/<tool>/version.py`

Two constants, no imports, so the op module can read them without pulling in the
subprocess machinery.

```python
TOOL_KIND: Final = "<tool>"      # the op kind, the root key, and the producer
TOOL_VERSION: Final = "<x.y>"    # the *integration's* compatibility version
```

Seed the version at the tool's release line so `tracks/<tool>.<x.y>-<digest>/`
reads legibly. It is **declared, never detected**: reading it from the installed
binary would re-mint every variant on every upstream patch. Bump it by hand when
output semantics change.

### 2. `tracking/<tool>/run.py`

One `ToolEnv` and the argv. The five-step location ladder — per-call conda env,
per-call bin, `MOSAIC_<TOOL>_CONDA_ENV`, `MOSAIC_<TOOL>_BIN`, `$PATH` — is
`tool_invocation`; you declare only what differs:

```python
_TOOL_ENV: Final = ToolEnv(
    tool="MyTool",
    conda_env_var="MOSAIC_MYTOOL_CONDA_ENV",
    bin_var="MOSAIC_MYTOOL_BIN",
    bin_mode="direct",          # or "sibling" when the bin names a directory entry
    not_found=MyToolNotFoundError,
)
```

Subclass `ToolNotFoundError` with a `default_message` giving the install hint,
and `ToolExitError` with a `tool_name`. Build argv, call `run_supervised` with
`env=subprocess_env()`, and return a small result dataclass. Keep the
`run_supervised` call at module scope here: it is the seam tests patch.

### 3. `tracking/model_refs.py`

One `ModelKindSpec` entry, keyed by your `TOOL_KIND`. Declaring the shape comes
before resolving against it, and it is the whole of what used to be a
per-tracker resolver — a checkpoint-name table, a config-token table, a
find-the-weights walk and a read-the-head-type scan, written once per tracker
until two of them said the same thing twice.

```python
"<tool>": ModelKindSpec(
    shape="directory",              # or "file", or "prefix" for a bare stem
    arity="one",                    # "ordered" when a reference is several, in order
    roles=(
        RoleSpec(role="config", names=("config.yaml",)),
        RoleSpec(role="weights", glob="ckpt/*.ckpt", prefer="best"),
    ),
    payload_prefix="<tool>",        # names the identity keys: <tool>_config, <tool>_weights
    config_names=("config.yaml",),  # where model_type is read from, for provenance
    model_types=("heatmap", ...),   # longest-first, matched against the config text
    label="MyTool model",           # how an error names it to a human
),
```

Four real decisions. **Shape** and **arity** are what a reference points at and
how many. **Roles** are the files identity is allowed to read — and that is the
point, not an optimization: whatever the tool writes back into its own model
directory afterwards is invisible by construction. Lightning Pose writes
`video_preds/` into the directory it was loaded from, so a whole-tree digest
would make the model stop matching its own cached output the first time
inference ran. A role with `in_identity=False` is read for provenance and not
named; a role with `required=False` may be absent, which is how SLEAP runs
without a config.

**`payload_prefix` is not optional for a directory.** `None` means the identity
*is* the bare weights digest — correct for a single `best.pt`, and a collision
for anything else holding the same bytes.

Order matters when `arity="ordered"`: SLEAP top-down is centroid then
centered-instance, and swapping them is a different model. The spec says so; you
do not implement it.

### 4. `tracking/<tool>/dataset_runs.py`

Resolve the model **first**, then mint, then drive:

```python
resolved = resolve_model_set(ds, [str(model_path)], TOOL_KIND)  # content, never a path
settings = my_settings(model_id=resolved.model_id, ...)   # scope-free
minted = mint_tracker_run(ds, kind=TOOL_KIND, version=TOOL_VERSION,
                          settings=settings, observed={...})
scope = ds.resolve_media_scope(groups, sequences, entries)

def run_one(job: EntryJob) -> MyIndexRow | None:
    ...   # your phases, then the bridge

return run_tracker(ds, kind=TOOL_KIND, target="mytool-run", minted=minted,
                   work_items=build_work_items(ds, scope, kind=TOOL_KIND),
                   index=my_index(my_index_path(ds)), run_entry=run_one, ...)
```

The ordering is not negotiable: an unresolvable model must abort before any run
root or tracks variant is written, because a recorded variant naming weights that
could not be found describes a run that never happened.

Inside `run_one`, per gated phase:

```python
reusable = reusable_output(job.ds, job.work_dir, "track",
                           params_hash=minted.params_hash,
                           video_path=job.item.video_path,
                           video_uid=job.item.video_uid)
if reusable is None:
    clear_phase_marker(job.work_dir, "track")
    clear_outputs(job.work_dir, TOOL_KIND, "track")
    phase_claim = claim(job.ctx, job.work_dir, "track", idle_timeout)
    result = run_my_tool(..., on_output=phase_activity(job.ctx, job.work_dir,
                                                       phase_claim, idle_timeout))
    marker = record_phase(job.ds, job.work_dir, "track", ctx=job.ctx, ...)
else:
    marker, output = reusable
```

Always pass `video_uid`. It is what notices a video replaced in place, and it is
what stops a rename from throwing away hours of work.

Then bridge: convert with your converter, and hand the frame to
`publish_tracks_table`. End the module with
`register_reconcilable_index(TOOL_KIND, my_index)`.

### 5. `tracking/ops/<tool>.py`

A `TrackerOpParams` subclass with your tool's knobs, and a registered `Op`:

```python
class MyToolParams(TrackerOpParams):
    model_path: str                                  # required, identity
    threshold: float = 0.5                           # identity
    batch_size: Annotated[int, HASH_EXCLUDE] = 4     # throughput only
```

Tag every knob that changes *how* a run happens rather than *what* it produces
with `HASH_EXCLUDE`; folding one into identity moves an identifier without moving
the output, which costs a recompute for nothing. Keep heavy imports inside
`run()`.

### 6. `core/pipeline/tracking_roots.py`

One row:

```python
TrackingRoot(
    key="mytool",
    retention="tracker",
    outputs=("*.predictions.json",),
    phase_outputs=(TrackingPhase("track", ("*.predictions.json",)),),
    path_columns=("video_abs_path", "predictions_path"),
    output_schema="mosaic_v1",
    joins_sources=False,
)
```

`outputs` is the sweeper's evidence a directory holds real output.
`phase_outputs` is what a re-run of each phase must delete first — not the same
thing, since it includes byproducts that are evidence of nothing. `path_columns`
is every path-bearing column on your row beyond `abs_path`; a column missing here
silently stops being portable across machines.

`joins_sources` is whether your tool can read an entry's several clips as **one
continuous video**. A recorder that chops a session into files leaves a boundary
that is a filesystem artifact rather than an event, so a tool that can span it
should not be shown only the first clip. Declare `True` and `build_work_items`
hands you every path in `video_order`; leave it `False` — the default, and right
for anything that opens one file — and it truncates the entry to its first clip
and says so on stderr. Only TREx declares `True` today, because its `source` is a
`PathArray` and its `VideoSource` sums the clip lengths into one frame index.

Declaring it is not free, and the two things it costs you are not obvious:

- **Reuse.** `item.source_uid` is the first clip's `video_uuid` for a single
  video and the *ordered composition digest* for several, so a clip added,
  removed or reordered invalidates the run. Pass it to `reusable_marker` /
  `record_phase` as `video_uid=`, not `item.video_uid` — that one names clip 0
  and cannot see the rest of the arrangement.
- **Adoption.** A pre-marker directory cannot say how many clips it covered, and
  a joined entry's directory looks exactly like a single-video one. Skip
  `adopt_completed_directory` when `item.n_sources > 1`, or an old directory will
  be adopted as though one clip's output were the whole session's.

And know what your tool does about frame rate. TREx takes it from the first clip
without checking the others, so mosaic reconstructs time per clip through
`mosaic.core.media.timeline` and drops the per-second columns the single rate
spoiled. `build_work_items` refuses clips that disagree on frame *geometry*
before your tool ever starts, but it deliberately accepts a rate disagreement:
that is a real property of real recordings, not an error.

`output_schema` is the schema your bridged tables answer to, and it is the only
place that answer is written — the bridge validates against it and records it on
every row. **State what your tracker measured, and nothing more.**

- `mosaic_v1` is almost certainly right. It wants `frame, time, id, group,
  sequence, X, Y` in **video pixels**, plus keypoints as `poseX*`/`poseY*`, with
  `X`/`Y` the body center — for a pose-only tracker, the mean of that frame's
  keypoints.
- It **forbids** `VX`, `VY`, `SPEED`, `ANGLE` and the rest. Do not compute them
  in your converter, however easy it looks. In the table they are
  indistinguishable from measurements, and the heading in particular is an
  inference with an arbitrary sign. `heading` and `speed-angvel` derive them
  where the method is chosen and recorded.
- If your tool genuinely *reports* one of them, declare a schema that `extends`
  `mosaic_v1` and `allows` it, as `trex_v2` does. That is a deliberate statement
  about your tracker, which is exactly what it should be.

### 7. Exports

Add your tool to `tracking/ops/__init__.py` and re-export `run_<tool>` and
`list_<tool>_runs` from `tracking/__init__.py`.

## Invariants that must hold

- Version **declared, not detected**, and out of the digest.
- Op run id and tracks variant id are **byte-identical**: pass settings to
  `tracker_variant_payload` unwrapped.
- The model is named by **content digest, never a path**, resolved before minting,
  and order-sensitive when there are several. Order-sensitivity is
  `arity="ordered"` in the spec, not something you implement; and identity reads
  only the roles the spec declares, so what a tool writes back into its own model
  directory cannot move it.
- Settings are **scope-free**: knobs only, no videos and no paths, so one value
  names one variant across every sequence.
- Identity versus throughput split with `HASH_EXCLUDE`.
- Placement (`MOSAIC_<TOOL>_*`) never reaches a run identifier.

## What the tests will tell you

`tests/test_tracker_conformance.py` is parametrized over every tracker root, so
your tracker inherits its assertions the moment its `TrackingRoot` row lands. It
will fail until you have registered the op, registered a reconcilable index,
declared phases and path columns, subclassed `TrackerRunRowBase` and
`TrackerOpParams`, and added both golden cases.

Three of its assertions are about the model, and they are the ones most likely
to surprise you if you worked from an older integration. Your runner may declare
no resolver of its own and no filename or head-type table for one to consult;
if your op declares a model parameter, your runner must bind the shared
`resolve_model` or `resolve_model_set` (compared by identity, so a same-named
local will not do); and if your tool's model is a directory, your `MODEL_KINDS`
entry must say so rather than inheriting the single-file default.

Add to the golden corpus a `<kind>/run-id-settings` case calling your real
settings builder with every argument explicit, and a `tracks/<kind>-variant`
case. The first is the one that matters: without it, renaming a key inside your
settings builder moves every run root and tracks variant on disk **with a fully
green suite**. Regenerate with `MOSAIC_UPDATE_GOLDEN=1 pytest
tests/test_op_identity_golden.py` and check the diff is additions only.

Then write `tests/test_<tool>_invocation.py` (the location ladder, patching
`toolenv.shutil.which` and `run_supervised`) and
`tests/test_<tool>_run_markers.py` (a fake over your module-level subprocess
wrappers: fresh run bridges, identical rerun reuses, `overwrite=True` recomputes,
different weights are a different run, a killed run leaves nothing trusted).

`tests/test_tracker_layout.py` is worth extending too — it pins what a run leaves
on disk, which is what a shared-machinery change must not move.

## Gotchas

1. **`PhaseName` is a closed `Literal`** in `core/pipeline/markers.py`. Reuse
   `"track"` for your one gated phase; a genuinely new phase name means editing
   that alias, and the sweeper follows automatically.
2. **Publish atomically anything gated on existence rather than a marker.** A
   tool that opens its output for writing truncates it immediately, so a killed
   step leaves a partial file that a later run trusts as complete. Write to a
   temp path and `os.replace`. This was a real bug in the SLEAP export.
3. **Re-derive counts from disk on reuse.** A reuse run that reports "skipped"
   overwrites a good index row with a zero.
4. **`dict` invariance.** Op params use `dict[str, JsonValue]`; a run-layer
   function receiving them must type the parameter `Mapping[str, object]`.
5. **Exact-set assertions.** `tests/test_cli.py` and `tests/test_tracking_ops.py`
   enumerate the op set, and the golden corpus asserts an exact family set.
