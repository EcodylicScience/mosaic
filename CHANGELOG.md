# Changelog

One entry per milestone of the hashing and data-consistency program, plus any
release that moves an identifier or a name on disk, saying what moved on the
surface another repository can observe. A paragraph to read instead of a diff to
interpret.

M0 and M1 predate this file; both carried their entry in the final commit
message of their branch, and for both the answer was **nothing**.

## 0.12.0 — the install has a default worth having

**No identifier moves, and no run on disk is re-addressed.** What changes is what
`pip install` brings and what the extras are called.

**A bare `pip install -e .` is now a complete analysis install.** PyWavelets,
h5py and PyTables became base dependencies: each gates *reading a file you
already have* — a spectral feature, a SLEAP analysis export, a DeepLabCut HDF5 —
and the three are about 30 MB together, which is not worth a way to get the
install wrong. `psutil` joined them because `temporal-stack`'s memory-headroom
check used to inherit it from `ultralytics` and would otherwise have become a
silent no-op. `seaborn` and `networkx` left; nothing imported either.

**Thirteen extras became nine, and the choice became two.** Install nothing, or
`[all]`. `wavelets`, `sleap` and `hdf5` are base now. `localizer` and `identity`
were both, once h5py moved, essentially "torch", so they merged into
`deep-learning`, which `pose` and `polo` both self-reference — choosing the POLO
fork no longer costs you the identity models. `gpu` became `faiss`, being what it
installs. `imgstore` left the extras entirely for the `test` dependency group:
reading a store has been native for some time, and the package is needed only to
*write* the fixture stores the suite builds.

**`recommended` became `all`, and bundles are now self-referential.**
`all = ["mosaic-behavior[pose,faiss]"]` cannot drift from its parts, where a
copied list could. `recommended`, `identity`, `localizer` and `gpu` still resolve
as aliases and are removed in 0.13 — they are kept only because pip *warns* about
an unknown extra and carries on, so a saved `.[recommended]` would otherwise
produce a working install with no torch in it and nothing to say so.

**A new `movement` extra declares a dependency that was never declared.**
`movement-smooth` and `movement-filter-interpolate` are registered
unconditionally and import `movement` lazily, so the package had no entry in
`pyproject.toml` at all, and the error message pointed at
`pip install 'mosaic[movement]'` — the wrong distribution name and an extra that
has never existed.

**`lightning-action` is capped at `<1.1`.** 1.1.0 requires `nvidia-dali-cuda110`
unconditionally and PyPI serves that as an sdist only, so the extra could not
install anywhere without CUDA, for a reason nothing named.

**One place now decides what a missing dependency says.**
`mosaic.optional_dependency.require` replaces ten hand-written `ImportError`
messages in four shapes, and `tests/test_optional_dependency_messages.py` checks
every extra a message names against what `pyproject.toml` declares. Two guards
that never existed were added — `global-tsne`'s faiss import raised a bare
`ModuleNotFoundError`, and the DeepLabCut `read_hdf` path is now covered by
PyTables being a base dependency rather than by a guard. Three `try/except
ImportError` blocks around the movement, lightning-action and feral imports were
deleted: none of those modules imports its optional dependency at module scope,
so the blocks caught nothing they were written for and would have swallowed a
real error, dropping a feature from the registry.

**The two-OpenCV story was told wrong, and is now told as two.** Installing
`albumentations`, `lightning-action` or `movement` beside mosaic is safe on the
documented conda environment — one conda-forge `py-opencv` registers both
distribution names for its single build, so they resolve with no wheel at all.
That collision is a plain-pip hazard. The separate hazard, `av` and any `cv2`
wheel each vendoring a complete ffmpeg, is unaffected by which OpenCV flavor is
installed and is not fixed by any dependency edit. `yolo-augment` stays opt-in
for the reason that was always sufficient: it changes what a YOLO or POLO
training run does, and nothing records which way a run went.

**Known, and pre-existing: `uv lock` does not resolve.** `ultralytics` publishes
nothing for win32 on Python 3.14, which `requires-python` admits, so `uv.lock`
is stale and `NOTICE` cannot be regenerated with it. No CI job reads the lock.

## Unreleased — no install of mosaic carries Ultralytics

**No identifier moves, and no run on disk is re-addressed.** The golden corpus has
no diff, no params field is added and no op version is bumped. What changes is
where `train-pose` and `train-points` run, and what `pip install` brings.

**`pip install -e ".[all]"` now resolves no AGPL-licensed dependency.** Pose and
point *training* were the last two paths that imported Ultralytics inside mosaic's
own process; both now drive the same runner the tracker and both inference ops
already use, in the environment their model belongs to. Nothing under
`src/mosaic/` imports Ultralytics outside
`src/mosaic/tracking/external/runner/`, no extra declares it, and
`tests/test_ultralytics_separation.py` asserts both with empty sets and a witness
that the detector still finds it where it really is.

**`pose` and `polo` become aliases for `deep-learning`, removed in 0.14.** A
saved `pip install -e ".[pose]"` still produces a working environment and no
longer produces an Ultralytics: build
`src/mosaic/tracking/external/ultralytics-env/` or `.../polo-env/` and name it
with `MOSAIC_ULTRALYTICS_BIN` or `MOSAIC_POLO_BIN`. `all` reaches
`deep-learning` directly, so no live bundle points into a deprecated alias.

**`yolo-augment` is removed rather than aliased.** `albumentations` is read by
whichever process runs the trainer, and after this that process is the external
environment -- so it is declared there, opted into with `uv sync --extra augment`,
and no alias in mosaic's own `pyproject.toml` could have put it where it is now
needed. Both environments also declare `opencv-python-headless` and override the
GUI wheel away, because `ultralytics` wants one build and `albumentations` the
other, and two builds of one import package in a process is what pitfall 8 is
about.

**A cancelled training run still stops at an epoch boundary.** This is the
behavior the move had to preserve rather than a new feature: Ultralytics cannot be
interrupted inside an epoch, so a killed process loses whichever one was running,
where a flag it reads between them leaves `last.pt` and `results.csv` complete.
Mosaic writes a file the runner stats at each boundary, and falls back to the kill
every other tool gets only after a grace long enough for an epoch. On a substrate
that imposes its own termination grace -- a queue pod -- that grace has to exceed
one epoch too, or the runtime's SIGKILL arrives first.

**`n_epochs` in the trained-model index now records what ran, not what was
asked.** A run stopped early by `patience` at forty of three hundred was recorded
as a three-hundred-epoch model. Anything reading that column sees a value that
means something narrower than it did.

**Two things a training run now does that it did not before.** It reports each
epoch into the run-log, so a queued job's progress is visible where previously the
denominator was set and the numerator never advanced; and it refreshes its own
run-root claim from the tool's output, so a run longer than the claim's window is
no longer read as abandoned by the next execution along.

## Unreleased — pose and point inference leave mosaic's process, and land in source pixels

**`infer-pose` and `infer-points` now run their model in an environment mosaic
does not install**, reached as a subprocess, the way the `ultralytics` tracker
already did. Nothing in `src/mosaic/tracking/pose_training/inference.py` imports
Ultralytics any more, and the separation guard's allowance is down to
`train.py`. Model training is what is left, and the entry below is where it goes.

**Both ops move to version `0.2`, because their output moved.** They used to
resize every frame at decode time to fit `imgsz` (640 by default) and return
coordinates in that smaller space — which then reached `tracks/`, where every
spatial column is supposed to be video pixels. Frames are now fed at their native
size, as the tracker's always have been. A table written by `infer-pose.0.1-*` or
`infer-points.0.1-*` holds coordinates scaled by `min(640/width, 640/height)` and
should be re-run; the two versions write into separate directories, so nothing is
overwritten and nothing is silently reinterpreted. No digest moves: the version is
a visible segment, not a hash term, and `tests/data/op_identity_golden.json` is
unchanged.

**POLO has an environment of its own**, `src/mosaic/tracking/external/polo-env/`,
located by `MOSAIC_POLO_CONDA_ENV` / `MOSAIC_POLO_BIN`. It cannot share one with
upstream: both ship under the distribution name `ultralytics`. Both environments
install the same `yolo` console script, so the `$PATH` step of the location
ladder cannot tell them apart — name the fork's environment explicitly. Point
detection refuses an upstream build by name rather than running it, which is what
that step would otherwise do silently.

**A checkpoint from the wrong fork is refused rather than crashing.** POLO pickles
its weights under a class upstream does not define, so an upstream build failed
inside `torch.load` before the task the checkpoint declares could be read, and the
refusal that already routed `locate` weights to `infer-points` was unreachable.
The probe now reports a load failure instead of raising it, and mosaic names the
environment those weights belong to. This fixes `mosaic track ultralytics` too.

**An imgstore recording has to be exported first**, for these two ops as it
already was for the tracker: they are handed a video path, so
`mosaic run --kind export-store` comes first and the error message names the
command. `infer-localizer` is unaffected — it is mosaic's own PyTorch, still runs
in process, still reads a store natively, and stays at version `0.1`.

**Two fixes the boundary forced.** Predictions are converted per batch instead of
being accumulated, so peak memory no longer grows with the length of a recording.
And a long video no longer expires its own claim: the runner's progress lines
refresh it, where an in-process op had no line to hang that on.

**One defect found and fixed in the shared runner.** Its prefetch producer offered
the end-of-stream sentinel once, with a timeout, so a consumer that had not freed
a queue slot within half a second never received it and blocked forever. Whether
that happened was decided by how long one batch took against that timer — never on
a fast GPU, reliably on a busy machine or on CPU. `mosaic track ultralytics` was
exposed to it too.

## Unreleased — a graph step names itself, and says no before doing the work

**No identifier moves.** What changes is the surface another repository sees: a
new argv form, a reserved exit code, two new file layouts under `.mosaic/`, and
one field renamed in a document nothing outside mosaic had yet written.

**A step is addressed rather than spelled out.**
`mosaic run --manifest <path> --graph-request <rid> --step <id> --execution-id
<eid>` runs one step of one submitted pipeline. It is strictly more expressive
than the spelled-out form: `--groups`, `--sequences`, the four frame filters and
`overlap_frames` all reach a feature's identity and none of them has a flag, so a
step that re-plans itself reads what a caller could not pass. **The request is
found from the manifest's parent, and there is deliberately no second path
flag** — a path mosaic-queue does not know about is one it cannot rewrite for a
substrate that mounts the dataset somewhere else, which would break precisely on
the machine a GPU step lands on.

**A refusal is a reserved exit code, not a new terminal status.** A step that
declines before doing any work exits **65** with `error_json` carrying
`{"reason": …}` from a closed set — `coverage_shortfall`, `upstream_empty`,
`schema_family_mismatch`, `variant_mismatch`, `version_moved`,
`parent_unrecorded`, `recipe_missing`, `digest_mismatch`. `terminal_status_for_exit`
maps 65 to `failed`, which is what it is. Nothing joins
`runlog.TERMINAL_STATUSES`: three repositories read that set and mosaic-api's
sweeper reaps it, the same reason `partial` was kept out of it.

**Two new places under `.mosaic/`.** `pipelines/requests/<request-id>.json` is
one submission — its narrowing, its `bind` pins, `allow_partial`,
`max_concurrent_steps`, the `step_id → execution_id` map and the version every
step's producer declared, all assigned before anything runs. `claims/` holds what
has been *tried*: per-entry and per-step failure counts under `entries/` and
`steps/`, and per-request exclusion decisions under `requests/`. That last split
is load-bearing — attempts are global so a resubmit does not reset the only bound
on retrying a sequence that cannot succeed, while the decision to proceed without
an entry is a scientific one and must not leak from one request to another.

**`Request.feature_versions` is now `step_versions`**, and covers op steps too.
The old name described only half of what belongs in it, and nothing outside
mosaic had written a request file yet, so the rename costs nothing now and would
have been a wire break later.

**Two behaviours that were wrong and are now refused by name.** `mosaic run
--kind ... --overwrite` accepted the flag and dropped it — `run_op` has no
overwrite — so it promised a recompute that never happened; it now fails. And a
dataset whose tracks tables span two schema families is refused with the families
named, at submit and at each step's start, instead of raising out of the middle
of an identifier hash.

`mosaic pipeline` gains `submit` and `status`. `submit` records the pipeline
against the dataset and prints the command each step is run by, so a graph can be
driven by a shell loop or a job array with no queue involved; `status` reads the
steps' run-logs and says how far the submission got, without touching the feature
registry.

**The run-log carries three more facts, because it is the only channel that can.**
`entries_written`, `cache_hit` and `tracks_variant` join `entries_failed` on
`RunLogSnapshot`. All three were already known inside a run and reached nothing:
`cache_hit` rode a non-serialized `Result` field, the entry count lived in a local
variable, and which tracks recipe a run read was recorded nowhere at all. A queue
cannot recover any of them — it spawns `mosaic run` with stdout *and* stderr on
`DEVNULL`, deliberately, because an undrained pipe deadlocks a chatty child — so
the run-log is where they have to be.

Three new event kinds rather than three new columns on an existing one, because
their knowers differ: an op reusing a trained model has a `cache_hit` and no
entry count, a partial feature run has a count and no claim to make, and the
variant is known at the start while the count is only final at the end. Nothing
joins `runlog.TERMINAL_STATUSES`, and an older reader is unaffected — an `ev` it
does not recognise falls off the end of the fold having advanced liveness and
changed nothing, which is the same property that made `entry_failed` safe to add.

**`entries_written` counts what the scope holds, not what the attempt computed.**
Cache hits count, so a resumed run and a fresh one report the same number over the
same scope — the point being that one number can drive a coverage reading without
the reader knowing which kind of run produced it. Two consequences worth stating:
it is last-write-wins where `entries_failed` accumulates, and for a tracker it is
attempted-minus-lost rather than the index-row count, because a failed bridge
still writes a row (the tool output is durable and adoptable, so a re-run redoes
only the conversion) and the two numbers therefore differ on exactly the partial
run where it matters.

`tracks_variant` is what a run **read**, never what an op produced. The queue
cannot derive it from the job spec: a step-addressed argv carries no
`--tracks-run-id`, because the step resolves its own variant out of the recipe.

`mosaic run --json` reports `entries_written` on the feature path, and the op path
stops reporting `cache_hit: null` — it now reads both back from its own log, which
is why `entry_failure_status` is now `attempt_facts`.

**Ultralytics tracking runs in an environment you build, and an upgraded worker
that has not built one fails every `mosaic track ultralytics` job.** That is the
operational fact this entry exists for, and mosaic-queue and mosaic-api deploy
this. The refusal is at the probe, before anything is minted, and it is
`UltralyticsNotFoundError` naming the two variables and the build command — so it
is loud rather than silent, and it costs nothing already computed. But it is
every job on the machine, not the first one to need something unusual.

The environment is built with `uv sync --python 3.12` in
`src/mosaic/tracking/external/ultralytics-env/`, and located by
`MOSAIC_ULTRALYTICS_CONDA_ENV` (a conda environment) or `MOSAIC_ULTRALYTICS_BIN`
(that environment's `yolo` script), the same ladder TRex, SLEAP and Lightning
Pose already use. With neither set mosaic looks for `yolo` on `$PATH`. Where the
tool sits is a property of the machine, so none of it reaches a `run_id`.

Ultralytics is AGPL-3.0, and a program that imports it is one work with it.
Mosaic now imports it nowhere on the tracking path: it spawns a program in that
environment and exchanges a JSON request file, a JSON response file and progress
lines on standard output.

**No identifier moves**, and no run on disk is re-addressed. The tracker's
settings, its version and its digest are what they were; a run recorded before
this change reads as current and reuses.

**The `pose` and `polo` extras are unchanged, and still install Ultralytics.**
The separation covers tracking only — `train-pose`, `train-points`, `infer-pose`
and `infer-points` still import Ultralytics inside mosaic's own process — so an
install that asked for one of those extras has an AGPL dependency in it exactly
as before, and a tracking run still ignores it and uses the built environment.

**An imgstore recording has to be exported first.** The tracker is handed a video
path like every other external tool, and a store is a directory of chunk files,
so `mosaic run --kind export-store` comes first — the same requirement TRex,
SLEAP and Lightning Pose already carry, and the error message names the command.
In-process tracking could open a store directly; that is the one capability this
costs.

## Unreleased — a table may declare centimetres, and three TREx readers say which

**mosaic refused data it could analyse perfectly well, over a number nobody
needed.** TRex has scaled its positional output by `cm_per_pixel` since long
before it began *recording* the factor (2025-02-18, TRex 2.0.0), so every older
export is centimetres with no record of by how much. Nothing can divide that back
out. "Tracks are pixels" therefore made a whole population of real data
unconvertible — and the workaround it invited, `cm_per_pixel = 1.0`, would have
written centimetres into a table whose schema promised pixels, which is the exact
silent lie the pixels rule exists to remove.

**New schema `mosaic_cm_v1`.** The same contract as `trex_v2` — same required
columns, same allowance for what TRex genuinely measures, `X`/`Y` the body
centre — in centimetres. It is deliberately **its own schema family**, extending
nothing: these columns mean the same *things* as `mosaic_v1`'s and not the same
numbers, so a scope resolving both is refused by `_refuse_mixed_schemas`, naming
both families and their entries. The required set is now shared through
`STANDARD_COLUMNS`, because the second family cannot inherit it — extending would
put it in the pixel family and make the two mix, which is the one thing they must
not do.

Pixels remain the default and what every modern tracker emits. `scale-to-cm`
converts px → cm as a recorded step; nothing converts cm → px without a factor,
because nothing can.

**New converter `trex_npz_cm`**, and TRex now has three readers with one recipe
each — `output_schema` is declared once per class, so a converter choosing its
schema per file would make that declaration a guess:

| the file | reader | emits |
| --- | --- | --- |
| records `cm_per_pixel` | `trex_npz` | `trex_v2`, pixels |
| does not, factor known | `trex_npz_scaled` | `trex_v2`, pixels |
| does not, factor gone | `trex_npz_cm` | `mosaic_cm_v1`, centimetres |

`trex_npz_cm` takes no parameters and converts nothing. It does still run
`name_the_body_centre`: the head-versus-centre defect is not a unit question and
does not deserve to ride along with one. `MissingTrexCalibrationError` now names
all three routes instead of one that did not exist.

**Three TREx 1.x field names were unclassified, and all of them mattered.** TRex
renamed them on the way to 2.x, and mosaic knew only the later spellings:

- `frame_segments` → `tracklets` and `segment_vxys` → `tracklet_vxys` are
  per-*tracklet*, and both are now in `OFF_AXIS_FIELDS`. Before this they were
  padded onto the frame axis — a value on a row that denies it — and, once
  `unscale_to_pixels` existed, refused the whole table as unclassified. Every
  file of a 720-file archive failed on exactly this.
- `segment_length` → `midline_segment_length` is a length in centimetres
  (`length(seg[1].pos - seg[0].pos) * cm_per_pixel`, verified in TRex's
  `OutputLibrary.cpp` at v1.1.9) and is now in `_LENGTH_FIELDS`, and in
  `scale-to-cm`'s classifier so the two stay in step.

**On disk:** nothing existing moves. `mosaic_cm_v1` is a new name on the
`std_format` column, `trex_npz_cm` writes its own
`tracks/convert-trex_npz_cm.<version>-<digest>/`, and a dataset holding both a
centimetre and a pixel conversion of one entry has two labelled variants —
`select_variant_rows` refuses to choose, and `Dataset.drop_entries(..., run_id=)`
retires one.

## Unreleased — `scale-to-cm` can be chained, and a TREx reader can be told the factor

**`scale-to-cm` could not be put in a pipeline, which is the one thing it is for.**
It returned only the scaled columns joined to the five metadata ones, so `ANGLE`,
`X#head`, the keypoint confidences and everything else non-length were dropped —
and `X`/`Y` came back as `X_cm`/`Y_cm`. A track feature chained onto that output
does not fail: `trajectory-smooth` reads `X`/`Y` literally and every positional
step is guarded on their presence, so it emits its input unchanged and reports
success. Nothing in the tree had ever chained it, and nothing had noticed.

A new `mode` decides. `"derive"` is the old behaviour and stays the default.
`"convert"` returns the **whole** table with every length column converted in
place under its own name, so the result is track-shaped and a whole pipeline can
run downstream of the conversion rather than around it. Emitting both spellings
at once — `X` in pixels beside `X_cm` — is the one thing neither mode does; that
is one table holding two coordinate systems, which is the failure the feature
exists to remove. `suffix` names nothing in `convert` mode and is refused there
rather than hashed, since two spellings would otherwise mint two identifiers for
one byte-identical table.

The invariant does not move. It is about *tracks*: the tables under `tracks/`,
which this feature never writes and which stay pixels in either mode. Its own
output is a derived table, and a derived table has always carried whatever unit
its feature computed.

**Its length classifier disagreed with the converter's, in both directions.**
`_LENGTH_NAMES` was missing `SPEED_SMOOTH`, `SPEED_OLD`, `ACCELERATION`,
`ACCELERATION_SMOOTH`, `BORDER_DISTANCE` and `NEIGHBOR_DISTANCE` — six columns the
TREx converter divides out into pixels and this never multiplied back, so a border
distance in pixels would be compared against a threshold in centimetres and keep
every frame while saying nothing. And it scaled `midline_length`, which TREx never
scales (its conversion is commented out in `OutputLibrary.cpp`), because that name
shares the `midline_` prefix with three genuine lengths. Both are fixed and two
tests now pin the correspondence in each direction; the second fails on the old
code.

**`scale-to-cm` 0.1 → 0.2.** The classifier is a module function, not a `Params`
field, so no digest can see that a default-params run now emits a different column
set. `"scale-to-cm/default"` in the golden corpus moves, and it is the only line
that does. No stored run is re-addressed: nothing in the corpus had run it.

**New `trex_npz_scaled` converter.** `MissingTrexCalibrationError` has always
ended by naming a remedy — "an older file has to be re-exported from its
`.results`, or **converted by a reader that is told the factor**" — and that
reader did not exist. It does now: the same conversion, schema and per-individual
merge as `trex_npz`, with a **required** `cm_per_pixel` param instead of a value
read off the table. Required rather than optional so a missing factor raises once,
before any file is opened, rather than being collected by the conversion loop and
reported as one skipped sequence per entry. A file that *does* record a factor and
disagrees raises `TrexCalibrationConflictError` rather than either value winning
silently.

A second `src_format`, not a flag on the first: a tracks variant identity names
exactly one producer and `converter_op` puts the format in the directory name, so
tables whose factor a human reconstructed stay addressable apart from tables whose
factor the exporter measured. Same reason `calms21_json` is a class rather than a
branch.

**On disk:** nothing existing moves. `trex_npz_scaled` writes to its own
`tracks/convert-trex_npz_scaled.<version>-<digest>/`, and two factors are two
variants. A dataset converting the same entries under both readers holds two
labelled variants, which `select_variant_rows` refuses to choose between —
retire one with `Dataset.drop_entries(..., run_id=...)`.

**For a caller with pre-2025 exports:** the factor is recoverable from the file
even though TRex did not write it down. TRex exports `tracklet_vxys` in px/s while
`SPEED#wcentroid` beside it is cm/s, so their ratio is the applied `cm_per_pixel`.
The converter deliberately does not do that division itself — it is noisy, and two
individuals of one recording would scale their halves of a merged table by subtly
different numbers. Recover it once, round it, state it; `params.json` records what
was used.

## Unreleased — `overlap_frames` reads across a continuous recording, or refuses

**`overlap_frames` did not do what it said, and said nothing about it.** It
concatenated the neighbouring sequences onto a run's input and then trimmed the
output by *row offsets measured on the input* — sound only for a feature returning
one row per input row in input order, which the `Feature` protocol has never
required and about half the library breaks by sorting, filtering or reducing. The
trim returned the right row *count* and the wrong rows. It also counted rows rather
than frames, so three individuals turned a request for three frames of context into
one; and its `core_start == 0` fast path skipped the trim outright for the first
sequence of every group whenever the feature dropped rows, writing the next
segment's rows into that entry's parquet. Two features documented a prohibition
against it; they were the only honest statements about the facility.

**The frame axis it needed did not exist.** Every converter numbers frames per file
from zero, so the three segments carried the same frame numbers: concatenating them
handed `apply` three rows for frame 7, and every feature that sorts or groups by
frame interleaved or merged three recordings. No trim can repair numbers computed
from that, so fixing only the trim would have made the result *look* correct.

**A group now declares that its sequences divide one recording**, with
`continuous_groups:` in the manifest, and mosaic verifies the claim against
`frame_min`/`frame_max` — two columns the tracks index now measures from each
parquet as it is written, the way `n_keypoints` is, blank meaning unknown. Within a
continuous group, neighbours are ordered by recorded frame extent rather than by
sequence name (which sorted `seg1, seg10, seg2` and closed silently over a missing
sequence), context is a window of N *frames*, the output is trimmed on the frame
interval the entry covers, and media resolves as one shared timeline so `frame`
still addresses the right clip. Anything else raises and names the two sequences and
their ranges. Features declare `accepts_overlap`; the two prohibitions are gone, and
both collective features now support it.

**Feature identity now covers the context width.** `compute_run_id` gains an
`_overlap_frames` term and `FEATURE_IDENTITY_SCHEME` moves 5 → 6. Before this,
`overlap_frames=300` and `overlap_frames=0` minted one identifier and one directory,
and the second run was served the first's parquet — the same shape of hole `_tracks`
and `_labels` closed, but for an *argument* that changes the numbers rather than an
input the digest failed to name. The term is omitted when zero, which every existing
run is, so the golden corpus moved zero lines and no directory is re-addressed;
`params.json` records it unconditionally so `mosaic reconcile` can still reproduce
an address. `Pipeline` step-cache prediction reads it too, where a missed term would
have reported "cached" over a directory built with different edge handling.

**On disk:** `tracks/index.csv` gains `frame_min` and `frame_max` (blank on existing
rows; `ds.measure_frame_extents()` fills them). `dataset.yaml` gains an optional
`continuous_groups`. `yield_sequences_with_overlap` is removed — a second
implementation of the same slicing with the same defects, reachable only from its
own tests, whose worked example taught the positional trim.

## Unreleased — tracks are pixels, and every tracks variant re-mints

**TREx tracks a session, not its first clip.** An entry whose media index holds
several videos — a recording a device split into clips — was tracked from
`video_order` 0 and the rest were dropped with a line on stderr. TREx now
receives all of them as one `PathArray`, converts them into a single `.pv` whose
frame index is continuous, and produces one set of identities across what used to
be an artificial boundary at every clip. SLEAP, Lightning Pose and Ultralytics
are unchanged: they still read the first clip and still say so, which is now
declared as `joins_sources` on each tracker's `TrackingRoot` rather than assumed.

Three observable consequences.

**`time` in `tracks/` is mosaic's for a joined entry, not TREx's.** TREx reads
one frame rate from the first clip and never checks the others, and it has no
per-frame timestamps for an `.mp4` at all — its timestamp path for video files is
compiled out. A real session measuring 30, then 29.95, then 31 fps was therefore
timed as if it were 30 throughout: about 3% wrong for most of the recording, and
accumulating. `time` is now reconstructed per clip from the measured rates
(`mosaic.core.media.timeline`), `frame_rate` names the rate in force at each
frame rather than one value, and the synthesised `timestamp` column is dropped
rather than re-minted from numbers mosaic never measured. `frame` is untouched.
When the clips *disagree* on rate, the per-second columns TREx derived against
the single rate — `SPEED`, `VX`, `ANGULAR_V` and their kin — are dropped rather
than rescaled; `speed-angvel` derives them from `X`/`Y` and the corrected `time`
with its method in a run identifier. A uniform-rate session keeps all of them.
The time axis assumes the clips are gapless, which mosaic cannot verify: the
probe records no creation timestamp, so a recorder that stopped between clips is
timed as though it had not.

**Four columns on `_tracking/trex/index.csv`.** `video_sources` (a JSON array of
root-relative paths, in `video_order`), `video_uuids` (comma-joined, never
sorted), `media_composition` (the digest of that arrangement) and
`n_source_videos`. `video_abs_path` still names the first clip, as every
tracker's row does. Older indexes gain the columns empty on the next write.

**A joined entry's markers record a composition digest in `source_uid`.** For a
single video it is still that video's `video_uuid`, so no existing run is
invalidated; for several it is the ordered digest, which is what notices a clip
being added, removed or reordered. A joined entry is also never adopted from a
pre-marker directory — that directory cannot say how many clips it covered.

Nothing else moves: no settings key was added, so every `run_id` and tracks
variant is unchanged, and `consumed_composition` on the tracks row stops
over-claiming, since the entry's whole media composition is now what was actually
tracked.

`mosaic sources add --kind media --layout per_sequence` is how several files
become one sequence; no new ingestion code was needed.

**Index locking moved to a sidecar, and every index directory gains one
zero-byte file.** `index_lock` held its lock on the index inode itself.
`atomic_write` renames a *new* inode over that path, which POSIX permits and
Windows does not — and WSL's `/mnt/*` mounts carry Windows semantics, so on a
dataset under `/mnt/c` every index write failed, reporting a missing temp file
that `atomic_write`'s own cleanup had already removed. The lock now lives on
`<index>.lock` beside the index — `media_raw/index.csv.lock`,
`tracks/index.csv.lock`, one per index — created on the first locked write,
never deleted, and never renamed over. Anything enumerating a dataset root will
see it. Nothing reads it, no identifier moves, and `index.csv` is no longer held
open at any point.

Two things travel with the move. The rule that a locked block may perform at
most one `atomic_write`, as its last act, is no longer load-bearing: it existed
because the first write dropped the block's grip on the inode it had locked, and
a sidecar is never renamed over. Where that shape survives in the code it is now
about throughput — not holding a lock tuned for a CSV rewrite across an ffprobe
pass. And the Windows branch, which kept its lock in `%TEMP%` keyed by a hash of
the index path, is gone: `%TEMP%` is per user, as `$TMPDIR` is per SLURM job and
per container, so a temp-directory lock silently failed to serialize the very
cases the lock exists for. Both platforms now lock the same file.

**Standardized tracks are now in video pixels, on every tracker, and `X`/`Y`
name the body centre.** Neither held before. TREx reports centimetres scaled by
`cm_per_pixel` and puts the *head* in its bare `X`; every other converter wrote
pixels and a keypoint mean. A feature reading `X` across trackers was comparing
a head to a centroid in two unit systems, and nothing on disk recorded either
fact. It went unnoticed because `cm_per_pixel` defaults to 1, where the two
units are the same number — the error appears the first time somebody calibrates.

**Every tracks variant re-mints, and every feature run built on tracks moves with
it.** All five converters bump: `trex_npz` 0.1 → 0.2, `sleap_analysis_h5`,
`deeplabcut` and `ultralytics_tracks` 0.1 → 0.2, `calms21_npy`/`calms21_json`
0.2 → 0.3. A tracks variant names the directory its tables live in, and the
resolved variants enter every feature identifier as the `_tracks` term, so
existing feature outputs become orphans rather than silently-reused caches.
Reconvert with `mosaic convert-tracks`.

**Three schemas, declared by the producer.** `mosaic_v1` is the tracker-neutral
standard; `trex_v2` is that plus what TREx genuinely measures, also in pixels;
`trex_v1` stays registered permanently because a real archived dataset is in it,
and its spatial columns are centimetres. A converter declares which it emits
through `TrackConverter.output_schema`, a tracker through
`TrackingRoot.output_schema`, and the tracks index records it in `std_format` —
a column that existed with no reader and now has one. `build_manifest` refuses a
scope resolving tables from two schema families, which is the mixture
`select_variant_rows` structurally could not see.

**Columns that were never measurements are gone.** `VX`, `VY`, `SPEED`, `ANGLE`
and a duplicated `X#wcentroid` were computed *by the converters* and presented
as tracker output. `mosaic_v1` forbids them. Heading is now the `heading`
feature (the method is a parameter, so the arbitrary-sign principal-component
fit is a choice rather than a silent fallback) and velocity is `speed-angvel`,
which gains `vx`/`vy`. CalMS21 also drops eighteen fabricated placeholder
columns, fifteen of them all-NaN floats that `feature_columns()` was pulling
into every matrix built from it.

**Three feature identifiers move.** `speed-angvel` 0.1 → 0.2 and
`nearest-neighbor` 0.1 → 0.2 move their version segment only, digests unchanged.
`nn-delta-response` 0.2 → 0.3 moves its digest too: its `speed_col` default
changes from `SPEED#wcentroid` to `SPEED`, and it now raises instead of
returning an empty frame — it was silently producing nothing on three of the
four trackers. `nearest-neighbor` gains an `nn_ego_unrotated` column recording
when its `_ego` offsets are world-frame because no heading was available.

**The media index gains a `cm_per_pixel` column.** Text, not numeric, so empty
means *uncalibrated* rather than `0.0`; set with
`Dataset.set_media_calibration(...)` and preserved across every rescan. Physical
units come from the new `scale-to-cm` feature, which refuses an uncalibrated
sequence rather than assuming 1.0. The media-index header moves, so a reader
pinning its column list sees one more.

**New: `mosaic upgrade-tracks`.** Rescales centimetre-era TREx tables whose raw
export has been reclaimed by `sweep-tracking`, reading the factor TREx wrote
into the file and refusing a table that does not record one. Dry-run by default.
Reconverting is still the better route where the `.npz` survives.

**For mosaic-api:** `parse_pose_columns` reads these tables, and
`scripts/conversion.py` globs `*.parquet` without consulting `tracks/index.csv`
— with two variants under `tracks/`, that can select the wrong recipe. Existing
`.pose` files are never regenerated (bound by filename co-location, with
`pose_frame_count` frozen at import), so they keep serving the old conversion.
A stored pipeline chain naming `"SPEED#wcentroid"` or `"X#wcentroid"` in a
free-string column parameter will start raising.

**The guppies archive stays on `trex_v1` and is deliberately not migrated.**

## Unreleased — mosaic-media 0.3.0 moves every media identifier

**Every `video_uuid` and `content_digest` re-mints.** `mosaic-media` advances its
identity scheme to 2, and the scheme is the first element hashed into both, so a
value stored under 0.2.x will not match what a re-probe now mints for the same
unchanged file. Run `mosaic reprobe-media --apply` once per dataset. It rewrites
the identity columns and re-points each derivative's `source_video_uuid` at its
source's new value, resolving the link from `source_path` rather than carrying
the old uuid, so derivative links survive. A derivative whose source no longer
resolves is reported under `derivative_links_unresolved` rather than silently
mislinked.

**Every transcode derivative is renamed.** The release changes what the command
builder emits — a stream copy that would drop frames now re-encodes, and a source
stating no frame rate raises rather than letting the muxer invent one — so
`TranscodeOp.version` moves to 0.2 and `recipe_hash` and the transcode run
identifier move with it. No dataset holds a transcode derivative, so nothing on
disk is orphaned.

**A `media_facts` cell written before this release no longer reconstructs.**
`MediaFacts` drops `timing_measured` and gains `timing_source`,
`coded_reordering_depth`, `discard_flagged_packets`,
`leading_non_keyframe_frames` and `max_timestamp_gap_frame_periods`. The same
re-probe rewrites the cells. The media-index header does not move: none of the
new fields is a flat column.

**An analysis read refuses a codec outside the measured frame-exact set**
(`h264`, `hevc`, `av1`, `vp9`, `vp8`), reporting
`unverified_frame_correspondence`. Such a source needs an analysis transcode
before tracking or frame extraction can read it, and that transcode emits AV1,
which satisfies the gate. A raw read still warns and proceeds.

### 0.11.0

**`global-identity-model` no longer builds a CNN from scratch.** It trained one
from raw crops with no prior, and individual animal identity is exactly the
regime where that loses: a few thousand egocentric crops per animal is far too
little to learn general visual features from nothing, and the features it needs —
markings, shape, texture — are ones an ImageNet-scale backbone already has. It
now puts a linear classification head on a pretrained timm backbone and trains
that instead, defaulting to the MIT-licensed
`timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k` — the same starting point
`global-identity-embedding` was already using to good effect without training at
all.

The backbone is frozen by default, so a fit is fast and cannot damage the
pretrained representation. `freeze_backbone=False` fine-tunes end to end for
datasets large enough to earn it. That choice also decides what a checkpoint
holds: a frozen run stores the head alone and refetches the backbone by name, so
the file is kilobytes rather than hundreds of megabytes.

**The `V200` and `V118_3` architectures are deleted, and with them the ability to
exchange identity checkpoints with T-Rex in either direction.** Mosaic can no
longer export weights for `visual_identification_model_path`, nor read a
checkpoint T-Rex saved. That interop was never safe to rely on: T-Rex picks its
architecture from its own `visual_identification_version` setting rather than
from the checkpoint, and loads with `strict=False` and merely warns, so every way
of getting it wrong produced a randomly-initialised network and a log line
instead of an error. Nothing else about the T-Rex integration changes —
`mosaic track trex` still drives the binary, and the track converters still read
its exports.

**`global-identity-model` keeps its slug and is at `0.3`.** `predict()` still
returns `(N, num_classes)` probabilities and the training history keeps its four
keys, so anything reading either is unaffected. Gone from its params:
`input_normalization`, `export_trex_weights` and `trex_weights_name`. New:
`model_name`, `freeze_backbone`, and a pre-fitted `model` reference, which this
was the only identity feature to lack — pinning one lets an inference run carry
its training set by reference instead of retraining. `image_size` now defaults to
`None`, following whatever the backbone declares, and `channels` to 3.

**Existing `global-identity-model` runs do not carry forward.** The network
changed outright, so a checkpoint written by `0.2` cannot be read by `0.3` at
all; the version bump moves the `run_id` so a stale one is never adopted in
place of a refit. Refit. The exported checkpoint is also deliberately no longer
named `identity_model.pth`: that filename resolves against the `train-identity`
model reference a T-Rex run reads, so under the old name a file this feature
wrote could still be handed to T-Rex and fail the silent way described above. It
is `identity_classifier.pth` now, which that reference cannot match.

Shared backbone plumbing — model-id resolution, preprocessing, device selection —
moved to `model_library/timm_backbone.py`, where the classifier, the embedding
model and the DINOv2 temporal model all read it from one place, replacing three
copies. The classification head's weights are now drawn from a seeded generator:
`nn.Linear` initializes from torch's global RNG, which is unseeded, so two runs
with identical params started from different weights and ended at different
predictions while the `run_id` matched and the cache hit.

**The MABe22 converters are gone, and with them the `mabe22_npy` format name.**
Both the track converter and the behavior label converter are removed, along with
the mouse-triplets notebook and the script that generated it. Neither converter
was ever exercised against real data beyond its own fixtures, and an untested
reader of a benchmark format is a claim mosaic could not stand behind. A dataset
holding `convert-mabe22_npy.*` tracks variants keeps its tables and its index
rows — nothing on disk is rewritten — but the format can no longer be converted,
and `get_track_converter("mabe22_npy")` now raises and names the formats that do
exist. CalMS21 is unaffected and remains the reference dataset the template
notebook is built on.

Two test files used `Mabe22Converter` as their vehicle for the converter-identity
seam rather than as their subject, and now use the SLEAP converter, which carries
`fps` and keeps the underscore in `sleap_analysis_h5` that the run-identifier
parsing test depends on. One MABe22 test was the only cover anywhere for a
converter refusing to guess among several sequences in one file; CalMS21 has the
same guard and had no test for it, so that test moved across rather than being
deleted with the format.

###  The manifest says where the data comes from

**`dataset.yaml` is at version 2, and declares its scan sources.** Roots were
pinned inside the dataset so that every `index.csv` travels with it; the
replacement for an outside root -- a *search directory* whose files are recorded
by absolute `abs_path` -- was an argument typed at the command line and
remembered nowhere. It is now `sources:` in the manifest, one entry per place the
dataset draws from, each carrying its whole recipe. `mosaic scan` with no
arguments rescans exactly that set, across media, tracks and labels.

A source may claim a directory to glob or an explicit list of files. The second
is what an import that selects some of a folder's contents needs, and no glob
expresses it.

**A scan now replaces what it claims and preserves everything else.** It used to
rewrite the index from whatever it had just walked, which meant scanning
directory A and then directory B kept only B, any scan destroyed rows pointing at
files outside the dataset, and one dataset could not hold two source formats at
once. All three are fixed by the same change. `--prune-unsourced` asks for the
old whole-file rebuild. A scan also no longer overwrites an identity a caller
*assigned*: it refreshes the measured cells and keeps the identity ones, so
declaring `media_raw` as a source cannot silently repartition a project the
control plane manages.

**Observable from another repository.** Newly created manifests are version 2.
Existing ones stay version 1 on disk until something saves; reading never writes.
`format`, `index_format`, `dataset_type`, `segment_duration` and `time_column`
are no longer written -- nothing read them -- but they are not deleted either:
unknown top-level keys are now preserved through a load-and-save round trip,
where `save()` previously wrote a fixed key list and annihilated everything else.
A manifest declaring a *newer* version raises rather than being read under the
wrong rules.

`mosaic index-media` and `mosaic index-tracks` are replaced by `mosaic scan`,
with `mosaic sources` to declare what it reads. New alongside them: `mosaic init`
(no command could create a dataset before), and `mosaic notes` / `mosaic tags`
for the dataset's own description. Tags are typed, carrying the same
`type` / `type_constraints` / `value` shape as mosaic-api's sequence and
individual tags. The library entry points `index_media`, `index_tracks_raw` and
`index_labels_raw` keep their signatures.

mosaic-api and mosaic-queue need no change: the whole `Dataset` surface either
uses is methods, and all of them are unchanged.

### A sequence stops resolving another sequence's media

**`resolve_media` matched a request against the media index's *filenames*, so a
sequence with no row of its own answered with another entry's video.** The last
matching tier tested containment against a row's `name` cell: an index holding
one row named `clip_a.mp4` resolved the sequence `clip` to it, registering one
sequence's extracted frames against another's recording. The test was also
regex-enabled -- `clip.a` matched `clipXa.mp4`, and an unbalanced bracket raised
`re.PatternError`.

That tier now compares the request against each row's own `sequence` cell,
case-insensitively, and tolerates the request carrying a media extension the
entry's own name lacks. Only a real one is stripped, from `VIDEO_EXTENSIONS`, so
`trial.1` no longer answers with entry `trial` -- entry names carry dots
routinely, and `cam1.left` names a recording rather than a suffixed `cam1`. A
dataset holding files outside that set no longer bridges a request written as
such a filename; it reports no match instead of the wrong entry. Matching
identity rather than filenames also matches an entry whole: a request landing on
a multi-file recording resolves every file in `video_order` rather than the one
chunk whose name fitted, and naming a chunk's filename resolves nothing.

**Where it used to guess, it now raises `AmbiguousMediaMatchError`.** Two groups
holding a sequence of the same name, asked for without a group, previously
concatenated both groups' media into one timeline; passing the group resolves it
as before. The exception subclasses `MediaProbeError`, so a caller that already
reports resolution faults per entry keeps going; one that re-resolves inside its
own handler must catch it there.

### A blank group stops being the word "nan"

**A dataset with no group converted its tables under one.** `convert_all_tracks`
read `tracks_raw/index.csv` with a bare `pd.read_csv`, so an empty `group` cell
arrived as a float NaN, and each `str()` of it downstream spelled the word:
`nan__seq.parquet` on disk, `group=nan` in `tracks/index.csv`, and a composition
lookup under a key nothing had recorded. `group` is empty on every dataset the
control plane creates, so this was the common path rather than an edge; only the
per-sequence merge normalized, and only TRex reached it.

**What to do about a dataset already holding one.** Re-converting writes the
correct `seq.parquet` beside the stale `nan__seq.parquet` and leaves the old
index row in place, because `("nan", seq)` and `("", seq)` are different
entries. `convert_all_tracks` already reports that: it names the entries this
conversion did not rewrite and gives the remedy, `ds.drop_entries([...],
delete_files=True)`. Nothing is repaired automatically — after the fact a group
spelled `nan` is indistinguishable from one a user genuinely named that, and
deleting tables a call did not write is what the migration rule forbids.

**`tracks_raw/index.csv` now reads like every other index.** Its frame reader
pins its text columns as strings, so a sequence named `001` reads back as
`"001"` rather than as the integer `1` — the failure `IndexCSV` describes, and
the numeric names are the CalMS21 and MABe convention. An index written before a
column existed still reads.

**`source_md5` carried the same word.** With `compute_md5=False` the empty
column reached `tracks/index.csv` as the literal `nan`; it is now empty.

**`index_labels_raw` refuses a format no converter claims.** It wrote any string
verbatim, and `convert_all_labels` then skipped those rows forever without ever
naming them. It now checks the format half of the label registry and lists what
exists. Register a custom converter before indexing under its format — the
order every doc already gives.

**The label converters load on demand.** `LABEL_CONVERTERS` filled only as a
side effect of `import mosaic.behavior.label_library`, which nothing in `core`
does — so a caller who reached a Dataset through `mosaic.core` alone was told
`Available: []`, and `migrate_labels_raw` matched no row and reported zero
migrated. The registry now fills itself when it is empty. A caller who
registered converters of their own keeps exactly those.

**One exception type for a format no converter claims.** `get_track_converter`
raised `KeyError`, whose repr adds quotes when the CLI interpolates it into a
message; both resolvers now raise `ValueError`, which `convert_all_labels`
already did.

**`convert_id_tags_from_csv` wrote nothing for a sequence with no group.** Its
`"category"` and `"multi"` branches group the CSV by `(group, sequence)`, and
`groupby` drops a row whose key is missing — so with `group` blank, which is
what most datasets have, every row was removed before the loop body ran. The
only symptom was a lower `Created N id_tags files` count. Nothing was written,
so re-running the conversion is the whole remedy. The same read now keeps a
sequence named `001` from arriving as the integer `1`; the rest of the CSV is
deliberately still inferred, because `id` keys the `.npz` that a tracks table's
integer `id` column is looked up in. A row naming no sequence is refused rather
than silently dropped.

**`write_labels_row` spelled a blank group `nan`.** The labels index writer
carried the same `str(group) if group is not None else ""` the tracks one did,
and the `is not None` guard does not catch a float NaN either.

## 0.10.0 — one tracker driver, and a model that may be a directory

**`mosaic trex` is gone; `mosaic track <kind>` replaces it.** It was 211
hand-wired lines mirroring params it could drift from, and it had no equivalent
for SLEAP or Lightning Pose. The new command reads a tracker's own parameter
schema, so scope and execution are flags and everything tool-specific is
`--set key=value`, validated against the schema rather than against a second
copy of it. `mosaic run --kind <tracker>` is unchanged and is still what the
executor shells out to, so nothing queued or scripted against it moves.

**The three trackers now run through one loop.** Everything a run does around
the tool — locating it, minting the run, scoping to work items, claiming an
entry, reusing a finished phase, bridging into `tracks/` — moved into
`tracking/common/`, and each tracker supplies only what is genuinely its own.
SLEAP and Lightning Pose picked up T-Rex's content-based reuse gate on the way,
so a video replaced in place now forces a recompute for all three rather than
one. What each leaves on disk is unchanged, pinned as a normalized snapshot.

**One count is renamed.** The tracker run indexes carried `n_tracks` for SLEAP
and `n_individuals` for Lightning Pose, which were not the same quantity; both
are now `n_ids`. An index written before this reads back under the new name
without being rewritten.

**A model reference may name a directory.** The resolver was shaped around a
single `best.pt`, so SLEAP and Lightning Pose — whose models are directories,
and whose top-down form is an ordered *pair* of them — each carried a private
resolver and could not be registered as a training run at all. A reference is
now an artifact described by a per-kind spec: what shape it points at, how many,
and which files inside it identity is allowed to read. Nothing a tool writes
back into its own model directory can move that identity, which is what makes
Lightning Pose's `video_preds/` harmless. Every identifier is preserved exactly;
no run is re-derived.

**Two observable consequences.** A trained-model row can record a directory
artifact, so a future framework-native training run has somewhere to say so.
And the `models/` root was registered under the wrong index shape — naming a
file nothing has ever written — so it was invisible to `make_portable`,
`rewrite_index_paths` and `reconcile`. It is reachable now, which means
`reconcile` will prune trained-model and converted-dataset rows whose artifact
is gone. `Dataset.reindex` still defaults to a dry run.

## 0.9.0 — identity backbone, generalized

**The MegaDescriptor identity feature was a generic backbone loader wearing one
backbone's name.** `MegaDescriptorNetwork` never contained anything specific to
MegaDescriptor: it loaded whatever `model_name` named through
`timm.create_model`, froze it, probed the embedding width, and decided identity
by cosine k-NN against per-identity prototypes. The name is now the mechanism —
`global-identity-embedding`, `EmbeddingIdentityNetwork`,
`identity_embedding_model.joblib` — and the backbone is a parameter, which is
what it always was. `model_name` now takes a bare timm architecture tag as well
as a Hugging Face hub id, so one feature reaches both catalogs.

**Preprocessing follows the backbone instead of being asserted.** Normalization
statistics and input size are read from the loaded model's own `pretrained_cfg`
through `timm.data.resolve_model_data_config`, rather than hardcoded to
ImageNet's triple and 384x384. A backbone that declares nothing falls back to
exactly the old constants, so the no-information path is the previous behaviour
rather than a third one. Interpolation and crop ratio are deliberately *not*
adopted: those describe timm's evaluation transform for a full image being
centre-cropped, and this feature's input is already a tight egocentric crop, so
honouring a 0.9 crop ratio would discard the border a discriminative marking may
sit in. `image_size` stays a parameter but defaults to `None`, meaning *follow
the backbone*; a value pins an override. The resolved configuration is written
into the exported checkpoint (`format_version` 1 → 2), so reloading a fitted
model reproduces the preprocessing it was fitted with even if the upstream
repository changes underneath it.

**The default weights are permissively licensed.** `model_name` defaults to
`timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k` (MIT) — the same Swin
architecture MegaDescriptor fine-tuned, carrying ImageNet-22k weights instead of
wildlife ones. `BVRA/MegaDescriptor-L-384` stays documented and is one parameter
away; it is CC-BY-NC-4.0 and remains the right choice for academic wildlife
re-identification, where it substantially outperforms an ImageNet backbone at
telling individual animals apart. Mosaic distributes no weights: whichever
backbone is named is fetched at run time under its own license. `docs/licensing.md`
gains the backbone table and says plainly that the default is the shippable
option, not the measured one. Unlike `kpms`, there is no acceptance gate — the
restricted component here is a value the user types, and the default is
permissive.

**A clean break, and an identity-neutral one.** `global-identity-megadescriptor`
resolves to nothing. The feature registry has no alias mechanism, and `mosaic
reconcile` categorically cannot carry a slug rename — it enumerates directories
under `features/`, and a run whose slug no longer resolves is reported
`unresolvable_pre_provenance` with nothing moved. But the rename moves no
digest: a feature's own slug is not in its hash payload, and every `Params`
field name survived, so a run that pins the three defaults that did change —
`model_name`, `image_size`, `weights_name` — mints byte-identical `0.1-8aebe700d2`
under the new name. A golden case pins exactly that. The feature therefore stays
at version `0.1`, and `FEATURE_IDENTITY_SCHEME` stays `5`: the shape of the
hashed payload is unchanged, only default values inside it. A dataset holding
runs under the old slug keeps them, unreachable by name; re-run under the new
slug, pinning `model_name` if the analysis used MegaDescriptor.

**Contract surface: nothing moved.** No sibling repository references any
identity slug — mosaic-api, mosaic-app, mosaic-queue and trex reference none of
the three. The `mosaic-behavior>=0.7.0` floor owed since M7 stands.

## 0.8.0 — T-Rex checkpoint interop

**The identity checkpoints mosaic wrote were not loadable by T-Rex, and the ones
T-Rex wrote were not loadable by mosaic.** Both directions failed on the module
tree. mosaic built the V200 as an `nn.Sequential`, so its state_dict keys were
positional (`0.weight`) where T-Rex's are named (`model.conv1.weight`). Reading a
real T-Rex checkpoint raised. Writing one was worse and quieter: T-Rex loads with
`load_state_dict(strict=False)` and only *warns* on a key mismatch, so a
mosaic-exported `.pth` passed to `visual_identification_model_path` produced a
**randomly-initialised network and a log line**. `GlobalIdentityModel` has
documented that export as T-Rex-loadable since it was introduced; it was not.
Both networks now mirror T-Rex's tree — a wrapper whose children are `normalize`
and `model` — so the keys agree by construction rather than by string surgery.

**`V118_3`'s `bn4` was the wrong normalization layer.** T-Rex uses
`nn.LayerNorm`; mosaic used `nn.BatchNorm1d(track_running_stats=False)`. The two
expose an identical state_dict — `weight` and `bias`, no running statistics — so
a checkpoint cannot distinguish them and the wrong one loaded clean while
computing different math. Measured against a real 4-mouse checkpoint, using
T-Rex's own TorchScript as the oracle: `max|Δlogit| = 10.36` and 12% argmax
agreement, plus a hard crash at batch size 1 and predictions that changed
depending on which other crops shared the batch. With `LayerNorm` the same
checkpoint reproduces T-Rex exactly. Anything inferred from that class's
predictions before 0.8 should be recomputed.

**Input normalization is now a stated contract, not an assumption.** T-Rex's
`Normalize.forward` differs across builds: some compute `(x / 255 - mean) / std`,
some pass raw `[0, 255]` through, and some ship the statistics in the checkpoint
as `normalize.*` buffers. mosaic assumed the first, silently. It is now the
`input_normalization` parameter (`"imagenet_scaled"` | `"raw255"`), recorded in
exported metadata and detected on load: buffers in the file win and their values
are used verbatim, metadata is consulted next, and a checkpoint that states
neither is genuinely ambiguous — mosaic keeps its previous behaviour and warns,
naming the override. Existing mosaic exports therefore keep their meaning.

**What moved, for a reader downstream:**

- `global-identity-model` is at `0.2`. Its `run_id` moves and every existing run
  recomputes. Network numerics are not part of the `run_id` payload, so without
  the bump `load_state` would have adopted a checkpoint the previous code wrote.
  One golden line moved; one was added, closing a `scope-a`/`scope-ab` pair the
  corpus claimed to keep but did not.
- A new `input_normalization` param on that feature.
- **The exported `.pth` changed shape**: named `model.*` keys, optional
  `normalize.*` buffers, and `input_normalization` / `architecture_version` /
  `model_type` / `class_labels` / `mosaic_checkpoint_version` metadata. Metadata
  stays primitive because T-Rex reads these files with `weights_only=True`.
- A TREx run that passes such a file to `visual_identification_model_path` **as a
  bare path** gets a new `run_id`: an unregistered path is identified by its
  digest, and re-exporting changes every byte. A run that passes a training
  `run_id` instead is unaffected. Same narrow population as the 0.5.0 note below.
- `input_shape` is `(W, H, C)` in both exporters. T-Rex compares it exactly, so
  the V118_3 exporter's previous `(H, W, C)` was a hard load failure there for
  any non-square crop. Files already written the other way still read correctly:
  the old exporter's `"architecture": "v200-native"` marker identifies them.
- `TRexNativeIdentityNetwork` is now `TRexV118_3IdentityNetwork`, in
  `model_library/trex_v118_3_identity.py`. **Renamed without an alias.** It was
  never a V200 — it is T-Rex's `V118_3`, as its own checkpoints say. Checkpoints
  written by mosaic ≤ 0.7 still load, through a positional-key shim that warns
  and is removed at 0.9.
- `V118_3` gained the three `Dropout2d(0.05)` layers T-Rex has and mosaic
  omitted. Inference is unaffected (dropout is identity in eval mode); training
  now regularizes as T-Rex does.

**Tests, where there were none.** Nothing previously constructed either network,
ran a forward pass, or exercised a checkpoint round trip — the whole suite stayed
green through all of the above. `tests/test_trex_checkpoint_interop.py` pins the
forward output against T-Rex's own classes, the normalization contract, batch
invariance, the key layout, the `(W, H, C)` orientation, and the legacy shim;
`tests/test_trex_checkpoint_real_weights.py` (slow, opt-in via
`MOSAIC_TREX_MODELS_DIR`) pins agreement with real deployed checkpoints using
T-Rex's TorchScript sidecar as the oracle — the only test that covers a build
whose preprocessing the current source no longer represents. torch runs in its
own CI job.

**A hazard these tests uncovered, which is not fixed here.** torch and xgboost
each bundle an OpenMP runtime, and a process holding both segfaults on macOS —
in either import order, inside whichever is first asked to do real work. xgboost
is a core dependency and torch the optional `identity` extra, so **a session that
trains an identity model and then an XGBoost classifier can crash**, with no
Python traceback. Nothing imported torch before these tests, so the suite had
never met it. `tests/conftest.py` now pins `OMP_NUM_THREADS=1`, which holds for
the suite; the usual `KMP_DUPLICATE_LIB_OK=TRUE` does *not* stop it here and is
documented as able to produce silently wrong results. The real fix is an
environment with one OpenMP runtime, which is outside this change.

**Owed.** mosaic still does not write the TorchScript `*_model.pth` sidecar that
T-Rex writes beside every checkpoint and falls back to when a state_dict load
fails. Reading one is supported; writing one needs a TorchScript-clean forward.

## 0.7.0 — M7, reconcile

**A dataset can now be brought forward after an identity change, in place.** Every
milestone up to here makes a change loud rather than silent -- a new scheme marker,
a moved digest -- but leaves the actual migration to a full recompute. `mosaic
reconcile` (and `Dataset.reconcile`) is the pass that recomputes each feature,
tracks, and label identifier from the current code, compares it against the one on
disk, and -- where the recorded provenance confirms the recipe is unchanged --
re-addresses the artifact under its new identifier rather than recomputing it: the
directory moves, the index rows and `params.json` are restamped, the scheme marker
is refreshed, and the index is backed up first. It runs bottom-up, so a moved
tracks or label variant carries its feature consumers to their new identifiers in
the same pass.

**It never guesses.** A run whose recipe cannot be confirmed unchanged -- one that
predates the scheme marker, whose upstream was never pinned, or whose digest moved
under the *current* scheme with nothing to explain it -- is reported and left where
it is, to be recomputed by an ordinary run. A version bump is a new recipe, not a
re-address: the recomputed identifier keeps the run's recorded version, so bumping a
feature or converter version leaves existing runs `ok`. The pass reads the
`.identity_scheme` marker each run was minted under, so it is idempotent and
resumable -- a re-run over an already-migrated dataset reports every run `ok`.

**Dry-run by default.** `mosaic reconcile --manifest <ds>.yaml` reports what would
move as a classified list (`ok`, `scheme_stale`, `identity_shift_relocatable`,
`identity_shift_recompute`, `unresolvable_pre_provenance`); `--apply` performs the
confirmed re-addresses and marker refreshes; `--only <kind>` narrows to one artifact
kind; `--json` emits the report. A full run also folds in the two cheap index-hygiene
passes -- dropping dangling rows (`reindex`) and rewriting non-portable `abs_path`
cells (`make_portable`). The heavier media and tracking passes stay their own
commands: `reprobe-media` (source drift), `prune-media` (stranded transcodes), and
`sweep-tracking` (expired working directories) probe or delete and carry their own
reports. `--force` is reserved for a future destructive path; the forward pass
itself produces nothing to delete.

**Contract surface: nothing moved.** M7 is mosaic-library-only. `Dataset.reconcile`
and the reconcile engine are imported by no sibling repo, and none of the five
`mosaic` symbols `mosaic-api` imports changed. The `mosaic-behavior>=0.6.0` floor
owed since M6 becomes `>=0.7.0`.

## 0.6.0 — M6, labels

**Converted labels now have the provenance and identity tracks already had.** A
converted label set used to live at `labels/<kind>/<group>__<seq>.npz`, flat, with
nothing recording which recipe produced it or what it was made from. It now lives
under `labels/<kind>/<run_id>/<group>__<seq>.npz` behind a single typed
`labels/<kind>/index.csv`, one index per kind, mirroring `tracks/`. The row carries
the converter identity, the op run behind it, and `consumed_source_roots` — which
distinguishes a **scored** set converted from an upload (`labels_raw`), a
**derived** set (the seam exists; no producer does this yet), and an **authored**
set minted in-process (id-tag columns, empty `run_id`, still flat).

**New source root: `labels_raw`.** It joins `media_raw` and `tracks_raw` as a
place that holds only what a person uploaded, with its own composition hash over
the raw-file checksums, so a label upload moves the identity of everything
converted from it exactly the way a track upload does. A dataset created before
this gains the root on load; nothing on disk moves for it.

**The label-converter extension point changed.** A `LabelConverter` now returns
`list[LabelEntry]` — data — and the `Dataset` writes the `.npz` and the typed row,
the same split `tracks` made. Converters carry a version and typed `Params`, are
keyed by `(source_format, label_kind)`, and register through
`core/label_converter.py` rather than `dataset.py`. A custom label converter
written against the old protocol must move to returning entries; the four built-in
converters and the templates already have. `convert_all_labels()` re-converts once
on a dataset converted before this release, because the output path changed.

**Feature identity now covers which label set a run read.** `compute_run_id` gains
a `_labels` term and `FEATURE_IDENTITY_SCHEME` moves 4 → 5. As with `_tracks`, the
term is omitted when there is nothing to say, so a run that reads no labels keeps
its identifier and the golden corpus moved zero lines — the shift is confined to
features that consume labels. `labels_run_id` is the additive selector, keyword-only
on `run_feature` and `Dataset.run_feature`, defaulting to today's behaviour.

**Migration both ways.** `migrate_labels_raw` copies label rows out of
`tracks_raw` into `labels_raw` (files stay where they are); `revert_labels`
flattens the variant directories back to the old flat layout and restores the
untyped index. Rearranging a sequence that carries labels is still refused — the
frame-index remap that would make it safe is not built.

**Contract surface: nothing moved.** M6 is mosaic-library only. The label-converter
registry, `LabelEntry` and the labels index are imported by no sibling repo, and
none of the five `mosaic` symbols `mosaic-api` imports changed. The
`mosaic-behavior>=0.5.0` floor owed since M5 is now `>=0.6.0`, and
`MINIMUM_MOSAIC_VERSION` in the contract test — still `"0.1.0"`, still passing
vacuously — is owed the same fix. Authoring internal scoring in Dolt and
projecting it into `labels_raw` on commit stays future work.

## 0.5.0 — M5, consolidation

**Tracker intermediates moved to `_tracking/`, and inference lost its root.**
A TREx, SLEAP or Lightning Pose run now writes under `_tracking/<tool>/<run_id>/`
instead of `tracks_raw/trex/…`, and model inference writes under
`_tracking/<infer-kind>/<run_id>/` instead of a top-level `predictions/`. So
`tracks_raw` holds only what a person uploaded, which is what lets a raw-track
scan tell source from byproduct.

**A dataset created before this keeps its own layout, deliberately.** Loading
one adds the `_tracking` roots it never declared — otherwise anything keyed on
them raises — but a `trex` root already reading `tracks_raw/trex` is **left
exactly where it is**. Repointing it would orphan every run on disk and strand
the index naming them. The consequence to know: on such a dataset the tracker
keeps writing inside `tracks_raw`, where a `*.npz` scan of that root will index
its per-individual intermediates as user raw tracks and fold them into the
sequence's composition. If that matters to you, move the directory and update
`roots.trex` in `dataset.yaml` by hand; nothing does it for you, and the sweeper
refuses to run at all while a tracker root sits inside a source root.

**New: `mosaic sweep-tracking`.** Deletes tracker working directories that are
finished and past their retention window — 14 days for tracker output, 3 for
inference audit parquets — dry-run by default. It never touches work in
progress: a directory a live execution holds, one this dataset's index does not
yet name, or one carrying no mosaic marker is reported and left alone. Rows go
before files, so an interrupted sweep leaves rows naming absent directories,
which `mosaic reindex` repairs.

**`mosaic reindex --root` covers every root, not just `features/`.** The
`_tracking` indexes were reached by no reindex, prune or portability pass at
all, so a working directory removed by hand left a row naming it forever.

**New: `promote_correction`.** A corrected track set copies into
`tracks_raw/<entry>/` as `corrected.rev<N>`, an append-only series — nothing is
overwritten, and each revision moves that sequence's composition, which is what
invalidates the artifacts built from it. Blocked while derivatives exist; forcing
promotes and deletes nothing.

**Roots must resolve inside the dataset.** `set_root` and `new_dataset_manifest`
refuse one that does not; a dataset already holding one still loads. **`abs_path`
is unchanged and stays able to point outside** — that is how a second dataset
references a video living inside a first without copying it, and it is the
mechanism that replaced shared-video membership.

**Contract surface: two additions, no removals.** `index_media` gains
`media_layout="per_sequence"`, which reads `(group, sequence)` from the entry
directory the control plane already writes — the default is unchanged, so no
existing dataset is re-identified. `Dataset.sweep_tracking` and
`Dataset.reindex` are new. `InferenceIndexRow`, `inference_index` and
`prediction_index_path` are **gone**; nothing outside mosaic imported them, and
every column of that row survives on the tracks row or in
`tracks/<variant>/params.json`.

Identity moved for one narrow population: a TREx run that sets
`visual_identification_model_path`, which is now resolved to the model's
identity rather than carried as a path. Both golden corpora otherwise gained
lines and moved none.

Owed to `mosaic-api`: raise the floor to `mosaic-behavior>=0.5.0`, and fix
`MINIMUM_MOSAIC_VERSION`, still `"0.1.0"` and so passing vacuously against the
`>=0.4.0` already declared.

## 0.4.0 — M3, source identity

**Every transcode derivative was renamed and relocated.** A derivative now lives
at `media/transcode/<video_uuid>.<recipe_hash>.<target>.mp4`, named for the video
it came from and the recipe that produced it. It used to be named for the video's
*position* within its sequence, directly under the media root — so reordering two
videos renamed both and re-encoded both, in place and without a transaction, and
a crash mid-loop left an index row pointing at another video's frames. The suffix
was additionally empty for a single-source sequence, so adding a second video
renamed the first one's derivative. Both are gone: a reorder now touches no
derivative file at all, and the transcode job is idempotent and skippable.

This is the most observable thing in the release. **A dataset transcoded before
it holds files no current name resolves to**, and clearing them is the whole
migration — `scripts/clear_transcode_derivatives.py`, dry-run by default, then a
re-run of the transcode job rebuilds whatever is wanted. Nothing is lost by
waiting; the old files are simply unreachable.

`media/index.csv` gained `recipe_hash`, additively, and reverting the release is
the reverse migration for the rows: reverted code reads the index through a
schema without that column, drops that cell and no other, and every forward link
still resolves. The *files* are the half a revert does not undo, and a revert
also reintroduces the path-keyed derivative matching this release replaced with
uuid-keyed matching — a known defect, which is the one way this differs from M2's
purely additive break.

**New: `mosaic prune-media`.** Deletes transcode derivatives that no forward link
reaches — the ones a retuned recipe leaves stranded. Dry-run by default, with an
age window so it cannot race a running encode, and it refuses to delete a
derivative whose source is no longer indexed, since that may be the last copy of
an archived video. `--relink` repairs instead of deleting where a current recipe
would reproduce the file. It does not replace the sweep script above: pre-rename
derivatives sit outside its blast radius, and the two reaches are disjoint.

**Transcoding is now refused on a dataset with no `media_raw` root.** There,
`media/index.csv` *is* the originals index, so the job was appending derivative
rows into it and writing links that routing then ignored — wasted work, and an
originals index left holding rows only `recipe_hash` distinguished from
originals.

**A new file per source root: `<root>/sequences.csv`.** One row per sequence,
holding the composition hash of what that sequence is made of — the ordered video
uids for `media_raw`, the raw-file checksums for `tracks_raw`. Written by the
four index writers as a projection of the index they just committed, and read
through `read_sequence_index`. Absent reads as empty, so a dataset that has never
been re-indexed behaves exactly as before.

**And one per dataset: `<base_dir>/sequences.csv`.** A sequence's *label*, as
distinct from its token. `Dataset.display_name` / `set_display_name` /
`display_names` are the surface; relabelling touches this file and nothing else,
so it cannot move a directory or invalidate a run. `(group, sequence)` remains
what every filename, directory and index join key is built from.

**Identifiers move for one population.** A `scope_dependent` feature that
declares it reads a source root now carries that root's per-entry composition in
its identifier (`FEATURE_IDENTITY_SCHEME` 3 → 4). No feature in the toolkit is in
that population today — the eight scope-dependent features all read tables — so
in practice nothing on disk re-derives. The golden corpus moved zero lines.

Separately, `train` and `infer` identifiers move **only** where a model was
handed in as a bare filesystem path. A path is a mutable key: swapping the
weights reused the identifier, and moving unchanged weights minted a new one.
Both are now named by the weights' content digest instead. A model referenced by
its training `run_id` is unaffected. `TRACKS_IDENTITY_SCHEME` 1 → 2 for the same
reason, on inference variants only.

**New columns, all additive.** `assignment_source` on the media index records how
each row learned its `(group, sequence)`. `consumed_roots` and
`consumed_composition` on the feature index record what each entry was made from.
`base_digest` on the trained-model index. Every one reads back `""` on a row that
predates it.

**Checksums are on by default.** `Dataset.index_tracks_raw` and
`write_tracks_raw_index` now hash each source file (`compute_md5=True`), because
the `tracks_raw` composition is over those checksums and an off-by-default column
left every sequence's composition unestablishable. A digest is carried forward
when path, size and mtime all match, so a re-index re-hashes only what changed;
`mosaic index-tracks --no-md5` opts out and accepts an unestablished composition.

**Imgstores now carry an identity.** A store's `video_uuid` is its
`__store.uuid`, marked `identity_scheme = "imgstore/1"` so a reader can tell a
declared identity from a measured one. It is a mint, not a derivation: it cannot
be re-derived, so a re-probe cannot audit it, and chunks edited in place go
undetected. Existing indexes pick it up through `index_media` /
`write_media_index`, not through `reprobe-media`. Transcoding a store is now
refused by name rather than by accident.

**Contract surface.** `MEDIA_INDEX_COLUMNS` gained `assignment_source`, and
mosaic-api imports that list in six test modules. `resolve_model` now returns a
`ResolvedModel` record rather than a `(path, run_id)` tuple; mosaic-api does not
import it. Nothing else on the five-import contract surface moved.

## 0.3.0 — M2, versioned tracks

**Tracks tables moved.** A standardized table now lives at
`tracks/<op>.<version>-<digest>/<group>__<seq>.parquet` instead of
`tracks/<group>__<seq>.parquet`. The directory names the recipe that produced
it, and is the same identity already recorded in the row's `run_id` and
described in that directory's `params.json`.

This closes a defect rather than only tidying a layout. All five producers — the
three converter branches, the TREx bridge, the inference bridge — used to target
one flat path behind an `exists() and not overwrite` skip, so a second producer
for a sequence was **discarded with a success return**. Two tracker runs with
different settings each built a complete run-addressed tree under `trex/`, and
only the first reached `tracks/`.

**Read paths are unaffected.** Every reader resolves `abs_path` out of
`tracks/index.csv`, and no glob in the package walks the tracks root. Nothing on
disk is moved or deleted: tables written before this release stay where they are
and their rows keep resolving, so reverting the release resolves both layouts.

**`convert_all_tracks()` re-converts once, on a dataset converted before this
release.** The output path changed, so the overwrite skips no longer find
anything. That call is cell one of the CalMS21 and MABe22 notebooks and of the
getting-started guide. It is not destructive — the old tables and rows stay — but
it is not free either, and it then moves every downstream identifier for
features that read tracks, because of the next item.

**Feature identity now covers which tracks recipe a run read.** A run consuming
`tracks` used to record the bare literal in its digest, so two tracker settings —
which produce different numbers in the tables — shared one identifier, one
directory and one cache entry. `compute_run_id` gains a `_tracks` term, and
`FEATURE_IDENTITY_SCHEME` moves 2 → 3.

The term is **omitted when there is nothing to say**, which is what protects
existing work: rows written before tracks carried an identity have an empty
`run_id` and contribute nothing, and an absent key digests differently from an
empty one. A dataset that is not re-converted keeps every identifier it has.
Verified against the archived guppies analysis, whose identifiers are unchanged
and are now pinned in the golden corpus so the check needs no drive.

**`tracks/index.csv` may hold several rows per entry.** Its dedup key is the
`(run_id, group, sequence)` triple its sibling indexes use. Which row an entry
*resolves* to is decided in one place: an unlabelled row loses to a labelled
one, and two genuinely different recipes for one entry raise rather than guess.

**New selector, additive.** `tracks_run_id` is keyword-only on `run_feature`,
`Dataset.run_feature` and `load_values`; `run_id` on `Dataset.load_tracks`,
`Dataset.drop_entries`, and both sequence iterators; `--tracks-run-id` on
`mosaic run` for feature runs. All default to today's behaviour.

**Contract surface: nothing moved.** The five `mosaic` symbols `mosaic-api`
imports — `Dataset`, `MediaIndexScope`, `new_dataset_manifest`, `from_safe_name`,
`FEATURES`, plus `core.pipeline.media_index` and `write_media_index` — are
untouched, and mosaic-api, mosaic-app and mosaic-queue reference neither the
`tracks` root nor its index. `mosaic-api` still declares
`mosaic-behavior>=0.1.0`; the floor is two releases stale and is doing nothing.
Bump it to `>=0.3.0`.
