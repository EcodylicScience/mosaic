# Changelog

One entry per milestone of the hashing and data-consistency program, plus any
release that moves an identifier or a name on disk, saying what moved on the
surface another repository can observe. A paragraph to read instead of a diff to
interpret.

M0 and M1 predate this file; both carried their entry in the final commit
message of their branch, and for both the answer was **nothing**.

## Unreleased — the manifest says where the data comes from

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

## Unreleased — a blank group stops being the word "nan"

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
