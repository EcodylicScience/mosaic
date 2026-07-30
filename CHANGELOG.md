# Changelog

One entry per milestone of the hashing and data-consistency program, saying what
moved on the surface another repository can observe. A paragraph to read instead
of a diff to interpret.

M0 and M1 predate this file; both carried their entry in the final commit
message of their branch, and for both the answer was **nothing**.

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
