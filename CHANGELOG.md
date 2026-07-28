# Changelog

One entry per milestone of the hashing and data-consistency program, saying what
moved on the surface another repository can observe. A paragraph to read instead
of a diff to interpret.

M0 and M1 predate this file; both carried their entry in the final commit
message of their branch, and for both the answer was **nothing**.

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
