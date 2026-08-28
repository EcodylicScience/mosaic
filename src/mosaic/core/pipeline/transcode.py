"""Multi-video transcode job with bidirectional derivative links.

``TranscodeOp`` is a registered op (``kind="transcode"``, ``domain="media"``) run
through :func:`mosaic.core.pipeline.ops.run_op`. It transcodes the originals of
one ``(group, sequence)`` entry for the analysis or playback target, running the
minimum operation each source needs. When a source is already clean for the
target, nothing is written for it.

Each performed transcode writes a per-target derivative under the ``transcode``
kind directory of the ``media`` root
(``transcode/<video_uuid>.<recipe_hash>.<target>.mp4``, so a source's analysis
and playback derivatives, and two recipes of either, all coexist) and links it
both ways:

* forward -- the original's row in the ``media_raw`` index gets its per-target
  forward-link column (``analysis_derivative_path`` or
  ``playback_derivative_path``) set to the derivative path relative to the
  ``media`` root, leaving the other target's column untouched;
* back -- the ``media`` index (one row per derivative) records the derivative's
  re-probed facts, with ``source_path`` pointing at the original relative to the
  ``media_raw`` root.

:meth:`Dataset.resolve_media` reads the ``analysis_derivative_path`` link to
route analysis reads to the clean derivative.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import pandas as pd
from mosaic_media import (
    CHROME_149,
    MediaFacts,
    PlaybackProfile,
    Thresholds,
    Verdict,
    derive,
    probe_media,
)
from mosaic_media.transcode import (
    ANALYSIS_ENCODING,
    PLAYBACK_ENCODING,
    EncodingParameters,
    Target,
    TranscodeError,
    TranscodeProgress,
    TranscodeResult,
    run_transcode,
)

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.media.facts_columns import (
    MEDIA_INDEX_COLUMNS,
    derivative_cell,
    derivative_column_for_target,
    facts_to_row,
    media_row_uuid,
    row_mapping,
    series_facts_or_none,
)
from mosaic.core.pipeline._utils import ResolvedScope, hash_params
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.media_index import (
    build_media_index_row,
    load_media_index_frame,
    write_media_index_rows,
)
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
    Params,
)
from mosaic.media_probe_config import media_thresholds

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

# Progress denominator per source file: fraction in [0, 1] maps onto this many
# ticks so the aggregate advances smoothly across all N sources.
_TICKS_PER_SOURCE = 1000

# media/ is organized by artifact kind, never by sequence: every child of the
# media root is a kind directory, so a future derived kind (audio, spectrograms)
# becomes a sibling rather than a new top-level root. Mirroring media_raw's
# per-sequence layout here would also let a sequence named "transcode" collide
# with a kind.
TRANSCODE_KIND_DIRECTORY = "transcode"


_TARGET_DESCRIPTION = (
    "Which derivative to write: 'analysis' is the one a tool decodes frame by "
    "frame, 'playback' the one a browser streams."
)

_ALLOW_HARDWARE_DESCRIPTION = (
    "Permit a hardware encoder where the machine offers a usable one. The "
    "encode falls back to the CPU encoder where it does not."
)


class TranscodeParams(Params):
    """What to transcode, and into what.

    **One job covers as many entries as it is given.** It used to cover exactly
    one, which made a transcode step in a graph an exception to the rule every
    other step follows -- one job per step with an entry list -- and would have
    meant a request file mapping one step to many attempts. The entries come
    from the run's scope rather than from this model, and widening what one job
    covers costs nothing on disk. The run identifier addresses no file.

    The trade is real and worth stating: a job over five hundred videos holds one
    queue slot for as long as it takes, where five hundred jobs would spread
    across machines. Narrow the entry list to shard it.

    The settings alone. Which entries a run covers is an argument to the run,
    and an unscoped one is refused by the op's declaration rather than by this
    model. The identities of the videos in scope are hashed in the coverage's
    place. A sequence rename therefore does not move the run.
    """

    # allow_hardware is excluded, and the reasoning cuts both ways. A hardware
    # encode and a CPU encode of the same source are not byte-identical, so this
    # is more than a throughput knob. But it is a permission rather than a
    # determinant: the encode gate takes hardware only when the machine can
    # actually open the encoder, so allow_hardware=True where it cannot falls
    # back to the CPU encoder, and hashing the flag would claim two runs differ
    # when both produced identical CPU output -- re-encoding a corpus for a flag
    # that did nothing.
    #
    # The honest discriminator is the encoder actually selected, and it is now
    # recorded, in the derivative's `encoder` index cell. It still cannot enter
    # this hash, whatever its merits: the hash names the destination path, so it
    # is needed before the transcode that chooses the encoder can run. What that
    # leaves standing is that a permitted run and a plain run write different
    # bytes to one path and each reuses the other's file. Closing that would make
    # the cheapest path in the job -- the reuse check -- pay a device probe; the
    # index cell says which encoder is there instead.
    target: Annotated[Target, Declared(_TARGET_DESCRIPTION)] = "analysis"
    allow_hardware: Annotated[
        bool, HASH_EXCLUDE, Declared(_ALLOW_HARDWARE_DESCRIPTION)
    ] = False


def relative_to_anchor(path: Path, anchor: Path) -> str:
    """POSIX-style path of *path* relative to *anchor* (falls back to relpath)."""
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def transcode_recipe_hash(
    params: TranscodeParams,
    encoding: EncodingParameters,
    profile: PlaybackProfile,
    thresholds: Thresholds,
) -> str:
    """The recipe every derivative of this job is named after.

    Every input that varies the output bytes and is not the source itself: the
    op's declared version, the hashable params (the target, after the excluded
    fields drop out), the encoding parameters, the playback profile, and the
    verdict thresholds. The last two matter because they reach ``derive``, whose
    verdict decides which operation the command builder emits -- the profile
    through the unsupported-container, unsupported-codec and
    client-dependent-decode reasons, and the thresholds through the seek-cost
    ones. A deployment that retunes either is running a different recipe.

    The thresholds are hashed whole even though only two of the five are
    overridable and both reach stream reasons only. Which field reaches which
    target is the upstream library's to decide, not a contract this module may
    assume, so a needless re-encode of an analysis derivative after a retune is
    accepted rather than guessed around.

    The upstream build -- ffmpeg, the encoder -- is deliberately absent: an
    installed tool's version is provenance, not identity, and folding it in
    would invalidate every derivative on every upstream patch.
    ``TranscodeOp.version`` is the declared compatibility segment standing in
    for it, which means **it is bumped by hand whenever an upstream change
    alters what the command builder emits.** That is the whole discipline behind
    the version term.

    It is the name's second segment, not only an index cell: an artifact's name
    is its recipe address, so two settings produce two files that coexist rather
    than one silently overwriting the other.

    ``hash_params`` truncates to 40 bits. That is thin for an identity keyed on
    user data and is not one: this value keys an op version, a target, and three
    sets of code constants, so a corpus holds a handful of distinct recipes
    rather than one per file, and the birthday bound over a hundred of them is
    around 5e-9. A future term that folds in per-file data addresses a different
    question and should not inherit this justification.
    """
    fingerprint = {
        "op_version": TranscodeOp.version,
        "params": params.identity_dump(),
        "encoding": dataclasses.asdict(encoding),
        # Sorted here rather than through asdict: json_ready serializes a set to
        # a list WITHOUT sorting, and sort_keys only orders dict keys, so a
        # frozenset hashed through it yields a different digest in every process
        # under hash randomization -- and this value is a filename.
        "profile": {
            "containers": sorted(profile.containers),
            "codecs": sorted(profile.codecs),
            "client_dependent_codecs": sorted(profile.client_dependent_codecs),
            "baseline_pixel_formats": sorted(profile.baseline_pixel_formats),
        },
        "thresholds": dataclasses.asdict(thresholds),
    }
    return hash_params(fingerprint)


def transcode_run_id(recipe_hash: str, source_uuids: Iterable[str]) -> str:
    """Ledger key: the recipe plus the sorted identities of the sources it ran over.

    Sources enter by ``video_uuid``, never by position, path, or size, because a
    transcode consumes one video rather than the sequence it sits in: a reorder
    or a rename must leave the identity where it is, or every ordering fix
    triggers a full re-encode producing byte-identical output. Sorted before
    hashing, since a set of sources has no inherent order.

    This value addresses nothing. It names no directory and gates no reuse --
    the filename carries the recipe and the derivative row records it -- so it
    reaches only the run log and the queue row.
    """
    fingerprint = {"recipe": recipe_hash, "sources": sorted(source_uuids)}
    return f"transcode-{hash_params(fingerprint)}"


def set_forward_link(
    ds: "Dataset",
    source: Path,
    source_video_uuid: str,
    derivative_rel: str,
    target: Target,
) -> None:
    """Point the original's ``media_raw`` row at its per-target derivative.

    Matches the source row by ``video_uuid`` alone. A row that does not carry
    the identity just measured is not this file's row, and writing the link by
    path would attach it to a row whose identity says it describes a different
    video. Writes only the column for *target*, leaving the other target's link
    untouched. Idempotent.

    Setting one cell rewrites the whole index, so the read and the write are one
    locked block: the op links each source after its own iteration, and two
    sources -- or two entries on a queue -- linking in parallel would otherwise
    each write a frame that never contained the other's cell. The loser's link is
    lost with no error, leaving a derivative file nothing references, which is
    exactly the state a pruner reads as garbage.
    """
    raw_root = ds.get_root(ds.resolve_media_root())
    index_path = raw_root / "index.csv"
    with index_lock(index_path):
        df = load_media_index_frame(index_path)
        matches = df["video_uuid"].fillna("").astype(str) == source_video_uuid
        if not bool(matches.any()):
            message = (
                f"no media_raw row carries video_uuid {source_video_uuid} for "
                f"{source}; re-probe the index before transcoding"
            )
            raise TranscodeError(message)
        df.loc[matches, derivative_column_for_target(target)] = derivative_rel
        write_media_index_rows(index_path, df)


def _derivative_row(
    ds: "Dataset",
    group: str,
    sequence: str,
    source: Path,
    output_path: Path,
    facts: MediaFacts,
    verdict: Verdict,
    video_order: int,
    source_video_uuid: str,
    recipe_hash: str,
    encoder: str,
) -> dict[str, object]:
    """Build the ``media`` index row describing one derivative."""
    raw_root = ds.get_root(ds.resolve_media_root())
    probe: dict[str, object] = {
        "width": facts.width,
        "height": facts.height,
        "fps": facts.fps,
        "codec": facts.codec_name,
        **facts_to_row(facts, verdict),
    }
    # facts_to_row leaves source_path/source_video_uuid/recipe_hash/encoder
    # empty; the back-link records the origin, the recipe it was produced under,
    # and the encoder that produced it. The encoder is not derivable from the
    # rest of the row: codec is measured and reads "av1" whichever encoder ran,
    # and the recipe hash records the recipe rather than the machine.
    #
    # No assignment_source: a derivative takes its (group, sequence) from the
    # source row it was made from and has no derivation of its own to record.
    # Nothing reads it here either -- the per-sequence composition is over
    # media_raw, and media/ holds no sequence semantics (rule P6).
    return build_media_index_row(
        path=output_path,
        stat=output_path.stat(),
        to_store_path=ds.relative_to_root,
        group=group,
        sequence=sequence,
        group_safe=to_safe_name(group) if group else "",
        sequence_safe=to_safe_name(sequence),
        probe=probe,
        source_path=relative_to_anchor(source, raw_root),
        source_video_uuid=source_video_uuid,
        recipe_hash=recipe_hash,
        encoder=encoder,
        video_order=video_order,
    )


def set_back_link(
    ds: "Dataset",
    group: str,
    sequence: str,
    source: Path,
    output_path: Path,
    facts: MediaFacts,
    verdict: Verdict,
    video_order: int,
    source_video_uuid: str,
    recipe_hash: str,
    encoder: str,
) -> None:
    """Record (or replace) the derivative's ``media`` index row (idempotent).

    Locked for the same reason as :func:`set_forward_link`: appending one row
    rewrites the whole derivative index, so two registrations in flight at once
    would each write a frame built from a read that predates the other. The row
    is built inside the lock too -- it stats the output file, and the answer must
    describe the same state the write commits.
    """
    index_path = ds.get_root("media") / "index.csv"
    with index_lock(index_path):
        df = load_media_index_frame(index_path)
        row = _derivative_row(
            ds,
            group,
            sequence,
            source,
            output_path,
            facts,
            verdict,
            video_order,
            source_video_uuid,
            recipe_hash,
            encoder,
        )
        abs_value = str(row["abs_path"])
        if not df.empty:
            df = df[df["abs_path"].astype(str) != abs_value]
        new_row = pd.DataFrame([row], columns=MEDIA_INDEX_COLUMNS)
        combined = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
        write_media_index_rows(index_path, combined)


def _refuse_without_media_raw(
    ds: "Dataset", entries: Sequence[tuple[str, str]]
) -> None:
    """Decline a dataset where a derivative would be written and never read.

    Without this it ran and quietly did harm. On a dataset with no ``media_raw``,
    ``get_root("media")`` and the originals index are the same file, so the
    back-link appended a derivative row *into* the originals index and the
    forward link went to the same place. ``route_derivatives`` is then False, so
    nothing ever read what was produced: the encode was wasted and the originals
    index was left holding rows that only the recipe hash distinguishes from
    originals. That is also the one dataset shape ``prune-media`` must decline,
    so refusing here keeps its decline a statement about history rather than
    about damage still being done.
    """
    if ds.resolve_media_root() == "media_raw":
        return
    named = ", ".join(f"{group}/{sequence}" for group, sequence in entries)
    raise TranscodeError(
        f"{named}: this dataset has no media_raw root, so media/index.csv is "
        f"its originals index; a derivative written there would never be read"
    )


def _sources_for(
    ds: "Dataset", entry: tuple[str, str]
) -> list[tuple[int, Path, "pd.Series"]]:
    """One entry's source videos, in the order the index records."""
    group, sequence = entry
    matched = ds.match_media_rows(group, sequence)
    return [
        (int(row.get("video_order", 0) or 0), ds.resolve_path(row["abs_path"]), row)
        for _, row in matched.iterrows()
    ]


def _source_uuids_for(ds: "Dataset", entry: tuple[str, str]) -> list[str]:
    """Every source's identity for one entry, read from the index rather than probed.

    The index is what routing reads, and keeping it current is the re-probe
    command's job. Resolved before any encoding, so a corpus that has not been
    re-probed fails immediately rather than half way through -- and so a planner
    can ask what a run will be called without opening a video.
    """
    group, sequence = entry
    uuids: list[str] = []
    for _, source_path, row in _sources_for(ds, entry):
        cells = row_mapping(row)
        # An explicit refusal, not an implicit one. Until a store minted a
        # video_uuid the empty-uuid check below was the only thing keeping one
        # out of this path -- and ffmpeg would have been handed a directory. Now
        # that check passes, so the kind has to be named. Whether a store can be
        # transcoded at all is a separate question.
        if str(cells.get("media_type", "")) == "imgstore":
            raise TranscodeError(
                f"{group}/{sequence}: {source_path} is an imgstore, which has "
                f"no elementary stream to transcode"
            )
        source_uuid = media_row_uuid(cells)
        if not source_uuid:
            raise TranscodeError(
                f"{group}/{sequence}: {source_path} has no video_uuid in the "
                f"media index; re-probe the index before transcoding"
            )
        uuids.append(source_uuid)
    return uuids


@register_op
class TranscodeOp(Op[TranscodeParams]):
    """Transcode the scoped entries' originals for a target, linking both ways.

    Reuse is decided per source by the recipe-addressed filename plus the forward
    link, and ``overwrite`` opens that gate. An attempt that passes it re-encodes
    and relinks a derivative already on disk. That is how a file written by a
    build whose output is no longer trusted is replaced.
    """

    kind = "transcode"
    domain = "media"
    category = "transcode"
    # 0.2: mosaic-media 0.3.0 changed what the command builder emits. A stream
    # copy that would drop frames now selects a re-encode, the new
    # presentation_timing_requires_decode reason takes a re-encode on both
    # targets, and a source stating no frame rate raises instead of leaving the
    # muxer to invent one. That is exactly the upstream change
    # transcode_recipe_hash's docstring says this segment is bumped by hand for.
    #
    # Not moved by the encode-gate change in mosaic-media 0.3.3. Hardware is now
    # taken only when the machine can open the encoder, and the one invocation
    # that changes is one that previously failed at encoder startup having
    # written nothing. So no derivative in existence came from a command that
    # change alters, and a bump would rename every derivative on every machine,
    # CPU-only ones included, to separate files from byte-identical output.
    version = "0.2"
    scope_takes = "at-least-one"
    scope_dependent = True
    Params = TranscodeParams

    def target(self, params: TranscodeParams, scope: ResolvedScope) -> str:
        entries = sorted(scope.entries)
        if len(entries) == 1:
            group, sequence = entries[0]
            return f"{group}/{sequence}"
        return f"{len(entries)} entries: {params.target}"

    def plan_identity(
        self,
        ds: "Dataset",
        params: TranscodeParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this transcode will be called, without encoding anything.

        The identity is the recipe plus the identities of every source it will
        read, so both halves are readable at planning time: the recipe is the
        params, and the sources are ``video_uuid`` cells the media index already
        holds. Nothing here is deferred -- a transcode reads originals, which
        exist before any graph runs.

        It **addresses nothing**: the derivative's filename carries the recipe
        and the source uuid, and reuse is gated on that plus the forward link. So
        this value names the attempt for the run log and the queue, and widening
        what one run covers moves no file.
        """
        entries = sorted(scope.entries)
        _refuse_without_media_raw(ds, entries)
        source_uuids = [
            uuid for entry in entries for uuid in _source_uuids_for(ds, entry)
        ]
        thresholds = media_thresholds()
        encoding = (
            ANALYSIS_ENCODING if params.target == "analysis" else PLAYBACK_ENCODING
        )
        recipe_hash = transcode_recipe_hash(params, encoding, CHROME_149, thresholds)
        return OpIdentity(run_id=transcode_run_id(recipe_hash, source_uuids))

    def run(
        self,
        ds: "Dataset",
        params: "TranscodeParams",
        scope: ResolvedScope,
        overwrite: bool,
        ctx: "JobContext",
    ) -> str:
        entries = sorted(scope.entries)
        _refuse_without_media_raw(ds, entries)

        # Named in one place, and before any encoding: a corpus that has not been
        # re-probed fails here rather than half way through.
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)

        encoding = (
            ANALYSIS_ENCODING if params.target == "analysis" else PLAYBACK_ENCODING
        )
        thresholds = media_thresholds()
        recipe_hash = transcode_recipe_hash(params, encoding, CHROME_149, thresholds)

        media_root = ds.get_root("media")
        transcode_root = media_root / TRANSCODE_KIND_DIRECTORY
        transcode_root.mkdir(parents=True, exist_ok=True)

        per_entry = [(entry, _sources_for(ds, entry)) for entry in entries]
        ctx.set_total(sum(len(sources) for _, sources in per_entry) * _TICKS_PER_SOURCE)

        done_ticks = 0
        for (group, sequence), sources in per_entry:
            try:
                done_ticks = self._transcode_entry(
                    ds,
                    ctx,
                    params,
                    group=group,
                    sequence=sequence,
                    sources=sources,
                    encoding=encoding,
                    thresholds=thresholds,
                    recipe_hash=recipe_hash,
                    media_root=media_root,
                    transcode_root=transcode_root,
                    done_ticks=done_ticks,
                    overwrite=overwrite,
                )
            except Cancelled:
                raise
            except Exception as exc:
                # One unreadable video must not end a batch of five hundred. The
                # entry is recorded as lost and the rest carry on, which is the
                # same split every per-entry producer makes: coverage describes
                # the artifact, the exit code describes the attempt.
                ctx.entry_failed(make_entry_key(group, sequence), exc)
                done_ticks += len(sources) * _TICKS_PER_SOURCE
                ctx.heartbeat(done=done_ticks)
        return run_id

    def _transcode_entry(
        self,
        ds: "Dataset",
        ctx: "JobContext",
        params: TranscodeParams,
        *,
        group: str,
        sequence: str,
        sources: "list[tuple[int, Path, pd.Series]]",
        encoding: "EncodingParameters",
        thresholds: "Thresholds",
        recipe_hash: str,
        media_root: Path,
        transcode_root: Path,
        done_ticks: int,
        overwrite: bool,
    ) -> int:
        """Transcode one entry's sources, returning the progress ticks so far."""
        source_uuids = _source_uuids_for(ds, (group, sequence))
        for i, (video_order, source, row) in enumerate(sources):
            ctx.check_cancel()
            ctx.progress.on_phase(
                "transcode", f"{group}/{sequence}[{i}]: {params.target}"
            )

            facts = series_facts_or_none(row)
            if facts is None:
                facts = probe_media(source)
            verdict = derive(facts, CHROME_149, thresholds)

            dest = (
                transcode_root / f"{source_uuids[i]}.{recipe_hash}.{params.target}.mp4"
            )

            already_linked = derivative_cell(
                row_mapping(row), params.target
            ) == relative_to_anchor(dest, media_root)
            if dest.exists() and already_linked and not overwrite:
                # The name carries the whole recipe, so an existing file at this
                # path is this recipe's output. The link is checked too, and it
                # is a completion marker only because registration writes the
                # back-link row first and the forward link last: an interrupted
                # registration leaves an unlinked file, which this re-links,
                # rather than a linked file with no row, which nothing repairs --
                # re-probing an index adds no row and never writes a link cell.
                #
                # `row` is the pre-loop snapshot from match_media_rows, not a
                # fresh read, and that is correct: each source's own link is
                # written after its own iteration, so no earlier iteration can
                # have changed this row's cell.
                ctx.progress.on_phase(
                    "transcode", f"{group}/{sequence}[{i}]: {params.target} reused"
                )
                ctx.heartbeat(done=done_ticks + (i + 1) * _TICKS_PER_SOURCE)
                continue

            def _on_progress(
                progress: TranscodeProgress, index: int = i, base: int = done_ticks
            ) -> None:
                if progress.fraction is not None:
                    done = (
                        base
                        + index * _TICKS_PER_SOURCE
                        + int(progress.fraction * _TICKS_PER_SOURCE)
                    )
                    ctx.heartbeat(done=done)

            result: TranscodeResult = run_transcode(
                source,
                dest,
                params.target,
                facts,
                verdict,
                profile=CHROME_149,
                thresholds=thresholds,
                encoding=encoding,
                allow_hardware=params.allow_hardware,
                on_progress=_on_progress,
                cancel_check=ctx.cancel_token.is_cancelled,
            )

            ctx.heartbeat(done=done_ticks + (i + 1) * _TICKS_PER_SOURCE)
            if not result.performed or result.output_path is None:
                continue
            if result.output_facts is None or result.output_verdict is None:
                message = (
                    f"transcode of {source} reported performed but returned no "
                    f"output facts/verdict"
                )
                raise TranscodeError(message)

            output_path = result.output_path
            derivative_rel = relative_to_anchor(output_path, media_root)
            # The back-link row first, the forward link last. The forward link is
            # what the reuse gate reads, so writing it last makes it a completion
            # marker: an interrupted registration leaves an unlinked file, which
            # the next run re-links, rather than a linked file with no row, which
            # nothing can repair.
            set_back_link(
                ds,
                group,
                sequence,
                source,
                output_path,
                result.output_facts,
                result.output_verdict,
                video_order,
                source_video_uuid=source_uuids[i],
                recipe_hash=recipe_hash,
                encoder=result.encoder_name,
            )
            set_forward_link(ds, source, source_uuids[i], derivative_rel, params.target)

        return done_ticks + len(sources) * _TICKS_PER_SOURCE
