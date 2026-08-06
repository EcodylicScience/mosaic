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
from collections.abc import Iterable
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

from mosaic.core.helpers import to_safe_name
from mosaic.core.media.facts_columns import (
    MEDIA_INDEX_COLUMNS,
    derivative_cell,
    derivative_column_for_target,
    facts_to_row,
    media_row_uuid,
    row_mapping,
    series_facts_or_none,
)
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.pipeline.media_index import (
    build_media_index_row,
    load_media_index_frame,
    write_media_index_rows,
)
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
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


class TranscodeParams(Params):
    """Parameters for one entry's transcode job."""

    # entry selects WHICH videos are transcoded, and the identities of those
    # videos are hashed in its place, so a sequence rename does not move the run.
    #
    # allow_hardware is excluded provisionally, and the reasoning cuts both ways.
    # A hardware encode and a CPU encode of the same source are not
    # byte-identical, so this is more than a throughput knob. But it is a
    # permission rather than a determinant: once the encode gate is
    # usability-checked (mosaic-media issue
    # encoder-gate-checks-listing-not-usability), allow_hardware=True on a
    # machine with no usable device silently falls back to the CPU encoder. So
    # including it would be wrong in both directions -- claiming two runs differ
    # when both produced identical CPU output, and claiming two match when one
    # used the hardware encoder. The honest discriminator is the encoder actually
    # selected, which nothing records; this exclusion holds until it does.
    entry: Annotated[tuple[str, str], HASH_EXCLUDE]
    target: Target = "analysis"
    allow_hardware: Annotated[bool, HASH_EXCLUDE] = False


def _relative_to(path: Path, anchor: Path) -> str:
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
    # facts_to_row leaves source_path/source_video_uuid/recipe_hash empty; the
    # back-link records the origin and the recipe it was produced under.
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
        source_path=_relative_to(source, raw_root),
        source_video_uuid=source_video_uuid,
        recipe_hash=recipe_hash,
        video_order=video_order,
    )


def _set_back_link(
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
        )
        abs_value = str(row["abs_path"])
        if not df.empty:
            df = df[df["abs_path"].astype(str) != abs_value]
        new_row = pd.DataFrame([row], columns=MEDIA_INDEX_COLUMNS)
        combined = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
        write_media_index_rows(index_path, combined)


@register_op
class TranscodeOp(Op[TranscodeParams]):
    """Transcode one entry's originals for a target and link the derivatives both ways."""

    kind = "transcode"
    domain = "media"
    category = "transcode"
    # 0.2: mosaic-media 0.3.0 changed what the command builder emits. A stream
    # copy that would drop frames now selects a re-encode, the new
    # presentation_timing_requires_decode reason takes a re-encode on both
    # targets, and a source stating no frame rate raises instead of leaving the
    # muxer to invent one. That is exactly the upstream change
    # transcode_recipe_hash's docstring says this segment is bumped by hand for.
    version = "0.2"
    Params = TranscodeParams

    def target(self, params: TranscodeParams) -> str:
        group, sequence = params.entry
        return f"{group}/{sequence}"

    def run(self, ds: "Dataset", params: TranscodeParams, ctx: "JobContext") -> str:
        group, sequence = params.entry
        # A second explicit refusal, for the same reason as the imgstore one
        # below: without it this ran and quietly did harm. On a dataset with no
        # `media_raw`, `get_root("media")` and the originals index are the same
        # file, so the back-link appended a derivative row *into* the originals
        # index and the forward link went to the same place. `route_derivatives`
        # is then False (`Dataset.media_routing_context`), so nothing ever read
        # what was produced: the encode was wasted and the originals index was
        # left holding rows that only `recipe_hash` distinguishes from originals.
        # That is also the one dataset shape `prune-media` must decline, so
        # refusing here keeps its decline a statement about history rather than
        # about damage still being done.
        if ds.resolve_media_root() != "media_raw":
            message = (
                f"{group}/{sequence}: this dataset has no media_raw root, so "
                f"media/index.csv is its originals index; a derivative written "
                f"there would never be read"
            )
            raise TranscodeError(message)
        matched = ds.match_media_rows(group, sequence)
        sources = [
            (int(row.get("video_order", 0) or 0), ds.resolve_path(row["abs_path"]), row)
            for _, row in matched.iterrows()
        ]
        # Every source's identity, read from the index rather than probed: the
        # index is what routing reads, and keeping it current is the re-probe
        # command's job. Resolved before any encoding so a corpus that has not
        # been re-probed fails immediately rather than half way through.
        source_uuids: list[str] = []
        for _, source_path, row in sources:
            cells = row_mapping(row)
            # An explicit refusal, not an implicit one. Until open item O5 a
            # store carried no video_uuid, so the empty-uuid check below was the
            # only thing keeping one out of this path -- and ffmpeg would have
            # been handed a directory. Now that a store mints its uuid that check
            # passes, so the kind has to be named. Whether a store can be
            # transcoded at all is a separate question O5 does not decide.
            if str(cells.get("media_type", "")) == "imgstore":
                message = (
                    f"{group}/{sequence}: {source_path} is an imgstore, which "
                    f"has no elementary stream to transcode"
                )
                raise TranscodeError(message)
            source_uuid = media_row_uuid(cells)
            if not source_uuid:
                message = (
                    f"{group}/{sequence}: {source_path} has no video_uuid in the "
                    f"media index; re-probe the index before transcoding"
                )
                raise TranscodeError(message)
            source_uuids.append(source_uuid)

        encoding = (
            ANALYSIS_ENCODING if params.target == "analysis" else PLAYBACK_ENCODING
        )
        thresholds = media_thresholds()
        recipe_hash = transcode_recipe_hash(params, encoding, CHROME_149, thresholds)
        run_id = transcode_run_id(recipe_hash, source_uuids)
        ctx.set_run_id(run_id)

        n_sources = len(sources)
        media_root = ds.get_root("media")
        transcode_root = media_root / TRANSCODE_KIND_DIRECTORY
        transcode_root.mkdir(parents=True, exist_ok=True)

        ctx.set_total(n_sources * _TICKS_PER_SOURCE)
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
            ) == _relative_to(dest, media_root)
            if dest.exists() and already_linked:
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
                ctx.heartbeat(done=(i + 1) * _TICKS_PER_SOURCE)
                continue

            def _on_progress(progress: TranscodeProgress, index: int = i) -> None:
                if progress.fraction is not None:
                    done = index * _TICKS_PER_SOURCE + int(
                        progress.fraction * _TICKS_PER_SOURCE
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

            ctx.heartbeat(done=(i + 1) * _TICKS_PER_SOURCE)
            if not result.performed or result.output_path is None:
                continue
            if result.output_facts is None or result.output_verdict is None:
                message = (
                    f"transcode of {source} reported performed but returned no "
                    f"output facts/verdict"
                )
                raise TranscodeError(message)

            output_path = result.output_path
            derivative_rel = _relative_to(output_path, media_root)
            # The back-link row first, the forward link last. The forward link is
            # what the reuse gate reads, so writing it last makes it a completion
            # marker: an interrupted registration leaves an unlinked file, which
            # the next run re-links, rather than a linked file with no row, which
            # nothing can repair.
            _set_back_link(
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
            )
            set_forward_link(ds, source, source_uuids[i], derivative_rel, params.target)

        return run_id
