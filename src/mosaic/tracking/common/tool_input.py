"""The files a tracker hands to an external tool, which are not always the source.

All four integrated trackers run their tool as a subprocess and give it paths to
open, so all four resolve through here. That works for a video file and fails for
an imgstore recording, which is a *directory* of chunk files: T-Rex converts it
to nothing and reports a missing ``.pv``, and SLEAP and Lightning Pose fail
comparably. mosaic's own readers handle a store natively, so the mismatch is only
ever at this boundary -- the moment a path leaves mosaic for a tool that does its
own decoding.

Two boundaries, one per way a tool reads (``TrackingRoot.reads``):

* :func:`resolve_tool_input` hands a one-file tool (TREx, SLEAP, Lightning Pose)
  one path. A plain video passes through untouched, a store resolves to the plain
  video ``export-store`` wrote for it, and several clips resolve to their join,
  or each raises naming the command that would produce it.
* :func:`entry_runner_sources` hands mosaic's Ultralytics runner (tracking,
  ``infer-pose``, ``infer-points``) the entry's files in order, which the runner
  reads on one frame axis: the clips themselves, and a store's chunk files when
  they hold the frames mosaic reads. Neither a join nor an export is built for
  it. The runner decodes each file with mosaic-media's reader in the tool's
  environment, so no Ultralytics code opens a path.

:func:`refuse_undecodable_codec` checks the codec of each file handed over. A tool
that declares a decode probe is tested in its environment by a
:class:`DecodeProbe`, which runs the probe program there and remembers for the
rest of the run a codec that decodes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic_media import SOFTWARE_DECODABLE_CODECS, MediaFacts

from mosaic.core.media.facts_columns import (
    derivative_path_for_target,
    media_row_entry,
    read_link_cell,
    row_mapping,
)
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.media.read_target import verified_read_facts
from mosaic.core.media.store_division import MediaRowKey, StoreWindow
from mosaic.core.media.store_rate import positive_rate, store_rate_mismatch
from mosaic.core.pipeline.joined_export import (
    current_join,
    join_codec,
    join_to_read,
    refuse_unidentified_clips,
)
from mosaic.core.pipeline.run_commands import mosaic_run_command
from mosaic.core.pipeline.store_export import (
    EXPORT_TARGET,
    export_codec,
    readable_chunks,
)
from mosaic.core.pipeline.stream_copy import header_codec
from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.core.pipeline.toolenv import (
    ToolEnv,
    captured_output,
    subprocess_env,
    tool_invocation,
)
from mosaic.core.pipeline.tracking_roots import (
    CONSERVATIVE_DECODER,
    DECODE_PROBE_IMPORT_FAILED,
    TRACKING_ROOTS,
    ToolCodecError,
    ToolDecoder,
    tracking_root,
)
from mosaic.core.pipeline.variant_source import VariantSource, preprocess_command
from mosaic.tracking.common.scope import refuse_unjoinable

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset, ResolvedScopeEntry
    from mosaic.tracking.common.scope import TrackerWorkItem

__all__ = [
    "DecodeProbe",
    "ProbeVerdict",
    "StoreExportMissingError",
    "ToolFile",
    "codecs_never_read",
    "entry_runner_sources",
    "entry_tool_input",
    "handed_codecs",
    "refuse_undecodable_codec",
    "refuse_unreadable_entry",
    "resolve_entry_input",
    "resolve_tool_input",
    "resolve_tool_inputs",
]

_ALLOW_CODECS_VAR: Final = "MOSAIC_ALLOW_TOOL_CODECS"
_DECODE_PROBE_TIMEOUT_SECONDS: Final = 300.0
"""How long one decode probe may run. DALI's start on a GPU takes most of it."""


class StoreExportMissingError(FileNotFoundError):
    """An imgstore has no exported video for a subprocess tool to open."""


class StoreExportStaleError(StoreExportMissingError):
    """An imgstore's exported video is at another frame rate than the store.

    A planner answers it as it answers a missing export, with the store's
    ``export-store`` run. The run does not pass ``--overwrite``. It replaces the
    export because the stale export's file name records an older recipe. A run
    of the current recipe reuses an export at the current recipe's name, and a
    stale export at that name is refused again on the next plan.
    """


def _operator_allowed_codecs() -> frozenset[str]:
    """The codecs that ``MOSAIC_ALLOW_TOOL_CODECS`` names, in lower case.

    ``MOSAIC_ALLOW_TOOL_CODECS`` is a comma-separated list of extra codec names.
    It exists because the refusal is an inference about a decoder mosaic does
    not own: someone who knows their tool environment links ``libdav1d`` is
    right, and should not have to re-encode a corpus to prove it. It widens the
    set and never narrows it, so the variable cannot turn a working run into a
    broken one. It also overrides ``never_reads``, for a reader that has since
    gained the codec.
    """
    extra = os.environ.get(_ALLOW_CODECS_VAR, "")
    return frozenset(part.strip().lower() for part in extra.split(",") if part.strip())


def _allowed_codecs(decoder: ToolDecoder) -> frozenset[str]:
    """Return the codecs that *decoder* may be handed.

    They are the baseline and the decoder's ``also_reads``, less its
    ``never_reads``, and then every codec that ``MOSAIC_ALLOW_TOOL_CODECS`` names.
    """
    declared = (SOFTWARE_DECODABLE_CODECS | decoder.also_reads) - decoder.never_reads
    return declared | _operator_allowed_codecs()


def codecs_never_read(decoder: ToolDecoder) -> frozenset[str]:
    """Return the codecs refused without a probe to a tool reading with *decoder*.

    They are the decoder's ``never_reads``, less the codecs that
    ``MOSAIC_ALLOW_TOOL_CODECS`` names. :func:`refuse_undecodable_codec` refuses a
    file in one of them in every environment. A caller that builds the tool's
    input therefore makes it in another codec.
    """
    return decoder.never_reads - _operator_allowed_codecs()


@dataclass(frozen=True, slots=True)
class ProbeVerdict:
    """Record the result of one decode probe in a tool's environment.

    Attributes:
        decoded: True when the probe exited 0.
        tested: The file that the probe was run on.
        environment: The interpreter's argv, joined by spaces.
        output: The probe's captured output, or the reason that it did not run,
            indented for a message.
        import_failed: True when the probe exited
            ``DECODE_PROBE_IMPORT_FAILED``. The interpreter did not import the
            tool's reader, and the file was not read.
    """

    decoded: bool
    tested: Path
    environment: str
    output: str
    import_failed: bool = False


class DecodeProbe:
    """Test a tool's environment for a codec, and keep the result when it decodes.

    A run creates one for its tool, from the placement that the run resolved, and
    passes it to the check of every entry. A pair of interpreter argv and codec
    that decodes is tested once. A refusal is not kept, and the next file in that
    codec is tested itself, because the SLEAP probe reads the file and one
    unreadable file says nothing about the next. A new run tests again, because
    an environment can be rebuilt between runs.

    Args:
        env: The tool's placement, as :meth:`ToolEnv.placed` returns it.
        timeout: The seconds that one probe may run. The default allows for
            DALI's start on a GPU.
    """

    def __init__(
        self, env: ToolEnv, *, timeout: float = _DECODE_PROBE_TIMEOUT_SECONDS
    ) -> None:
        self.env: ToolEnv = env
        self.timeout: float = timeout
        self._verdicts: dict[tuple[tuple[str, ...], str], ProbeVerdict] = {}

    def verdict(
        self,
        program: str,
        path: Path,
        codec: str,
        *,
        kind: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ProbeVerdict:
        """Return whether the environment decodes *codec*, tested on *path*.

        A call runs *program* on *path* unless an earlier call decoded *codec*
        with the same interpreter, whose result it returns. When the program
        decodes the file, one line on standard error names the tool, the codec
        and the environment, and the result is kept. A refusal and a cancelled
        probe are not kept.

        Args:
            program: The probe that the tool's decoder declares.
            path: The file that the tool is about to open.
            codec: The codec of *path*.
            kind: The tool's op kind, named in the line printed.
            cancel_check: Polled while the probe runs. The probe is stopped
                when it returns True.

        Returns:
            The result, from this call's probe or from an earlier one that
            decoded.

        Raises:
            ToolNotFoundError: The subclass that the tool declares, when its
                environment cannot be located. The tool's run raises the same.
            ProcessCancelled: When *cancel_check* returns True during the probe.
        """
        interpreter = tuple(tool_invocation(self.env, executable="python"))
        remembered = self._verdicts.get((interpreter, codec))
        if remembered is not None:
            return remembered
        verdict = _run_decode_probe(
            interpreter,
            program,
            path,
            timeout=self.timeout,
            cancel_check=cancel_check,
        )
        if verdict.decoded:
            self._verdicts[(interpreter, codec)] = verdict
            print(
                f"[{kind}] A test in the environment of {kind} decoded {codec} "
                f"from {path.name}, and {kind} is handed {codec} files for the "
                f"rest of this run. Environment: {verdict.environment}",
                file=sys.stderr,
            )
        return verdict


def _run_decode_probe(
    interpreter: tuple[str, ...],
    program: str,
    path: Path,
    *,
    timeout: float,
    cancel_check: Callable[[], bool] | None,
) -> ProbeVerdict:
    """Run *program* on *path* with *interpreter*, and return the result.

    A timeout, or an interpreter that does not start, counts as a failure to
    decode, and the result's output states the reason. A cancel raises
    ``ProcessCancelled`` instead of returning a result.
    """
    environment = " ".join(interpreter)
    try:
        stdout, stderr, returncode = run_supervised(
            [*interpreter, "-c", program, str(path)],
            env=subprocess_env(),
            cancel_check=cancel_check,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        reason = f"  The test did not finish within {timeout:g} seconds."
        return ProbeVerdict(
            decoded=False, tested=path, environment=environment, output=reason
        )
    except OSError as error:
        reason = f"  The interpreter did not start: {error}"
        return ProbeVerdict(
            decoded=False, tested=path, environment=environment, output=reason
        )
    return ProbeVerdict(
        decoded=returncode == 0,
        tested=path,
        environment=environment,
        output=captured_output(stdout, stderr),
        import_failed=returncode == DECODE_PROBE_IMPORT_FAILED,
    )


def refuse_undecodable_codec(
    ds: "Dataset",
    path: Path,
    *,
    kind: str,
    group: str,
    sequence: str,
    variant: VariantSource | None = None,
    decode_probe: DecodeProbe | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Raise unless *kind*'s tool decodes *path*, by declaration or by test.

    Each tool declares its decoder on its
    :class:`~mosaic.core.pipeline.tracking_roots.TrackingRoot`, because the answer
    differs per tool. TREx links the libavcodec of its environment, and
    Ultralytics reads through mosaic-media's PyAV. Both declare AV1. SLEAP reads
    through OpenCV, and the Linux OpenCV wheel from PyPI does not decode AV1.
    Lightning Pose reads through DALI's ``fn.readers.video``, whose list of
    codecs omits AV1. It reads AV1 on no GPU, and declares it in ``never_reads``.

    A codec is allowed when it is in the baseline or in the tool's
    ``also_reads``, and not in its ``never_reads``, or when
    ``MOSAIC_ALLOW_TOOL_CODECS`` names it. A codec in ``never_reads`` is refused
    at once, without a probe, because no environment would pass one. Otherwise a
    tool that declares a probe is tested in its environment through
    *decode_probe*. Exit 0 allows the codec. A
    non-zero exit, a timeout or an interpreter that does not start raises with the
    probe's output. A tool without a probe, and a caller without *decode_probe*,
    are refused by the declaration.

    The check reads the file that the tool opens. A clip that is joined into one
    file for the tool may be in any codec, because the tool does not open it.

    A tool without a decoder for a file reads zero frames and exits 0. The refusal
    stops the run before it records that empty result as a success.

    A kind without a registered root gets the conservative declaration, which
    lists only the baseline.

    Args:
        ds: The dataset, read for the recipe of *variant* and the indexed key of
            its entry.
        path: The file that the tool opens.
        kind: The tool's op kind, which selects its declaration.
        group: The entry's group, named in a refusal.
        sequence: The entry's sequence, named in a refusal.
        variant: The media variant that *path* is the file of, or ``None``. A
            refusal of a variant also names making the variant again in H.264
            from the recipe that *ds* recorded for it, over the indexed key of
            the entry.
        decode_probe: The probe of the tool's environment, created by the run
            from the placement that it resolved. ``None`` for a caller without
            one.
        cancel_check: The run's cancel check, polled while a probe runs.

    Raises:
        ToolCodecError: If the codec is refused.
        ProcessCancelled: When *cancel_check* returns True during a probe.
    """
    root = TRACKING_ROOTS.get(kind)
    decoder = root.decoder if root is not None else CONSERVATIVE_DECODER
    codec = header_codec(path)
    if not codec or codec in _allowed_codecs(decoder):
        return
    remedy = f"\n    {decoder.remedy}." if decoder.remedy else ""
    if codec in codecs_never_read(decoder):
        finding = (
            f". That reader decodes {codec} in no environment, so it is not tested."
        )
        setting = f"If this environment does decode {codec}, set"
    elif not decoder.probe or decode_probe is None:
        finding = (
            f", and its declaration does not list {codec}. A tool without a "
            f"decoder for a file reads zero frames and exits 0, and without this "
            f"refusal its run would record an empty result as a success."
        )
        setting = f"To declare that this environment decodes {codec}, set"
    else:
        verdict = decode_probe.verdict(
            decoder.probe, path, codec, kind=kind, cancel_check=cancel_check
        )
        if verdict.decoded:
            return
        measured = (
            f"    Environment: {verdict.environment}\n"
            f"{textwrap.indent(verdict.output, '  ')}"
        )
        if verdict.import_failed:
            finding = (
                f". The interpreter that was to test it did not import that "
                f"reader, because it is not an environment of {kind} or the "
                f"reader's installation in it is broken:\n{measured}"
            )
            remedy = (
                f"\n    Place the run in an environment that {kind} is installed "
                f"in, or repair the reader in the environment this run was "
                f"placed in."
            )
        else:
            finding = (
                f". A test in the environment of {kind} did not decode {codec} in "
                f"{verdict.tested.name}:\n{measured}"
            )
        setting = "To skip the test, set"
    remake = (
        "" if variant is None else _remake_remedy(ds, kind, group, sequence, variant)
    )
    message = (
        f"[{kind}] ({group}, {sequence}) resolves to {path.name}, which is "
        f"{codec}. {kind} decodes with {decoder.stack}{finding}{remedy}{remake}\n"
        f"    {setting} {_ALLOW_CODECS_VAR}={codec}."
    )
    raise ToolCodecError(message)


def _remake_remedy(
    ds: "Dataset", kind: str, group: str, sequence: str, variant: VariantSource
) -> str:
    """Return how to make *variant* again in a codec that *kind* reads.

    The command names the entry by the indexed key of the entry media that the
    variant was made from.
    """
    entry = ds.indexed_entry(variant.entry_row_keys[0], (group, sequence))
    run_id = variant.run_id
    return (
        f"\n    Or make the media variant in a codec that {kind} reads. Run the "
        f'recipe of {run_id} with "codec" set to "h264" in a copy of it, and name '
        f"the variant that it writes in media. The recipe of {run_id} is run "
        f"with:\n{preprocess_command(ds, run_id, [entry])}"
    )


def resolve_tool_inputs(
    ds: "Dataset",
    item: "TrackerWorkItem",
    *,
    kind: str,
    decode_probe: DecodeProbe | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[Path, ...]:
    """Every path *kind*'s external tool should open for *item*, in order.

    Each clip is resolved independently first: a store becomes its registered
    export, a plain video passes through, and a sequence mixing the two is fine
    here because an export *is* a plain video by the time the tool sees it.

    **A multi-clip entry then resolves to exactly one path: the join of those
    clips.** Not the list. Handing a tool several files was wrong in both
    directions -- the three tools that cannot take a list tracked clip 0 and
    dropped the rest of the recording, and TREx, which can, under-counts every
    file it opens and so lost the tail of each clip, leaving a table whose
    ``frame`` column no longer addressed the video. One file has one
    unambiguous frame index and needs neither tool to be trusted with the
    arrangement. See :mod:`mosaic.core.pipeline.joined_export`.

    A single-clip entry is unchanged, and is the overwhelming majority: it
    resolves to its one file with nothing built and nothing required.

    An item that reads a media variant resolves to the variant file. It is a plain
    video that mosaic wrote, one file for the whole entry. An export and a join do
    not apply to it.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        item: The work item whose source paths are being resolved.
        kind: The tracker's kind, so a failure names the tool the user invoked.
        decode_probe: The probe of the tool's environment, for a tool that
            declares one. See :func:`refuse_undecodable_codec`.
        cancel_check: The run's cancel check, polled while a probe runs.

    Raises:
        StoreExportMissingError: If a source is a store with no export
            registered, with a link pointing at a file that is gone, or with an
            export at another rate (:class:`StoreExportStaleError`).
        JoinedExportMissingError: If the entry has several clips and no joined
            export has been built for exactly that clip set.
    """
    handed = (
        item.variant.path
        if item.variant is not None
        else entry_tool_input(
            ds,
            item.group,
            item.sequence,
            item.video_paths,
            item.source_facts,
            windows=item.video_windows,
            kind=kind,
        )
    )
    refuse_undecodable_codec(
        ds,
        handed,
        kind=kind,
        group=item.group,
        sequence=item.sequence,
        variant=item.variant,
        decode_probe=decode_probe,
        cancel_check=cancel_check,
    )
    return (handed,)


def resolve_tool_input(
    ds: "Dataset",
    item: "TrackerWorkItem",
    *,
    kind: str,
    decode_probe: DecodeProbe | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Path:
    """The path *kind*'s external tool should open for *item*'s first clip.

    The single-source view of :func:`resolve_tool_inputs`, for the trackers that
    read one video file. One rule, two views -- a second implementation is how
    the two would come to disagree about what a store resolves to.
    """
    return resolve_tool_inputs(
        ds, item, kind=kind, decode_probe=decode_probe, cancel_check=cancel_check
    )[0]


def resolve_entry_input(
    ds: "Dataset",
    group: str,
    sequence: str,
    source: Path,
    window: StoreWindow | None,
    *,
    kind: str,
) -> Path:
    """*source* itself, or the export registered for it when it is a store.

    The entry named by *group* and *sequence* rather than a
    :class:`~mosaic.tracking.common.scope.TrackerWorkItem`, because the inference
    ops reach this boundary too and build no work items: they walk a media scope
    directly. Those two names are all a work item ever supplied here.

    The store row is found by path and *window* rather than by camera, because
    the path alone identifies the store within a multi-camera sequence, and the
    window identifies a division of it.
    """
    if not is_imgstore(source):
        return source
    why = f"which {kind} cannot open -- it reads a video file, not a store directory"
    return _store_export(ds, group, sequence, source, window, kind=kind, why=why)


def _store_export(
    ds: "Dataset",
    group: str,
    sequence: str,
    store: Path,
    window: StoreWindow | None,
    *,
    kind: str,
    why: str,
) -> Path:
    """Return the export registered for *store*, or raise naming the command.

    *why* says why *kind* needs the export, after "is an imgstore recording,".
    The store's row (:meth:`~mosaic.core.dataset.Dataset.media_rows_of`), the
    row of *window* for a division, gives the export. The command names the
    entry by the row's key, which a run's scope resolves.

    The export's back-link row records the rate it was measured at, and an
    export at another rate than the store's row is refused. ``export-store``
    before 0.3 stamped a copy at its chunks' declared rate.
    """
    key = MediaRowKey.of(str(store), window)
    row = ds.media_rows_of([key]).get(key)
    export = _linked_export(ds, row)
    entry = (group, sequence) if row is None else media_row_entry(row)
    if export is None:
        command = mosaic_run_command("export-store", [entry])
        message = (
            f"[{kind}] ({group}, {sequence}) is an imgstore recording, {why}. "
            f"Export it first:\n    {command}"
        )
        raise StoreExportMissingError(message)
    if not export.is_file():
        message = (
            f"[{kind}] ({group}, {sequence}) links to an exported video "
            f"at {export}, which does not exist; re-run 'mosaic run --kind "
            f"export-store' to rebuild it"
        )
        raise StoreExportMissingError(message)
    _refuse_stale_export(
        ds, row, export, entry, label=f"[{kind}] ({group}, {sequence})"
    )
    return export


def _refuse_stale_export(
    ds: "Dataset",
    row: Mapping[str, object] | None,
    export: Path,
    entry: tuple[str, str],
    *,
    label: str,
) -> None:
    """Raise when *export* is at another rate than the store whose *row* links it.

    The rates are the ``fps`` cells of the store's row and of the export's
    back-link row, compared under
    :func:`~mosaic.core.media.store_rate.store_rate_mismatch`. A missing row or
    cell compares nothing.
    """
    if row is None:
        return
    _, derivatives = ds.media_routing_context()
    back_link = ds.derivative_row(export, derivatives)
    if back_link is None:
        return
    store_rate = positive_rate(read_link_cell(row, "fps"))
    export_rate = positive_rate(read_link_cell(row_mapping(back_link), "fps"))
    if store_rate is None or export_rate is None:
        return
    if store_rate_mismatch([store_rate, export_rate]) is None:
        return
    command = mosaic_run_command("export-store", [entry])
    message = (
        f"{label} links to an exported video at {export_rate:.3f} fps, and "
        f"the store records {store_rate:.3f} fps. A tool times each frame of "
        f"the export by the export's rate. Export the store again:\n"
        f"    {command} --overwrite"
    )
    raise StoreExportStaleError(message)


def entry_tool_input(
    ds: "Dataset",
    group: str,
    sequence: str,
    sources: Sequence[Path],
    facts: Sequence[MediaFacts],
    *,
    windows: Sequence[StoreWindow | None],
    kind: str,
) -> Path:
    """Return the one file that *kind*'s tool opens for an entry's media.

    Each clip is resolved first, so a store without an export is refused before
    the join is looked up. One clip is the file itself, or its export. Several
    clips are their current join, found by
    :func:`~mosaic.core.pipeline.joined_export.current_join` over the clips'
    routed facts, and the clips themselves are not handed over.

    The codec gate is not run here. A clip that is joined may be in any codec,
    because the tool opens the join, and the caller checks the codec of the file
    returned.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        group: The entry's group.
        sequence: The entry's sequence.
        sources: The entry's clips, in ``video_order``.
        facts: The clips' routed facts, parallel to *sources*.
        windows: The chunks of each store in *sources* that the entry reads,
            parallel to them, ``None`` for a file read whole.
        kind: The op that hands the file over, named in a refusal.

    Raises:
        StoreExportMissingError: If a clip is a store with no export registered,
            with a link pointing at a file that is gone, or with an export at
            another rate (:class:`StoreExportStaleError`).
        JoinedExportMissingError: If the entry has several clips and no single
            current join of exactly those clips.
    """
    clips = [
        resolve_entry_input(ds, group, sequence, source, window, kind=kind)
        for source, window in zip(sources, windows, strict=True)
    ]
    if len(clips) < 2:
        return clips[0]
    why = (
        f"{kind} is handed one video file, and a tool that joins clips itself "
        f"loses frames at every boundary, so mosaic joins them first."
    )
    return current_join(
        ds, group, sequence, sources, facts, windows=windows, asker=kind, why=why
    )


@dataclass(frozen=True, slots=True)
class ToolFile:
    """One file that mosaic's runner reads for an entry, with its gated facts.

    Attributes:
        path: The file.
        facts: Its facts, gated for the read. The runner places the file on the
            entry's frame axis by their ``frame_count``.
    """

    path: Path
    facts: MediaFacts


def entry_runner_sources(
    ds: "Dataset",
    group: str,
    sequence: str,
    sources: Sequence[Path],
    facts: Sequence[MediaFacts],
    *,
    windows: Sequence[StoreWindow | None],
    kind: str,
    variant: VariantSource | None = None,
) -> tuple[ToolFile, ...]:
    """Return the files that mosaic's runner reads for an entry, in order.

    The runner reads the files one after another on one frame axis, so nothing
    is joined or exported for it, and the frames it reads are the frames of the
    join or the export it would otherwise be handed:

    * a plain clip is itself, with the facts the media index stored for it;
    * a store whose chunk files hold the frames mosaic reads
      (:func:`~mosaic.core.pipeline.store_export.readable_chunks`) is its chunk
      files, the chunks of its window for a division, each probed and gated as
      mosaic's own store reader gates a chunk.
      Each chunk's measured frame count must equal the count the store's index
      gives it, or the chunks do not hold the store's frame axis;
    * any other store is its registered export, as a one-file tool reads it.

    A media variant's file arrives as the entry's one plain clip, with the
    variant's stored facts.

    Each file is checked against the codec that *kind* declares it reads. A
    store's chunks share one codec, so its first chunk is checked.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        group: The entry's group.
        sequence: The entry's sequence.
        sources: The entry's clips, in ``video_order``, or the variant's file.
        facts: The stored facts of *sources*, parallel to them, or empty to
            probe every file.
        windows: The chunks of each store in *sources* that the entry reads,
            parallel to them, ``None`` for a file read whole.
        kind: The op whose runner reads the files, named in a refusal.
        variant: The media variant that *sources* is the file of, or ``None``. A
            codec refusal of a variant names how to remake it.

    Raises:
        StoreExportMissingError: If a store's chunks cannot be read directly and
            it lacks an export, or its export is at another rate.
        ToolCodecError: If a file is in a codec that *kind* does not read.
        MediaProbeError: If a file's verdict says it needs a transcode before it
            can be read for analysis.
    """
    stored: list[MediaFacts | None] = list(facts) if facts else [None] * len(sources)
    files: list[ToolFile] = []
    for source, source_facts, window in zip(sources, stored, windows, strict=True):
        handed = (
            _store_files(ds, group, sequence, source, window, kind=kind)
            if is_imgstore(source)
            else [ToolFile(source, _analysis_facts(source, source_facts))]
        )
        refuse_undecodable_codec(
            ds,
            handed[0].path,
            kind=kind,
            group=group,
            sequence=sequence,
            variant=variant,
        )
        files.extend(handed)
    return tuple(files)


def _analysis_facts(path: Path, stored: MediaFacts | None) -> MediaFacts:
    """Return *path*'s facts gated for an analysis read, probing only when absent."""
    return verified_read_facts(path, stored, "analysis")[0]


def _unreadable_chunks_why(kind: str) -> str:
    """Why *kind*'s runner reads a store through its export, for a refusal."""
    return (
        f"whose chunk files are not the frames mosaic reads from it (a raw, "
        f"image-directory, Bayer or YUV store), so {kind} reads its export"
    )


def _store_files(
    ds: "Dataset",
    group: str,
    sequence: str,
    store: Path,
    window: StoreWindow | None,
    *,
    kind: str,
) -> list[ToolFile]:
    """Return the files the runner reads for *store*: its chunks, or its export.

    The chunks are those of *window* for a division of the store.
    """
    spans = readable_chunks(store, window)
    if not spans:
        why = _unreadable_chunks_why(kind)
        export = _store_export(ds, group, sequence, store, window, kind=kind, why=why)
        return [ToolFile(export, _analysis_facts(export, None))]

    # Gated as mosaic's own store reader gates a chunk (`NativeStore`), so the
    # runner reads a store exactly as far as mosaic does.
    measured = verified_read_facts([path for path, _ in spans], None, "raw")
    for (path, count), chunk_facts in zip(spans, measured, strict=True):
        if chunk_facts.frame_count != count:
            why = (
                f"whose chunk {path.name} holds {chunk_facts.frame_count} frames by "
                f"its own measure and {count} by the store's index, so {kind} reads "
                f"its export, whose frames the export verified"
            )
            export = _store_export(
                ds, group, sequence, store, window, kind=kind, why=why
            )
            return [ToolFile(export, _analysis_facts(export, None))]
    return [
        ToolFile(path, chunk_facts)
        for (path, _), chunk_facts in zip(spans, measured, strict=True)
    ]


def handed_codecs(
    ds: "Dataset", entry: "ResolvedScopeEntry", *, kind: str
) -> frozenset[str]:
    """Return the codecs of the files that *kind*'s tool is handed for *entry*.

    The files are those of :func:`entry_tool_input` and
    :func:`entry_runner_sources`, without the codec gate, and the answer comes
    without a probe:

    * an in-process reader is not handed a file;
    * a one-file tool is handed one clip as itself, and several clips as their
      join, in :func:`~mosaic.core.pipeline.joined_export.join_codec`;
    * mosaic's runner is handed every clip;
    * mosaic's runner reads a store whose chunks contain the frames
      (:func:`~mosaic.core.pipeline.store_export.readable_chunks`) chunk by
      chunk, in its first chunk's codec, even when the store has an export;
    * any other store is read through its export: the linked file while it
      exists, by its header, and otherwise the export that ``export-store`` would
      write (:func:`~mosaic.core.pipeline.store_export.export_codec`).

    A clip's codec is the one that its routed facts record. A store's facts name the
    imgstore format. For a store, the answer therefore comes from its chunks or its
    export. A file whose header does not read is left out, as the gate passes it.

    Args:
        ds: The dataset, read for the media index, the stores' metadata and the
            ``media`` root.
        entry: One camera of an entry, resolved.
        kind: The op whose tool reads the entry.

    Returns:
        The codecs, empty for an in-process reader.

    Raises:
        KeyError: If a tracking root is not registered for *kind*.
        MediaProbeError: If a store's metadata cannot be read.
        TranscodeError: If ``ffprobe`` cannot read the header of a store's first
            chunk, for a store whose export is still to be written.
    """
    reads = tracking_root(kind).reads
    if reads == "in-process":
        return frozenset()
    resolved = entry.resolved
    paths, facts = resolved.paths, resolved.facts
    if reads == "one-file" and len(paths) > 1:
        return frozenset({join_codec(facts)} - {""})
    codecs: set[str] = set()
    for path, clip, window, key in zip(
        paths, facts, resolved.windows, resolved.row_keys, strict=True
    ):
        if not is_imgstore(path):
            codecs.add(clip.codec_name)
            continue
        spans = readable_chunks(path, window) if reads == "entry-files" else []
        if spans:
            codecs.add(header_codec(spans[0][0]))
            continue
        export = _linked_export(ds, ds.media_rows_of([key]).get(key))
        if export is not None and export.is_file():
            codecs.add(header_codec(export))
        else:
            codecs.add(export_codec(path, window))
    return frozenset(codecs - {""})


def refuse_unreadable_entry(
    ds: "Dataset", entry: "ResolvedScopeEntry", *, kind: str
) -> None:
    """Raise as *kind*'s run raises before its tool reads *entry*, without a probe.

    The calls are the run's own: those of
    :func:`~mosaic.tracking.common.scope.build_work_items` for a tracker, then
    :func:`entry_tool_input` for a one-file tool, the store half of
    :func:`entry_runner_sources` for mosaic's runner, and
    :func:`~mosaic.core.pipeline.joined_export.join_to_read` for an in-process
    reader.

    Args:
        ds: The dataset, read for the media index, the stores' metadata and the
            ``media`` root.
        entry: One camera of an entry, resolved.
        kind: The op whose tool reads the entry.

    Raises:
        KeyError: If a tracking root is not registered for *kind*.
        JoinedSourceMismatchError: If the clips cannot be read as one video.
        EntryJoinMissingError: If the clips need a single current join and have
            none, with the cause.
        StoreExportMissingError: If a store that the tool reads through its export
            has none, its linked file is gone, or the export is at another rate.
        MediaProbeError: If a store's metadata cannot be read.
    """
    root = tracking_root(kind)
    group, sequence = entry.group, entry.sequence
    paths, facts, windows = (
        entry.resolved.paths,
        entry.resolved.facts,
        entry.resolved.windows,
    )
    refuse_unjoinable(
        ds,
        kind,
        group,
        sequence,
        paths,
        facts,
        windows=windows,
        hands_over_path=root.reads != "in-process",
    )
    if root.retention == "tracker":
        refuse_unidentified_clips(
            ds, group, sequence, paths, facts, windows=windows, asker=kind
        )
    if root.reads == "in-process":
        _ = join_to_read(ds, entry, asker=kind)
    elif root.reads == "one-file":
        _ = entry_tool_input(
            ds, group, sequence, paths, facts, windows=windows, kind=kind
        )
    else:
        for store, window in zip(paths, windows, strict=True):
            if is_imgstore(store) and not readable_chunks(store, window):
                why = _unreadable_chunks_why(kind)
                _ = _store_export(
                    ds, group, sequence, store, window, kind=kind, why=why
                )


def _linked_export(ds: "Dataset", row: Mapping[str, object] | None) -> Path | None:
    """Return the export that a store's *row* links, or ``None`` when it links none.

    *row* is found by the store's file
    (:meth:`~mosaic.core.dataset.Dataset.media_rows_of`), whatever name a caller
    gives the entry. It is ``None`` when the index does not list the store.
    """
    if row is None:
        return None
    return derivative_path_for_target(row, EXPORT_TARGET, ds.get_root("media"))
