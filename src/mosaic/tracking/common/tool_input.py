"""The path a tracker hands to an external tool, which is not always the source.

All four integrated trackers run their tool as a subprocess and give it a path to
open, so all four resolve through here. That works for a video file and fails for
an imgstore recording, which is a *directory* of chunk files: T-Rex converts it
to nothing and reports a missing ``.pv``, and SLEAP, Lightning Pose and
Ultralytics fail comparably. mosaic's own readers handle a store natively, so the
mismatch is only ever at this boundary -- the moment a path leaves mosaic for a
tool that does its own decoding.

:func:`resolve_tool_input` is that boundary. A plain video passes through
untouched; a store resolves to the plain video ``export-store`` wrote for it, or
raises naming the command that would produce one.

Ultralytics used to be the exception, decoding through ``open_frame_reader`` and
reading a store directly -- genuinely the better path, and one that worked with
no export on disk. That capability is gone: Ultralytics is AGPL-3.0, so it runs
in an environment of its own and opens a path like every other tool, and tracking
a store now costs an ``export-store`` run and a copy of the pixels first. Pose and
point *inference* pay the same cost for the same reason. The heatmap localizer
does not: it is mosaic's own PyTorch and still reads a store natively.

:func:`refuse_undecodable_codec` checks the codec of each file handed over. A tool
that declares a decode probe is tested in its environment by a
:class:`DecodeProbe`, which runs the probe program there once per run.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from mosaic_media import SOFTWARE_DECODABLE_CODECS
from mosaic_media.ffmpeg import run_to_completion
from mosaic_media.transcode import TranscodeError

from mosaic.core.media.facts_columns import derivative_path_for_target, row_mapping
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.pipeline.joined_export import current_join
from mosaic.core.pipeline.store_export import EXPORT_TARGET
from mosaic.core.pipeline.subprocess_util import run_supervised
from mosaic.core.pipeline.tracking_roots import (
    CONSERVATIVE_DECODER,
    TRACKING_ROOTS,
    ToolDecoder,
)
from mosaic.core.pipeline.variant_source import preprocess_command
from mosaic.tracking.common.toolenv import (
    ToolEnv,
    captured_output,
    subprocess_env,
    tool_invocation,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.tracking.common.scope import TrackerWorkItem

__all__ = [
    "DecodeProbe",
    "ProbeVerdict",
    "StoreExportMissingError",
    "ToolCodecError",
    "refuse_undecodable_codec",
    "resolve_entry_input",
    "resolve_tool_input",
    "resolve_tool_inputs",
]

_ALLOW_CODECS_VAR: Final = "MOSAIC_ALLOW_TOOL_CODECS"
_CODEC_PROBE_TIMEOUT_SECONDS: Final = 120.0
_DECODE_PROBE_TIMEOUT_SECONDS: Final = 300.0
"""How long one decode probe may run. DALI's start on a GPU takes most of it."""


class StoreExportMissingError(FileNotFoundError):
    """An imgstore has no exported video for a subprocess tool to open."""


class ToolCodecError(TranscodeError):
    """A tool is being handed a file its decoder stack may not open.

    Its own class rather than a reuse of the two above, and for the same reason
    they are separate from each other: the remedy differs. Those two say build
    the file; this one says the file exists and is in the wrong codec.
    """


def _stream_codec(path: Path) -> str:
    """*path*'s video codec, from its header.

    A header read, not :func:`~mosaic_media.probe_media` -- that scans every
    packet, which is minutes over a joined session, and the codec is in the
    first few bytes. This runs once per file handed to a tool.

    An unreadable header answers ``""``, which passes. A file a tool cannot open
    at all is the tool's own error to report, with its own message; inventing a
    codec refusal for it here would name the wrong cause.
    """
    try:
        out = run_to_completion(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(path),
            ],
            timeout=_CODEC_PROBE_TIMEOUT_SECONDS,
            action=f"reading the codec of {path.name}",
            error_type=TranscodeError,
        )
    except TranscodeError:
        return ""
    return out.strip().splitlines()[0].strip().lower() if out.strip() else ""


def _allowed_codecs(decoder: ToolDecoder) -> frozenset[str]:
    """What *decoder* may be handed: the baseline, its own extras, and the override.

    ``MOSAIC_ALLOW_TOOL_CODECS`` is a comma-separated list of extra codec names.
    It exists because the refusal is an inference about a decoder mosaic does
    not own: someone who knows their tool environment links ``libdav1d`` is
    right, and should not have to re-encode a corpus to prove it. It widens the
    set and never narrows it, so the variable cannot turn a working run into a
    broken one.
    """
    extra = os.environ.get(_ALLOW_CODECS_VAR, "")
    named = {part.strip().lower() for part in extra.split(",") if part.strip()}
    return SOFTWARE_DECODABLE_CODECS | decoder.also_reads | named


@dataclass(frozen=True, slots=True)
class ProbeVerdict:
    """Record the answer of one decode probe in a tool's environment.

    Attributes:
        decoded: True when the probe exited 0.
        tested: The file that the probe was run on. A later file in the same
            codec is answered from this one.
        environment: The interpreter's argv, joined by spaces.
        output: The probe's captured output, or the reason that it did not run,
            indented for a message.
    """

    decoded: bool
    tested: Path
    environment: str
    output: str


class DecodeProbe:
    """Test a tool's environment for a codec once per run, and remember the answer.

    A run creates one for its tool, from the placement that the run resolved, and
    passes it to the check of every entry. Each pair of interpreter argv and codec
    is tested once. A new run tests again, because an environment can be rebuilt
    between runs.

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
        """Return whether the environment decodes *codec*, tested on *path* once.

        The first call for an interpreter and a codec runs *program* on *path*.
        When the program decodes the file, one line on standard error names the
        tool, the codec and the environment. Later calls return the remembered
        answer. A cancelled probe is not remembered.

        Args:
            program: The probe that the tool's decoder declares.
            path: The file that the tool is about to open.
            codec: The codec of *path*.
            kind: The tool's op kind, named in the line printed.
            cancel_check: Polled while the probe runs. The probe is stopped
                when it returns True.

        Returns:
            The answer, from this call's probe or from an earlier one.

        Raises:
            ToolNotFoundError: The subclass that the tool declares, when its
                environment cannot be located. The tool's run raises the same.
            ProcessCancelled: When *cancel_check* fires during the probe.
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
        self._verdicts[(interpreter, codec)] = verdict
        if verdict.decoded:
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
    """Run *program* on *path* with *interpreter*, and return its answer.

    A timeout or an interpreter that does not start is an answer that the file
    was not decoded. Its output then states the reason. A cancel raises
    ``ProcessCancelled`` and gives no answer.
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
    output = captured_output(stdout, stderr)
    return ProbeVerdict(
        decoded=returncode == 0, tested=path, environment=environment, output=output
    )


def refuse_undecodable_codec(
    ds: "Dataset",
    path: Path,
    *,
    kind: str,
    group: str,
    sequence: str,
    variant: str = "",
    decode_probe: DecodeProbe | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> None:
    """Raise unless *kind*'s tool decodes *path*, by declaration or by test.

    Each tool declares its decoder on its
    :class:`~mosaic.core.pipeline.tracking_roots.TrackingRoot`, because the answer
    differs per tool. TREx links the libavcodec of its environment, and
    Ultralytics reads through mosaic-media's PyAV. Both declare AV1. SLEAP reads
    through OpenCV, and the Linux OpenCV wheel from PyPI does not decode AV1.
    Lightning Pose reads through DALI's ``fn.readers.video``, which in DALI 2.3
    does not handle AV1 on any GPU.

    A codec is allowed when it is in the baseline, in the tool's ``also_reads``,
    or in ``MOSAIC_ALLOW_TOOL_CODECS``. Otherwise a tool that declares a probe is
    tested in its environment through *decode_probe*. Exit 0 allows the codec. A
    non-zero exit, a timeout or an interpreter that does not start raises with the
    probe's output. A tool without a probe, and a caller without *decode_probe*,
    are refused by the declaration.

    The check reads the file that the tool opens. A clip that is joined into one
    file for the tool may be in any codec, because the tool does not open it.

    A tool without a decoder for a file reads zero frames and exits 0, and its run
    records an empty result as a success. The refusal stops the run before that
    result is recorded.

    A kind without a registered root gets the conservative declaration, which
    lists only the baseline.

    Args:
        ds: The dataset, read for the recipe of *variant*.
        path: The file that the tool opens.
        kind: The tool's op kind, which selects its declaration.
        group: The entry's group, named in a refusal.
        sequence: The entry's sequence, named in a refusal.
        variant: The run id of the media variant that *path* is the file of, or
            empty. A refusal of a variant also names making the variant again in
            H.264 from the recipe that *ds* recorded for it.
        decode_probe: The probe of the tool's environment, created by the run
            from the placement that it resolved. ``None`` for a caller without
            one.
        cancel_check: The run's cancel check, polled while a probe runs.

    Raises:
        ToolCodecError: If the codec is refused.
        ProcessCancelled: When *cancel_check* fires during a probe.
    """
    root = TRACKING_ROOTS.get(kind)
    decoder = root.decoder if root is not None else CONSERVATIVE_DECODER
    codec = _stream_codec(path)
    if not codec or codec in _allowed_codecs(decoder):
        return
    verdict = (
        decode_probe.verdict(
            decoder.probe, path, codec, kind=kind, cancel_check=cancel_check
        )
        if decoder.probe and decode_probe is not None
        else None
    )
    if verdict is not None and verdict.decoded:
        return
    if verdict is None:
        finding = (
            f", and its declaration does not list {codec}. A tool without a "
            f"decoder for a file reads zero frames and exits 0, and its run "
            f"records an empty result as a success."
        )
        setting = f"To declare that this environment decodes {codec}, set"
    else:
        tested = (
            path.name
            if verdict.tested == path
            else f"{verdict.tested} earlier in this run"
        )
        finding = (
            f". A test in the environment of {kind} did not decode {codec} in "
            f"{tested}:\n"
            f"    Environment: {verdict.environment}\n"
            f"{textwrap.indent(verdict.output, '  ')}"
        )
        setting = "To skip the test, set"
    remedy = f"\n    {decoder.remedy}." if decoder.remedy else ""
    remake = (
        f"\n    Or make the media variant in a codec that {kind} reads. Run the "
        f'recipe of {variant} with "codec" set to "h264" in a copy of it, and name '
        f"the variant that it writes in media. The recipe of {variant} is run "
        f"with:\n"
        f"{preprocess_command(ds, variant, [(group, sequence)])}"
        if variant
        else ""
    )
    message = (
        f"[{kind}] ({group}, {sequence}) resolves to {path.name}, which is "
        f"{codec}. {kind} decodes with {decoder.stack}{finding}{remedy}{remake}\n"
        f"    {setting} {_ALLOW_CODECS_VAR}={codec}."
    )
    raise ToolCodecError(message)


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
    video that mosaic wrote, one file for the whole entry. An export and a join
    therefore do not apply to it.

    Args:
        ds: The dataset, read for the media index and the ``media`` root.
        item: The work item whose source paths are being resolved.
        kind: The tracker's kind, so a failure names the tool the user invoked.
        decode_probe: The probe of the tool's environment, for a tool that
            declares one. See :func:`refuse_undecodable_codec`.
        cancel_check: The run's cancel check, polled while a probe runs.

    Raises:
        StoreExportMissingError: If a source is a store with no export
            registered, or with a link pointing at a file that is gone.
        JoinedExportMissingError: If the entry has several clips and no joined
            export has been built for exactly that clip set.
    """
    handed: tuple[Path, ...]
    if item.variant is not None:
        handed = (item.variant.path,)
    else:
        clips = tuple(
            resolve_entry_input(ds, item.group, item.sequence, source, kind=kind)
            for source in item.video_paths
        )
        # Several clips resolve to their join and the clips themselves are
        # dropped, so the codec gate runs over what is returned rather than
        # inside the comprehension above: an AV1 derivative about to be
        # re-encoded into a uniform join is not a file any tool will open, and
        # refusing it here would block a run that would have worked.
        handed = clips if len(clips) < 2 else (_joined_input(ds, item, kind=kind),)
    for target in handed:
        refuse_undecodable_codec(
            ds,
            target,
            kind=kind,
            group=item.group,
            sequence=item.sequence,
            variant=item.media,
            decode_probe=decode_probe,
            cancel_check=cancel_check,
        )
    return handed


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
    ds: "Dataset", group: str, sequence: str, source: Path, *, kind: str
) -> Path:
    """*source* itself, or the export registered for it when it is a store.

    The entry named by *group* and *sequence* rather than a
    :class:`~mosaic.tracking.common.scope.TrackerWorkItem`, because the inference
    ops reach this boundary too and build no work items: they walk a media scope
    directly. Those two names are all a work item ever supplied here.

    The store row is found by path rather than by camera, because the path
    alone identifies the store within a multi-camera sequence.
    """
    if not is_imgstore(source):
        return source

    export = _registered_export(ds, group, sequence, source)
    if export is None:
        message = (
            f"[{kind}] ({group}, {sequence}) is an imgstore recording, "
            f"which {kind} cannot open -- it reads a video file, not a store "
            f"directory. Export it first:\n"
            f"    mosaic run -m <manifest> --kind export-store --params "
            f'\'{{"entry": ["{group}", "{sequence}"]}}\''
        )
        raise StoreExportMissingError(message)
    if not export.is_file():
        message = (
            f"[{kind}] ({group}, {sequence}) links to an exported video "
            f"at {export}, which does not exist; re-run 'mosaic run --kind "
            f"export-store' to rebuild it"
        )
        raise StoreExportMissingError(message)
    return export


def _joined_input(ds: "Dataset", item: "TrackerWorkItem", *, kind: str) -> Path:
    """The one video holding *item*'s clips, or a refusal naming how to build it.

    The lookup is :func:`~mosaic.core.pipeline.joined_export.current_join`'s,
    keyed by ``item.source_uid`` -- the value the reuse gate already computes.
    """
    why = (
        f"{kind} is handed one video file, and a tool that joins clips itself "
        f"loses frames at every boundary, so mosaic joins them first."
    )
    return current_join(
        ds,
        item.group,
        item.sequence,
        item.source_uid,
        item.n_sources,
        asker=kind,
        why=why,
    )


def _registered_export(
    ds: "Dataset", group: str, sequence: str, source: Path
) -> Path | None:
    """The export linked from *source*'s own store row, or ``None`` if unlinked."""
    matched = ds.match_media_rows(group, sequence)
    media_root = ds.get_root("media")
    for _, row in matched.iterrows():
        # Through row_mapping rather than indexing the Series: a Series subscript
        # is untyped, and the path is compared, not merely printed.
        cells = row_mapping(row)
        if ds.resolve_path(str(cells["abs_path"])) != source:
            continue
        return derivative_path_for_target(cells, EXPORT_TARGET, media_root)
    return None
