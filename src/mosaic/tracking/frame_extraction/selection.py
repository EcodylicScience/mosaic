"""Defines the select-frames op, which proposes frames to annotate from tracks.

:class:`SelectFramesParams` names one tracks variant and the track-quality
criteria of :mod:`mosaic.core.track_quality` that flag its rows and frames.
:func:`select_entry_frames` reads one entry's tracks table and its tracks-index
row and returns the rows of the entry's ``selection.parquet``: at most one frame
per event, chosen in an order seeded by the entry alone.

:class:`SelectFramesOp` claims each scoped entry of the variant, writes its
``selection.parquet`` and its row of the selections index, and rewrites the
run's ``extract-frames.params.json`` from the run's finished rows. An unknown
variant, and a scope without an entry of the variant, are refused before any
claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, ClassVar, Final, Self

import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.entry import Entry, make_entry_key
from mosaic.core.helpers import text_cell
from mosaic.core.keypoint_refs import resolve_keypoint_refs
from mosaic.core.params import Declared, DeclaredModel, Params, Probability
from mosaic.core.pipeline._utils import atomic_write, hash_params
from mosaic.core.pipeline.entry_claim import discard_if_empty, open_entry, release_entry
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.identity_scheme import write_identity_scheme
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    finished_rows,
    index_records,
    row_file_exists,
)
from mosaic.core.pipeline.index_lock import index_lock
from mosaic.core.pipeline.job import Cancelled
from mosaic.core.pipeline.op_identity import OP_IDENTITY_SCHEME, op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import (
    CellRow,
    frame_axis_verdict,
    read_frames_read,
    read_known_tail_loss,
    read_media_frames,
    read_tracks_index,
    row_in_scope,
    select_variant_rows,
)
from mosaic.core.pipeline.writers import read_parquet_table, write_parquet_atomic
from mosaic.core.pose_columns import pose_column_pairs
from mosaic.core.scope import Scope
from mosaic.core.seeded_digest import seeded_digest
from mosaic.core.track_quality import (
    EVENT_COLUMNS,
    ROW_CRITERIA,
    Criterion,
    QualityRules,
    body_length,
    events,
    frame_flags,
    row_flags,
    select,
)

from .dataset_runs import (
    EntryFrames,
    ExtractFramesParams,
    entry_frames_from_mapping,
    normalized_frames,
)
from .selection_index import (
    EXTRACT_PARAMS_FILE_NAME,
    SELECTION_FILE_NAME,
    SELECTION_KEY_COLUMNS,
    SELECT_FRAMES_KIND,
    FrameSelectionRow,
    frame_selection_index,
    run_selected_frames,
    selection_index_path,
    selections_kind_root,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext

__all__ = [
    "PER_FRAME_ID_PRODUCERS",
    "SELECTION_COLUMNS",
    "DuplicateCriterion",
    "EntrySelection",
    "JumpCriterion",
    "LowScoreCriterion",
    "MissingKeypointsCriterion",
    "SelectFramesConflict",
    "SelectFramesNotFound",
    "SelectFramesOp",
    "SelectFramesParams",
    "select_entry_frames",
]

SELECTION_COLUMNS: Final[tuple[str, ...]] = ("group", "sequence", *EVENT_COLUMNS)
"""The columns of an entry's ``selection.parquet``, in order."""

PER_FRAME_ID_PRODUCERS: Final = frozenset(
    {"infer-pose", "infer-points", "infer-localizer"}
)
"""The producers whose ids number the detections of each frame.

Such an id does not follow one animal across frames. Their tables skip ``jump``,
and their row criteria form events per frame.
"""

_JUMP_ENABLED_DESCRIPTION = "Whether a step of one id between frames is flagged."

_JUMP_BODY_LENGTHS_DESCRIPTION = (
    "The longest step of one id between consecutive frames that is not a jump."
)

_LOW_SCORE_ENABLED_DESCRIPTION = "Whether a row of low confidence is flagged."

_LOW_SCORE_THRESHOLD_DESCRIPTION = (
    "A present row whose mean finite keypoint confidence is below this is flagged."
)

_LOW_SCORE_QUANTILE_DESCRIPTION = (
    "A present row whose mean finite keypoint confidence is below this quantile "
    "of the entry's means is flagged. A threshold given alone clears it."
)

_MISSING_KEYPOINTS_ENABLED_DESCRIPTION = (
    "Whether a present row with a non-finite keypoint is flagged."
)

_MISSING_KEYPOINTS_DESCRIPTION = (
    "The keypoints, by index or by name, that a present row must have finite. "
    "Empty reads every keypoint. Each entry resolves them against its table "
    "whether or not the criterion is enabled, and a reference that does not "
    "resolve fails the entry."
)

_DUPLICATE_ENABLED_DESCRIPTION = (
    "Whether a frame with two rows of one animal is flagged."
)

_DUPLICATE_BODY_LENGTHS_DESCRIPTION = (
    "Two present rows of a frame whose centers lie closer than this are a "
    "duplicate pair, when their heads and their tails do too where head and "
    "tail are given."
)

_SOURCE_TRACKS_RUN_ID_DESCRIPTION = (
    "The run id of the tracks variant to read. A tracker or infer op must have "
    "published it from the full, untrimmed media, because a converted, "
    "resampled or trimmed variant does not record a frame axis."
)

_N_FRAMES_DESCRIPTION = "The most frames chosen per entry."

_MIN_SEPARATION_FRAMES_DESCRIPTION = (
    "The smallest distance allowed between two chosen frame indices of an "
    "entry, and between a chosen and an excluded one."
)

_MIN_EVENT_FRAMES_DESCRIPTION = "The shortest run of a criterion that is an event."

_N_EXPECTED_DESCRIPTION = (
    "The animals expected in every frame. A frame that detects fewer is flagged "
    "fewer_than_expected, and unset leaves that criterion off."
)

_HEAD_DESCRIPTION = (
    "The head keypoint of the body axis, by index or by name. With tail, it "
    "measures the body length, and two rows whose heads lie apart do not form "
    "a duplicate pair."
)

_TAIL_DESCRIPTION = (
    "The tail keypoint of the body axis, by index or by name. With head, it "
    "measures the body length, and two rows whose tails lie apart do not form "
    "a duplicate pair."
)

_KEYPOINT_NAMES_DESCRIPTION = (
    "The variant's keypoint names in index order, read only to resolve a name "
    "given in head, tail or missing_keypoints.keypoints."
)

_BODY_LENGTH_PX_DESCRIPTION = (
    "The body length that scales every criterion given in body lengths, in "
    "place of the median length measured from the keypoints."
)

_JUMP_DESCRIPTION = (
    "The row criterion of a step between frames. Skipped for a producer whose "
    "ids number each frame's detections."
)

_LOW_SCORE_DESCRIPTION = (
    "The row criterion of low keypoint confidence. Skipped for a table without "
    "confidence columns."
)

_MISSING_KEYPOINTS_FIELD_DESCRIPTION = "The row criterion of a non-finite keypoint."

_DUPLICATE_DESCRIPTION = "The frame criterion of two rows of one animal."

_EXCLUDE_DESCRIPTION = (
    "Frames already annotated, by entry. None of them is chosen, and every "
    "chosen frame is at least min_separation_frames from each of them."
)

_RANDOM_STATE_DESCRIPTION = (
    "Seeds the order in which an entry's candidates are taken, together with "
    "the entry's key."
)


class JumpCriterion(DeclaredModel):
    """Flags a row whose id steps far from its center at the frame before.

    Only consecutive frames are compared. A step across a gap does not count.
    """

    enabled: Annotated[bool, Declared(_JUMP_ENABLED_DESCRIPTION)] = True
    body_lengths: Annotated[
        float,
        Field(gt=0.0),
        Declared(_JUMP_BODY_LENGTHS_DESCRIPTION, unit="body lengths"),
    ] = 0.8


class LowScoreCriterion(DeclaredModel):
    """Flags a present row whose mean finite keypoint confidence is low.

    ``threshold`` and ``quantile`` are alternatives. A ``threshold`` given
    without a ``quantile`` sets ``quantile`` to None, and the two given together
    are refused.
    """

    enabled: Annotated[bool, Declared(_LOW_SCORE_ENABLED_DESCRIPTION)] = True
    threshold: Annotated[float | None, Declared(_LOW_SCORE_THRESHOLD_DESCRIPTION)] = (
        None
    )
    quantile: Annotated[
        Probability | None, Declared(_LOW_SCORE_QUANTILE_DESCRIPTION)
    ] = 0.05

    @model_validator(mode="after")
    def _one_rule(self) -> Self:
        """Keep one of ``threshold`` and ``quantile``, and refuse both or neither.

        A ``quantile`` of None given beside a ``threshold`` is the cleared
        state, which a recorded dump of these parameters contains.
        """
        if self.threshold is not None:
            if "quantile" in self.model_fields_set and self.quantile is not None:
                msg = (
                    "low_score takes a threshold or a quantile, and both are "
                    "given. Leave one of them unset."
                )
                raise ValueError(msg)
            self.quantile = None
        elif self.enabled and self.quantile is None:
            msg = (
                "low_score is enabled with neither a threshold nor a quantile. "
                "Give one of them, or set enabled to false."
            )
            raise ValueError(msg)
        return self


class MissingKeypointsCriterion(DeclaredModel):
    """Flags a present row with any of ``keypoints`` non-finite."""

    enabled: Annotated[bool, Declared(_MISSING_KEYPOINTS_ENABLED_DESCRIPTION)] = True
    keypoints: Annotated[
        tuple[int | str, ...], Declared(_MISSING_KEYPOINTS_DESCRIPTION)
    ] = ()


class DuplicateCriterion(DeclaredModel):
    """Flags a frame with two present rows close enough to be one animal.

    The rows of a duplicate pair count as one detected animal toward
    ``n_expected``.
    """

    enabled: Annotated[bool, Declared(_DUPLICATE_ENABLED_DESCRIPTION)] = True
    body_lengths: Annotated[
        float,
        Field(gt=0.0),
        Declared(_DUPLICATE_BODY_LENGTHS_DESCRIPTION, unit="body lengths"),
    ] = 0.2


class SelectFramesParams(Params):
    """Declares which frames of a tracks variant are proposed for annotation.

    Each criterion flags rows or frames, a run of flagged frames is an event,
    and each event offers its first frame. Every distance threshold is a
    multiple of the entry's body length.

    A tracks table numbers its keypoints and does not name them. A name given
    in ``head``, ``tail`` or ``missing_keypoints.keypoints`` is resolved
    against ``keypoint_names``, which is refused unless one of them is a name.
    """

    source_tracks_run_id: Annotated[
        str, Field(min_length=1), Declared(_SOURCE_TRACKS_RUN_ID_DESCRIPTION)
    ]
    n_frames: Annotated[int, Field(ge=1), Declared(_N_FRAMES_DESCRIPTION)] = 20
    min_separation_frames: Annotated[
        int,
        Field(ge=0),
        Declared(_MIN_SEPARATION_FRAMES_DESCRIPTION, unit="frames"),
    ] = 30
    min_event_frames: Annotated[
        int, Field(ge=1), Declared(_MIN_EVENT_FRAMES_DESCRIPTION, unit="frames")
    ] = 1
    n_expected: Annotated[
        int | None, Field(ge=1), Declared(_N_EXPECTED_DESCRIPTION)
    ] = None
    head: Annotated[int | str | None, Declared(_HEAD_DESCRIPTION)] = None
    tail: Annotated[int | str | None, Declared(_TAIL_DESCRIPTION)] = None
    keypoint_names: Annotated[
        tuple[str, ...] | None,
        Field(min_length=1),
        Declared(_KEYPOINT_NAMES_DESCRIPTION),
    ] = None
    body_length_px: Annotated[
        float | None, Field(gt=0.0), Declared(_BODY_LENGTH_PX_DESCRIPTION, unit="px")
    ] = None
    jump: Annotated[JumpCriterion, Declared(_JUMP_DESCRIPTION)] = JumpCriterion()
    low_score: Annotated[LowScoreCriterion, Declared(_LOW_SCORE_DESCRIPTION)] = (
        LowScoreCriterion()
    )
    missing_keypoints: Annotated[
        MissingKeypointsCriterion, Declared(_MISSING_KEYPOINTS_FIELD_DESCRIPTION)
    ] = MissingKeypointsCriterion()
    duplicate: Annotated[DuplicateCriterion, Declared(_DUPLICATE_DESCRIPTION)] = (
        DuplicateCriterion()
    )
    exclude: Annotated[tuple[EntryFrames, ...], Declared(_EXCLUDE_DESCRIPTION)] = ()
    random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 0

    @field_validator("exclude")
    @classmethod
    def _normalized_exclude(
        cls, value: tuple[EntryFrames, ...]
    ) -> tuple[EntryFrames, ...]:
        """Sort by entry and by frame, and refuse an entry named twice."""
        return normalized_frames(value, field="exclude")

    @model_validator(mode="after")
    def _names_have_a_list(self) -> Self:
        """Refuse a keypoint name without ``keypoint_names``, and the list unread."""
        references: list[tuple[str, int | str | None]] = [
            ("head", self.head),
            ("tail", self.tail),
        ]
        references += [
            ("missing_keypoints.keypoints", keypoint)
            for keypoint in self.missing_keypoints.keypoints
        ]
        named = [(field, ref) for field, ref in references if isinstance(ref, str)]
        if named and self.keypoint_names is None:
            field, name = named[0]
            msg = (
                f"{field} names keypoint {name!r}, and keypoint_names is unset. "
                "Set keypoint_names to the variant's keypoint names in index "
                "order, or give the keypoint by index."
            )
            raise ValueError(msg)
        if not named and self.keypoint_names is not None:
            msg = (
                "keypoint_names resolves a keypoint name given in head, tail or "
                "missing_keypoints.keypoints, and none of them is a name. Leave "
                "keypoint_names unset, or give the keypoints by name."
            )
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class EntrySelection:
    """Records the frames chosen for one entry and the scale they were judged at.

    Attributes:
        frames: The rows of the entry's ``selection.parquet``, with
            :data:`SELECTION_COLUMNS`, sorted by frame.
        n_candidates: The events whose first frames were candidates.
        skipped_criteria: The enabled row criteria left unevaluated, in
            criterion order. ``jump`` is skipped for a producer of
            :data:`PER_FRAME_ID_PRODUCERS`, and ``low_score`` for a table
            without confidence columns.
        body_length_px: The body length that scaled the criteria, None when
            the keypoints measure none and ``body_length_px`` is unset.
    """

    frames: pd.DataFrame
    n_candidates: int
    skipped_criteria: tuple[Criterion, ...]
    body_length_px: float | None


def _entry_label(group: str, sequence: str) -> str:
    """Return the entry as a message names it."""
    return f"({group}, {sequence})"


def _media_span(row: CellRow, entry: str) -> int:
    """Return the frame count of the entry's media, which the row must record.

    Raises:
        ValueError: The row records no ``media_frames``, or its tool read a
            frame count that :func:`frame_axis_verdict` calls a mismatch.
    """
    variant = text_cell(row.get("run_id"))
    producer = text_cell(row.get("producer"))
    media = read_media_frames(row)
    if media is None:
        msg = (
            f"tracks variant {variant} was made by {producer} and does not "
            f"record a frame axis for {entry}. select-frames needs a variant that "
            "a tracker or infer op published from the full media."
        )
        raise ValueError(msg)
    read = read_frames_read(row)
    if read is None:
        return media
    verdict = frame_axis_verdict(
        producer, read=read, media=media, known_tail_loss=read_known_tail_loss(row)
    )
    if verdict == "mismatch":
        msg = (
            f"the tool that made tracks variant {variant} read {read} frames of "
            f"{entry}, and its media contains {media}. Frames extracted from the "
            "media would be offset from the tracks."
        )
        raise ValueError(msg)
    return media


@dataclass(frozen=True, slots=True)
class _Keypoints:
    """The keypoint references of the parameters, resolved against one table."""

    head: int | None
    tail: int | None
    missing: tuple[int, ...]


def _resolved_keypoints(
    params: SelectFramesParams, n_keypoints: int, entry: str
) -> _Keypoints:
    """Resolve ``head``, ``tail`` and ``missing_keypoints.keypoints``.

    Raises:
        ValueError: ``keypoint_names`` lists a count other than *n_keypoints*,
            or :func:`resolve_keypoint_refs` refuses a reference.
    """
    names = params.keypoint_names
    if names is not None and len(names) != n_keypoints:
        msg = (
            f"keypoint_names lists {len(names)} keypoint names, and the table of "
            f"{entry} has {n_keypoints} keypoints."
        )
        raise ValueError(msg)

    def one(ref: int | str | None, who: str) -> int | None:
        if ref is None:
            return None
        return resolve_keypoint_refs([ref], names, n_keypoints, who=who)[0]

    missing = resolve_keypoint_refs(
        params.missing_keypoints.keypoints,
        names,
        n_keypoints,
        who="missing_keypoints.keypoints",
    )
    return _Keypoints(
        head=one(params.head, "head"),
        tail=one(params.tail, "tail"),
        missing=tuple(missing),
    )


def _rules(
    params: SelectFramesParams, keypoints: _Keypoints, *, by_id: bool
) -> QualityRules:
    """Return the thresholds of the enabled criteria, ``jump`` only *by_id*."""
    jump = params.jump
    low_score = params.low_score
    return QualityRules(
        jump_body_lengths=jump.body_lengths if jump.enabled and by_id else None,
        low_score_threshold=low_score.threshold if low_score.enabled else None,
        low_score_quantile=low_score.quantile if low_score.enabled else None,
        missing_keypoints=(
            keypoints.missing if params.missing_keypoints.enabled else None
        ),
        duplicate_body_lengths=(
            params.duplicate.body_lengths if params.duplicate.enabled else None
        ),
        n_expected=params.n_expected,
        head=keypoints.head,
        tail=keypoints.tail,
    )


def _excluded(params: SelectFramesParams, entry: Entry) -> tuple[int, ...]:
    """Return the frames that the parameters exclude for *entry*."""
    for item in params.exclude:
        if item.entry == entry:
            return item.indices
    return ()


def select_entry_frames(
    table: pd.DataFrame, row: CellRow, params: SelectFramesParams
) -> EntrySelection:
    """Choose the frames of one entry from the events of its tracks table.

    The span is the row's ``media_frames``. The criteria run on every frame of
    it, their runs of flagged frames become events, and :func:`select` takes
    event onsets in a permutation seeded by ``random_state`` and the entry's
    key. The entry's ``exclude`` frames start the chosen set. An entry's picks
    therefore depend on its table, row and parameters only.

    Args:
        table: The entry's tracks table.
        row: The table's tracks-index row, read by column name.
        params: The run's parameters.

    Returns:
        The chosen frames, the candidates they were chosen from, the criteria
        skipped and the body length used.

    Raises:
        ValueError: The entry fails. The row records no ``media_frames``, its
            tool read a frame count that mismatches the media, a keypoint
            reference does not resolve against the table, a criterion given in
            body lengths finds none measured and ``body_length_px`` unset, or a
            row lies outside the media's frames.
    """
    group = text_cell(row.get("group"))
    sequence = text_cell(row.get("sequence"))
    entry = _entry_label(group, sequence)
    span = _media_span(row, entry)
    n_keypoints = len(pose_column_pairs(str(name) for name in table.columns))
    keypoints = _resolved_keypoints(params, n_keypoints, entry)
    by_id = text_cell(row.get("producer")) not in PER_FRAME_ID_PRODUCERS
    rules = _rules(params, keypoints, by_id=by_id)
    scale = (
        params.body_length_px
        if params.body_length_px is not None
        else body_length(table, keypoints.head, keypoints.tail)
    )
    flagged_rows = row_flags(table, rules, scale)
    flagged_frames = frame_flags(table, span, rules, scale, flagged_rows)
    candidates = events(
        flagged_rows,
        flagged_frames,
        by_id=by_id,
        min_event_frames=params.min_event_frames,
    )
    chosen = select(
        candidates,
        n_frames=params.n_frames,
        min_separation=params.min_separation_frames,
        exclude=_excluded(params, (group, sequence)),
        seed=seeded_digest(params.random_state, make_entry_key(group, sequence)),
    )
    frames = chosen.assign(group=group, sequence=sequence)
    enabled: dict[Criterion, bool] = {
        "jump": params.jump.enabled,
        "low_score": params.low_score.enabled,
        "missing_keypoints": params.missing_keypoints.enabled,
    }
    skipped: tuple[Criterion, ...] = tuple(
        criterion
        for criterion in ROW_CRITERIA
        if enabled[criterion] and criterion not in flagged_rows.columns
    )
    return EntrySelection(
        frames=frames[list(SELECTION_COLUMNS)].reset_index(drop=True),
        n_candidates=len(candidates),
        skipped_criteria=skipped,
        body_length_px=scale,
    )


# --- The run ---------------------------------------------------------------------


class SelectFramesNotFound(ClassifiedError, LookupError):
    """Reports a tracks variant that no tracks-index row names."""

    error_kind: ClassVar[ErrorKind] = "not_found"


class SelectFramesConflict(ClassifiedError, ValueError):
    """Refuses a scope of which no entry has a row in the tracks variant."""

    error_kind: ClassVar[ErrorKind] = "conflict"


_SKIP_REASONS: Final[Mapping[Criterion, str]] = {
    "jump": "its ids number the detections of each frame",
    "low_score": "its table has no confidence columns",
}
"""Why an entry's table leaves each skippable criterion unevaluated."""


@dataclass(frozen=True, slots=True)
class _ScopedRows:
    """Lists the source variant's rows of the scoped entries, read before any claim.

    Attributes:
        rows: One tracks-index row per scoped entry that the variant has a row
            for, in entry order.
        skipped: How many scoped entries the variant has no row for.
    """

    rows: tuple[dict[str, str], ...]
    skipped: int


@dataclass(frozen=True, slots=True)
class _RunFacts:
    """Records the cells that every index row of one run shares, and its directory."""

    run_id: str
    run_root: Path
    params_hash: str


def _identity_payload(params: SelectFramesParams) -> dict[str, object]:
    """Return the payload that a select-frames run id digests."""
    return {"params": params.identity_dump()}


def _scoped_rows(ds: Dataset, variant: str, scope: ResolvedScope) -> _ScopedRows:
    """Read the rows of the tracks variant *variant* that *scope* covers.

    An unset scope covers every entry of the variant.

    Raises:
        SelectFramesNotFound: No tracks-index row names *variant*.
        SelectFramesConflict: No scoped entry has a row in *variant*.
    """
    rows = index_records(select_variant_rows(read_tracks_index(ds), variant))
    if not rows:
        message = (
            f"No tracks-index row names the tracks variant {variant!r}. "
            "`mosaic inventory --kind tracks-variant` lists the variants."
        )
        raise SelectFramesNotFound(message)
    asked = scope.op_entries
    wanted = None if asked is None else set(asked)
    scoped = sorted(
        (row for row in rows if row_in_scope(row, wanted)),
        key=lambda row: (row.get("group", ""), row.get("sequence", "")),
    )
    if not scoped:
        named = ", ".join(make_entry_key(*entry) for entry in sorted(wanted or ()))
        message = (
            f"None of the scoped entries has a row in the tracks variant "
            f"{variant!r}. The scope names {named or 'no entry'}."
        )
        raise SelectFramesConflict(message)
    skipped = 0 if wanted is None else len(wanted) - len(scoped)
    return _ScopedRows(rows=tuple(scoped), skipped=skipped)


def _extract_frames_params(
    frames: Mapping[Entry, tuple[int, ...]],
) -> dict[str, object]:
    """Return the params of the ``list`` extraction of *frames*, as JSON values.

    ``mosaic run --kind extract-frames --params @<file>`` reads them from the
    run's ``extract-frames.params.json``, and they name the run that
    ``extract_frames(ds, method="list", frames=frames)`` names.
    """
    params = ExtractFramesParams(
        method="list", frames=entry_frames_from_mapping(frames)
    )
    return params.model_dump(mode="json", include={"method", "frames"})


def _rebuild_extract_params(
    ds: Dataset, index: IndexCSV[FrameSelectionRow], facts: _RunFacts
) -> None:
    """Rewrite the run's extract-frames params from its finished rows.

    The caller holds the lock of *index*. A finished row whose file is gone is
    left out. Without a frame to list, the params file is removed.
    """
    frames = run_selected_frames(
        ds, index.read_holding_lock(), facts.run_id, skip_missing=True
    )
    path = facts.run_root / EXTRACT_PARAMS_FILE_NAME
    if not frames:
        path.unlink(missing_ok=True)
        return
    text = json.dumps(_extract_frames_params(frames), indent=2) + "\n"
    atomic_write(path, lambda target: target.write_text(text))


def _write_entry(
    ds: Dataset,
    params: SelectFramesParams,
    facts: _RunFacts,
    index: IndexCSV[FrameSelectionRow],
    row: Mapping[str, str],
    seq_dir: Path,
) -> EntrySelection:
    """Select one entry's frames, and record its selection and finished row.

    The row is appended to *index* and finished, and the run's extract-frames
    params are rebuilt from every finished row of the run, under one lock of
    *index*. Two executions over different entries of one run therefore both
    remain in the params.

    Raises:
        ValueError: The entry fails, for a reason that
            :func:`select_entry_frames` names.
        OSError: The table cannot be read, or a file cannot be written.
    """
    table = read_parquet_table(ds.resolve_path(row.get("abs_path", "")))
    chosen = select_entry_frames(table, row, params)
    path = seq_dir / SELECTION_FILE_NAME
    _ = write_parquet_atomic(chosen.frames, path)
    scale = chosen.body_length_px
    record = FrameSelectionRow(
        run_id=facts.run_id,
        abs_path=Path(ds.relative_to_root(path)),
        group=row.get("group", ""),
        sequence=row.get("sequence", ""),
        source_tracks_run_id=params.source_tracks_run_id,
        producer=row.get("producer", ""),
        n_candidates=chosen.n_candidates,
        n_selected=len(chosen.frames),
        skipped_criteria=json.dumps(list(chosen.skipped_criteria)),
        body_length_px="" if scale is None else repr(float(scale)),
        params_hash=facts.params_hash,
    )
    with index_lock(index.path):
        index.append_holding_lock([record])
        index.mark_finished(facts.run_id)
        _rebuild_extract_params(ds, index, facts)
    return chosen


def _forget_entry(
    ds: Dataset,
    index: IndexCSV[FrameSelectionRow],
    facts: _RunFacts,
    entry: Entry,
    seq_dir: Path,
) -> None:
    """Remove a failed entry from the run, and rebuild the run's extract-frames params.

    The entry's rows of the run are dropped, then its ``selection.parquet``. An
    overwrite removes the entry's directory before it selects again. A failed
    entry would otherwise keep a finished row naming a file that is gone, which
    ``selected_frames`` raises on while the params still list its old frames.
    """
    if not index.path.exists():
        return
    _ = index.drop_entries(scope=Scope(entries=[entry]), run_id=facts.run_id)
    (seq_dir / SELECTION_FILE_NAME).unlink(missing_ok=True)
    with index_lock(index.path):
        _rebuild_extract_params(ds, index, facts)


@register_op
class SelectFramesOp(Op[SelectFramesParams]):
    """Propose frames to annotate from a tracker or infer op's tracks of the full media.

    Each event of the track-quality criteria offers its first frame. A converted
    variant records no frame axis, and each of its entries fails, as does each
    entry of a resampled or trimmed variant. The frames chosen for an entry are
    written to its ``selection.parquet``, and the run's
    ``extract-frames.params.json`` lists every entry's frames as the params of a
    ``list`` extraction.
    """

    kind = SELECT_FRAMES_KIND
    category = "extract"
    domain = "tracking"
    version = "0.1"
    writes_tracks = False
    scope_takes = "any"
    scope_dependent = False
    Params = SelectFramesParams

    def plan_identity(
        self,
        ds: Dataset,
        params: SelectFramesParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """Name the run from its params alone.

        The params name the tracks variant, so the identity reads neither the
        dataset nor the scope.
        """
        return OpIdentity(
            run_id=op_run_id(self.kind, self.version, _identity_payload(params))
        )

    def run(
        self,
        ds: Dataset,
        params: SelectFramesParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        """Select the frames of each scoped entry of the variant, and record each one.

        Without *overwrite*, an entry with a finished row of this run and its
        ``selection.parquet`` is reused. Under *overwrite*, every entry is
        selected again. An entry that another execution holds is skipped. A
        failed entry leaves no row and no selection in the run, and the run's
        ``extract-frames.params.json`` is rebuilt without it. A run whose params
        file is absent at its end rebuilds it from the run's finished rows,
        unless the selections index does not exist yet.
        """
        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        variant = params.source_tracks_run_id
        scoped = _scoped_rows(ds, variant, scope)
        ctx.tracks_variant([variant])
        if scoped.skipped:
            noun = "entry has" if scoped.skipped == 1 else "entries have"
            print(
                f"[{SELECT_FRAMES_KIND}] {scoped.skipped} scoped {noun} no row in the "
                f"tracks variant {variant}; skipping them."
            )

        run_root = selections_kind_root(ds) / run_id
        write_identity_scheme(run_root, OP_IDENTITY_SCHEME)
        facts = _RunFacts(
            run_id=run_id,
            run_root=run_root,
            params_hash=hash_params(_identity_payload(params)),
        )
        index = frame_selection_index(selection_index_path(ds))
        records = index_records(index.read()) if index.path.exists() else []
        finished = finished_rows(records, run_id, SELECTION_KEY_COLUMNS)
        total = len(scoped.rows)
        ctx.set_total(total)

        # An entry that another execution holds is not this run's to lose. A
        # reused entry is attempted, so `entries_written` counts it, as the
        # run-log counts a cache hit's entries.
        attempted: set[str] = set()
        lost: set[str] = set()
        reused: set[str] = set()
        try:
            for done, row in enumerate(scoped.rows, start=1):
                ctx.check_cancel()
                entry = (row.get("group", ""), row.get("sequence", ""))
                key = make_entry_key(*entry)
                opened = open_entry(
                    ds, ctx, run_root, key, kind=SELECT_FRAMES_KIND, overwrite=overwrite
                )
                if opened is not None:
                    seq_dir = opened[0]
                    attempted.add(key)
                    try:
                        if not overwrite and row_file_exists(ds, finished.get(entry)):
                            reused.add(key)
                        else:
                            chosen = _write_entry(
                                ds, params, facts, index, row, seq_dir
                            )
                            for criterion in chosen.skipped_criteria:
                                reason = _SKIP_REASONS.get(criterion, "")
                                print(
                                    f"[{SELECT_FRAMES_KIND}] {key} skips {criterion}: "
                                    f"{reason}."
                                )
                    except Cancelled:
                        raise
                    except Exception as exc:
                        ctx.entry_failed_unless_cancelled(key, exc)
                        lost.add(key)
                        _forget_entry(ds, index, facts, entry, seq_dir)
                    finally:
                        release_entry(seq_dir, ctx.execution_id)
                        discard_if_empty(seq_dir)
                ctx.progress.on_entry_end(done, total, key)
                ctx.heartbeat(done)
        finally:
            # Reused entries count. A rerun and a first run over one scope then
            # report the same coverage.
            ctx.entries_written(len(attempted) - len(lost))

        if len(reused) == total:
            ctx.cache_hit()
        if attempted and lost == attempted:
            message = (
                f"[{SELECT_FRAMES_KIND}] every one of {len(attempted)} attempted "
                f"entries failed: {', '.join(sorted(lost))}. The run "
                f"run_id={run_id} selected no frames. The per-entry errors are in "
                "this attempt's run-log."
            )
            raise AllEntriesFailed(message)
        params_path = run_root / EXTRACT_PARAMS_FILE_NAME
        if not params_path.is_file() and index.path.exists():
            with index_lock(index.path):
                _rebuild_extract_params(ds, index, facts)
        held = total - len(attempted)
        if not params_path.is_file() and held:
            noun = "entry" if held == 1 else "entries"
            print(
                f"[{SELECT_FRAMES_KIND}] the selection of run_id={run_id} has no frame "
                f"to extract yet: another execution holds {held} scoped {noun}, "
                f"and {EXTRACT_PARAMS_FILE_NAME} is not written."
            )
        elif not params_path.is_file():
            print(
                f"[{SELECT_FRAMES_KIND}] the selection of run_id={run_id} is empty: no "
                f"entry has a frame to extract, and {EXTRACT_PARAMS_FILE_NAME} "
                "is not written."
            )
        print(
            f"[{SELECT_FRAMES_KIND}] completed run_id={run_id}: "
            f"{len(attempted) - len(lost)} of {total} entries selected, "
            f"{len(reused)} reused -> {run_root}"
        )
        return run_id
