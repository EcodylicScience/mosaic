"""``label-predictions``: read a label kind of predictions onto the rows of its input.

The bridge between predictions made outside mosaic, converted by
``convert-labels`` into a kind such as ``predictions``, and the steps that score
or tune a classifier. Its output is the input's identity columns, then
``predicted_label``, then a ``prob_<class>`` column per class when the variant
has probabilities, which ``evaluate-labels``, ``tune-threshold`` and
``probability-ensemble`` read as they read a classifier's output.

**Which event a row takes.** Each row is matched by its own frame and
individuals (:func:`~mosaic.core.pipeline.loading.events_for_rows`), as
``evaluate-labels`` matches its labels. A track row ``(id, -1)`` takes an
individual event of its id, a pair event focal on it or a scene event; a pair
row takes its own direction, an individual event of its focal or a scene event;
a row without ids takes the events of its frame once reverse pair events are
dropped. The row's probabilities come from the event that gave its label. A row
two or more events match is refused with its entry, since its prediction would
depend on the order of the events in the file. A row no event matches has a
null ``predicted_label`` and NaN probabilities, unless the kind's background
class fills it, as ``evaluate-labels`` reads the same kind.

**The fold table.** A file's ``fold`` column marks the rows predicted out of
fold, by the model of fold ``k >= 0``. When a row of the variant has one, the
run writes ``cv_folds.parquet`` in its run root, from every entry of the
variant whatever the request's scope, in the layout a classifier writes. It is
by entry when no file has a ``block`` column and every row of each listed entry
has one fold of 0 or more. Otherwise it is by block: an entry's predicted frames
are sorted and cut into maximal runs of one ``(fold, block)`` with a fold of 0 or
more, each a block from its first frame to its last plus one, named by the
file's ``block`` value or else by the run's position in the entry. Two runs of
one ``block`` value in an entry are refused. ``n_rows`` and ``rows_<class>``
count the out-of-fold rows, and an entry without one is not listed.

**Label controls.** ``predictions.mask`` makes the masked classes unlabeled, so
their rows have a null prediction and NaN probabilities. ``predictions.remap``
rewrites ``predicted_label`` and is refused on a variant with probabilities,
because its ``prob_<class>`` columns name the file's classes, from which every
consumer of them recomputes the class. A step downstream reads the mask in the
bridge's recorded params and refuses a truth class that the bridge writes no
probability for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.helpers import UNLABELED
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Conditional,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.loading import events_for_rows
from mosaic.core.pipeline.types import (
    DependencyLookup,
    EmitsLevel,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    Result,
    TrackInput,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.pipeline.writers import write_parquet_atomic

from .classifier_common import (
    FOLD_TABLE_FILE,
    CvBlock,
    block_fold_table,
    fold_table,
)
from .helpers import ensure_columns, entry_identity, meta_columns
from .registry import register_feature

__all__ = ["LabelPredictions", "PredictedEntry", "prediction_fold_table"]

_NAME: Final = "label-predictions"

_PREDICTIONS_DESCRIPTION = (
    "The label kind of predictions read onto the rows, resolved from "
    "labels/<kind>/index.csv, where a {step: <id>} reference names a "
    "convert-labels step and pins the variant it writes."
)

_MISSING_PREDICTIONS_DESCRIPTION = (
    "What an entry without a file of the variant writes: null writes its rows "
    "without a prediction, and error fails the entry."
)


@dataclass(frozen=True, slots=True)
class PredictedEntry:
    """The predictions of one entry that its fold table counts.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        frames: The frame of each event.
        labels: The predicted class of each event.
        label_ids: The classes of the entry's file.
        folds: The fold of each event, ``-1`` in sample, or ``None`` for a file
            without folds.
        blocks: The block of each event, ``-1`` in sample, or ``None`` for a
            file without blocks.
    """

    group: str
    sequence: str
    frames: npt.NDArray[np.int64]
    labels: npt.NDArray[np.int64]
    label_ids: tuple[int, ...]
    folds: npt.NDArray[np.int64] | None
    blocks: npt.NDArray[np.int64] | None

    @property
    def key(self) -> tuple[str, str]:
        """The entry's ``(group, sequence)``."""
        return self.group, self.sequence

    @property
    def out_of_fold(self) -> npt.NDArray[np.bool_]:
        """Whether each event was predicted out of fold."""
        if self.folds is None:
            return np.zeros(len(self.frames), dtype=np.bool_)
        return self.folds >= 0


def prediction_fold_table(entries: Sequence[PredictedEntry]) -> pd.DataFrame | None:
    """The ``cv_folds.parquet`` of a variant's predictions, ``None`` without folds.

    The module docstring gives the rule that chooses the layout and cuts the
    blocks.

    Args:
        entries: Every entry of the variant.

    Returns:
        The table by entry (:func:`fold_table`) or by block
        (:func:`block_fold_table`), counting each class of the entries'
        ``label_ids``, or ``None`` when no event is out of fold.

    Raises:
        ValueError: An entry's file gives one ``block`` value to two runs.
    """
    listed = sorted(
        (entry for entry in entries if entry.out_of_fold.any()),
        key=lambda entry: entry.key,
    )
    if not listed:
        return None
    classes = sorted({int(c) for entry in entries for c in entry.label_ids})
    by_entry = all(entry.blocks is None for entry in entries) and all(
        entry.out_of_fold.all() and np.unique(_folds(entry)).size == 1
        for entry in listed
    )
    if by_entry:
        fold_of = {entry.key: int(_folds(entry)[0]) for entry in listed}
        counted = [(entry.key, entry.labels) for entry in listed]
        return fold_table(fold_of, _class_counts(counted, classes), classes)
    blocks: list[CvBlock] = []
    counted_blocks: list[tuple[tuple[str, str, int], npt.NDArray[np.int64]]] = []
    for entry in listed:
        for block, labels in _entry_blocks(entry):
            blocks.append(block)
            counted_blocks.append(((block.group, block.sequence, block.block), labels))
    blocks.sort(key=lambda block: (block.group, block.sequence, block.block))
    return block_fold_table(blocks, _class_counts(counted_blocks, classes), classes)


def _folds(entry: PredictedEntry) -> npt.NDArray[np.int64]:
    """The folds of an entry that has them."""
    assert entry.folds is not None
    return entry.folds


def _entry_blocks(
    entry: PredictedEntry,
) -> list[tuple[CvBlock, npt.NDArray[np.int64]]]:
    """Each block of *entry* with the predicted classes of its events."""
    order = np.argsort(entry.frames, kind="stable")
    frames = entry.frames[order]
    folds = _folds(entry)[order]
    labels = entry.labels[order]
    named = None if entry.blocks is None else entry.blocks[order]
    change = folds[1:] != folds[:-1]
    if named is not None:
        change |= named[1:] != named[:-1]
    starts = np.concatenate([[0], np.flatnonzero(change) + 1])
    stops = np.concatenate([starts[1:], [len(frames)]])
    found: list[tuple[CvBlock, npt.NDArray[np.int64]]] = []
    seen: set[int] = set()
    for first, stop in zip(starts.tolist(), stops.tolist(), strict=True):
        fold = int(folds[first])
        if fold < 0:
            continue
        index = len(found) if named is None else int(named[first])
        if index in seen:
            msg = (
                f"[{_NAME}] entry ({entry.group!r}, {entry.sequence!r}) gives block "
                f"{index} to two runs of frames, split by frames of another fold or "
                "block, and a block is one run of frames"
            )
            raise ValueError(msg)
        seen.add(index)
        block = CvBlock(
            entry.group,
            entry.sequence,
            index,
            int(frames[first]),
            int(frames[stop - 1]) + 1,
            fold,
        )
        found.append((block, labels[first:stop]))
    return found


def _class_counts[K: tuple[object, ...]](
    counted: Sequence[tuple[K, npt.NDArray[np.int64]]], classes: list[int]
) -> pd.DataFrame:
    """The events of each key per class, indexed by the keys, one column per class."""
    keys = [key for key, _ in counted]
    table = pd.DataFrame(
        [
            [int(np.count_nonzero(labels == cls)) for cls in classes]
            for _, labels in counted
        ],
        columns=pd.Index(classes),
        dtype=np.int64,
    )
    table.index = pd.MultiIndex.from_tuples(keys)
    return table


@dataclass(frozen=True, slots=True)
class _VariantFile:
    """What ``load_state`` reads of one file of the variant."""

    entry: PredictedEntry
    pred_classes: tuple[int, ...] | None


def _read_file(key: tuple[str, str], path: Path) -> _VariantFile:
    """The classes, probabilities and folds of the file of entry *key*."""
    with np.load(path, allow_pickle=True) as npz:
        names = set(npz.files)
        pred_classes = (
            tuple(int(c) for c in np.asarray(npz["pred_classes"]).ravel())
            if "pred_probabilities" in names
            else None
        )
        label_ids = (
            tuple(int(c) for c in np.asarray(npz["label_ids"]).ravel())
            if "label_ids" in names
            else ()
        )
        folds = (
            np.asarray(npz["pred_fold"], dtype=np.int64).ravel()
            if "pred_fold" in names
            else None
        )
        blocks = (
            np.asarray(npz["pred_block"], dtype=np.int64).ravel()
            if "pred_block" in names
            else None
        )
        if folds is None:
            frames = np.zeros(0, dtype=np.int64)
            labels = np.zeros(0, dtype=np.int64)
        else:
            frames = np.asarray(npz["frames"], dtype=np.int64).ravel()
            labels = np.asarray(npz["labels"], dtype=np.int64).ravel()
    return _VariantFile(
        PredictedEntry(
            group=key[0],
            sequence=key[1],
            frames=frames,
            labels=labels,
            label_ids=label_ids,
            folds=folds,
            blocks=blocks,
        ),
        pred_classes,
    )


@final
@register_feature
class LabelPredictions:
    """Read a label kind of predictions onto the rows of one input.

    Writes ``predicted_label`` from each row's matching event, and its
    ``prob_<class>`` when the variant has probabilities, so a scoring or tuning
    step reads predictions made outside mosaic as it reads a classifier's. A
    variant with out-of-fold rows gives the run a ``cv_folds.parquet``.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.label_predictions.LabelPredictions.Params`.
    """

    category = "per-frame"
    name = _NAME
    version = "0.1"
    parallelizable = True
    scope_dependent = False  # each entry is read from its own file
    fits = False  # learns nothing; the fold table is read from the variant
    temporal_reach = 0  # a row depends on its own frame's prediction
    relates: RelatesLevel = "none"
    accepts_overlap = False  # computes within a row
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    follows_ids: bool = True  # labels a row by its ids
    requires: ClassVar[tuple[Requirement, ...]] = (Column("frame"),)
    provides: ClassVar[Provides] = Provides(
        adds=(Column("predicted_label"),),
        conditional=(*META_COLUMNS_KEPT, Conditional(Prefix("prob_"))),
    )

    class Inputs(Inputs[TrackInput | Result]):
        @model_validator(mode="after")
        def _one_input(self) -> Self:
            """Read the predictions onto the rows of one input."""
            if len(self.root) != 1:
                msg = (
                    f"{_NAME} reads its predictions onto the rows of one input, "
                    f"and got {len(self.root)} inputs"
                )
                raise ValueError(msg)
            return self

    class Params(Params):
        predictions: Annotated[
            GroundTruthLabelsSource, Declared(_PREDICTIONS_DESCRIPTION)
        ] = Field(default_factory=lambda: GroundTruthLabelsSource(kind="predictions"))
        missing_predictions: Annotated[
            Literal["null", "error"], Declared(_MISSING_PREDICTIONS_DESCRIPTION)
        ] = "null"

    def __init__(
        self,
        inputs: LabelPredictions.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._files: DependencyLookup = {}
        self._classes: tuple[int, ...] | None = None
        self._fold_table: pd.DataFrame | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._files = dict(dependency_lookups.get("predictions", {}))
        self._fold_table = None
        read = [_read_file(key, path) for key, path in sorted(self._files.items())]
        self._classes = self._variant_classes(read)
        entries = [file.entry for file in read]
        if not any(entry.out_of_fold.any() for entry in entries):
            return True
        if (run_root / FOLD_TABLE_FILE).exists():
            return True
        self._fold_table = prediction_fold_table(entries)
        return False

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        if self._fold_table is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        _ = write_parquet_atomic(self._fold_table, run_root / FOLD_TABLE_FILE)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_columns(df, ["frame"])
        n_rows = len(df)
        labels = np.full(n_rows, UNLABELED, dtype=np.int64)
        positions = np.full(n_rows, -1, dtype=np.int64)
        table: npt.NDArray[np.float64] | None = None
        group, sequence = entry_identity(df)
        path = self._files.get((group, sequence))
        if path is None:
            if self.params.missing_predictions == "error":
                msg = (
                    f"[{self.name}] the predictions variant has no file for entry "
                    f"({group!r}, {sequence!r}). Convert its predictions, or set "
                    'missing_predictions to "null" to write its rows without one'
                )
                raise ValueError(msg)
        elif n_rows:
            events = events_for_rows(path, df)
            self._refuse_doubled(df, events.matches, (group, sequence))
            labels = self.params.predictions.relabel(events.labels)
            positions = events.positions
            if self._classes is not None:
                with np.load(path, allow_pickle=True) as npz:
                    table = np.asarray(npz["pred_probabilities"], dtype=np.float64)

        out = df[meta_columns(df)].reset_index(drop=True)
        labeled = labels != UNLABELED
        predicted = pd.array(labels, dtype="Int64")
        predicted[~labeled] = pd.NA
        out["predicted_label"] = predicted
        if self._classes is not None:
            values = np.full((n_rows, len(self._classes)), np.nan, dtype=np.float64)
            gathered = labeled & (positions >= 0)
            if table is not None:
                values[gathered] = table[positions[gathered]]
            for column, cls in enumerate(self._classes):
                out[f"prob_{cls}"] = values[:, column]
        return out

    # --- Refusals ---

    def _variant_classes(self, read: list[_VariantFile]) -> tuple[int, ...] | None:
        """The probability classes of the variant, ``None`` for one without.

        Raises:
            ValueError: Files list different classes, only some files have
                probabilities, or a remap is given to a variant with them.
        """
        with_probabilities = [file for file in read if file.pred_classes is not None]
        if not with_probabilities:
            return None
        if len(with_probabilities) != len(read):
            without = next(file for file in read if file.pred_classes is None)
            having = with_probabilities[0]
            msg = (
                f"[{self.name}] entry {having.entry.key} of the predictions variant "
                f"has probabilities and entry {without.entry.key} has none. Every "
                "file of a variant gives probabilities, or none does, because a "
                "consumer of the prob_<class> columns needs them on every entry"
            )
            raise ValueError(msg)
        first = with_probabilities[0]
        for file in with_probabilities[1:]:
            if file.pred_classes != first.pred_classes:
                msg = (
                    f"[{self.name}] entry {first.entry.key} lists classes "
                    f"{list(first.pred_classes or ())} and entry {file.entry.key} "
                    f"lists {list(file.pred_classes or ())}. Every file of a "
                    "variant lists the same probability classes"
                )
                raise ValueError(msg)
        if self.params.predictions.remap is not None:
            msg = (
                f"[{self.name}] predictions.remap is refused on a variant with "
                "probabilities: its prob_<class> columns name the file's classes, "
                "from which a consumer recomputes the class. Remap the classes "
                "in the file, or convert it without probabilities"
            )
            raise ValueError(msg)
        return first.pred_classes

    def _refuse_doubled(
        self,
        df: pd.DataFrame,
        matches: npt.NDArray[np.int64],
        entry: tuple[str, str],
    ) -> None:
        """Refuse a row that two or more events match.

        Raises:
            ValueError: Naming the entry, and the frame, ids and match count of
                the first such row by frame.
        """
        doubled = np.flatnonzero(matches >= 2)
        if doubled.size == 0:
            return
        frames = df["frame"].to_numpy()
        row = int(doubled[np.argmin(frames[doubled])])
        ids = ", ".join(
            f"{column} {df[column].iloc[row]}"
            for column in ("id", "id1", "id2")
            if column in df.columns
        )
        where = f"frame {frames[row]}" + (f" with {ids}" if ids else "")
        msg = (
            f"[{self.name}] entry ({entry[0]!r}, {entry[1]!r}): the row at {where} "
            f"is matched by {int(matches[row])} events, and its prediction would "
            "depend on their order in the file"
        )
        raise ValueError(msg)
