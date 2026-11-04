"""The ``predictions_table`` label converter: per-frame predictions from a file.

A ``.parquet`` file, or a ``.csv`` file with a header row and comma separators,
holds one prediction per row. Its columns:

- ``sequence`` (text): the row's sequence. A blank cell, or a file without the
  column, takes the sequence of the file's ``labels_raw/index.csv`` row.
- ``group`` (text, optional): the row's group under ``group_from="infile"``, or
  when the raw row names none.
- ``frame`` (integer, at least 0): the row's frame in the entry's tracks table.
- ``id``, or ``id1`` and ``id2`` (integers, at least 0, optional): the
  individual of an individual row, or the focal and the other individual of a
  pair row. A row without ids is a scene-level row.
- ``predicted_label``: the predicted class, an integer id, or a name of
  ``class_ids``. Without the column, a row predicts the class of its highest
  probability, the lowest class id on a tie.
- ``prob_<class>`` (float in [0, 1] or NaN): the probability of a class, named
  by its id, or by a name of ``class_ids``.
- ``fold`` (integer, at least -1, or empty): the fold whose model predicted the
  row out of fold. ``-1`` or empty marks a row predicted in sample.
- ``block`` (integer, at least 0, or empty, only beside ``fold``): the row's
  cross-validation block.

Other columns are ignored. A file must give ``predicted_label`` or a
``prob_<class>`` column.

``(id1, id2)`` is the direction of a pair row: ``id1`` is the focal individual.
A pair feature writes each pair once per perspective, with ``id1`` and ``id2``
swapped between the two, so the same order aligns a prediction with its pair
row. A ``perspective`` column is therefore not read, and a file that writes each
pair once with the direction in ``perspective`` repeats a ``(frame, id1, id2)``
key and is refused.

The frames are the tracks table's frames, so a division's predictions are not
moved by its first frame. One set of predictions writes each entry once: the
converter refuses one entry key from two converted entries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import field_validator

from mosaic.core.helpers import UNLABELED
from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
    entry_group,
)
from mosaic.core.params import Declared

__all__ = ["PredictionsTableConverter", "PredictionsTableParams"]

_FILES_DESCRIPTION = (
    "A case-sensitive pattern matched against each raw file's path as "
    "labels_raw/index.csv records it, relative to the dataset root inside the "
    "dataset, so that only matching files convert."
)

_CLASS_IDS_DESCRIPTION = (
    "The class id of each class name the file uses in predicted_label and in "
    "its prob_<name> columns, which are integer ids when this is unset."
)

_TEXT_COLUMNS = ("sequence", "group")
_ID_COLUMNS = ("id", "id1", "id2")
_PROBABILITY_PREFIX = "prob_"


class PredictionsTableParams(LabelConvertParams):
    """Parameters of the ``predictions_table`` converter.

    ``files`` and ``class_ids`` change the labels, so both are hashed.
    """

    files: Annotated[str, Declared(_FILES_DESCRIPTION)] = "*"
    class_ids: Annotated[dict[str, int] | None, Declared(_CLASS_IDS_DESCRIPTION)] = None

    @field_validator("class_ids")
    @classmethod
    def _class_ids_are_distinct_and_not_negative(
        cls, value: dict[str, int] | None
    ) -> dict[str, int] | None:
        if value is None:
            return value
        negative = sorted(name for name, class_id in value.items() if class_id < 0)
        if negative:
            msg = (
                f"class_ids must map each name to an id of at least 0, not "
                f"{negative[0]!r} to {value[negative[0]]}; -1 reads as unlabeled"
            )
            raise ValueError(msg)
        names_of: dict[int, list[str]] = {}
        for name, class_id in value.items():
            names_of.setdefault(class_id, []).append(name)
        shared = sorted(
            (class_id, sorted(names))
            for class_id, names in names_of.items()
            if len(names) > 1
        )
        if shared:
            class_id, names = shared[0]
            msg = f"class_ids names one id twice: {names} are all {class_id}"
            raise ValueError(msg)
        return value


@dataclass(frozen=True, slots=True)
class _Rows:
    """The entry and frame of each row of one file, for naming a refused row."""

    path: Path
    groups: npt.NDArray[np.object_]
    sequences: npt.NDArray[np.object_]
    frames: npt.NDArray[np.int64]

    def first(self, mask: npt.NDArray[np.bool_]) -> int:
        """The row of *mask* first in ``(group, sequence, frame)`` order."""
        rows = np.flatnonzero(mask)
        order = np.lexsort(
            (
                self.frames[rows],
                self.sequences[rows].astype(str),
                self.groups[rows].astype(str),
            )
        )
        return int(rows[order[0]])

    def at(self, row: int, reason: str) -> ValueError:
        """The refusal of *row* for *reason*."""
        return _refusal(
            self.path, str(self.sequences[row]), str(self.frames[row]), reason
        )

    def refuse(self, mask: npt.NDArray[np.bool_], reason: str) -> None:
        """Raise naming the first row of *mask*, when *mask* holds one."""
        if mask.any():
            raise self.at(self.first(mask), reason)

    def everywhere(self, reason: str) -> ValueError:
        """The refusal of the whole file for *reason*, naming its first row."""
        return self.at(self.first(np.ones(len(self.frames), dtype=bool)), reason)


def _refusal(path: Path, sequence: str, frame: str, reason: str) -> ValueError:
    return ValueError(f"{path}: sequence {sequence!r}, frame {frame}: {reason}.")


def _read(path: Path, text_columns: tuple[str, ...]) -> pd.DataFrame:
    """The table of *path*, with each of *text_columns* read as text."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        table = pd.read_parquet(path)
        for column in text_columns:
            if column in table.columns:
                table[column] = [_text(value) for value in table[column].tolist()]
        return table
    if suffix == ".csv":
        table = pd.read_csv(path, float_precision="round_trip")
        present = [column for column in text_columns if column in table.columns]
        if present:
            text = pd.read_csv(path, usecols=present, dtype=str, keep_default_na=False)
            for column in present:
                table[column] = text[column].tolist()
        return table
    msg = f"{path}: a predictions table is a .parquet or a .csv file, not {suffix!r}."
    raise ValueError(msg)


def _text(value: object) -> str:
    """A cell as text, empty for a missing value (None, NaN, NA or NaT)."""
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _numbers(
    series: pd.Series,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """The values of *series* as floats, NaN where empty, and its non-numeric cells."""
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(
        series.dtype
    ):
        numeric = series.to_numpy(dtype=np.float64, na_value=np.nan)
        return numeric, np.zeros(len(numeric), dtype=bool)
    text = np.asarray([_text(value).strip() for value in series.tolist()], dtype=str)
    empty = text == ""
    numeric = pd.to_numeric(
        pd.Series(np.where(empty, "nan", text)), errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    return numeric, np.isnan(numeric) & ~empty


def _whole(
    series: pd.Series,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """The integers of *series*, its empty cells, and its cells of no integer."""
    numeric, textual = _numbers(series)
    empty = np.isnan(numeric) & ~textual
    finite = np.isfinite(numeric)
    safe = np.where(finite, numeric, 0.0)
    integral = finite & (safe == np.floor(safe))
    invalid = textual | (~empty & ~integral)
    return np.where(integral, safe, 0.0).astype(np.int64), empty, invalid


def _frame_text(value: object) -> str:
    """A frame cell as a refusal names it."""
    text = _text(value).strip()
    if not text:
        return "(empty)"
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _probability_columns(
    columns: list[str], class_ids: dict[str, int] | None, rows: _Rows
) -> dict[int, str]:
    """The ``prob_<class>`` column of each class, ascending by class."""
    from mosaic.behavior.feature_library.classifier_common import (
        PROBABILITY_COLUMN,
        probability_classes,
    )

    prefixed = [column for column in columns if column.startswith(_PROBABILITY_PREFIX)]
    found: dict[int, str] = {}
    for column in prefixed:
        if class_ids is not None:
            name = column.removeprefix(_PROBABILITY_PREFIX)
            if name not in class_ids:
                raise rows.everywhere(
                    f"the column {column} names no class of class_ids"
                )
            class_id = class_ids[name]
        else:
            match = PROBABILITY_COLUMN.match(column)
            if match is None:
                raise rows.everywhere(
                    f"the column {column} names no integer class; give class_ids "
                    "to name classes"
                )
            class_id = int(match[1])
            if class_id == UNLABELED:
                raise rows.everywhere(
                    f"the column {column} names the class -1, which reads as unlabeled"
                )
        if class_id in found:
            raise rows.everywhere(
                f"the columns {found[class_id]} and {column} name one class"
            )
        found[class_id] = column
    if class_ids is not None:
        return {class_id: found[class_id] for class_id in sorted(found)}
    return {class_id: found[class_id] for class_id in probability_classes(prefixed)}


def _probabilities(
    table: pd.DataFrame, probability_columns: dict[int, str], rows: _Rows
) -> npt.NDArray[np.float64]:
    """The probabilities of each row, one column per class."""
    probabilities = np.empty((len(table), len(probability_columns)), dtype=np.float64)
    for position, column in enumerate(probability_columns.values()):
        values, textual = _numbers(table[column])
        outside = ~np.isnan(values) & ((values < 0.0) | (values > 1.0))
        rows.refuse(textual | outside, f"{column} is a probability in [0, 1] or NaN")
        probabilities[:, position] = values
    return probabilities


def _predicted(
    series: pd.Series, class_ids: dict[str, int] | None, rows: _Rows
) -> npt.NDArray[np.int64]:
    """The class of each row's ``predicted_label`` cell."""
    if class_ids is not None:
        names = [_text(value).strip() for value in series.tolist()]
        rows.refuse(
            np.asarray([not name for name in names], dtype=bool),
            "predicted_label is empty",
        )
        unknown = np.asarray([name not in class_ids for name in names], dtype=bool)
        if unknown.any():
            row = rows.first(unknown)
            raise rows.at(
                row, f"the predicted_label {names[row]!r} is no name of class_ids"
            )
        return np.asarray([class_ids[name] for name in names], dtype=np.int64)
    values, empty, invalid = _whole(series)
    rows.refuse(empty, "predicted_label is empty")
    rows.refuse(
        invalid,
        "predicted_label is an integer class id; give class_ids to name classes",
    )
    rows.refuse(values == UNLABELED, "the predicted_label -1 reads as unlabeled")
    return values


def _derived(
    probabilities: npt.NDArray[np.float64], classes: list[int], rows: _Rows
) -> npt.NDArray[np.int64]:
    """The class of each row's highest probability, the lowest class on a tie."""
    from mosaic.behavior.feature_library.classifier_common import predicted_labels

    rows.refuse(
        np.isnan(probabilities).any(axis=1),
        "a row without predicted_label has a NaN probability, so it predicts no class",
    )
    return predicted_labels(probabilities, classes, None, classes[0]).astype(np.int64)


def _folds(
    table: pd.DataFrame, columns: list[str], rows: _Rows
) -> tuple[npt.NDArray[np.int64] | None, npt.NDArray[np.int64] | None]:
    """Each row's fold and block, ``-1`` in sample, or ``None`` without the column."""
    if "block" in columns and "fold" not in columns:
        raise rows.everywhere("a block column needs a fold column")
    if "fold" not in columns:
        return None, None
    values, empty, invalid = _whole(table["fold"])
    rows.refuse(
        invalid | (~empty & (values < UNLABELED)),
        "fold is an integer of at least -1, or empty",
    )
    folds = np.where(empty, UNLABELED, values).astype(np.int64)
    blocks: npt.NDArray[np.int64] | None = None
    if "block" in columns:
        values, empty, invalid = _whole(table["block"])
        rows.refuse(
            invalid | (~empty & (values < 0)),
            "block is an integer of at least 0, or empty",
        )
        rows.refuse(
            ~empty & (folds == UNLABELED),
            "a row predicted in sample, of fold -1, has no block",
        )
        rows.refuse(
            empty & (folds != UNLABELED), "a row predicted out of fold needs a block"
        )
        blocks = np.where(empty, UNLABELED, values).astype(np.int64)

    frames = pd.DataFrame(
        {"group": rows.groups, "sequence": rows.sequences, "frame": rows.frames}
    )
    for name, assigned in (("fold", folds), ("block", blocks)):
        if assigned is None:
            continue
        distinct = (
            frames.assign(value=assigned)
            .groupby(["group", "sequence", "frame"])["value"]
            .transform("nunique")
        )
        rows.refuse(
            distinct.to_numpy(dtype=np.int64) > 1,
            f"two {name}s at one frame of one entry; a {name} belongs to a frame, "
            "not to an individual",
        )
    if blocks is not None:
        rows.refuse(
            _split_blocks(frames, folds, blocks),
            "one fold and block in two separate runs of the entry's frames",
        )
    return folds, blocks


def _split_blocks(
    frames: pd.DataFrame,
    folds: npt.NDArray[np.int64],
    blocks: npt.NDArray[np.int64],
) -> npt.NDArray[np.bool_]:
    """The rows that start a second run of an entry's frames of one fold and block."""
    entry = ["group", "sequence"]
    distinct = (
        frames.assign(fold=folds, block=blocks)
        .drop_duplicates([*entry, "frame"])
        .sort_values([*entry, "frame"], kind="stable")
    )
    run_key = [*entry, "fold", "block"]
    starts = distinct[
        (distinct[run_key] != distinct[run_key].shift()).any(axis=1).to_numpy()
    ]
    starts = starts[starts["block"] >= 0]
    repeated = starts[starts.duplicated(run_key)]
    split = set(
        zip(repeated["group"], repeated["sequence"], repeated["frame"], strict=True)
    )
    every_row = zip(
        frames["group"].tolist(),
        frames["sequence"].tolist(),
        frames["frame"].tolist(),
        strict=True,
    )
    return np.asarray([key in split for key in every_row], dtype=bool)


class PredictionsTableConverter(LabelConverter[PredictionsTableParams]):
    """Read a per-frame table of predictions, with optional probabilities and folds."""

    src_format = "predictions_table"
    label_kind = "predictions"
    label_format = "individual_pair_v1"
    version = "0.1"
    Params = PredictionsTableParams
    refuses_shared_entry_keys = True
    frames_on_tracks_axis = True

    def selects(
        self, raw_row: Mapping[str, object], params: PredictionsTableParams
    ) -> bool:
        """Whether the raw file's recorded path matches ``params.files``."""
        return fnmatchcase(str(raw_row.get("abs_path", "")), params.files)

    def convert(
        self,
        src_path: Path,
        params: PredictionsTableParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read one predictions file into one :class:`LabelEntry` per entry.

        Raises:
            ValueError: The file breaks a rule of the table, naming the file, the
                sequence and the first offending frame.
        """
        class_ids = params.class_ids
        text_columns = _TEXT_COLUMNS + (("predicted_label",) if class_ids else ())
        table = _read(src_path, text_columns)
        if table.empty:
            msg = f"{src_path}: a predictions table has no rows."
            raise ValueError(msg)
        columns = [str(column) for column in table.columns]
        if "frame" not in columns:
            msg = f"{src_path}: a predictions table needs a frame column."
            raise ValueError(msg)
        rows = _entry_rows(src_path, table, params, raw_row)

        individuals = _individuals(table, columns, rows)
        probability_columns = _probability_columns(columns, class_ids, rows)
        if "predicted_label" not in columns and not probability_columns:
            raise rows.everywhere(
                "a predictions table needs a predicted_label or a prob_<class> column"
            )
        classes = list(probability_columns)
        probabilities = (
            _probabilities(table, probability_columns, rows)
            if probability_columns
            else None
        )
        if "predicted_label" in columns:
            labels = _predicted(table["predicted_label"], class_ids, rows)
            uncovered = ~np.isin(labels, classes)
            if probabilities is not None and uncovered.any():
                row = rows.first(uncovered)
                raise rows.at(
                    row,
                    f"the predicted_label {labels[row]} is none of the probability "
                    f"classes {classes}, so no probability covers it",
                )
        else:
            assert probabilities is not None
            labels = _derived(probabilities, classes, rows)
        if not classes:
            classes = sorted(
                class_ids.values() if class_ids else {int(label) for label in labels}
            )

        keys = pd.DataFrame(
            {
                "group": rows.groups,
                "sequence": rows.sequences,
                "frame": rows.frames,
                "id1": individuals[:, 0],
                "id2": individuals[:, 1],
            }
        )
        rows.refuse(
            keys.duplicated(keep=False).to_numpy(dtype=bool),
            "two rows predict one individual, pair or scene at one frame",
        )
        folds, blocks = _folds(table, columns, rows)

        names_of = {class_id: name for name, class_id in (class_ids or {}).items()}
        label_names = tuple(
            names_of.get(class_id, str(class_id)) for class_id in classes
        )
        entries: list[LabelEntry] = []
        for group, sequence in sorted(
            set(zip(rows.groups.tolist(), rows.sequences.tolist(), strict=True))
        ):
            members = np.flatnonzero(
                (rows.groups == group) & (rows.sequences == sequence)
            )
            order = members[
                np.lexsort(
                    (
                        individuals[members, 1],
                        individuals[members, 0],
                        rows.frames[members],
                    )
                )
            ]
            payload: dict[str, object] = {
                "label_format": self.label_format,
                "group": str(group),
                "sequence": str(sequence),
                "frames": rows.frames[order].astype(np.int64),
                "labels": labels[order].astype(np.int32),
                "individual_ids": individuals[order].astype(np.int32),
                "label_ids": np.asarray(classes, dtype=np.int64),
                "label_names": np.asarray(label_names, dtype=object),
                "background_id": np.int32(UNLABELED),
            }
            if probabilities is not None:
                payload["pred_classes"] = np.asarray(classes, dtype=np.int64)
                payload["pred_probabilities"] = probabilities[order]
            if folds is not None:
                payload["pred_fold"] = folds[order]
            if blocks is not None:
                payload["pred_block"] = blocks[order]
            entries.append(
                LabelEntry(
                    group=str(group),
                    sequence=str(sequence),
                    payload=payload,
                    n_frames=int(rows.frames[order].max()) + 1,
                    label_ids=tuple(classes),
                    label_names=label_names,
                )
            )
        return entries


def _entry_rows(
    path: Path,
    table: pd.DataFrame,
    params: PredictionsTableParams,
    raw_row: Mapping[str, object],
) -> _Rows:
    """The entry and frame of each row, refusing a row of no sequence or frame."""
    n_rows = len(table)
    raw_sequence = _text(raw_row.get("sequence")).strip()
    cells = (
        np.asarray(table["sequence"].tolist(), dtype=object)
        if "sequence" in table.columns
        else np.full(n_rows, "", dtype=object)
    )
    sequences = np.where(cells == "", raw_sequence, cells).astype(object)
    group_cells = (
        [str(cell) for cell in table["group"].tolist()]
        if "group" in table.columns
        else [""] * n_rows
    )
    groups = np.asarray(
        [entry_group(params.group_from, raw_row, cell) for cell in group_cells],
        dtype=object,
    )
    frames, empty, invalid = _whole(table["frame"])
    bad_frames = empty | invalid | (frames < 0)
    nameless = sequences == ""
    if bad_frames.any() or nameless.any():
        row = int(np.flatnonzero(bad_frames | nameless)[0])
        reason = (
            "a frame is an integer of at least 0"
            if bad_frames[row]
            else "the row names no sequence, and neither does the file's "
            "labels_raw/index.csv row"
        )
        raise _refusal(
            path, str(sequences[row]), _frame_text(table["frame"].iloc[row]), reason
        )
    return _Rows(path, groups, sequences, frames)


def _individuals(
    table: pd.DataFrame, columns: list[str], rows: _Rows
) -> npt.NDArray[np.int64]:
    """The ``(id, -1)``, ``(id1, id2)`` or ``(-1, -1)`` of each row."""
    has_id = "id" in columns
    has_pair = "id1" in columns or "id2" in columns
    if has_id and has_pair:
        raise rows.everywhere("a table has an id column or id1 and id2, not both")
    if has_pair and not ("id1" in columns and "id2" in columns):
        raise rows.everywhere("a pair row needs both id1 and id2")
    ids: dict[str, npt.NDArray[np.int64]] = {}
    for column in _ID_COLUMNS:
        if column in columns:
            values, empty, invalid = _whole(table[column])
            rows.refuse(
                empty | invalid | (values < 0),
                f"{column} is an integer of at least 0 on every row",
            )
            ids[column] = values
    unset = np.full(len(table), UNLABELED, dtype=np.int64)
    if has_id:
        return np.column_stack([ids["id"], unset])
    if has_pair:
        return np.column_stack([ids["id1"], ids["id2"]])
    return np.column_stack([unset, unset])
