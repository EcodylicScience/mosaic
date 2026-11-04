"""The ``predictions_table`` converter reads a per-frame file of predictions.

Each test writes a small table as ``.parquet`` or ``.csv`` and reads it either
through the converter directly or through a dataset's conversion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.label_library.predictions_table import (
    PredictionsTableConverter,
    PredictionsTableParams,
)
from mosaic.core.dataset import Dataset
from mosaic.core.label_converter import LabelEntry, get_label_converter
from mosaic.core.pipeline.labels_index import read_labels_index
from tests.helpers import make_dataset, tree_snapshot

Table = dict[str, list[object]]

RAW_ROW: dict[str, object] = {"group": "", "sequence": "s1", "abs_path": "p.csv"}


def write_table(directory: Path, name: str, table: Table | pd.DataFrame) -> Path:
    """Write *table* as *name*, a ``.parquet`` or a ``.csv``, under *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    frame = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
    path = directory / name
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def convert(
    path: Path,
    raw_row: Mapping[str, object] = RAW_ROW,
    /,
    **params: object,
) -> list[LabelEntry]:
    """Convert *path* with *params* as the raw row *raw_row* names it."""
    return PredictionsTableConverter().convert(
        path, PredictionsTableParams.model_validate(params), raw_row
    )


def listed(value: object) -> object:
    """*value* as a list when it is an array of at least one dimension."""
    array = np.asarray(value)
    return array.tolist() if array.ndim else value


def payload(entry: LabelEntry) -> dict[str, object]:
    """*entry*'s payload with each array as a list, for comparison."""
    return {key: listed(value) for key, value in entry.payload.items()}


def dtypes(entry: LabelEntry) -> dict[str, str]:
    """The dtype of each array or numpy scalar of *entry*'s payload."""
    return {
        key: str(np.asarray(value).dtype)
        for key, value in entry.payload.items()
        if not isinstance(value, str)
    }


def full_table() -> Table:
    """Two sequences of pair rows with probabilities, folds and blocks."""
    return {
        "sequence": ["s1", "s1", "s1", "s1", "s2", "s2"],
        "frame": [0, 0, 1, 2, 0, 5],
        "id1": [0, 1, 0, 0, 0, 0],
        "id2": [1, 0, 1, 1, 1, 1],
        "predicted_label": [1, 0, 1, 0, 0, 1],
        "prob_0": [0.1, 0.7, 0.3, 0.6, 1 / 3, 0.2],
        "prob_1": [0.9, 0.3, 0.7, 0.4, 2 / 3, 0.8],
        "fold": [0, 0, 0, 1, 2, np.nan],
        "block": [0, 0, 0, 1, 2, np.nan],
    }


# --- the payload ---------------------------------------------------------------


def test_parquet_and_csv_of_one_table_give_equal_payloads(tmp_path: Path) -> None:
    from_parquet = convert(write_table(tmp_path, "p.parquet", full_table()))
    from_csv = convert(write_table(tmp_path, "p.csv", full_table()))

    assert [payload(entry) for entry in from_parquet] == [
        payload(entry) for entry in from_csv
    ]
    assert [dtypes(entry) for entry in from_parquet] == [
        dtypes(entry) for entry in from_csv
    ]


def test_the_payload_of_a_file_holds_its_classes_probabilities_and_folds(
    tmp_path: Path,
) -> None:
    first, second = convert(write_table(tmp_path, "p.parquet", full_table()))

    assert (first.group, first.sequence, first.n_frames) == ("", "s1", 3)
    assert (second.sequence, second.n_frames) == ("s2", 6)
    assert payload(first) == {
        "label_format": "individual_pair_v1",
        "group": "",
        "sequence": "s1",
        "frames": [0, 0, 1, 2],
        "labels": [1, 0, 1, 0],
        "individual_ids": [[0, 1], [1, 0], [0, 1], [0, 1]],
        "label_ids": [0, 1],
        "label_names": ["0", "1"],
        "background_id": -1,
        "pred_classes": [0, 1],
        "pred_probabilities": [[0.1, 0.9], [0.7, 0.3], [0.3, 0.7], [0.6, 0.4]],
        "pred_fold": [0, 0, 0, 1],
        "pred_block": [0, 0, 0, 1],
    }
    assert dtypes(first) == {
        "frames": "int64",
        "labels": "int32",
        "individual_ids": "int32",
        "label_ids": "int64",
        "label_names": "object",
        "background_id": "int32",
        "pred_classes": "int64",
        "pred_probabilities": "float64",
        "pred_fold": "int64",
        "pred_block": "int64",
    }
    assert np.asarray(second.payload["pred_fold"]).tolist() == [2, -1]
    assert np.asarray(second.payload["pred_block"]).tolist() == [2, -1]
    assert (first.label_ids, first.label_names) == ((0, 1), ("0", "1"))


def test_a_file_and_its_rows_shuffled_give_equal_payloads(tmp_path: Path) -> None:
    table = pd.DataFrame(full_table())
    shuffled = table.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert not shuffled.equals(table)

    in_order = convert(write_table(tmp_path, "a.parquet", table))
    out_of_order = convert(write_table(tmp_path, "b.parquet", shuffled))

    assert [payload(entry) for entry in in_order] == [
        payload(entry) for entry in out_of_order
    ]


def test_a_float64_probability_from_parquet_is_stored_bit_for_bit(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    probabilities = rng.random(50)
    table: Table = {
        "frame": list(range(50)),
        "prob_0": list(1.0 - probabilities),
        "prob_1": list(probabilities),
    }

    (entry,) = convert(write_table(tmp_path, "p.parquet", table))

    stored = np.asarray(entry.payload["pred_probabilities"])
    assert stored[:, 1].tobytes() == probabilities.tobytes()
    assert stored[:, 0].tobytes() == (1.0 - probabilities).tobytes()


def test_a_file_without_a_sequence_column_is_the_raw_row_s_sequence(
    tmp_path: Path,
) -> None:
    table: Table = {"frame": [4, 1], "predicted_label": [2, 0]}

    (entry,) = convert(
        write_table(tmp_path, "p.csv", table),
        {"group": "g", "sequence": "from_row", "abs_path": "p.csv"},
    )

    assert (entry.group, entry.sequence, entry.n_frames) == ("g", "from_row", 5)
    assert payload(entry)["frames"] == [1, 4]
    assert payload(entry)["labels"] == [0, 2]
    assert payload(entry)["label_ids"] == [0, 2]
    assert "pred_probabilities" not in entry.payload
    assert "pred_fold" not in entry.payload


def test_a_blank_sequence_cell_takes_the_raw_row_s_sequence(tmp_path: Path) -> None:
    table: Table = {"sequence": ["", "s2"], "frame": [0, 0], "predicted_label": [0, 1]}

    entries = convert(write_table(tmp_path, "p.csv", table))

    assert [entry.sequence for entry in entries] == ["s1", "s2"]


def test_a_null_cell_of_a_string_column_in_parquet_is_blank(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "group": pd.array(["g", None], dtype="string"),
            "sequence": pd.array([None, "s2"], dtype="string"),
            "frame": [0, 0],
            "predicted_label": [0, 1],
        }
    )
    path = write_table(tmp_path, "p.parquet", table)

    expected = [("", "s2"), ("g", "s1")]
    assert [(e.group, e.sequence) for e in convert(path)] == expected
    assert [(e.group, e.sequence) for e in convert(path, group_from="infile")] == (
        expected
    )


def test_a_null_cell_of_a_string_predicted_label_in_parquet_is_empty(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame(
        {"frame": [0, 1], "predicted_label": pd.array(["0", None], dtype="string")}
    )
    path = write_table(tmp_path, "p.parquet", table)

    with pytest.raises(ValueError, match=r"frame 1: predicted_label is empty"):
        _ = convert(path)


def test_individual_pair_and_scene_rows_carry_their_ids(tmp_path: Path) -> None:
    individuals = convert(
        write_table(
            tmp_path,
            "i.csv",
            {"frame": [0, 0], "id": [3, 1], "predicted_label": [0, 1]},
        )
    )
    pairs = convert(
        write_table(
            tmp_path,
            "p.csv",
            {"frame": [0], "id1": [2], "id2": [5], "predicted_label": [0]},
        )
    )
    scene = convert(
        write_table(tmp_path, "s.csv", {"frame": [0, 1], "predicted_label": [0, 1]})
    )

    assert payload(individuals[0])["individual_ids"] == [[1, -1], [3, -1]]
    assert payload(individuals[0])["labels"] == [1, 0]
    assert payload(pairs[0])["individual_ids"] == [[2, 5]]
    assert payload(scene[0])["individual_ids"] == [[-1, -1], [-1, -1]]


def test_class_ids_name_the_classes_of_the_labels_and_the_probabilities(
    tmp_path: Path,
) -> None:
    class_ids = {"other": 0, "attack": 3}
    with_probabilities: Table = {
        "frame": [0, 1],
        "predicted_label": ["attack", "other"],
        "prob_other": [0.2, 0.9],
        "prob_attack": [0.8, 0.1],
    }
    labels_alone: Table = {"frame": [0, 1], "predicted_label": ["other", "other"]}

    (probable,) = convert(
        write_table(tmp_path, "p.csv", with_probabilities), class_ids=class_ids
    )
    (plain,) = convert(
        write_table(tmp_path, "l.parquet", labels_alone), class_ids=class_ids
    )

    assert payload(probable)["labels"] == [3, 0]
    assert payload(probable)["pred_classes"] == [0, 3]
    assert payload(probable)["pred_probabilities"] == [[0.2, 0.8], [0.9, 0.1]]
    assert payload(probable)["label_names"] == ["other", "attack"]
    assert payload(plain)["label_ids"] == [0, 3]
    assert payload(plain)["label_names"] == ["other", "attack"]
    assert (plain.label_ids, plain.label_names) == ((0, 3), ("other", "attack"))


def test_without_predicted_labels_each_row_takes_its_most_probable_class(
    tmp_path: Path,
) -> None:
    table: Table = {
        "frame": [0, 1, 2],
        "prob_0": [0.2, 0.5, 0.25],
        "prob_1": [0.8, 0.5, 0.25],
        "prob_2": [0.0, 0.0, 0.5],
    }

    (entry,) = convert(write_table(tmp_path, "p.csv", table))

    assert payload(entry)["labels"] == [1, 0, 2]


def test_a_tie_of_named_classes_takes_the_lowest_class_id(tmp_path: Path) -> None:
    table: Table = {"frame": [0], "prob_b": [0.5], "prob_a": [0.5]}

    (entry,) = convert(
        write_table(tmp_path, "p.csv", table), class_ids={"a": 4, "b": 2}
    )

    assert payload(entry)["labels"] == [2]


def test_a_predicted_label_that_is_not_the_most_probable_is_kept(
    tmp_path: Path,
) -> None:
    table: Table = {
        "frame": [0],
        "predicted_label": [1],
        "prob_0": [0.6],
        "prob_1": [0.4],
    }

    (entry,) = convert(write_table(tmp_path, "p.csv", table))

    assert payload(entry)["labels"] == [1]


def test_a_whole_number_float_is_an_integer(tmp_path: Path) -> None:
    table: Table = {
        "frame": [0.0, 2.0],
        "id": [1.0, 1.0],
        "predicted_label": [1.0, 0.0],
    }

    (entry,) = convert(write_table(tmp_path, "p.parquet", table))

    assert payload(entry)["frames"] == [0, 2]
    assert payload(entry)["individual_ids"] == [[1, -1], [1, -1]]
    assert payload(entry)["labels"] == [1, 0]


# --- groups --------------------------------------------------------------------


def test_group_from_filename_takes_the_raw_row_s_group_when_it_names_one(
    tmp_path: Path,
) -> None:
    table: Table = {
        "group": ["", "infile"],
        "sequence": ["s1", "s2"],
        "frame": [0, 0],
        "predicted_label": [0, 1],
    }
    path = write_table(tmp_path, "p.csv", table)
    named = {"group": "row", "sequence": "", "abs_path": "p.csv"}
    unnamed = {"group": "", "sequence": "", "abs_path": "p.csv"}

    assert [(e.group, e.sequence) for e in convert(path, named)] == [
        ("row", "s1"),
        ("row", "s2"),
    ]
    assert [(e.group, e.sequence) for e in convert(path, unnamed)] == [
        ("", "s1"),
        ("infile", "s2"),
    ]


def test_group_from_infile_takes_each_row_s_group(tmp_path: Path) -> None:
    table: Table = {
        "group": ["a", "b", "a"],
        "sequence": ["s", "s", "s"],
        "frame": [0, 0, 1],
        "predicted_label": [0, 1, 1],
    }
    raw_row = {"group": "row", "sequence": "", "abs_path": "p.csv"}

    entries = convert(
        write_table(tmp_path, "p.csv", table), raw_row, group_from="infile"
    )

    assert [(e.group, e.sequence, e.n_frames) for e in entries] == [
        ("a", "s", 2),
        ("b", "s", 1),
    ]


# --- refusals ------------------------------------------------------------------


def _with(**columns: list[object]) -> Table:
    """A table of three rows of sequence ``s1`` at frames 0, 1 and 2, with *columns*."""
    return {"frame": [0, 1, 2], "predicted_label": [0, 1, 0], **columns}


def _without(table: Table, *columns: str) -> Table:
    return {name: values for name, values in table.items() if name not in columns}


MINUS_ONE_COLUMN: Table = {"prob_-1": [0.5, 0.5, 0.5], "prob_0": [0.5, 0.5, 0.5]}

REFUSED_TABLES: dict[str, tuple[Table, dict[str, object], str]] = {
    "id beside id1": (
        _with(id=[0, 0, 0], id1=[0, 0, 0], id2=[1, 1, 1]),
        {},
        r"frame 0: .*id.*id1",
    ),
    "id1 without id2": (_with(id1=[0, 0, 0]), {}, r"frame 0: .*id1.*id2"),
    "id2 without id1": (_with(id2=[0, 0, 0]), {}, r"frame 0: .*id1.*id2"),
    "an empty id": (_with(id=[0, np.nan, 0]), {}, r"frame 1: .*id"),
    "a negative id": (_with(id=[0, 0, -1]), {}, r"frame 2: .*id"),
    "a negative id2": (_with(id1=[0, 0, 0], id2=[1, -2, 1]), {}, r"frame 1: .*id2"),
    "a fractional id": (_with(id=[0, 1.5, 0]), {}, r"frame 1: .*id"),
    "a textual id": (_with(id=["a", "0", "0"]), {}, r"frame 0: .*id"),
    "a negative frame": (
        {"frame": [0, -3], "predicted_label": [0, 1]},
        {},
        r"frame -3: .*frame",
    ),
    "a fractional frame": (
        {"frame": [0, 2.5], "predicted_label": [0, 1]},
        {},
        r"frame 2\.5: .*frame",
    ),
    "an empty frame": (
        {"frame": [0, np.nan], "predicted_label": [0, 1]},
        {},
        r"frame \(empty\): .*frame",
    ),
    "two rows of one id at one frame": (
        {"frame": [4, 2, 2], "id": [0, 1, 1], "predicted_label": [0, 1, 0]},
        {},
        r"frame 2: .*two rows",
    ),
    "two rows of one pair at one frame": (
        {
            "frame": [3, 3, 3],
            "id1": [0, 1, 0],
            "id2": [1, 0, 1],
            "predicted_label": [0, 1, 0],
        },
        {},
        r"frame 3: .*two rows",
    ),
    "two scene rows at one frame": (
        {"frame": [1, 1], "predicted_label": [0, 1]},
        {},
        r"frame 1: .*two rows",
    ),
    "an empty predicted_label": (
        {"frame": [0, 1], "predicted_label": [0, np.nan]},
        {},
        r"frame 1: .*predicted_label",
    ),
    "a predicted_label of -1": (
        {"frame": [0, 1], "predicted_label": [-1, 0]},
        {},
        r"frame 0: .*-1",
    ),
    "a fractional predicted_label": (
        {"frame": [0, 1], "predicted_label": [0, 0.5]},
        {},
        r"frame 1: .*predicted_label",
    ),
    "a name without class_ids": (
        {"frame": [0, 1], "predicted_label": ["attack", "other"]},
        {},
        r"frame 0: .*predicted_label",
    ),
    "a name outside class_ids": (
        {"frame": [0, 1], "predicted_label": ["attack", "mount"]},
        {"class_ids": {"attack": 1}},
        r"frame 1: .*'mount'",
    ),
    "a predicted_label outside the probability classes": (
        _with(
            prob_0=[1.0, 0.0, 1.0], prob_1=[0.0, 1.0, 0.0], predicted_label=[0, 2, 0]
        ),
        {},
        r"frame 1: .*2",
    ),
    "a probability above 1": (
        _with(prob_0=[0.5, 1.5, 0.5], prob_1=[0.5, 0.5, 0.5]),
        {},
        r"frame 1: .*prob_0",
    ),
    "a negative probability": (
        _with(prob_0=[0.5, 0.5, -0.1], prob_1=[0.5, 0.5, 0.5]),
        {},
        r"frame 2: .*prob_0",
    ),
    "a NaN probability without a predicted_label": (
        _without(
            _with(prob_0=[0.5, np.nan, 0.5], prob_1=[0.5, 0.5, 0.5]), "predicted_label"
        ),
        {},
        r"frame 1: .*NaN",
    ),
    "two columns of one class": (
        _with(prob_1=[0.5, 0.5, 0.5], prob_01=[0.5, 0.5, 0.5]),
        {},
        r"frame 0: .*prob_1 and prob_01 name one class",
    ),
    "a textual probability": (
        _with(prob_0=["0.5", "high", "0.5"], prob_1=["0.5", "0.5", "0.5"]),
        {},
        r"frame 1: .*prob_0",
    ),
    "a prob_-1 column": (
        _with(**MINUS_ONE_COLUMN),
        {},
        r"frame 0: .*-1",
    ),
    "a prob_ column of no class": (
        _with(prob_0=[0.5, 0.5, 0.5], prob_attack=[0.5, 0.5, 0.5]),
        {},
        r"frame 0: .*prob_attack",
    ),
    "a prob_ column outside class_ids": (
        {
            "frame": [0],
            "predicted_label": ["a"],
            "prob_a": [0.5],
            "prob_0": [0.5],
        },
        {"class_ids": {"a": 0}},
        r"frame 0: .*prob_0",
    ),
    "two folds at one frame": (
        {
            "frame": [0, 0, 1],
            "id": [0, 1, 0],
            "predicted_label": [0, 1, 0],
            "fold": [0, 1, 1],
        },
        {},
        r"frame 0: .*fold",
    ),
    "a fold below -1": (_with(fold=[0, -2, 0]), {}, r"frame 1: .*fold"),
    "two blocks at one frame": (
        {
            "frame": [0, 1, 1],
            "id": [0, 0, 1],
            "predicted_label": [0, 1, 0],
            "fold": [0, 0, 0],
            "block": [0, 0, 1],
        },
        {},
        r"frame 1: .*block",
    ),
    "a block without a fold": (_with(block=[0, 0, 0]), {}, r"frame 0: .*block.*fold"),
    "a block on an in-sample row": (
        _with(fold=[0, -1, 0], block=[0, 0, 0]),
        {},
        r"frame 1: .*block",
    ),
    "an empty block of an out-of-fold row": (
        _with(fold=[0, 0, 0], block=[0, np.nan, 0]),
        {},
        r"frame 1: .*block",
    ),
    "a negative block": (
        _with(fold=[0, 0, 0], block=[0, -3, 0]),
        {},
        r"frame 1: .*block",
    ),
    "one fold and block in two runs": (
        {
            "frame": [0, 1, 2, 3],
            "predicted_label": [0, 1, 0, 1],
            "fold": [0, 1, 0, 0],
            "block": [0, 1, 0, 0],
        },
        {},
        r"frame 2: .*block",
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSED_TABLES))
@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_a_refused_table_names_its_file_sequence_and_first_offending_frame(
    tmp_path: Path, case: str, suffix: str
) -> None:
    table, params, message = REFUSED_TABLES[case]
    path = write_table(tmp_path, f"refused{suffix}", table)

    with pytest.raises(
        ValueError, match=rf"refused\{suffix}: sequence 's1', {message}"
    ):
        _ = convert(path, **params)


STRUCTURALLY_REFUSED: dict[str, tuple[Table, str]] = {
    "no rows": ({"frame": [], "predicted_label": []}, r"no rows"),
    "no frame": ({"predicted_label": [0]}, r"frame"),
    "no class": ({"frame": [0], "id": [0]}, r"predicted_label.*prob_"),
}


@pytest.mark.parametrize("case", sorted(STRUCTURALLY_REFUSED))
def test_a_file_without_rows_frames_or_classes_is_refused(
    tmp_path: Path, case: str
) -> None:
    table, message = STRUCTURALLY_REFUSED[case]
    path = write_table(tmp_path, "refused.csv", table)

    with pytest.raises(ValueError, match=rf"refused\.csv.*{message}"):
        _ = convert(path)


def test_a_row_of_no_sequence_is_refused(tmp_path: Path) -> None:
    path = write_table(tmp_path, "refused.csv", {"frame": [7], "predicted_label": [0]})

    with pytest.raises(
        ValueError, match=r"refused\.csv: sequence '', frame 7: .*sequence"
    ):
        _ = convert(path, {"group": "", "sequence": "", "abs_path": "refused.csv"})


def test_a_file_of_another_suffix_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "refused.tsv"
    _ = path.write_text("frame\tpredicted_label\n0\t0\n")

    with pytest.raises(ValueError, match=r"refused\.tsv.*\.parquet.*\.csv"):
        _ = convert(path)


@pytest.mark.parametrize(
    "class_ids", [{"a": -1}, {"a": -2}, {"a": 0, "b": 0}], ids=["-1", "-2", "shared"]
)
def test_class_ids_refuse_a_negative_id_and_two_names_of_one_id(
    class_ids: dict[str, int],
) -> None:
    with pytest.raises(ValidationError, match="class_ids"):
        _ = PredictionsTableParams(class_ids=class_ids)


# --- identity ------------------------------------------------------------------


def test_the_variant_id_moves_with_files_and_class_ids_only(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    def variant(**params: object) -> str:
        return ds.label_conversion_variant("predictions", "predictions_table", params)

    default = variant()
    assert re.fullmatch(r"convert-labels-predictions_table\.0\.1-[0-9a-f]{10}", default)
    assert variant(group_from="infile") == default
    assert variant(strict_schema=True) == default
    assert variant(files="labels_raw/predictions/a/*") != default
    assert variant(class_ids={"a": 0}) != default


def test_the_converter_is_registered_for_predictions() -> None:
    converter = get_label_converter("predictions_table", "predictions")

    assert isinstance(converter, PredictionsTableConverter)
    assert converter.refuses_shared_entry_keys
    assert converter.frames_on_tracks_axis


# --- conversion in a dataset ---------------------------------------------------


def predictions_dataset(tmp_path: Path, files: Mapping[str, Table]) -> Dataset:
    """A dataset whose ``labels_raw/predictions`` holds *files*, indexed."""
    ds = make_dataset(tmp_path / "ds")
    root = ds.get_root("labels_raw") / "predictions"
    for name, table in files.items():
        _ = write_table(root / Path(name).parent, Path(name).name, table)
    ds.index_labels_raw(
        [root],
        patterns=["*.parquet", "*.csv"],
        src_format="predictions_table",
    )
    return ds


def written(ds: Dataset) -> list[str]:
    root = ds.get_root("labels") / "predictions"
    return sorted(path.name for path in root.rglob("*.npz"))


def scene(sequence: str, *labels: int) -> Table:
    return {
        "sequence": [sequence] * len(labels),
        "frame": list(range(len(labels))),
        "predicted_label": list(labels),
    }


def test_files_select_one_of_two_sets_of_predictions(tmp_path: Path) -> None:
    ds = predictions_dataset(
        tmp_path,
        {"xgb/run.parquet": scene("s1", 0, 1), "own/run.csv": scene("s1", 1, 1, 1)},
    )
    before = tree_snapshot(ds.base_dir)
    selection: dict[str, object] = {"files": "labels_raw/predictions/xgb/*"}

    rows = ds.label_rows_to_convert("predictions", "predictions_table", selection)
    assert tree_snapshot(ds.base_dir) == before
    outcome = ds.convert_all_labels(
        kind="predictions", source_format="predictions_table", params=selection
    )

    assert rows["abs_path"].tolist() == ["labels_raw/predictions/xgb/run.parquet"]
    assert outcome.written == 1
    (path,) = read_labels_index(ds, "predictions")["abs_path"].tolist()
    with np.load(ds.resolve_path(str(path)), allow_pickle=True) as npz:
        assert np.asarray(npz["labels"]).tolist() == [0, 1]


def test_a_files_pattern_that_matches_no_row_is_refused_at_run(
    tmp_path: Path,
) -> None:
    ds = predictions_dataset(tmp_path, {"xgb/run.csv": scene("s1", 0)})

    with pytest.raises(
        ValueError, match=r"No rows in labels_raw/index\.csv.*'nothing/\*'"
    ):
        _ = ds.convert_all_labels(
            kind="predictions",
            source_format="predictions_table",
            params={"files": "nothing/*"},
        )

    assert written(ds) == []


def test_two_files_naming_one_sequence_are_refused_naming_both(
    tmp_path: Path,
) -> None:
    ds = predictions_dataset(
        tmp_path, {"a/first.csv": scene("s1", 0), "b/second.csv": scene("s1", 1)}
    )

    with pytest.raises(ValueError, match=r"'s1'.*first\.csv.*second\.csv"):
        _ = ds.convert_all_labels(kind="predictions", source_format="predictions_table")

    assert written(ds) == []


def test_one_file_naming_one_entry_key_twice_is_refused(tmp_path: Path) -> None:
    table: Table = {
        "group": ["a", "", "a"],
        "sequence": ["b", "a__b", "b"],
        "frame": [0, 0, 1],
        "predicted_label": [0, 1, 0],
    }
    ds = predictions_dataset(tmp_path, {"one/run.csv": table})

    with pytest.raises(ValueError, match=r"run\.csv.*run\.csv"):
        _ = ds.convert_all_labels(
            kind="predictions",
            source_format="predictions_table",
            params={"group_from": "infile"},
        )

    assert written(ds) == []


def test_a_file_of_several_sequences_writes_one_entry_each(tmp_path: Path) -> None:
    table = scene("s1", 0, 1)
    for column, values in scene("s2", 1).items():
        table[column] = [*table[column], *values]
    ds = predictions_dataset(tmp_path, {"one/run.parquet": table})

    outcome = ds.convert_all_labels(
        kind="predictions", source_format="predictions_table"
    )

    assert outcome.written == 2
    assert written(ds) == ["s1.npz", "s2.npz"]
