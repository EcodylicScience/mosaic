"""The ``lmt_sqlite`` ``id_tags`` converter writes a database's animals as individual tags.

Each test writes a Live Mouse Tracker database, indexes it as a labels source and
converts it through a dataset. The files load without unpickling, and
``id-tag-columns`` attaches their text to the tracks of the same entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mosaic.behavior.feature_library.id_tag_columns import IdTagColumns
from mosaic.behavior.label_library.lmt_sqlite import LmtAnimalsConverter
from mosaic.core.dataset import Dataset
from mosaic.core.label_converter import LABEL_CONVERTERS, LabelConvertParams
from mosaic.core.pipeline.labels_index import read_labels_index
from mosaic.core.pipeline.run import run_feature
from tests.helpers import (
    DEFAULT_ANIMAL_COLUMNS,
    NINE_ANIMAL_COLUMNS,
    LmtAnimal,
    LmtRecording,
    default_lmt_recording,
    file_digests,
    make_dataset,
    result_outputs,
    write_lmt_sqlite,
)

_RFIDS = ["000004395862", "000004395870", "000004395881"]


def _lmt_labels_dataset(
    base: Path,
    recording: LmtRecording | None = None,
    *,
    animal_columns: Sequence[str] = DEFAULT_ANIMAL_COLUMNS,
    tracks: bool = False,
) -> Dataset:
    """A dataset whose labels source, and with *tracks* its tracks source, is ``raw/``."""
    ds = make_dataset(base)
    raw = ds.base_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    recording = recording or default_lmt_recording()
    _ = write_lmt_sqlite(
        raw / "recording.sqlite",
        animals=recording.animals,
        frames=recording.frames,
        events=recording.events,
        detections=recording.detections,
        log=recording.log,
        animal_columns=animal_columns,
    )
    _ = ds.index_labels_raw([raw], patterns=["*.sqlite"], src_format="lmt_sqlite")
    if tracks:
        _ = ds.index_tracks_raw([raw], patterns=["*.sqlite"], src_format="lmt_sqlite")
        assert ds.convert_all_tracks().failed == 0
    return ds


def _convert(ds: Dataset) -> Path:
    outcome = ds.convert_all_labels(kind="id_tags", source_format="lmt_sqlite")
    assert outcome.written == 1
    index = read_labels_index(ds, "id_tags")
    (path,) = index["abs_path"].tolist()
    return ds.resolve_path(str(path))


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_the_converter_declares_its_contract() -> None:
    converter = LABEL_CONVERTERS[("lmt_sqlite", "id_tags")]

    assert converter is LmtAnimalsConverter
    assert converter.label_format == "id_tags_v1"
    assert converter.version == "0.1"
    assert converter.Params is LabelConvertParams
    assert converter.refuses_shared_entry_keys


def test_the_animals_are_written_as_text_arrays_keyed_by_id(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")

    arrays = _arrays(_convert(ds))

    assert sorted(arrays) == ["genotype", "ids", "name", "rfid"]
    assert arrays["ids"].dtype == np.int64
    assert arrays["ids"].tolist() == [1, 2, 3]
    assert arrays["rfid"].dtype.kind == "U"
    assert arrays["rfid"].tolist() == _RFIDS
    assert arrays["name"].tolist() == ["mouse one", "mouse two", "mouse three"]


def test_a_null_cell_is_empty_text(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")

    genotype = _arrays(_convert(ds))["genotype"]

    assert genotype.dtype.kind == "U"
    assert genotype.tolist() == ["WT", "KO", ""]


def test_the_ids_ascend_whatever_the_row_order(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    reordered = replace(recording, animals=list(reversed(recording.animals)))
    ds = _lmt_labels_dataset(tmp_path / "ds", reordered)

    arrays = _arrays(_convert(ds))

    assert arrays["ids"].tolist() == [1, 2, 3]
    assert arrays["rfid"].tolist() == _RFIDS


def test_the_nine_column_table_writes_nine_arrays(tmp_path: Path) -> None:
    recording = default_lmt_recording()
    extras = {
        "AGE": "12",
        "SEX": "male",
        "STRAIN": "C57",
        "SETUP": "setup 2",
        "TREATMENT": None,
    }
    animals: list[LmtAnimal] = [
        replace(animal, extra=extras) for animal in recording.animals
    ]
    ds = _lmt_labels_dataset(
        tmp_path / "ds",
        replace(recording, animals=animals),
        animal_columns=NINE_ANIMAL_COLUMNS,
    )

    arrays = _arrays(_convert(ds))

    assert sorted(arrays) == [
        "age",
        "genotype",
        "ids",
        "name",
        "rfid",
        "setup",
        "sex",
        "strain",
        "treatment",
    ]
    assert arrays["age"].tolist() == ["12", "12", "12"]
    assert arrays["setup"].tolist() == ["setup 2"] * 3
    assert arrays["treatment"].tolist() == ["", "", ""]
    assert arrays["rfid"].tolist() == _RFIDS


def test_the_index_row_records_the_format_and_the_vocabulary(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    _ = _convert(ds)

    (row,) = (row for _, row in read_labels_index(ds, "id_tags").iterrows())

    assert row["label_format"] == "id_tags_v1"
    assert row["label_kind"] == "id_tags"
    assert (row["group"], row["sequence"]) == ("", "recording")
    assert row["n_frames"] == "3"
    assert row["label_ids"] == "1,2,3"
    assert row["label_names"] == "genotype,name,rfid"


def test_id_tag_columns_attaches_the_rfid_as_text_with_its_zeros(
    tmp_path: Path,
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds", tracks=True)
    _ = _convert(ds)

    result = run_feature(ds, IdTagColumns())

    (table,) = result_outputs(ds, result).values()
    by_id = table.groupby("id")[["rfid", "genotype"]].first()
    assert by_id["rfid"].tolist() == _RFIDS
    assert all(type(value) is str for value in by_id["rfid"])
    assert by_id["genotype"].tolist() == ["WT", "KO", ""]


def test_conversion_leaves_the_database_unchanged(tmp_path: Path) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    path = raw / "recording.sqlite"
    digests = file_digests(raw)
    mtime_ns = path.stat().st_mtime_ns

    _ = _convert(ds)

    assert file_digests(raw) == digests
    assert path.stat().st_mtime_ns == mtime_ns
    assert sorted(item.name for item in raw.iterdir()) == ["recording.sqlite"]


@pytest.mark.parametrize("suffix", ["-journal", "-wal"])
def test_a_journal_sibling_refuses_the_conversion_and_writes_nothing(
    tmp_path: Path, suffix: str
) -> None:
    ds = _lmt_labels_dataset(tmp_path / "ds")
    sibling = ds.base_dir / "raw" / f"recording.sqlite{suffix}"
    _ = sibling.write_bytes(b"")

    with pytest.raises(ValueError, match=f"recording.sqlite{suffix}"):
        _ = ds.convert_all_labels(kind="id_tags", source_format="lmt_sqlite")

    assert read_labels_index(ds, "id_tags").empty
