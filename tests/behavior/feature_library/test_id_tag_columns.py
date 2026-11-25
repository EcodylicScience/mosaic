"""``id-tag-columns`` attaches an id file's values as they are stored.

A file written by a direct ``save_id_labels`` call can hold NumPy scalars; each
attaches as the Python value it stores.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.behavior.feature_library.id_tag_columns import IdTagColumns
from mosaic.core.dataset import Dataset

from tests.helpers import make_dataset


def _attached(
    ds: Dataset, per_id_labels: dict[object, dict[str, object]]
) -> pd.DataFrame:
    """The columns ``id-tag-columns`` attaches to one entry whose ids are 0 and 1."""
    path = ds.save_id_labels("id_tags", "g", "s", per_id_labels)
    feature = IdTagColumns()
    feature.load_state(Path("."), {}, {"labels": {("g", "s"): path}})
    rows = pd.DataFrame({"group": "g", "sequence": "s", "frame": [0, 1], "id": [0, 1]})
    return feature.apply(rows)


def test_a_numpy_string_field_stays_text_and_attaches(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    out = _attached(
        ds,
        {0: {"genotype": np.str_("WT")}, 1: {"genotype": np.str_("KO")}},
    )

    assert out["genotype"].tolist() == ["WT", "KO"]
    assert all(type(value) is str for value in out["genotype"])


def test_a_numeric_looking_string_keeps_its_zeros(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    out = _attached(
        ds,
        {
            0: {"rfid": np.str_("000004395862")},
            1: {"rfid": np.str_("000004395863")},
        },
    )

    assert out["rfid"].tolist() == ["000004395862", "000004395863"]


def test_a_numpy_float_stays_a_float(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    out = _attached(
        ds, {0: {"weight": np.float64(2.5)}, 1: {"weight": np.float64(3.75)}}
    )

    assert out["weight"].tolist() == [2.5, 3.75]


def test_a_numpy_bool_stays_a_bool(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    out = _attached(ds, {0: {"focal": np.bool_(True)}, 1: {"focal": np.bool_(False)}})

    assert out["focal"].tolist() == [True, False]
    assert out["focal"].dtype == bool


def test_a_numpy_integer_becomes_an_int(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    out = _attached(ds, {0: {"cohort": np.int64(7)}, 1: {"cohort": np.int64(8)}})

    assert out["cohort"].tolist() == [7, 8]


def test_numpy_ids_key_the_tags(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    out = _attached(
        ds,
        {np.int64(0): {"genotype": "WT"}, np.int64(1): {"genotype": "KO"}},
    )

    assert out["genotype"].tolist() == ["WT", "KO"]


def test_a_csv_authored_file_attaches_as_before(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    csv = tmp_path / "meta.csv"
    csv.write_text(
        "group,sequence,id,genotype,weight,cohort\ng,s,0,WT,2.5,7\ng,s,1,KO,3.75,8\n"
    )
    (path,) = ds.convert_id_tags_from_csv(csv_path=csv, csv_type="multi")

    feature = IdTagColumns()
    feature.load_state(Path("."), {}, {"labels": {("g", "s"): path}})
    rows = pd.DataFrame({"group": "g", "sequence": "s", "frame": [0, 1], "id": [0, 1]})
    out = feature.apply(rows)

    assert out["cohort"].tolist() == [7, 8]
    assert out["genotype"].tolist() == ["WT", "KO"]
    assert out["weight"].tolist() == [2.5, 3.75]
