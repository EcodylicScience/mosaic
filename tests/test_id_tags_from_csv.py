"""Per-individual metadata read out of a user CSV into ``id_tags`` files.

The identity columns of that CSV key a ``groupby`` and name the file it writes,
and ``groupby`` drops a row whose key is missing. A blank ``group`` -- which is
what every dataset the control plane creates has -- was therefore removed before
the loop body ran, so the sequences that needed tags got none and the only
symptom was a lower "Created N id_tags files" count.

The rest of the CSV is deliberately *not* read as text: ``id`` becomes the key
of the ``.npz`` that a tracks table's integer ``id`` column is looked up in, and
a blank ``focal_id`` has to stay missing rather than become an empty string.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mosaic.core.dataset import Dataset

from tests.helpers import make_dataset


def _csv(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n")
    return path


def _tag_files(ds: Dataset) -> list[str]:
    root = ds.get_root("labels") / "id_tags"
    return sorted(p.name for p in root.glob("*.npz")) if root.exists() else []


def _tags(path: Path) -> dict[object, dict[str, object]]:
    """The ``.npz`` payload back as ``{id: {field: value}}``."""
    with np.load(path, allow_pickle=True) as npz:
        ids = list(npz["ids"])
        fields = [
            key for key in npz.files if key != "ids" and not key.startswith("meta__")
        ]
        return {
            identity: {field: npz[field][position] for field in fields}
            for position, identity in enumerate(ids)
        }


# --- the rows a blank group used to take with it ------------------------------


def test_a_blank_group_still_gets_its_tags(tmp_path: Path) -> None:
    """The common case: no group column filled in, one file per sequence."""
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "cats.csv",
        """
group,sequence,id,category
,s1,0,resident
,s1,1,intruder
,s2,0,resident
""",
    )

    created = ds.convert_id_tags_from_csv(csv_path=csv, csv_type="category")

    assert len(created) == 2
    assert _tag_files(ds) == ["s1.npz", "s2.npz"]


def test_several_fields_still_reach_a_blank_group(tmp_path: Path) -> None:
    """The second groupby, which dropped the same rows for the same reason."""
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "meta.csv",
        """
group,sequence,id,strain,sex
,s1,0,A,f
,s1,1,B,m
""",
    )

    created = ds.convert_id_tags_from_csv(
        csv_path=csv, csv_type="multi", field_columns=["strain", "sex"]
    )

    assert len(created) == 1
    assert _tag_files(ds) == ["s1.npz"]
    assert _tags(created[0])[0]["strain"] == "A"


def test_a_group_that_is_named_still_names_the_file(tmp_path: Path) -> None:
    """The path that already worked, so the fix is not a trade."""
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "cats.csv",
        """
group,sequence,id,category
cohortA,s1,0,resident
,s2,0,resident
""",
    )

    created = ds.convert_id_tags_from_csv(csv_path=csv, csv_type="category")

    assert len(created) == 2
    assert _tag_files(ds) == ["cohortA__s1.npz", "s2.npz"]


# --- what the identity columns are allowed to be ------------------------------


def test_a_numeric_sequence_name_keeps_its_zeros(tmp_path: Path) -> None:
    """``001`` inferred as the integer 1 names an entry no tracks table has.

    The numeric names are the CalMS21 and MABe convention, so this is reachable
    rather than theoretical.
    """
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "cats.csv",
        """
group,sequence,id,category
,001,0,resident
,002,0,intruder
""",
    )

    _ = ds.convert_id_tags_from_csv(csv_path=csv, csv_type="category")

    assert _tag_files(ds) == ["001.npz", "002.npz"]


def test_a_focal_row_names_its_sequence_as_written(tmp_path: Path) -> None:
    """The per-row branch reads the same two cells and must answer alike."""
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "focal.csv",
        """
group,sequence,focal_id
,001,1
""",
    )

    created = ds.convert_id_tags_from_csv(
        csv_path=csv, csv_type="focal", all_ids=[0, 1]
    )

    assert _tag_files(ds) == ["001.npz"]
    tags = _tags(created[0])
    assert tags[0]["focal"] is False
    assert tags[1]["focal"] is True


def test_a_row_naming_no_sequence_is_refused(tmp_path: Path) -> None:
    """Refused rather than dropped, and refused rather than written unnamed.

    Once a blank key no longer removes the row, a blank *sequence* would reach
    ``make_entry_key("", "")`` and write ``.npz`` with no stem. Saying so beats
    trading one silent outcome for another.
    """
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "cats.csv",
        """
group,sequence,id,category
,s1,0,resident
,,0,intruder
""",
    )

    with pytest.raises(ValueError, match="no sequence named"):
        _ = ds.convert_id_tags_from_csv(csv_path=csv, csv_type="category")

    assert _tag_files(ds) == [], "nothing is written when a row is refused"


# --- what the rest of the CSV must keep being ---------------------------------


def test_an_id_stays_the_number_it_was(tmp_path: Path) -> None:
    """Why only the identity columns are pinned as text.

    ``id`` becomes the ``.npz`` key that ``id-tag-columns`` looks up against a
    tracks table's integer ``id`` column, so reading the whole CSV as strings
    would attach nothing while appearing to succeed. This is green today and
    exists to go red if that read is ever widened.
    """
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "cats.csv",
        """
group,sequence,id,category
,s1,0,resident
,s1,1,intruder
""",
    )

    created = ds.convert_id_tags_from_csv(csv_path=csv, csv_type="category")

    assert sorted(_tags(created[0]).keys()) == [0, 1]


def test_a_missing_focal_id_is_left_missing(tmp_path: Path) -> None:
    """The other reason the read is narrow: a blank must not become ``""``.

    ``int("")`` raises, so a whole-CSV text read would turn a sequence with no
    focal individual recorded into a crash.
    """
    ds = make_dataset(tmp_path / "ds")
    csv = _csv(
        tmp_path / "focal.csv",
        """
group,sequence,focal_id
,s1,1
,s2,
""",
    )

    created = ds.convert_id_tags_from_csv(
        csv_path=csv, csv_type="focal", all_ids=[0, 1]
    )

    assert len(created) == 2
    by_name = {path.name: path for path in created}
    assert all(value["focal"] is False for value in _tags(by_name["s2.npz"]).values())
