"""A tag table is saved as a revision of the ``tags`` series, and only a change
makes a new one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from mosaic.core.annotations.projection import (
    TAG_TABLE_KEY,
    TAGS_SERIES,
    write_tag_table_revision,
)
from mosaic.core.annotations.tag_table import read_tag_table, tag_table_payload
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    read_revision_manifest,
)
from mosaic.core.typed_attribute import TypedAttributeType
from tests.helpers import make_dataset, tag_table

_TYPES: Final[dict[str, TypedAttributeType]] = {"condition": "categorical"}


def test_a_first_save_writes_the_payload_and_its_manifest(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    table = tag_table(
        {
            ("g", "s1"): {"condition": "treated"},
            ("g", "s2"): {"condition": "control"},
            ("h", "s3"): {"condition": "treated"},
        },
        _TYPES,
    )

    saved = write_tag_table_revision(ds, table=table, origin={"source": "sheet.csv"})

    revision_dir = ds.get_root("labels_raw") / "tags" / "dataset" / "rev1"
    assert (TAGS_SERIES, TAG_TABLE_KEY) == ("tags", "dataset")
    assert (saved.revision, saved.written) == (1, True)
    assert saved.path == revision_dir / "tags.mosaic.json"
    assert saved.path.read_bytes() == tag_table_payload(table)
    assert read_tag_table(saved.path) == table
    assert json.loads((revision_dir / "manifest.json").read_text())["origin"] == {
        "source": "sheet.csv"
    }
    assert read_revision_manifest(saved.path).origin == {"source": "sheet.csv"}
    index = read_label_series(ds, TAGS_SERIES)
    assert len(index) == 1 and int(index["n_records"].iloc[0]) == 3


def test_a_save_of_the_same_table_writes_nothing(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    table = tag_table({("g", "s1"): {"condition": "treated"}}, _TYPES)
    first = write_tag_table_revision(ds, table=table, origin={"a": 1})

    again = write_tag_table_revision(ds, table=table, origin={"a": 2})

    assert (again.revision, again.written) == (1, False)
    assert again.digest == first.digest
    assert not (first.path.parent.parent / "rev2").exists()


def test_a_changed_table_is_the_next_revision(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "52", name="project")
    first = write_tag_table_revision(
        ds,
        table=tag_table({("g", "s1"): {"condition": "treated"}}, _TYPES),
        origin={},
    )
    before = first.path.read_bytes()

    second = write_tag_table_revision(
        ds,
        table=tag_table({("g", "s1"): {"condition": "control"}}, _TYPES),
        origin={},
    )

    assert (second.revision, second.written) == (2, True)
    assert second.path == first.path.parent.parent / "rev2" / "tags.mosaic.json"
    assert first.path.read_bytes() == before
