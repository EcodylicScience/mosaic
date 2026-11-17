"""A selection of entries by sequence tag lowers to a scope against one revision."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from mosaic.core.annotations.sequence_tags import (
    SequenceTagRevisionNotFound,
    SequenceTagTableUnavailable,
    read_sequence_tag_table,
    resolve_sequence_tag_revision,
)
from mosaic.core.annotations.tag_selection import (
    TagSelectionInvalid,
    selection_tag_table,
    tagged_recordings,
    tagged_scope,
)
from mosaic.core.annotations.tag_table import TagTable
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.file_digest import file_digest
from mosaic.core.pipeline.label_series import TAGS_SERIES
from mosaic.core.pipeline.label_series_index import series_index_path
from mosaic.core.pipeline.tag_reads import ResolvedTagTable
from mosaic.core.recording import RecordingKey
from mosaic.core.scope import Scope
from mosaic.core.typed_attribute import TypedAttributeType
from tests.helpers import (
    index_a_second_origin,
    MediaClip,
    make_dataset,
    tag_table,
    write_media_index,
    write_tags,
)

_ENTRIES: Final = (
    ("g1", "s1"),
    ("g1", "s2"),
    ("g1", "s3"),
    ("g2", "s4"),
    ("g2", "s5"),
    ("c", "c1"),
    ("c", "c2"),
    ("c", "c3"),
)

_TYPES: Final[dict[str, TypedAttributeType]] = {
    "site": "categorical",
    "n": "int",
    "code": "categorical",
    "flag": "label",
}


def _table() -> TagTable:
    return tag_table(
        {
            ("g1", "s1"): {"site": "a", "n": 3, "code": "03", "flag": None},
            ("g1", "s2"): {"site": "a", "n": 5, "code": "3"},
            ("g1", "s3"): {"site": "b", "n": 3},
            ("g2", "s4"): {"site": "a", "n": 5, "flag": None},
            ("g2", "s5"): {"site": "b"},
            ("c", "c1"): {"site": "x"},
            ("c", "c2"): {"site": "x"},
            ("c", "c3"): {"site": "y"},
        },
        _TYPES,
    )


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset,
        [
            MediaClip(group=group, sequence=sequence, filename=f"{sequence}.mp4")
            for group, sequence in _ENTRIES
        ],
    )
    dataset.manifest.continuous_groups = ("c",)
    return dataset


@pytest.fixture
def resolved(ds: Dataset) -> ResolvedTagTable:
    _ = write_tags(ds, _table())
    return selection_tag_table(ds)


class TestTheScope:
    def test_it_holds_the_selected_entries_sorted(
        self, resolved: ResolvedTagTable
    ) -> None:
        scope = tagged_scope(resolved, {"site": ["b", "a"]}, groups=["g1", "g2"])

        assert scope == Scope(
            entries=[
                ("g1", "s1"),
                ("g1", "s2"),
                ("g1", "s3"),
                ("g2", "s4"),
                ("g2", "s5"),
            ]
        )

    def test_names_combine_as_and_and_values_as_or(
        self, resolved: ResolvedTagTable
    ) -> None:
        scope = tagged_scope(resolved, {"site": ["a", "b"], "n": [3]})

        assert scope.entries == [("g1", "s1"), ("g1", "s3")]

    def test_a_label_takes_no_values_and_matches_where_assigned(
        self, resolved: ResolvedTagTable
    ) -> None:
        scope = tagged_scope(resolved, {"flag": []})

        assert scope.entries == [("g1", "s1"), ("g2", "s4")]

    def test_groups_keep_the_selected_entries_of_those_groups(
        self, resolved: ResolvedTagTable
    ) -> None:
        scope = tagged_scope(resolved, {"site": ["a"]}, groups=["g2"])

        assert scope.entries == [("g2", "s4")]

    def test_groups_that_hold_no_selected_entry_are_refused(
        self, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid, match="no entry") as raised:
            _ = tagged_scope(resolved, {"site": ["a"]}, groups=["c"])

        assert raised.value.error_kind == "invalid"


class TestTheValuesAreReadByTheTagsType:
    @pytest.mark.parametrize("text", ["3", "03"])
    def test_text_for_an_int_tag_selects_the_entries_holding_the_number(
        self, resolved: ResolvedTagTable, text: str
    ) -> None:
        scope = tagged_scope(resolved, {"n": [text]})

        assert scope.entries == [("g1", "s1"), ("g1", "s3")]

    def test_a_leading_zero_for_a_categorical_tag_is_kept(
        self, resolved: ResolvedTagTable
    ) -> None:
        assert tagged_scope(resolved, {"code": ["03"]}).entries == [("g1", "s1")]
        assert tagged_scope(resolved, {"code": ["3"]}).entries == [("g1", "s2")]

    def test_a_number_for_a_categorical_tag_is_refused(
        self, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid, match="'code'") as raised:
            _ = tagged_scope(resolved, {"code": [3]})

        assert raised.value.error_kind == "invalid"

    def test_text_that_is_not_a_number_is_refused_for_an_int_tag(
        self, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid, match="a whole number"):
            _ = tagged_scope(resolved, {"n": ["three"]})

    def test_an_unknown_tag_is_refused_listing_the_defined_names(
        self, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid, match="'nope'.*'site'"):
            _ = tagged_scope(resolved, {"nope": ["a"]})

    def test_a_label_given_values_is_refused(self, resolved: ResolvedTagTable) -> None:
        with pytest.raises(TagSelectionInvalid, match="takes no values"):
            _ = tagged_scope(resolved, {"flag": ["yes"]})

    def test_a_selection_of_no_tag_is_refused(self, resolved: ResolvedTagTable) -> None:
        with pytest.raises(
            TagSelectionInvalid, match="a tag selection names at least one tag"
        ) as raised:
            _ = tagged_scope(resolved, {})

        assert raised.value.error_kind == "invalid"


class TestTheRevision:
    def test_no_revision_indexed_says_how_to_import_tags(self, ds: Dataset) -> None:
        with pytest.raises(
            TagSelectionInvalid, match="mosaic sequence-tags import"
        ) as raised:
            _ = selection_tag_table(ds)

        assert raised.value.error_kind == "invalid"

    def test_a_revision_that_is_not_indexed_is_refused_naming_it(
        self, ds: Dataset
    ) -> None:
        _ = write_tags(ds, _table())

        with pytest.raises(TagSelectionInvalid, match="revision 7"):
            _ = selection_tag_table(ds, 7)

    def test_a_named_revision_is_the_one_read(self, ds: Dataset) -> None:
        _ = write_tags(ds, _table())
        _ = write_tags(
            ds, tag_table({("g1", "s1"): {"site": "z"}}, {"site": "categorical"})
        )

        assert selection_tag_table(ds).revision == 2
        first = selection_tag_table(ds, 1)

        assert first.revision == 1
        assert tagged_scope(first, {"site": ["a"]}).entries == [
            ("g1", "s1"),
            ("g1", "s2"),
            ("g2", "s4"),
        ]

    def test_the_resolver_returns_none_when_nothing_is_indexed(
        self, ds: Dataset
    ) -> None:
        assert resolve_sequence_tag_revision(ds, kind="test") is None
        assert read_sequence_tag_table(ds, kind="test") is None

    def test_the_resolver_raises_for_a_revision_that_is_not_indexed(
        self, ds: Dataset
    ) -> None:
        _ = write_tags(ds, _table())

        with pytest.raises(SequenceTagRevisionNotFound, match="revision 4"):
            _ = resolve_sequence_tag_revision(ds, 4, kind="test")
        with pytest.raises(SequenceTagRevisionNotFound, match="revision 4"):
            _ = read_sequence_tag_table(ds, 4, kind="test")

    def test_the_table_read_carries_its_revision_and_digest(self, ds: Dataset) -> None:
        saved = write_tags(ds, _table())

        read = read_sequence_tag_table(ds, kind="test")

        assert read is not None
        assert read.revision == 1
        assert read.path == saved.path
        assert read.payload_digest == file_digest(saved.path)
        assert read.origin_uuid == (ds.uuid or "")


def _missing_payload(ds: Dataset) -> None:
    index = pd.read_csv(
        series_index_path(ds, TAGS_SERIES), dtype=str, keep_default_na=False
    )
    ds.resolve_path(str(index["abs_path"].iloc[0])).unlink()


def _edited_payload(ds: Dataset) -> None:
    index = pd.read_csv(
        series_index_path(ds, TAGS_SERIES), dtype=str, keep_default_na=False
    )
    path = ds.resolve_path(str(index["abs_path"].iloc[0]))
    _ = path.write_bytes(path.read_bytes() + b" ")


def _unreadable_payload(ds: Dataset) -> None:
    path = series_index_path(ds, TAGS_SERIES)
    index = pd.read_csv(path, dtype=str, keep_default_na=False)
    payload = ds.resolve_path(str(index["abs_path"].iloc[0]))
    _ = payload.write_bytes(b"not a tag table")
    index["digest"] = file_digest(payload)
    index.to_csv(path, index=False)


class TestAnUnavailableTable:
    @pytest.mark.parametrize(
        "damage",
        [index_a_second_origin, _missing_payload, _edited_payload, _unreadable_payload],
        ids=["two-origins", "missing-payload", "edited-in-place", "unreadable"],
    )
    def test_each_state_is_a_conflict_that_names_the_reason(
        self, ds: Dataset, damage: Callable[[Dataset], object]
    ) -> None:
        _ = write_tags(ds, _table())
        damage(ds)

        for read in (
            lambda: selection_tag_table(ds),
            lambda: read_sequence_tag_table(ds, kind="test"),
        ):
            with pytest.raises(SequenceTagTableUnavailable) as raised:
                _ = read()
            assert isinstance(raised.value, ClassifiedError)
            assert raised.value.error_kind == "conflict"
            assert raised.value.error_reason() == "sequence_tags_unavailable"

    def test_the_unavailable_error_is_still_a_value_error(self) -> None:
        assert issubclass(SequenceTagTableUnavailable, ValueError)


class TestAnEmptySelection:
    def test_it_names_the_tag_the_values_and_the_revision(self, ds: Dataset) -> None:
        _ = write_tags(ds, tag_table({("g1", "s1"): {"site": "q"}}, _TYPES))
        _ = write_tags(ds, _table())
        read = selection_tag_table(ds)

        with pytest.raises(TagSelectionInvalid) as raised:
            _ = tagged_scope(read, {"site": ["zzz", "yyy"]})

        message = str(raised.value)
        assert "'site'" in message
        assert "'zzz'" in message
        assert "'yyy'" in message
        assert "revision 2" in message
        assert raised.value.error_kind == "invalid"

    def test_the_detail_holds_the_revision_and_the_tags(
        self, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid) as raised:
            _ = tagged_scope(resolved, {"site": ["zzz"], "flag": []})

        assert raised.value.error_detail() == {
            "revision": 1,
            "tags": {"site": ["zzz"], "flag": []},
        }


class TestTheRecordings:
    def test_each_undivided_entry_is_its_own_recording(
        self, ds: Dataset, resolved: ResolvedTagTable
    ) -> None:
        keys = tagged_recordings(ds, resolved, {"site": ["a"]})

        assert keys == (
            RecordingKey("g1", "s1"),
            RecordingKey("g1", "s2"),
            RecordingKey("g2", "s4"),
        )

    def test_a_divided_recording_covered_whole_is_one_key(
        self, ds: Dataset, resolved: ResolvedTagTable
    ) -> None:
        keys = tagged_recordings(ds, resolved, {"site": ["x", "y", "b"]})

        assert keys == (
            RecordingKey("c", ""),
            RecordingKey("g1", "s3"),
            RecordingKey("g2", "s5"),
        )

    def test_a_divided_recording_covered_in_part_is_refused(
        self, ds: Dataset, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid) as raised:
            _ = tagged_recordings(ds, resolved, {"site": ["x"]})

        message = str(raised.value)
        assert "recording 'c'" in message
        assert "'x'" in message
        assert "'y'" in message
        assert "c3" in message
        assert raised.value.error_kind == "invalid"

    def test_an_empty_selection_is_refused(
        self, ds: Dataset, resolved: ResolvedTagTable
    ) -> None:
        with pytest.raises(TagSelectionInvalid, match="revision 1"):
            _ = tagged_recordings(ds, resolved, {"site": ["zzz"]})

    def test_a_label_is_reported_as_assigned_or_not_assigned(self, ds: Dataset) -> None:
        _ = write_tags(
            ds,
            tag_table(
                {
                    ("c", "c1"): {"flag": None},
                    ("c", "c2"): {},
                    ("c", "c3"): {},
                },
                {"flag": "label"},
            ),
        )
        read = selection_tag_table(ds)

        with pytest.raises(TagSelectionInvalid) as raised:
            _ = tagged_recordings(ds, read, {"flag": []})

        message = str(raised.value)
        assert "c1=assigned" in message
        assert "c2=not assigned" in message
        assert "None" not in message
