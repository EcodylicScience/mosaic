"""A CSV of sequence tags becomes the next revision of the tag table, and the table lists."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.annotations import sequence_tags
from mosaic.core.annotations.sequence_tag_wire import (
    SequenceTagChange,
    SequenceTagListing,
    SequenceTagValueCount,
)
from mosaic.core.annotations.sequence_tags import (
    KeyForm,
    SequenceTagCsv,
    SequenceTagCsvInvalid,
    SequenceTagImportRefused,
    SequenceTagTableUnavailable,
    SequenceTagsMoved,
    import_sequence_tags,
    list_sequence_tags,
)
from mosaic.core.annotations.tag_table import (
    ArenaRow,
    IndividualRow,
    SequenceTags,
    TagDefinition,
    TagTable,
    read_tag_table,
)
from mosaic.core.dataset import Dataset
from mosaic.core.scan_findings import FindingLog
from mosaic.core.typed_attribute import TypedAttributeType
from mosaic.core.pipeline.label_series_index import (
    read_label_series,
    read_revision_manifest,
)
from tests.helpers import (
    MediaClip,
    make_dataset,
    modules_loaded_by,
    tag_table,
    write_media_index,
    write_tags,
)

_ENTRIES: Final = (("g1", "s1"), ("g1", "s2"), ("g2", "s3"), ("", "solo"))


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
    return dataset


def _csv(tmp_path: Path, text: str, name: str = "tags.csv") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def _table(ds: Dataset) -> TagTable:
    listing = list_sequence_tags(ds)
    assert listing.revision is not None
    index = read_label_series(ds, "tags")
    path = ds.resolve_path(
        str(index[index["revision"] == listing.revision]["abs_path"].iloc[0])
    )
    return read_tag_table(path)


def _tags_by_entry(table: TagTable) -> dict[tuple[str, str], dict[str, object]]:
    return {(s.group, s.sequence): dict(s.tags) for s in table.sequences}


def _lock_files(ds: Dataset) -> list[Path]:
    return sorted(ds.base_dir.rglob("*.lock"))


class TestTheFirstImport:
    def test_it_writes_the_first_revision_with_the_origin_of_the_file(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        path = _csv(tmp_path, "entry,site\ng1:s1,a\ng2:s3,b\n")

        result = import_sequence_tags(ds, path)

        assert (result.revision, result.written, result.dry_run) == (1, True, False)
        assert result.rows_matched == 2
        assert result.findings == ()
        index = read_label_series(ds, "tags")
        record = index.iloc[0]
        digest = read_revision_manifest(
            ds.resolve_path(str(record["abs_path"]))
        ).origin["sha256"]
        assert isinstance(digest, str)
        manifest = read_revision_manifest(ds.resolve_path(str(record["abs_path"])))
        assert manifest.origin == {
            "system": "mosaic-csv",
            "file": "tags.csv",
            "sha256": digest,
        }
        assert manifest.origin_ref == f"csv:{digest[:10]}"
        assert record["origin_ref"] == f"csv:{digest[:10]}"

    def test_the_table_lists_every_recording_entry_tagged_or_not(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))

        table = _table(ds)

        assert _tags_by_entry(table) == {
            ("g1", "s1"): {"site": "a"},
            ("g1", "s2"): {},
            ("g2", "s3"): {},
            ("", "solo"): {},
        }

    def test_the_changes_name_each_tag_with_its_counts(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,site,n\ng1:s1,a,1\ng1:s2,b,2\n")
        )

        assert result.changes == (
            SequenceTagChange(
                name="n",
                type="int",
                change="added",
                entries_set=2,
                entries_cleared=0,
            ),
            SequenceTagChange(
                name="site",
                type="categorical",
                change="added",
                entries_set=2,
                entries_cleared=0,
            ),
        )

    def test_a_row_that_names_no_entry_is_a_finding_and_the_rest_is_written(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,site\ng1:s1,a\nnope:x,b\n")
        )

        assert result.rows_matched == 1
        assert [(f.problem, f.line) for f in result.findings] == [("row_unmatched", 3)]
        assert _tags_by_entry(_table(ds))[("g1", "s1")] == {"site": "a"}


class TestAgain:
    def test_the_same_file_again_writes_nothing(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        path = _csv(tmp_path, "entry,site\ng1:s1,a\n")
        _ = import_sequence_tags(ds, path)

        again = import_sequence_tags(ds, path)

        assert (again.revision, again.written) == (1, False)
        assert {change.change for change in again.changes} == {"unchanged"}
        assert len(read_label_series(ds, "tags")) == 1

    def test_the_order_of_the_rows_does_not_change_the_revision(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        other = make_dataset(tmp_path / "other")
        write_media_index(
            other,
            [
                MediaClip(group=group, sequence=sequence, filename=f"{sequence}.mp4")
                for group, sequence in _ENTRIES
            ],
        )
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,k\ng1:s1,a\ng2:s3,b\n"))
        _ = import_sequence_tags(
            other, _csv(tmp_path, "entry,k\ng2:s3,b\ng1:s1,a\n", "other.csv")
        )

        def payload(dataset: Dataset) -> bytes:
            index = read_label_series(dataset, "tags")
            return dataset.resolve_path(str(index["abs_path"].iloc[0])).read_bytes()

        assert payload(ds) == payload(other)


class TestASecondFile:
    def test_a_column_the_file_does_not_name_keeps_its_values(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(
            ds, _csv(tmp_path, "entry,genotype\ng1:s1,wt\ng1:s2,ko\n")
        )

        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,sex\ng1:s1,f\n", "sex.csv")
        )

        assert result.revision == 2
        assert _tags_by_entry(_table(ds))[("g1", "s1")] == {
            "genotype": "wt",
            "sex": "f",
        }
        by_name = {change.name: change for change in result.changes}
        assert by_name["genotype"].change == "unchanged"
        assert by_name["sex"].change == "added"

    def test_replace_removes_the_tags_the_file_does_not_name(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(
            ds, _csv(tmp_path, "entry,genotype\ng1:s1,wt\ng1:s2,ko\n")
        )

        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,sex\ng1:s1,f\n", "sex.csv"), replace=True
        )

        table = _table(ds)
        assert [d.name for d in table.definitions.sequence] == ["sex"]
        assert _tags_by_entry(table)[("g1", "s2")] == {}
        removed = next(c for c in result.changes if c.name == "genotype")
        assert (removed.change, removed.entries_cleared) == ("removed", 2)

    def test_a_column_the_file_names_is_the_whole_truth_for_it(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\ng1:s2,b\n"))

        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,site\ng1:s1,c\n", "site.csv")
        )

        assert _tags_by_entry(_table(ds))[("g1", "s2")] == {}
        change = result.changes[0]
        assert (change.change, change.entries_set, change.entries_cleared) == (
            "changed",
            1,
            1,
        )


class TestKeptRows:
    def test_individual_and_arena_rows_are_kept_unchanged(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        base = tag_table({("g1", "s1"): {}}, {})
        row = IndividualRow(id=1, individual="fish 1", uid="F1", tags={"focal": True})
        arena = ArenaRow(
            arena="left", ordinal=0, individual="fish 1", uid="F1", tags={}
        )
        seeded = replace(
            base,
            definitions=replace(
                base.definitions,
                individual=(
                    *base.definitions.individual,
                    *_individual_definition(),
                ),
            ),
            sequences=(
                SequenceTags(
                    group="g1",
                    sequence="s1",
                    tags={},
                    individuals={"ultralytics.0.4-1a2b3c4d5e": (row,)},
                    arenas=(arena,),
                ),
            ),
        )
        _ = write_tags(ds, seeded)

        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))

        table = _table(ds)
        kept = next(s for s in table.sequences if s.sequence == "s1")
        assert kept.individuals == {"ultralytics.0.4-1a2b3c4d5e": (row,)}
        assert kept.arenas == (arena,)
        assert table.definitions.individual == seeded.definitions.individual


def _individual_definition() -> tuple[TagDefinition, ...]:
    return (TagDefinition("focal", "bool"),)


class TestTypes:
    def test_a_type_declared_changes_the_type_of_a_defined_tag(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,n\ng1:s1,1\ng1:s2,2\n"))
        assert {d.name: d.type for d in _table(ds).definitions.sequence} == {"n": "int"}

        result = import_sequence_tags(
            ds,
            _csv(tmp_path, "entry,n\ng1:s1,1\ng1:s2,2\n", "n.csv"),
            types={"n": "float"},
        )

        assert {d.name: d.type for d in _table(ds).definitions.sequence} == {
            "n": "float"
        }
        assert result.changes[0].change == "changed"
        assert result.changes[0].type == "float"

    def test_a_defined_type_is_kept_without_a_declaration(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(
            ds, _csv(tmp_path, "entry,n\ng1:s1,a\n"), types={"n": "text"}
        )

        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,n\ng1:s1,1\n", "n.csv"))

        assert {d.name: d.type for d in _table(ds).definitions.sequence} == {
            "n": "text"
        }
        assert _tags_by_entry(_table(ds))[("g1", "s1")] == {"n": "1"}

    def test_an_invalid_value_refuses_before_anything_is_written(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="line 3, column 'n'"):
            _ = import_sequence_tags(
                ds, _csv(tmp_path, "entry,n\ng1:s1,1\ng1:s2,x\n"), types={"n": "int"}
            )

        assert not (ds.get_root("labels_raw") / "tags").exists()


class TestARevisionWrittenMeanwhile:
    def test_it_refuses_and_writes_nothing(
        self, ds: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))
        reader = sequence_tags.read_sequence_tag_csv

        def racing(
            dataset: Dataset,
            path: Path,
            *,
            key: tuple[str, KeyForm] | None,
            types: Mapping[str, TypedAttributeType],
            kept: Mapping[str, TypedAttributeType],
            findings: FindingLog,
        ) -> SequenceTagCsv:
            read = reader(
                dataset, path, key=key, types=types, kept=kept, findings=findings
            )
            _ = write_tags(
                ds, tag_table({("g1", "s1"): {"other": "x"}}, {"other": "text"})
            )
            return read

        monkeypatch.setattr(sequence_tags, "read_sequence_tag_csv", racing)

        with pytest.raises(SequenceTagsMoved, match="revision 1.*revision 2"):
            _ = import_sequence_tags(
                ds, _csv(tmp_path, "entry,site\ng1:s1,b\n", "b.csv")
            )

        assert len(read_label_series(ds, "tags")) == 2
        assert list_sequence_tags(ds).revision == 2


class TestARevisionOfAnotherOrigin:
    def _saved_elsewhere(self, ds: Dataset) -> None:
        _ = write_tags(
            ds,
            tag_table({("g1", "s1"): {"genotype": "wt"}}, {"genotype": "categorical"}),
        )
        ds.manifest.uuid = "another-uuid"

    def test_the_import_is_refused_and_writes_nothing(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        self._saved_elsewhere(ds)

        with pytest.raises(SequenceTagImportRefused, match="another dataset"):
            _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))

        assert len(read_label_series(ds, "tags")) == 1

    def test_the_listing_shows_the_revision_that_runs_read(self, ds: Dataset) -> None:
        self._saved_elsewhere(ds)

        listing = list_sequence_tags(ds)

        assert listing.revision == 1
        assert [tag.name for tag in listing.tags] == ["genotype"]

    def test_two_origins_refuse_the_import_and_fail_the_listing(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        self._saved_elsewhere(ds)
        _ = write_tags(ds, tag_table({("g1", "s1"): {"n": 1}}, {"n": "int"}))

        with pytest.raises(SequenceTagImportRefused, match="indexed from 2 datasets"):
            _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))
        with pytest.raises(
            SequenceTagTableUnavailable, match="indexed from 2 datasets"
        ):
            _ = list_sequence_tags(ds)

        assert len(read_label_series(ds, "tags")) == 2


class TestADryRun:
    def test_it_reports_the_changes_and_writes_nothing(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        locks = _lock_files(ds)

        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"), dry_run=True
        )

        assert (result.revision, result.written, result.dry_run) == (None, False, True)
        assert [(c.name, c.change) for c in result.changes] == [("site", "added")]
        assert not (ds.get_root("labels_raw") / "tags").exists()
        assert _lock_files(ds) == locks

    def test_it_leaves_an_existing_series_as_it_was(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))
        locks = _lock_files(ds)

        result = import_sequence_tags(
            ds, _csv(tmp_path, "entry,site\ng1:s1,b\n", "b.csv"), dry_run=True
        )

        assert (result.revision, result.written) == (1, False)
        assert result.changes[0].change == "changed"
        assert list_sequence_tags(ds).revision == 1
        assert _lock_files(ds) == locks


class TestRefusals:
    def test_a_managed_dataset_is_refused(self, ds: Dataset, tmp_path: Path) -> None:
        ds.manifest.preserved["managed_by"] = "mosaic-api"

        with pytest.raises(SequenceTagImportRefused, match="managed"):
            _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))

        assert not (ds.get_root("labels_raw") / "tags").exists()

    def test_a_revision_that_names_a_project_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = write_tags(
            ds,
            tag_table({("g1", "s1"): {}}, {}),
            origin={"system": "mosaic-api", "project_id": "p-1"},
        )

        with pytest.raises(SequenceTagImportRefused, match="project_id"):
            _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))

        assert list_sequence_tags(ds).revision == 1


class TestListing:
    def test_a_dataset_without_a_revision_lists_nothing(self, ds: Dataset) -> None:
        listing = list_sequence_tags(ds)

        assert listing == SequenceTagListing(
            revision=None, digest=None, entries=0, tags=()
        )
        assert not (ds.get_root("labels_raw") / "tags").exists()

    def test_it_counts_entries_per_value_sorted_by_name_and_value(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(
            ds,
            _csv(
                tmp_path,
                "entry,site,flag,n\ng1:s1,b,x,2\ng1:s2,a,x,10\ng2:s3,b,,10\nsolo,,,\n",
            ),
            types={"flag": "label"},
        )

        listing = list_sequence_tags(ds)

        assert listing.revision == 1
        assert listing.entries == 4
        assert listing.digest is not None
        assert [(t.name, t.type, t.entries) for t in listing.tags] == [
            ("flag", "label", 2),
            ("n", "int", 3),
            ("site", "categorical", 3),
        ]
        by_name = {tag.name: tag.values for tag in listing.tags}
        assert by_name["flag"] == (SequenceTagValueCount(value=True, entries=2),)
        assert by_name["n"] == (
            SequenceTagValueCount(value=2, entries=1),
            SequenceTagValueCount(value=10, entries=2),
        )
        assert by_name["site"] == (
            SequenceTagValueCount(value="a", entries=1),
            SequenceTagValueCount(value="b", entries=2),
        )

    def test_it_reads_the_latest_revision_and_writes_nothing(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        _ = import_sequence_tags(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n"))
        locks = _lock_files(ds)
        index = ds.get_root("labels_raw") / "tags" / "index.csv"
        before = index.read_bytes()

        listing = list_sequence_tags(ds)

        assert listing.revision == 1
        assert index.read_bytes() == before
        assert _lock_files(ds) == locks


def test_the_wire_models_load_neither_pandas_nor_the_pipeline() -> None:
    loaded = modules_loaded_by(
        "import mosaic.core.annotations.sequence_tag_wire",
        ("pandas", "mosaic.core.pipeline", "mosaic.core.dataset"),
    )

    assert loaded == []
