"""``mosaic sequence-tags``: import a CSV of sequence tags and list the table's tags."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.annotations.sequence_tag_wire import (
    SequenceTagImport,
    SequenceTagListing,
)
from mosaic.core.dataset import Dataset
from tests.helpers import MediaClip, invoke_ok, make_dataset, write_media_index

_ENTRIES = (("g1", "s1"), ("g1", "s2"), ("g2", "s3"))


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


def _csv(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "tags.csv"
    _ = path.write_text(text, encoding="utf-8")
    return path


def _import(ds: Dataset, csv: Path, *options: str) -> list[str]:
    return [
        "sequence-tags",
        "import",
        str(csv),
        "-m",
        str(ds.manifest_path),
        *options,
    ]


def test_an_import_prints_its_changes_and_a_second_import_says_nothing_changed(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,site,age\ng1:s1,a,3\ng1:s2,b,4\n")

    first = invoke_ok(_import(ds, csv))
    second = invoke_ok(_import(ds, csv))

    assert "Wrote revision 1" in first.stdout
    assert "added" in first.stdout and "site" in first.stdout and "age" in first.stdout
    assert "Nothing changed" in second.stdout
    assert "revision 1" in second.stdout


def test_json_output_validates_as_the_import_report(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\ng2:s3,b\n")

    result = invoke_ok(_import(ds, csv, "--json"))

    report = SequenceTagImport.model_validate_json(result.stdout)
    assert (report.revision, report.written, report.dry_run) == (1, True, False)
    assert report.rows_matched == 2
    assert [change.name for change in report.changes] == ["site"]


def test_an_unmatched_row_prints_its_warning_and_exits_zero(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\ng9:nope,b\n")

    result = invoke_ok(_import(ds, csv, "--json"))

    report = SequenceTagImport.model_validate_json(result.stdout)
    assert [finding.problem for finding in report.findings] == ["row_unmatched"]
    assert "[WARN]" in result.stderr and "g9:nope" in result.stderr
    assert report.written


def test_a_type_refusal_exits_nonzero_naming_the_line_and_the_column(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,n\ng1:s1,1\ng1:s2,two\n")

    result = CliRunner().invoke(app, _import(ds, csv, "--type", "n=int"))

    assert result.exit_code != 0
    assert "line 3" in result.stderr and "'n'" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_refused_key_exits_nonzero_with_the_message(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,sequence,site\ng1:s1,s1,a\n")

    result = CliRunner().invoke(app, _import(ds, csv))

    assert result.exit_code != 0
    assert "entry" in result.stderr and "sequence" in result.stderr


def test_a_malformed_key_option_exits_nonzero(ds: Dataset, tmp_path: Path) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\n")

    result = CliRunner().invoke(app, _import(ds, csv, "--key", "entry"))

    assert result.exit_code == 2
    assert "--key" in result.stderr and "NAME=FORM" in result.stderr


def test_a_malformed_type_option_exits_nonzero(ds: Dataset, tmp_path: Path) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\n")

    result = CliRunner().invoke(app, _import(ds, csv, "--type", "site"))

    assert result.exit_code == 2
    assert "--type" in result.stderr and "NAME=TYPE" in result.stderr


def test_an_unknown_type_name_exits_two(ds: Dataset, tmp_path: Path) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\n")

    result = CliRunner().invoke(app, _import(ds, csv, "--type", "site=colour"))

    assert result.exit_code == 2
    assert "--type" in result.stderr and "categorical" in result.stderr


def test_key_and_type_options_name_the_column_and_the_type(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "seq,n\ns1,1\ns2,2\n")

    result = invoke_ok(
        _import(ds, csv, "--key", "seq=sequence", "--type", "n=float", "--json")
    )

    report = SequenceTagImport.model_validate_json(result.stdout)
    assert [(change.name, change.type) for change in report.changes] == [("n", "float")]


def test_a_missing_csv_exits_nonzero(ds: Dataset, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, _import(ds, tmp_path / "absent.csv"))

    assert result.exit_code != 0
    assert "absent.csv" in result.stderr


def test_a_managed_dataset_is_refused_with_the_message(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\n")
    ds.manifest.preserved["managed_by"] = "mosaic-api"
    ds.save()

    result = CliRunner().invoke(app, _import(ds, csv))

    assert result.exit_code != 0
    assert "managed" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_dry_run_prints_the_changes_and_writes_nothing(
    ds: Dataset, tmp_path: Path
) -> None:
    csv = _csv(tmp_path, "entry,site\ng1:s1,a\n")

    result = invoke_ok(_import(ds, csv, "--dry-run"))
    listing = SequenceTagListing.model_validate_json(
        invoke_ok(
            ["sequence-tags", "list", "-m", str(ds.manifest_path), "--json"]
        ).stdout
    )

    assert "Dry run" in result.stdout and "site" in result.stdout
    assert listing.revision is None


def test_replace_removes_the_tags_the_csv_does_not_name(
    ds: Dataset, tmp_path: Path
) -> None:
    _ = invoke_ok(_import(ds, _csv(tmp_path, "entry,site\ng1:s1,a\n")))
    other = tmp_path / "other.csv"
    _ = other.write_text("entry,age\ng1:s1,3\n", encoding="utf-8")

    result = invoke_ok(_import(ds, other, "--replace", "--json"))

    report = SequenceTagImport.model_validate_json(result.stdout)
    assert {change.name: change.change for change in report.changes} == {
        "age": "added",
        "site": "removed",
    }


def test_list_json_validates_as_the_listing(ds: Dataset, tmp_path: Path) -> None:
    _ = invoke_ok(_import(ds, _csv(tmp_path, "entry,site\ng1:s1,a\ng1:s2,a\n")))

    result = invoke_ok(["sequence-tags", "list", "-m", str(ds.manifest_path), "--json"])

    listing = SequenceTagListing.model_validate_json(result.stdout)
    assert listing.revision == 1
    assert [(tag.name, tag.entries) for tag in listing.tags] == [("site", 2)]
    assert [(v.value, v.entries) for v in listing.tags[0].values] == [("a", 2)]


def test_the_plain_listing_names_each_tag_with_its_values(
    ds: Dataset, tmp_path: Path
) -> None:
    empty = invoke_ok(["sequence-tags", "list", "-m", str(ds.manifest_path)])
    assert "No sequence tags" in empty.stdout

    _ = invoke_ok(_import(ds, _csv(tmp_path, "entry,site\ng1:s1,a\ng1:s2,b\n")))
    shown = invoke_ok(["sequence-tags", "list", "-m", str(ds.manifest_path)])

    assert "site" in shown.stdout and "categorical" in shown.stdout
    assert "a" in shown.stdout and "b" in shown.stdout
