"""A CSV of sequence tags is read into typed values, one set for each entry it names."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.annotations.sequence_tags import (
    KeyForm,
    SequenceTagCsv,
    SequenceTagCsvInvalid,
    parse_key_option,
    read_sequence_tag_csv,
)
from mosaic.core.annotations.tag_table import TagDefinition
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.scan_findings import FindingLog
from mosaic.core.typed_attribute import (
    OPTION_MAX_LENGTH,
    OPTIONS_MAX_COUNT,
    TypedAttributeType,
)
from tests.helpers import MediaClip, make_dataset, write_media_index

_NIGHT: Final[tuple[Entry, ...]] = (("night", "n1"), ("night", "n2"), ("night", "n3"))
"""The members of the divided recording, in recording order."""


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """Two groups sharing a sequence name, a divided recording and an undivided one."""
    dataset = make_dataset(tmp_path / "ds")
    write_media_index(
        dataset,
        [
            MediaClip(group="g1", sequence="s1", filename="g1_s1.mp4"),
            MediaClip(group="g1", sequence="s2", filename="g1_s2.mp4"),
            MediaClip(group="g2", sequence="s1", filename="g2_s1.mp4"),
            MediaClip(group="g2", sequence="s3", filename="g2_s3.mp4"),
            MediaClip(group="night", sequence="n1", filename="n1.mp4"),
            MediaClip(group="night", sequence="n2", filename="n2.mp4"),
            MediaClip(group="night", sequence="n3", filename="n3.mp4"),
            MediaClip(group="", sequence="solo", filename="solo.mp4"),
        ],
    )
    dataset.set_continuous_groups(["night"])
    return dataset


def _read(
    ds: Dataset,
    tmp_path: Path,
    text: str,
    *,
    key: tuple[str, KeyForm] | None = None,
    types: Mapping[str, TypedAttributeType] | None = None,
    kept: Mapping[str, TypedAttributeType] | None = None,
    findings: FindingLog | None = None,
) -> SequenceTagCsv:
    path = tmp_path / "tags.csv"
    path.write_bytes(text.encode("utf-8"))
    return read_sequence_tag_csv(
        ds,
        path,
        key=key,
        types=types or {},
        kept=kept or {},
        findings=findings or FindingLog(kind="labels", source=path.name),
    )


def _types(csv: SequenceTagCsv) -> dict[str, TypedAttributeType]:
    return {column.name: column.type for column in csv.columns}


class TestKeyForms:
    def test_an_entry_column_reads_tokens(self, ds: Dataset, tmp_path: Path) -> None:
        csv = _read(ds, tmp_path, "entry,site\ng1:s1,a\nsolo,b\nnight:n2,c\n")

        assert csv.key_form == "entry"
        assert csv.values == {
            ("g1", "s1"): {"site": "a"},
            ("", "solo"): {"site": "b"},
            ("night", "n2"): {"site": "c"},
        }
        assert csv.matched_rows == 3

    def test_a_group_and_a_sequence_column_name_the_entry(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "group,sequence,site\ng2,s1,a\n,solo,b\n")

        assert csv.key_form == "sequence"
        assert csv.values == {("g2", "s1"): {"site": "a"}, ("", "solo"): {"site": "b"}}

    def test_a_bare_sequence_names_the_one_entry_that_has_it(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "sequence,site\ns2,a\ns3,b\nsolo,c\n")

        assert csv.key_form == "sequence"
        assert set(csv.values) == {("g1", "s2"), ("g2", "s3"), ("", "solo")}

    def test_a_recording_column_names_a_group_or_an_entry_token(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "recording,site\nnight,a\ng1:s1,b\nsolo,c\n")

        assert csv.key_form == "recording"
        assert set(csv.values) == {*_NIGHT, ("g1", "s1"), ("", "solo")}

    def test_a_recording_row_sets_every_sequence_of_a_divided_recording(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "recording,site\nnight,a\n")

        assert csv.values == {entry: {"site": "a"} for entry in _NIGHT}
        assert csv.matched_rows == 1

    @pytest.mark.parametrize(
        ("name", "form", "text", "expected"),
        [
            ("where", "entry", "where,site\ng1:s2,a\n", {("g1", "s2")}),
            ("where", "sequence", "where,site\ns2,a\n", {("g1", "s2")}),
            ("where", "recording", "where,site\nnight,a\n", set(_NIGHT)),
        ],
    )
    def test_key_names_the_column_and_how_its_values_read(
        self,
        ds: Dataset,
        tmp_path: Path,
        name: str,
        form: KeyForm,
        text: str,
        expected: set[Entry],
    ) -> None:
        csv = _read(ds, tmp_path, text, key=(name, form))

        assert csv.key_form == form
        assert set(csv.values) == expected
        assert _types(csv) == {"site": "categorical"}

    def test_a_header_with_two_forms_is_refused_naming_them(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="entry.*recording"):
            _ = _read(ds, tmp_path, "entry,recording,site\ng1:s1,x,a\n")

    def test_a_header_with_no_form_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="--key"):
            _ = _read(ds, tmp_path, "where,site\ng1:s1,a\n")

    def test_key_beside_a_form_in_the_header_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="recording"):
            _ = _read(
                ds,
                tmp_path,
                "where,recording,site\ng1:s1,x,a\n",
                key=("where", "entry"),
            )

    def test_key_naming_a_column_the_header_lacks_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'where'"):
            _ = _read(ds, tmp_path, "site\na\n", key=("where", "entry"))

    def test_a_bare_sequence_in_two_groups_is_refused_naming_both(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"line 2.*'g1'.*'g2'"):
            _ = _read(ds, tmp_path, "sequence,site\ns1,a\n")

    def test_two_rows_naming_one_entry_are_refused_with_both_lines(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"lines 2 and 4"):
            _ = _read(ds, tmp_path, "entry,site\ng1:s1,a\ng1:s2,b\ng1:s1,c\n")

    def test_a_recording_row_and_a_second_row_of_it_are_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"lines 2 and 3"):
            _ = _read(ds, tmp_path, "recording,site\nnight,a\nnight,b\n")


class TestUnmatchedRows:
    def test_a_row_naming_no_entry_is_one_finding_and_the_rest_resolve(
        self, ds: Dataset, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = FindingLog(kind="labels", source="tags.csv")

        csv = _read(
            ds, tmp_path, "entry,site\ng1:s1,a\ng9:nope,b\nsolo,c\n", findings=log
        )

        (finding,) = log.findings
        assert (finding.kind, finding.source, finding.problem, finding.line) == (
            "labels",
            "tags.csv",
            "row_unmatched",
            3,
        )
        assert "line 3" in finding.message and "g9:nope" in finding.message
        assert "line 3" in capsys.readouterr().err
        assert set(csv.values) == {("g1", "s1"), ("", "solo")}
        assert csv.matched_rows == 2

    def test_an_empty_key_cell_is_unmatched(self, ds: Dataset, tmp_path: Path) -> None:
        log = FindingLog(kind="labels", source="tags.csv")

        csv = _read(ds, tmp_path, "entry,site\n,a\nsolo,b\n", findings=log)

        (finding,) = log.findings
        assert (finding.problem, finding.line) == ("row_unmatched", 2)
        assert set(csv.values) == {("", "solo")}

    def test_a_sequence_in_no_group_is_unmatched_in_the_pair_form(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        log = FindingLog(kind="labels", source="tags.csv")

        csv = _read(
            ds, tmp_path, "group,sequence,site\ng2,s2,a\ng2,s1,b\n", findings=log
        )

        assert [finding.line for finding in log.findings] == [2]
        assert set(csv.values) == {("g2", "s1")}

    def test_an_entry_that_no_row_names_has_no_value(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "entry,site\ng1:s1,a\n")

        assert ("g1", "s2") not in csv.values


class TestInference:
    @pytest.mark.parametrize(
        ("cells", "expected"),
        [
            (["true", "false", "true"], "bool"),
            (["3", "-4", "0"], "int"),
            (["3.5", "4"], "float"),
            (["1e3", "2"], "float"),
            (["03", "5"], "categorical"),
            (["True", "False", "TRUE"], "bool"),
            (["True", "yes"], "categorical"),
            (["red", "blue", "red"], "categorical"),
            (["nan", "1"], "categorical"),
        ],
    )
    def test_a_column_takes_the_first_type_that_every_cell_reads_as(
        self, ds: Dataset, tmp_path: Path, cells: list[str], expected: str
    ) -> None:
        entries = ["g1:s1", "g1:s2", "g2:s1"]
        rows = "".join(f"{entries[i]},{cell}\n" for i, cell in enumerate(cells))

        csv = _read(ds, tmp_path, f"entry,x\n{rows}")

        assert _types(csv) == {"x": expected}

    def test_a_leading_zero_keeps_its_text(self, ds: Dataset, tmp_path: Path) -> None:
        csv = _read(ds, tmp_path, "entry,code\ng1:s1,03\ng1:s2,5\n")

        assert csv.values == {("g1", "s1"): {"code": "03"}, ("g1", "s2"): {"code": "5"}}

    def test_typed_values_are_python_values(self, ds: Dataset, tmp_path: Path) -> None:
        csv = _read(
            ds,
            tmp_path,
            "entry,flag,n,x\ng1:s1,true,3,3.5\ng1:s2,false,4,4\n",
        )

        assert csv.values == {
            ("g1", "s1"): {"flag": True, "n": 3, "x": 3.5},
            ("g1", "s2"): {"flag": False, "n": 4, "x": 4.0},
        }
        assert _types(csv) == {"flag": "bool", "n": "int", "x": "float"}

    def test_more_distinct_values_than_options_make_a_text_column(
        self, tmp_path: Path
    ) -> None:
        count = OPTIONS_MAX_COUNT + 1
        wide = make_dataset(tmp_path / "wide")
        write_media_index(
            wide,
            [
                MediaClip(group="", sequence=f"q{index}", filename=f"q{index}.mp4")
                for index in range(count)
            ],
        )
        body = "".join(f"q{index},v{index}\n" for index in range(count))

        at_limit = _read(
            wide,
            tmp_path,
            "entry,x\n" + "".join(f"q{index},v{index}\n" for index in range(count - 1)),
        )
        over = _read(wide, tmp_path, "entry,x\n" + body)

        assert _types(at_limit) == {"x": "categorical"}
        assert _types(over) == {"x": "text"}

    def test_a_value_over_the_option_length_makes_a_text_column(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        long = "a" * (OPTION_MAX_LENGTH + 1)
        limit = "b" * OPTION_MAX_LENGTH

        csv = _read(ds, tmp_path, f"entry,x,y\ng1:s1,{long},{limit}\ng1:s2,b,c\n")

        assert _types(csv) == {"x": "text", "y": "categorical"}

    def test_an_empty_cell_is_no_assignment_and_does_not_decide_the_type(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "entry,n\ng1:s1,3\ng1:s2,\n")

        assert _types(csv) == {"n": "int"}
        assert csv.values == {("g1", "s1"): {"n": 3}, ("g1", "s2"): {}}

    def test_an_unmatched_row_does_not_decide_the_type(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "entry,n\ng1:s1,3\nnowhere,x\n")

        assert _types(csv) == {"n": "int"}

    def test_a_column_whose_only_value_is_on_an_unmatched_row_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'n'.*--type"):
            _ = _read(ds, tmp_path, "entry,n\ng1:s1,\nnowhere,x\n")

    def test_a_column_with_no_value_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'n'.*--type"):
            _ = _read(ds, tmp_path, "entry,n\ng1:s1,\ng1:s2,\n")

    def test_a_kept_type_does_not_exempt_a_column_with_no_value(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'n'"):
            _ = _read(ds, tmp_path, "entry,n\ng1:s1,\n", kept={"n": "int"})

    def test_a_declared_type_defines_a_column_with_no_value(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "entry,n\ng1:s1,\n", types={"n": "int"})

        assert _types(csv) == {"n": "int"}
        assert csv.values == {("g1", "s1"): {}}


class TestDeclaredAndKeptTypes:
    def test_a_kept_int_reads_a_leading_zero_as_the_number(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "entry,n\ng1:s1,03\ng1:s2,4\n", kept={"n": "int"})

        assert _types(csv) == {"n": "int"}
        assert csv.values[("g1", "s1")] == {"n": 3}

    def test_a_declared_type_wins_over_a_kept_type_and_inference(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(
            ds,
            tmp_path,
            "entry,n\ng1:s1,3\n",
            types={"n": "text"},
            kept={"n": "int"},
        )

        assert _types(csv) == {"n": "text"}
        assert csv.values == {("g1", "s1"): {"n": "3"}}

    def test_a_declared_float_reads_a_whole_number_as_a_float(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "entry,x\ng1:s1,3\n", types={"x": "float"})

        value = csv.values[("g1", "s1")]["x"]
        assert value == 3.0 and isinstance(value, float)

    def test_a_declared_bool_refuses_yes_naming_the_line_and_the_column(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"line 3.*'flag'.*'yes'"):
            _ = _read(
                ds,
                tmp_path,
                "entry,flag\ng1:s1,true\ng1:s2,yes\n",
                types={"flag": "bool"},
            )

    def test_a_bool_is_read_in_any_case(self, ds: Dataset, tmp_path: Path) -> None:
        csv = _read(ds, tmp_path, "entry,blind\ng1:s1,False\ng1:s2,TRUE\n")

        assert csv.values[("g1", "s1")]["blind"] is False
        assert csv.values[("g1", "s2")]["blind"] is True

    def test_a_declared_bool_refusal_names_the_spellings_it_accepts(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(
            SequenceTagCsvInvalid, match=r"'yes' is not true or false, in any case"
        ):
            _ = _read(ds, tmp_path, "entry,flag\ng1:s1,yes\n", types={"flag": "bool"})

    def test_a_declared_int_refuses_a_fraction(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"line 2.*'n'"):
            _ = _read(ds, tmp_path, "entry,n\ng1:s1,3.5\n", types={"n": "int"})

    def test_a_declared_categorical_refuses_a_value_over_the_option_length(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        long = "a" * (OPTION_MAX_LENGTH + 1)

        with pytest.raises(SequenceTagCsvInvalid, match=r"line 2.*'site'"):
            _ = _read(
                ds,
                tmp_path,
                f"entry,site\ng1:s1,{long}\n",
                types={"site": "categorical"},
            )

    def test_a_label_column_assigns_its_non_empty_cells(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(
            ds,
            tmp_path,
            "entry,focal\ng1:s1,yes\ng1:s2,\ng2:s1,x\n",
            types={"focal": "label"},
        )

        assert _types(csv) == {"focal": "label"}
        assert csv.values == {
            ("g1", "s1"): {"focal": None},
            ("g1", "s2"): {},
            ("g2", "s1"): {"focal": None},
        }

    def test_a_type_for_a_column_the_csv_lacks_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'missing'"):
            _ = _read(ds, tmp_path, "entry,site\ng1:s1,a\n", types={"missing": "int"})

    def test_a_type_for_the_key_column_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'entry'"):
            _ = _read(ds, tmp_path, "entry,site\ng1:s1,a\n", types={"entry": "text"})

    def test_a_value_on_an_unmatched_row_is_not_checked(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        log = FindingLog(kind="labels", source="tags.csv")

        csv = _read(
            ds,
            tmp_path,
            "entry,n\ng1:s1,3\nnowhere,many\n",
            types={"n": "int"},
            findings=log,
        )

        assert csv.values == {("g1", "s1"): {"n": 3}}
        assert len(log.findings) == 1


class TestTheFile:
    def test_the_result_names_the_file_and_its_digest(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        text = "entry,site\ng1:s1,a\n"

        csv = _read(ds, tmp_path, text)

        assert csv.file_name == "tags.csv"
        assert csv.sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_a_byte_order_mark_and_padded_cells_are_ignored(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "﻿ entry , site \n g1:s1 , a b \n")

        assert csv.values == {("g1", "s1"): {"site": "a b"}}

    def test_a_quoted_cell_over_two_lines_leaves_the_next_line_right(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        log = FindingLog(kind="labels", source="tags.csv")

        _ = _read(
            ds,
            tmp_path,
            'entry,note\ng1:s1,"two\nlines"\nnowhere,x\n',
            findings=log,
        )

        (finding,) = log.findings
        assert finding.line == 4

    def test_a_quoted_cell_over_two_lines_names_its_first_line(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"line 3"):
            _ = _read(
                ds,
                tmp_path,
                'entry,n\ng1:s1,3\ng1:s2,"4\n5"\n',
                types={"n": "int"},
            )

    def test_blank_lines_count_toward_a_row_s_line(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        log = FindingLog(kind="labels", source="tags.csv")

        _ = _read(ds, tmp_path, "entry,site\n\ng1:s1,a\n\nnowhere,x\n", findings=log)

        assert [finding.line for finding in log.findings] == [5]

    def test_a_file_without_a_header_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="no header"):
            _ = _read(ds, tmp_path, "")

    def test_a_missing_file_is_refused(self, ds: Dataset, tmp_path: Path) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="nope.csv"):
            _ = read_sequence_tag_csv(
                ds,
                tmp_path / "nope.csv",
                key=None,
                types={},
                kept={},
                findings=FindingLog(kind="labels", source="nope.csv"),
            )

    def test_a_repeated_column_is_refused(self, ds: Dataset, tmp_path: Path) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="'site'"):
            _ = _read(ds, tmp_path, "entry,site,site\ng1:s1,a,b\n")

    def test_a_column_without_a_name_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="no name"):
            _ = _read(ds, tmp_path, "entry,,site\ng1:s1,a,b\n")

    def test_a_row_longer_than_the_header_is_refused(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match=r"line 2"):
            _ = _read(ds, tmp_path, "entry,site\ng1:s1,a,extra\n")

    def test_the_columns_keep_the_header_order(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "zeta,entry,alpha\nz,g1:s1,a\n")

        assert csv.columns == (
            TagDefinition("zeta", "categorical"),
            TagDefinition("alpha", "categorical"),
        )


class TestReservedNames:
    @pytest.mark.parametrize(
        "name", ["group", "id", "frame", "time", "individual", "uid"]
    )
    def test_each_reserved_name_is_refused_as_a_column(
        self, ds: Dataset, tmp_path: Path, name: str
    ) -> None:
        with pytest.raises(
            SequenceTagCsvInvalid,
            match=rf"tag '{name}' has the name of a column that this step writes",
        ):
            _ = _read(ds, tmp_path, f"where,{name}\ng1:s1,a\n", key=("where", "entry"))

    def test_a_sequence_column_beside_another_key_is_refused_as_a_second_key(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="entry, sequence"):
            _ = _read(ds, tmp_path, "entry,sequence\ng1:s1,a\n")

    def test_the_group_and_sequence_columns_of_the_pair_are_the_key(
        self, ds: Dataset, tmp_path: Path
    ) -> None:
        csv = _read(ds, tmp_path, "group,sequence,site\ng1,s1,a\n")

        assert _types(csv) == {"site": "categorical"}


class TestKeyOption:
    @pytest.mark.parametrize("form", ["entry", "sequence", "recording"])
    def test_a_name_and_a_form_are_read(self, form: str) -> None:
        assert parse_key_option(f"where={form}") == ("where", form)

    def test_the_form_is_after_the_last_equals_sign(self) -> None:
        assert parse_key_option("a=b=entry") == ("a=b", "entry")

    @pytest.mark.parametrize("text", ["where", "=entry", "where=", "where=group"])
    def test_anything_else_is_refused(self, text: str) -> None:
        with pytest.raises(SequenceTagCsvInvalid, match="NAME=FORM"):
            _ = parse_key_option(text)
