"""``mosaic summarize``: one run's rows counted and described by group."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline import run_summary
from mosaic.core.pipeline.run_summary import (
    RunSummary,
    RunSummaryWire,
    run_summary_wire,
    summarize_run,
)
from mosaic.core.pipeline.summary_arguments import KeyValue, SummaryTest
from mosaic.core.pipeline.tag_reads import ResolvedTagTable
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from mosaic.core.selections import save_selection
from tests.helpers import (
    add_tracks_variant,
    grown_run,
    invoke_ok,
    labeled_run,
    rewrite_index,
    tag_table,
    tree_snapshot,
    write_tags,
)

runner = CliRunner()


def table(**columns: object) -> pd.DataFrame:
    """One entry's rows, from its columns."""
    return pd.DataFrame(dict(columns))


def strain_run(tmp_path: Path) -> tuple[Dataset, Result[str]]:
    """Two strains over four entries, with an integer label and two value columns."""
    return labeled_run(
        tmp_path,
        {
            ("g", "a"): table(
                strain=["A"] * 3,
                n_inds=[1] * 3,
                value=[1.0, 2.0, 3.0],
                other=[4.0, 4.5, 5.0],
            ),
            ("g", "b"): table(
                strain=["A"] * 3,
                n_inds=[1] * 3,
                value=[2.0, 3.0, 4.0],
                other=[4.2, 4.6, 5.1],
            ),
            ("h", "c"): table(
                strain=["B"] * 3,
                n_inds=[2] * 3,
                value=[5.0, 6.0, 7.0],
                other=[1.0, 2.0, 3.0],
            ),
            ("h", "d"): table(
                strain=["B"] * 3,
                n_inds=[2] * 3,
                value=[6.0, 7.0, 9.0],
                other=[1.5, 2.5, 3.5],
            ),
        },
    )


def ref_of(result: Result[str]) -> str:
    return f"feature/{result.feature}/{result.run_id}"


def summarize(ds: Dataset, result: Result[str], *options: str) -> list[str]:
    return ["summarize", ref_of(result), "--manifest", str(ds.manifest_path), *options]


@pytest.mark.parametrize(
    ("options", "library"),
    [
        pytest.param(
            ["--by", "strain"], partial(summarize_run, by=["strain"]), id="by"
        ),
        pytest.param(
            ["--by", "strain", "--by", "n_inds"],
            partial(summarize_run, by=["strain", "n_inds"]),
            id="two keys",
        ),
        pytest.param(
            ["--by", "strain", "--column", "other"],
            partial(summarize_run, by=["strain"], columns=["other"]),
            id="column",
        ),
        pytest.param(
            ["--by", "strain", "--column", "val*"],
            partial(summarize_run, by=["strain"], columns=["val*"]),
            id="column pattern",
        ),
        pytest.param(
            ["--by", "strain", "--test", "kruskal", "--test", "mannwhitney"],
            partial(summarize_run, by=["strain"], tests=["kruskal", "mannwhitney"]),
            id="tests",
        ),
        pytest.param(
            ["--by", "strain", "--where", "n_inds=1"],
            partial(summarize_run, by=["strain"], where={"n_inds": [1]}),
            id="where an integer key from text",
        ),
        pytest.param(
            ["--by", "n_inds", "--where", "strain=A", "--where", "strain=B"],
            partial(summarize_run, by=["n_inds"], where={"strain": ["A", "B"]}),
            id="where alternatives",
        ),
        pytest.param(
            ["--by", "strain", "--group", "h"],
            partial(summarize_run, by=["strain"], scope=Scope(groups=["h"])),
            id="group",
        ),
        pytest.param(
            ["--by", "strain", "--entry", "g:a", "--entry", "h:c"],
            partial(
                summarize_run,
                by=["strain"],
                scope=Scope(entries=[("g", "a"), ("h", "c")]),
            ),
            id="entries",
        ),
        pytest.param([], partial(summarize_run), id="no key"),
    ],
)
def test_json_is_the_library_summary_as_the_wire(
    tmp_path: Path,
    options: list[str],
    library: Callable[[Dataset, Result[str]], RunSummary],
) -> None:
    ds, result = strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, *options, "--json"))

    expected = run_summary_wire(library(ds, result), ref_of(result))
    assert json.loads(printed.stdout) == expected.model_dump(mode="json")


def test_a_selection_token_summarizes_the_entries_it_holds(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)
    _ = save_selection(ds, "strain-b", [("h", "c"), ("h", "d")])

    printed = invoke_ok(
        summarize(ds, result, "--by", "strain", "--entry", "@strain-b", "--json")
    )

    expected = run_summary_wire(
        summarize_run(ds, result, by=["strain"], scope=Scope(groups=["h"])),
        ref_of(result),
    )
    assert json.loads(printed.stdout) == expected.model_dump(mode="json")


def test_a_missing_selection_is_refused_by_summarize(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)

    refused = runner.invoke(
        app, summarize(ds, result, "--by", "strain", "--entry", "@nope", "--json")
    )

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert envelope.kind == "not_found"
    assert envelope.reason == "selection_not_found"


def test_where_matches_an_integer_key_from_its_text(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)

    printed = invoke_ok(
        summarize(ds, result, "--by", "strain", "--where", "n_inds=1", "--json")
    )

    wire = json.loads(printed.stdout)
    assert wire["groups"] == [{"key": {"strain": "A"}, "n_rows": 6, "n_entries": 2}]


def test_where_keeps_a_text_key_that_looks_like_a_number(tmp_path: Path) -> None:
    ds, result = labeled_run(
        tmp_path,
        {
            ("g", "a"): table(code=["1", "1"], value=[1.0, 2.0]),
            ("g", "b"): table(code=["2", "2"], value=[3.0, 4.0]),
        },
    )

    printed = invoke_ok(
        summarize(ds, result, "--by", "code", "--where", "code=1", "--json")
    )

    wire = json.loads(printed.stdout)
    assert [group["key"] for group in wire["groups"]] == [{"code": "1"}]


def test_where_keeps_a_boolean_key_from_its_text(tmp_path: Path) -> None:
    ds, result = labeled_run(
        tmp_path,
        {
            ("g", "a"): table(fed=[True, True], value=[1.0, 2.0]),
            ("g", "b"): table(fed=[False, False], value=[3.0, 4.0]),
        },
    )

    printed = invoke_ok(
        summarize(ds, result, "--by", "fed", "--where", "fed=true", "--json")
    )

    wire = json.loads(printed.stdout)
    assert [group["key"] for group in wire["groups"]] == [{"fed": True}]


@pytest.mark.parametrize(
    ("key", "cells", "kept"),
    [
        pytest.param("fed", [True, False], True, id="boolean key"),
        pytest.param("n_inds", [1, 2], 1, id="integer key"),
    ],
)
@pytest.mark.parametrize(
    "tokens", [["1", "true"], ["true", "1"]], ids=["1 then true", "true then 1"]
)
def test_where_keeps_every_reading_of_values_that_compare_equal(
    tmp_path: Path, key: str, cells: list[object], kept: object, tokens: list[str]
) -> None:
    ds, result = labeled_run(
        tmp_path,
        {
            ("g", "a"): table(**{key: [cells[0]] * 2}, value=[1.0, 2.0]),
            ("g", "b"): table(**{key: [cells[1]] * 2}, value=[3.0, 4.0]),
        },
    )
    where = [option for token in tokens for option in ("--where", f"{key}={token}")]

    printed = invoke_ok(summarize(ds, result, "--by", key, *where, "--json"))

    wire = json.loads(printed.stdout)
    assert [group["key"] for group in wire["groups"]] == [{key: kept}]


@pytest.mark.parametrize("token", ["strain", "=A"])
def test_a_where_without_a_key_and_a_value_reports_invalid(
    tmp_path: Path, token: str
) -> None:
    ds, result = strain_run(tmp_path)

    refused = runner.invoke(
        app, summarize(ds, result, "--by", "strain", "--where", token, "--json")
    )

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert (envelope.kind, envelope.detail) == ("invalid", {"field": "where"})


def test_a_tracks_variant_ref_reports_invalid_naming_the_ref(tmp_path: Path) -> None:
    ds, _ = strain_run(tmp_path)
    variant = "convert-trex.0.2-1111111111"
    add_tracks_variant(ds, variant, "seq_a", std_format="mosaic_v1")

    refused = runner.invoke(
        app,
        [
            "summarize",
            f"tracks-variant/{variant}",
            "--manifest",
            str(ds.manifest_path),
            "--json",
        ],
    )

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert (envelope.kind, envelope.detail) == ("invalid", {"field": "ref"})


def test_a_malformed_ref_reports_invalid(tmp_path: Path) -> None:
    ds, _ = strain_run(tmp_path)

    refused = runner.invoke(
        app, ["summarize", "nonsense", "--manifest", str(ds.manifest_path), "--json"]
    )

    assert refused.exit_code == 1
    assert ErrorEnvelope.model_validate_json(refused.stdout).kind == "invalid"


def test_a_per_frame_run_reports_invalid(tmp_path: Path) -> None:
    ds, result = labeled_run(
        tmp_path,
        {("g", "a"): table(frame=[0, 1, 2], strain=["A"] * 3, value=[1.0, 2.0, 3.0])},
    )

    refused = runner.invoke(app, summarize(ds, result, "--by", "strain", "--json"))

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert envelope.kind == "invalid"
    assert envelope.detail is not None
    assert envelope.detail["feature"] == result.feature


def test_an_unknown_run_reports_not_found(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)

    refused = runner.invoke(
        app,
        [
            "summarize",
            f"feature/{result.feature}/0.1-0000000000",
            "--manifest",
            str(ds.manifest_path),
            "--json",
        ],
    )

    assert refused.exit_code == 1
    assert ErrorEnvelope.model_validate_json(refused.stdout).kind == "not_found"


def test_an_unknown_test_reports_invalid(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)

    refused = runner.invoke(app, summarize(ds, result, "--test", "anova", "--json"))

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert (envelope.kind, envelope.detail) == ("invalid", {"field": "test"})


def test_the_plain_output_prints_the_groups_the_statistics_and_each_test(
    tmp_path: Path,
) -> None:
    ds, result = strain_run(tmp_path)

    printed = invoke_ok(
        summarize(
            ds,
            result,
            "--by",
            "strain",
            "--column",
            "value",
            "--test",
            "kruskal",
            "--test",
            "mannwhitney",
        )
    )

    lines = printed.stdout.splitlines()
    assert f"ref: {ref_of(result)}" in lines
    assert "strain\tn_rows\tn_entries" in lines
    assert "A\t6\t2" in lines
    assert "B\t6\t2" in lines
    assert "strain\tcolumn\tn\tn_missing\tmean\tsd\tsem\tmedian" in lines
    assert any(line.startswith("A\tvalue\t6\t0\t2.5\t") for line in lines)
    assert "column\tk\tn\tstatistic\tdf\tp\treason" in lines
    assert any(line.startswith("value\t2\t12\t") for line in lines)
    assert (
        "column\tstrain_a\tstrain_b\tn_a\tn_b\tstatistic\tp\tp_holm\tcomparisons\treason"
        in lines
    )
    assert any(line.startswith("value\tA\tB\t6\t6\t") for line in lines)
    assert printed.stderr == ""


def test_a_test_that_was_not_asked_prints_no_table(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, "--by", "strain"))

    assert "statistic" not in printed.stdout


def test_stderr_notes_what_the_summary_leaves_out(tmp_path: Path) -> None:
    ds, result = labeled_run(
        tmp_path,
        {
            ("g", "a"): table(strain=["A", "A", None], value=[1.0, 2.0, 3.0]),
            ("g", "b"): table(strain=["B", "B"], value=[4.0, 5.0]),
            ("g", "c"): table(
                strain=pd.Series([], dtype="object"),
                value=pd.Series([], dtype="float64"),
            ),
        },
    )

    printed = invoke_ok(summarize(ds, result, "--by", "strain"))

    notes = printed.stderr.splitlines()
    assert "note: rows left out because a key is null: 1" in notes
    assert "note: entries with no row: g:c" in notes


def test_stderr_notes_the_entries_a_partial_run_lacks_and_that_it_is_unfinished(
    tmp_path: Path,
) -> None:
    ds, result = strain_run(tmp_path)

    def lose_d_and_leave_c_going(frame: pd.DataFrame) -> pd.DataFrame:
        kept = frame[frame["sequence"] != "d"].reset_index(drop=True)
        kept.loc[kept["sequence"] == "c", "finished_at"] = ""
        return kept

    rewrite_index(ds, result, lose_d_and_leave_c_going)
    arguments = summarize(
        ds, result, "--by", "strain", "--entry", "h:c", "--entry", "h:d"
    )

    refused = runner.invoke(app, [*arguments, "--json"])
    assert refused.exit_code == 1
    assert ErrorEnvelope.model_validate_json(refused.stdout).kind == "conflict"

    printed = invoke_ok([*arguments, "--allow-partial"])

    notes = printed.stderr.splitlines()
    assert "note: entries missing from the run: h:d" in notes
    assert "note: the run is not finished" in notes


def test_a_finished_run_notes_nothing(tmp_path: Path) -> None:
    ds, result = strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, "--by", "strain", "--allow-partial"))

    assert printed.stderr == ""


# --- sequence tags -----------------------------------------------------------


def sites(a: str, b: str) -> dict[tuple[str, str], dict[str, str]]:
    """The site of each entry of :func:`strain_run`; ``g/a`` and ``h/c`` share ``a``."""
    return {
        ("g", "a"): {"site": a},
        ("g", "b"): {"site": b},
        ("h", "c"): {"site": a},
        ("h", "d"): {"site": b},
    }


def tagged_strain_run(tmp_path: Path) -> tuple[Dataset, Result[str]]:
    """:func:`strain_run` with revision 1 of a ``site`` tag, ``x`` on ``g/a`` and ``h/c``."""
    ds, result = strain_run(tmp_path)
    _ = write_tags(ds, tag_table(sites("x", "y"), {"site": "categorical"}))
    return ds, result


def write_swapped_sites(ds: Dataset) -> None:
    """Save revision 2 of the tags, which gives ``g/a`` and ``h/c`` the site ``y``."""
    _ = write_tags(ds, tag_table(sites("y", "x"), {"site": "categorical"}))


def summary_of(printed: str) -> RunSummaryWire:
    return RunSummaryWire.model_validate_json(printed)


def test_a_tag_key_prints_the_wire_with_the_keys_and_the_revision_read(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, "--by", "site", "--json"))

    wire = summary_of(printed.stdout)
    assert wire.tag_keys == ["site"]
    assert wire.tag_revision is not None
    assert wire.tag_revision.revision == 1
    assert sorted(group.key["site"] for group in wire.groups) == ["x", "y"]
    expected = run_summary_wire(summarize_run(ds, result, by=["site"]), ref_of(result))
    assert wire == expected


def test_a_tag_revision_reads_the_revision_it_names_after_a_later_one_exists(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)
    write_swapped_sites(ds)

    first = summary_of(
        invoke_ok(
            summarize(ds, result, "--by", "site", "--tag-revision", "1", "--json")
        ).stdout
    )
    latest = summary_of(
        invoke_ok(summarize(ds, result, "--by", "site", "--json")).stdout
    )

    assert first.tag_revision is not None
    assert first.tag_revision.revision == 1
    assert latest.tag_revision is not None
    assert latest.tag_revision.revision == 2
    entries_of_x = {
        revision: next(
            group.n_entries for group in wire.groups if group.key["site"] == "x"
        )
        for revision, wire in ((1, first), (2, latest))
    }
    assert entries_of_x == {1: 2, 2: 2}
    assert first.tag_revision.digest != latest.tag_revision.digest


def test_a_tag_revision_that_is_not_indexed_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    refused = runner.invoke(
        app, summarize(ds, result, "--by", "site", "--tag-revision", "9", "--json")
    )

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert envelope.kind == "invalid"
    assert "9" in envelope.message


def test_a_tag_selection_reads_the_entries_that_the_tag_holds(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    by_tag = invoke_ok(
        summarize(ds, result, "--by", "strain", "--tag", "site=x", "--json")
    )
    by_entry = invoke_ok(
        summarize(
            ds,
            result,
            "--by",
            "strain",
            "--entry",
            "g:a",
            "--entry",
            "h:c",
            "--json",
        )
    )

    by_entry_document = json.loads(by_entry.stdout)
    by_tag_document = json.loads(by_tag.stdout)
    assert by_entry_document.pop("tag_revision") is None
    assert by_tag_document.pop("tag_revision")["revision"] == 1
    assert by_tag_document == by_entry_document
    assert summary_of(by_tag.stdout).n_entries == 2


def test_a_tag_selection_and_a_tag_key_read_the_revision_the_selection_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds, result = tagged_strain_run(tmp_path)
    summarize_ref = run_summary.summarize_ref

    def after_a_new_revision(
        opened: Dataset,
        ref: str,
        *,
        by: Sequence[str],
        columns: str | Sequence[str] | None,
        tests: Sequence[SummaryTest],
        where: Mapping[str, Sequence[KeyValue]] | None,
        scope: Scope | None,
        allow_partial: bool,
        tag_revision: int | None,
        tag_table: ResolvedTagTable | None,
    ) -> RunSummaryWire:
        write_swapped_sites(ds)
        return summarize_ref(
            opened,
            ref,
            by=by,
            columns=columns,
            tests=tests,
            where=where,
            scope=scope,
            allow_partial=allow_partial,
            tag_revision=tag_revision,
            tag_table=tag_table,
        )

    monkeypatch.setattr(run_summary, "summarize_ref", after_a_new_revision)

    printed = invoke_ok(
        summarize(ds, result, "--tag", "site=x", "--by", "site", "--json")
    )

    wire = summary_of(printed.stdout)
    assert wire.n_entries == 2
    assert [group.key for group in wire.groups] == [{"site": "x"}]
    assert wire.tag_revision is not None
    assert wire.tag_revision.revision == 1


def test_a_tag_selection_records_its_revision_without_a_tag_key(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    printed = invoke_ok(
        summarize(ds, result, "--by", "group", "--tag", "site=x", "--json")
    )

    wire = summary_of(printed.stdout)
    assert wire.tag_keys == []
    assert wire.tag_revision is not None
    assert wire.tag_revision.revision == 1


def test_the_plain_output_prints_the_revision_a_tag_selection_read(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, "--by", "group", "--tag", "site=x"))

    assert "tag_revision: 1" in printed.stdout.splitlines()
    assert "tag_keys" not in printed.stdout


def test_a_tag_selection_reads_the_revision_that_tag_revision_names(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)
    write_swapped_sites(ds)

    printed = invoke_ok(
        summarize(
            ds,
            result,
            "--tag",
            "site=x",
            "--tag-revision",
            "1",
            "--by",
            "site",
            "--json",
        )
    )

    wire = summary_of(printed.stdout)
    assert [group.key for group in wire.groups] == [{"site": "x"}]
    assert wire.groups[0].n_entries == 2
    assert wire.tag_revision is not None
    assert wire.tag_revision.revision == 1


def test_a_tag_beside_an_entry_is_refused(tmp_path: Path) -> None:
    ds, result = tagged_strain_run(tmp_path)

    refused = runner.invoke(
        app, summarize(ds, result, "--tag", "site=x", "--entry", "g:a")
    )

    assert refused.exit_code == 1
    assert "--entry" in refused.stderr
    assert "--tag" in refused.stderr


def test_a_tag_that_is_not_defined_is_refused_naming_the_tags(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    refused = runner.invoke(app, summarize(ds, result, "--tag", "nope=x", "--json"))

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert envelope.kind == "invalid"
    assert "nope" in envelope.message
    assert "site" in envelope.message


def test_a_key_that_is_neither_a_column_nor_a_tag_is_refused_naming_both(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    refused = runner.invoke(app, summarize(ds, result, "--by", "nope", "--json"))

    assert refused.exit_code == 1
    envelope = ErrorEnvelope.model_validate_json(refused.stdout)
    assert envelope.kind == "invalid"
    assert "nope" in envelope.message
    assert "strain" in envelope.message
    assert "site" in envelope.message


def test_the_plain_output_prints_the_revision_the_tag_keys_were_read_from(
    tmp_path: Path,
) -> None:
    ds, result = tagged_strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, "--by", "site"))

    assert "tag_keys: site (revision 1)" in printed.stdout.splitlines()


def test_a_run_without_a_tag_key_prints_no_tag_revision(tmp_path: Path) -> None:
    ds, result = tagged_strain_run(tmp_path)

    printed = invoke_ok(summarize(ds, result, "--by", "strain"))

    assert "tag_keys" not in printed.stdout


def test_a_summary_by_tag_writes_nothing(tmp_path: Path) -> None:
    ds, result = tagged_strain_run(tmp_path)
    before = tree_snapshot(ds.base_dir)

    _ = invoke_ok(summarize(ds, result, "--tag", "site=x", "--by", "site", "--json"))

    assert tree_snapshot(ds.base_dir) == before


def test_an_unscoped_summary_of_a_run_that_missed_added_entries_cautions(
    tmp_path: Path,
) -> None:
    ds, result = grown_run(tmp_path)
    caution = (
        f"run {result.run_id} covers 3 of the dataset's 5 entries. A run over the "
        f"other 2 adds them to what an unscoped read returns. Read through a saved "
        f"selection to keep this set"
    )

    printed = invoke_ok(summarize(ds, result))
    as_json = invoke_ok(summarize(ds, result, "--json"))

    assert f"note: {caution}" in printed.stderr.splitlines()
    assert json.loads(as_json.stdout)["cautions"] == [caution]


def test_a_summary_through_a_selection_prints_no_growth_caution(
    tmp_path: Path,
) -> None:
    ds, result = grown_run(tmp_path)
    _ = save_selection(ds, "earlier", [("", f"seq_{index}") for index in range(3)])

    printed = invoke_ok(summarize(ds, result, "--entry", "@earlier"))
    as_json = invoke_ok(summarize(ds, result, "--entry", "@earlier", "--json"))

    assert printed.stderr == ""
    assert json.loads(as_json.stdout)["cautions"] == []
