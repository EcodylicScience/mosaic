"""``mosaic summarize``: one run's rows counted and described by group."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.error_envelope import ErrorEnvelope
from mosaic.core.pipeline.run_summary import RunSummary, run_summary_wire, summarize_run
from mosaic.core.pipeline.types import Result
from mosaic.core.scope import Scope
from tests.helpers import add_tracks_variant, invoke_ok, labeled_run, rewrite_index

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
