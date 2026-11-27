"""A recipe whose step reads an individual summary and a recording summary.

A step that derives a column from a per-individual summary and a per-recording
summary plans with no problem and runs. Without ``bin_s`` the recording's row
joins each individual's row by the entry. With ``bin_s`` both summaries hold
``bin_start`` and ``bin_end``, and each individual's row of a bin joins the
recording's row of that bin.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.graph import Recipe, run_pipeline, validation_report
from tests.helpers import add_tracks_variant, make_dataset, track_table

VARIANT = "convert-trex.0.2-1111111111"
SEQUENCES = ("seq_a", "seq_b")
INDIVIDUALS = 2
FRAMES = 40
BIN_SECONDS = 0.5


def _recipe(bin_s: float | None) -> Recipe:
    """``derive-columns`` over ``individual-aggregate`` and ``sequence-aggregate``.

    The individual summary averages ``X`` and the recording summary averages
    ``Y``, so the derived column ``gap`` reads one column of each.
    """
    binned = {} if bin_s is None else {"bin_s": bin_s}
    return Recipe.model_validate(
        {
            "schema_version": 1,
            "name": "individual beside recording",
            "steps": [
                {
                    "id": "individuals",
                    "type": "feature",
                    "feature": "individual-aggregate",
                    "inputs": ["tracks"],
                    "params": {"columns": ["X"], "aggs": ["mean"], **binned},
                },
                {
                    "id": "recording",
                    "type": "feature",
                    "feature": "sequence-aggregate",
                    "inputs": ["tracks"],
                    "params": {"columns": ["Y"], "aggs": ["mean"], **binned},
                },
                {
                    "id": "gap",
                    "type": "feature",
                    "feature": "derive-columns",
                    "inputs": [{"step": "individuals"}, {"step": "recording"}],
                    "params": {"columns": [{"name": "gap", "expr": "X_mean - Y_mean"}]},
                },
            ],
        }
    )


@pytest.fixture
def tracked(tmp_path: Path) -> Dataset:
    dataset = make_dataset(tmp_path / "tracked")
    add_tracks_variant(
        dataset, VARIANT, *SEQUENCES, n_rows=FRAMES, std_format="mosaic_v1"
    )
    return dataset


def _run_table(dataset: Dataset, run_id: str, sequence: str) -> pd.DataFrame:
    found = list(
        Path(dataset.get_root("features")).glob(f"*/{run_id}/{sequence}.parquet")
    )
    assert len(found) == 1, f"{sequence} of run {run_id}: {found}"
    return pd.read_parquet(found[0])


def _tracks(sequence: str) -> pd.DataFrame:
    return track_table("", sequence, n_rows=FRAMES, n_ids=INDIVIDUALS)


def _gap_run(dataset: Dataset, recipe: Recipe) -> str:
    done = run_pipeline(dataset, recipe)
    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran", "ran"]
    run_id = done.outcomes[-1].run_id
    assert run_id is not None
    return run_id


@pytest.mark.parametrize("bin_s", [None, BIN_SECONDS])
def test_a_recipe_reading_an_individual_summary_and_a_recording_summary_plans_clean(
    bin_s: float | None,
) -> None:
    report = validation_report(_recipe(bin_s).model_dump(mode="json"))

    assert report.problems == ()
    assert report.cautions == ()


def test_each_individual_row_gets_its_recordings_value(tracked: Dataset) -> None:
    run_id = _gap_run(tracked, _recipe(None))

    for sequence in SEQUENCES:
        table = _run_table(tracked, run_id, sequence)
        tracks = _tracks(sequence)
        recording_mean = tracks["Y"].mean()
        per_individual = tracks.groupby("id")["X"].mean()
        assert sorted(table["id"]) == sorted(per_individual.index)
        for _, row in table.iterrows():
            assert row["Y_mean"] == pytest.approx(recording_mean)
            assert row["gap"] == pytest.approx(
                per_individual[row["id"]] - recording_mean
            )


def test_each_individual_row_of_a_bin_gets_its_bins_value(tracked: Dataset) -> None:
    run_id = _gap_run(tracked, _recipe(BIN_SECONDS))

    for sequence in SEQUENCES:
        table = _run_table(tracked, run_id, sequence)
        tracks = _tracks(sequence)
        bins = np.floor(tracks["time"] / BIN_SECONDS) * BIN_SECONDS
        per_bin = tracks.assign(bin_start=bins).groupby("bin_start")["Y"].mean()
        assert len(table) == len(per_bin) * tracks["id"].nunique()
        for _, row in table.iterrows():
            assert row["Y_mean"] == pytest.approx(per_bin[row["bin_start"]])
            assert row["bin_end"] == pytest.approx(row["bin_start"] + BIN_SECONDS)
