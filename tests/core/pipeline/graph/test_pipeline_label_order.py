"""A label conversion runs after every step that writes tracks.

The labels of a converter of events in seconds are placed on the tracks that exist
when it runs, so a recipe that declares the label conversion first has to run it
second. The order is the same for every label converter, which keeps it from
reading a registry. No reference carries that order, and the planner derives it. The labels
variant names the tracks variant the labels are placed on, so a plan names it from the
tracks step above the conversion, or leaves it unresolved when it cannot.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    IdentityUnresolved,
    Recipe,
    ancestors_of,
    descendants_of,
    edges,
    plan_pipeline,
    run_pipeline,
    topological_order,
)
from mosaic.core.pipeline.labels_identity import read_labels_variant
from mosaic.core.pipeline.labels_index import read_labels_index
from tests.helpers import (
    add_tracks_table,
    dropped_frame_tracks,
    make_dataset,
    write_tracktor_recording,
)

PLACE = "obs__ant"
OBSERVATION = "20211117111002_900s_obs"
SUBJECT = "ant-DGRP101-female-2to4-6-1"
STEM = f"20211117111002_900s_{PLACE}-DGRP101-female-2to4-6-1"
"""A Tracktor recording whose entry is the sequence BORIS names for the events below."""

TIMED_LABELS: dict[str, JsonValue] = {
    "id": "labels",
    "type": "op",
    "kind": "convert-labels",
    "params": {"source_format": "boris_pandas_pickle"},
}
RATE_LABELS: dict[str, JsonValue] = {
    "id": "labels",
    "type": "op",
    "kind": "convert-labels",
    "params": {"source_format": "calms21_npy"},
}
CONVERT: dict[str, JsonValue] = {
    "id": "tracks",
    "type": "op",
    "kind": "convert-tracks",
    "params": {"source_format": "tracktor_corrected_tsv"},
}
RESAMPLE: dict[str, JsonValue] = {
    "id": "resampled",
    "type": "op",
    "kind": "resample-tracks",
    "params": {
        "target_fps": 30.0,
        "source_tracks_run_id": "convert-tracktor_corrected_tsv.0.1-aaaaaaaaaa",
    },
    "after": ["tracks"],
}


def _order(*steps: dict[str, JsonValue]) -> list[str]:
    recipe = Recipe.model_validate({"steps": list(steps)})
    return [step.id for step in topological_order(recipe)]


def test_a_timed_label_conversion_declared_first_runs_after_the_conversion_of_tracks() -> (
    None
):
    assert _order(TIMED_LABELS, CONVERT) == ["tracks", "labels"]


def test_it_runs_after_every_step_that_writes_tracks() -> None:
    assert _order(TIMED_LABELS, RESAMPLE, CONVERT) == ["tracks", "resampled", "labels"]


def test_a_label_conversion_at_a_fixed_rate_runs_after_the_tracks_too() -> None:
    assert _order(RATE_LABELS, CONVERT) == ["tracks", "labels"]


def test_the_derived_order_is_not_a_reference() -> None:
    recipe = Recipe.model_validate({"steps": [TIMED_LABELS, CONVERT]})

    assert edges(recipe) == ()
    assert "tracks" in ancestors_of(recipe, "labels")
    assert "labels" in descendants_of(recipe, "tracks")


def test_a_tracks_writer_that_reads_the_labels_stays_below_them() -> None:
    reader: dict[str, JsonValue] = {
        "id": "after_labels",
        "type": "op",
        "kind": "resample-tracks",
        "params": {"source_tracks_run_id": "x", "marker": {"step": "labels"}},
    }

    assert _order(TIMED_LABELS, CONVERT, reader) == ["tracks", "labels", "after_labels"]


def _dataset_with_timed_events(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    _ = write_tracktor_recording(raw, stem=STEM, write_csv=False)
    _ = ds.index_tracks_raw(
        [raw],
        patterns=["*_tracked_fixed_final.tsv"],
        src_format="tracktor_corrected_tsv",
    )
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    pd.DataFrame(
        {
            "Observation id": [OBSERVATION],
            "Subject": [SUBJECT],
            "Behavior": ["approach"],
            "Behavior type": ["STATE"],
            "Start (s)": [1.1],
            "Stop (s)": [1.2],
            "FPS": [30.0],
        }
    ).to_pickle(uploads / "boris.pkl")
    ds.index_labels_raw([uploads], patterns=["*.pkl"], src_format="boris_pandas_pickle")
    return ds


@pytest.fixture
def declared_first() -> Recipe:
    return Recipe.model_validate({"steps": [TIMED_LABELS, CONVERT]})


def test_a_plan_lists_the_conversion_of_tracks_first(
    tmp_path: Path, declared_first: Recipe
) -> None:
    plan = plan_pipeline(_dataset_with_timed_events(tmp_path), declared_first)

    assert [step.step_id for step in plan.steps] == ["tracks", "labels"]


def test_a_run_places_the_labels_on_the_tracks_the_recipe_converts(
    tmp_path: Path, declared_first: Recipe
) -> None:
    ds = _dataset_with_timed_events(tmp_path)

    run = run_pipeline(ds, declared_first)

    assert [outcome.step_id for outcome in run.outcomes] == ["tracks", "labels"]
    (roots,) = read_labels_index(ds, "behavior")["consumed_source_roots"]
    assert "tracks_raw" in str(roots)


def test_a_plan_names_the_labels_variant_the_run_writes(
    tmp_path: Path, declared_first: Recipe
) -> None:
    ds = _dataset_with_timed_events(tmp_path)

    planned = plan_pipeline(ds, declared_first)
    _ = run_pipeline(ds, declared_first)

    (written,) = set(read_labels_index(ds, "behavior")["run_id"])
    labels = planned.step("labels")
    assert labels.run_id == written
    assert labels.labels_variant == written
    assert labels.run_id == ds.label_conversion_variant(
        "behavior",
        "boris_pandas_pickle",
        tracks_run_id=planned.step("tracks").tracks_variant,
    )


def test_a_plan_with_several_tracks_steps_and_no_pin_leaves_the_labels_unresolved(
    tmp_path: Path,
) -> None:
    ds = _dataset_with_timed_events(tmp_path)
    recipe = Recipe.model_validate({"steps": [TIMED_LABELS, CONVERT, RESAMPLE]})

    labels = plan_pipeline(ds, recipe).step("labels")

    assert labels.run_id is None
    assert isinstance(labels.reason, IdentityUnresolved)
    assert "tracks_run_id" in labels.reason.because


def test_a_pinned_tracks_variant_resolves_the_labels_whatever_the_steps_above(
    tmp_path: Path,
) -> None:
    ds = _dataset_with_timed_events(tmp_path)
    pin = "resample-tracks.0.2-bbbbbbbbbb"
    pinned: dict[str, JsonValue] = {
        **TIMED_LABELS,
        "params": {"source_format": "boris_pandas_pickle", "tracks_run_id": pin},
    }
    recipe = Recipe.model_validate({"steps": [pinned, CONVERT, RESAMPLE]})

    labels = plan_pipeline(ds, recipe).step("labels")

    assert labels.run_id == ds.label_conversion_variant(
        "behavior", "boris_pandas_pickle", tracks_run_id=pin
    )


def test_a_run_uses_the_tracks_variant_the_plan_named_beside_an_older_one(
    tmp_path: Path, declared_first: Recipe
) -> None:
    ds = _dataset_with_timed_events(tmp_path)
    older = dropped_frame_tracks(ids=(0,), sequence=STEM)
    add_tracks_table(
        ds, "convert-tracktor_raw_csv.0.1-bbbbbbbbbb", older, std_format="mosaic_v1"
    )

    planned = plan_pipeline(ds, declared_first)
    _ = run_pipeline(ds, declared_first)

    converted = planned.step("tracks").tracks_variant
    (written,) = set(read_labels_index(ds, "behavior")["run_id"])
    assert written == planned.step("labels").run_id
    record = read_labels_variant(ds.get_root("labels") / "behavior" / written)
    assert record is not None
    assert record.params["tracks_variant"] == converted
