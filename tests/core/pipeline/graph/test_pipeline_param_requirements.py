"""Requirements that a step's params name, checked at plan time and before apply.

A keypoint feature names its columns from its params: ``pair-facing`` from a
prefix and two indices, ``pair-egocentric`` and ``pair-posedistance-pca`` from a
``PoseConfig``. Tables whose keypoints are renamed plan and run. Tables lacking
a keypoint that the params select are refused at plan time and before apply,
each refusal naming the column.

Other features read a column only under some params, as ``approach-avoidance``
reads ``ANGLE`` while its orientation gate is set. The column is required under
those params alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.graph import (
    Recipe,
    RecipeRefused,
    build_feature,
    editor_catalog,
    plan_pipeline,
    read_recipe,
    run_pipeline,
    validation_report,
)
from mosaic.core.pipeline.column_contract import (
    MissingInputColumns,
    check_input_columns,
)
from mosaic.core.pipeline.manifest import tracks_table_columns
from mosaic.core.pipeline.run import run_feature
from mosaic.core.scope import Scope
from tests.helpers import add_tracks_variant, make_dataset

VARIANT = "convert-sleap.0.1-2222222222"

_RENAMED_POSE: dict[str, JsonValue] = {
    "pose_n": 5,
    "x_prefix": "kpX",
    "y_prefix": "kpY",
}

RENAMED: dict[str, dict[str, JsonValue]] = {
    "pair-facing": {
        "x_prefix": "kpX",
        "y_prefix": "kpY",
        "pose_head_index": 0,
        "pose_abdomen_index": 1,
    },
    "pair-egocentric": {"pose": _RENAMED_POSE, "neck_idx": 0, "tail_base_idx": 1},
    "pair-posedistance-pca": {"pose": _RENAMED_POSE},
}
"""Params that read five keypoints named ``kpX<i>`` and ``kpY<i>``."""


def _dataset(
    tmp_path: Path,
    *,
    keypoints: int,
    pose_prefixes: tuple[str, str] = ("poseX", "poseY"),
) -> Dataset:
    dataset = make_dataset(tmp_path / "keypoints")
    dataset.meta["fps_default"] = 30.0
    add_tracks_variant(
        dataset,
        VARIANT,
        "seq_a",
        "seq_b",
        std_format="mosaic_v1",
        keypoints=keypoints,
        pose_prefixes=pose_prefixes,
    )
    return dataset


def _recipe(feature: str, params: dict[str, JsonValue]) -> Recipe:
    return read_recipe(
        {
            "steps": [
                {"id": "step", "type": "feature", "feature": feature, "params": params}
            ]
        }
    )


@pytest.mark.parametrize("feature", sorted(RENAMED))
def test_keypoints_under_the_prefixes_the_params_name_plan_and_run(
    tmp_path: Path, feature: str
) -> None:
    dataset = _dataset(tmp_path, keypoints=5, pose_prefixes=("kpX", "kpY"))
    params = RENAMED[feature]

    report = validation_report(_recipe(feature, params).model_dump(mode="json"))
    plan = plan_pipeline(dataset, _recipe(feature, params))
    result = run_feature(dataset, build_feature(feature, ["tracks"], params))

    assert report.valid
    assert plan.problems == ()
    assert result.entries_written == 2
    assert result.failed_entries == ()


@pytest.mark.parametrize("feature", sorted(RENAMED))
def test_a_keypoint_that_the_params_select_and_the_tables_lack_is_refused(
    tmp_path: Path, feature: str
) -> None:
    """Each default selects keypoint 3, and the tables hold keypoints 0 to 2."""
    dataset = _dataset(tmp_path, keypoints=3)
    columns = tracks_table_columns(dataset, VARIANT, Scope())[("", "seq_a")]

    with pytest.raises(RecipeRefused) as caught:
        _ = plan_pipeline(dataset, _recipe(feature, {}))
    with pytest.raises(MissingInputColumns, match="lacks 'poseX3'"):
        check_input_columns(build_feature(feature, ["tracks"], {}), columns)

    messages = [problem.message for problem in caught.value.problems]
    assert any(message.startswith("needs 'poseX3'") for message in messages)


def test_a_refusal_names_only_the_providers_that_the_step_can_read(
    tmp_path: Path,
) -> None:
    """``egocentric-crop`` writes ``poseX*`` crop columns and no track-shaped table.

    ``pair-facing`` reads a track-shaped table alone and cannot read them.
    """
    dataset = _dataset(tmp_path, keypoints=3)

    with pytest.raises(RecipeRefused) as caught:
        _ = plan_pipeline(dataset, _recipe("pair-facing", {}))

    messages = [problem.message for problem in caught.value.problems]
    assert messages
    assert all("egocentric-crop" not in message for message in messages)
    assert all(
        message.endswith("Neither the tracks nor a registered feature provides it")
        for message in messages
    )


def test_a_fit_over_an_entry_lacking_a_selected_keypoint_names_it(
    tmp_path: Path,
) -> None:
    """``pair-posedistance-pca`` fits over every entry before it applies."""
    dataset = _dataset(tmp_path, keypoints=3)

    with pytest.raises(MissingInputColumns, match="lacks 'poseX3'"):
        _ = run_feature(dataset, build_feature("pair-posedistance-pca", ["tracks"], {}))


def test_the_catalog_lists_each_keypoint_that_the_defaults_select() -> None:
    """A ``PoseConfig`` default is read as the object it dumps to."""
    entries = {entry.name: entry for entry in editor_catalog().entries}
    requires = entries["pair-egocentric"].requires

    assert requires is not None
    assert [need.label for need in requires][-2:] == ["'poseX6'", "'poseY6'"]


# --- requirements under a condition on the params ---------------------------------


def test_a_column_read_under_the_default_params_is_refused_where_tables_lack_it(
    tmp_path: Path,
) -> None:
    """``approach-avoidance`` reads ``ANGLE`` under its default orientation gate."""
    dataset = _dataset(tmp_path, keypoints=3)

    with pytest.raises(RecipeRefused) as caught:
        _ = plan_pipeline(dataset, _recipe("approach-avoidance", {}))
    ungated = plan_pipeline(
        dataset, _recipe("approach-avoidance", {"orientation_gate_cos": None})
    )

    (problem,) = caught.value.problems
    assert problem.message.startswith("needs 'ANGLE'")
    assert ungated.problems == ()


_TRACK_COLUMNS: tuple[str, ...] = ("frame", "id", "group", "sequence", "X", "Y")


@pytest.mark.parametrize(
    ("feature", "params", "columns", "missing"),
    [
        ("collective-motion-metrics", {"heading_source": "orientation"}, (), "ANGLE"),
        (
            "local-order-metrics",
            {"heading_source": "orientation", "radius": 5.0},
            (),
            "ANGLE",
        ),
        ("scale-to-cm", {}, ("ftime",), "group"),
        ("pair-interaction-filter", {"pose_head_index": 2}, ("ANGLE",), "poseX2"),
        (
            "pair-interaction-filter",
            {"use_pixel_coords": False},
            ("ANGLE", "x_pixels", "y_pixels", "poseX0", "poseY0"),
            "X",
        ),
    ],
    ids=["cmm-orientation", "lom-orientation", "scale", "head", "body-center"],
)
def test_a_column_read_under_some_params_is_required_under_them(
    feature: str,
    params: dict[str, JsonValue],
    columns: tuple[str, ...],
    missing: str,
) -> None:
    present = [*_TRACK_COLUMNS, *columns]
    if missing in present:
        present.remove(missing)
    if missing == "X":
        present.remove("Y")

    with pytest.raises(MissingInputColumns, match=f"lacks '{missing}'"):
        check_input_columns(build_feature(feature, ["tracks"], params), present)


@pytest.mark.parametrize(
    ("feature", "params", "columns"),
    [
        ("collective-motion-metrics", {"heading_source": "auto"}, ()),
        ("local-order-metrics", {"heading_source": "velocity", "radius": 5.0}, ()),
        ("scale-to-cm", {"cm_per_pixel": 0.1, "columns": ["ftime"]}, ("ftime",)),
        ("pair-interaction-filter", {}, ("ANGLE",)),
        (
            "pair-interaction-filter",
            {"pose_head_index": 2},
            ("ANGLE", "poseX2", "poseY2"),
        ),
    ],
    ids=["cmm-auto", "lom-velocity", "scale-calibrated", "pixels", "head"],
)
def test_a_column_read_under_other_params_is_not_required(
    feature: str, params: dict[str, JsonValue], columns: tuple[str, ...]
) -> None:
    """``scale-to-cm`` with its factor set reads neither group nor sequence."""
    present = [*_TRACK_COLUMNS, *columns]
    if feature == "scale-to-cm":
        present = ["id", *columns]

    check_input_columns(build_feature(feature, ["tracks"], params), present)


@pytest.mark.parametrize("feature", ["approach-avoidance", "pair-position"])
def test_heading_beside_the_tracks_supplies_the_angle_that_a_feature_reads(
    tmp_path: Path, feature: str
) -> None:
    """``mosaic_v1`` tracks record no ``ANGLE``, and ``heading`` writes it."""
    dataset = _dataset(tmp_path, keypoints=3)
    recipe = read_recipe(
        {
            "steps": [
                {"id": "heading", "type": "feature", "feature": "heading"},
                {
                    "id": "step",
                    "type": "feature",
                    "feature": feature,
                    "inputs": ["tracks", {"step": "heading"}],
                },
            ]
        }
    )

    report = validation_report(recipe.model_dump(mode="json"))
    done = run_pipeline(dataset, recipe)

    assert report.valid
    assert [outcome.state for outcome in done.outcomes] == ["ran", "ran"]
