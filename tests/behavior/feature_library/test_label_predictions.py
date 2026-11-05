"""``label-predictions`` reads a predictions label kind onto the rows of its input.

Each row takes the event that labels it, by its own frame and individuals, and
the probabilities of that event. A row two events match is refused with its
entry. A variant with out-of-fold rows gives the run a ``cv_folds.parquet``,
by entry or by block, that the fold readers of the classifiers read.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.classifier_common import (
    FOLD_TABLE_FILE,
    FoldLookup,
    read_fold_table,
)
from mosaic.behavior.feature_library.evaluate_labels import (
    METRICS_FILE,
    EvaluateLabels,
)
from mosaic.behavior.feature_library.label_predictions import (
    LabelPredictions,
    PredictedEntry,
    prediction_fold_table,
)
from mosaic.behavior.feature_library.pair_position import PairPositionFeatures
from mosaic.core.dataset import Dataset
from mosaic.core.evaluation import EvaluationReport
from mosaic.core.helpers import make_entry_key
from mosaic.core.pipeline.graph import step_type_detail
from mosaic.core.pipeline.index import feature_run_root
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types import InputStream, Result
from tests.helpers import (
    REPO_ROOT,
    add_labels,
    add_predictions,
    add_tracks_table,
    make_dataset,
    write_predictions,
)

FRAMES = 6
ENTRY = ("g", "s")


def _track_rows(frames: int = FRAMES, ids: tuple[int, ...] = (0, 1)) -> pd.DataFrame:
    """Rows of each individual at every frame, as the tracks give them."""
    return pd.DataFrame(
        {
            "group": ENTRY[0],
            "sequence": ENTRY[1],
            "frame": np.repeat(np.arange(frames), len(ids)),
            "id": np.tile(ids, frames),
        }
    )


def _pair_rows(frames: int = FRAMES) -> pd.DataFrame:
    """The rows ``pair-position`` writes for individuals 0 and 1, focal on each in turn."""
    tracks = _track_rows(frames)
    moving = tracks.assign(
        time=tracks["frame"] / 10.0,
        X=10.0 * tracks["id"] + tracks["frame"],
        Y=5.0 * tracks["id"] + 0.5 * tracks["frame"],
        ANGLE=0.1 * tracks["frame"] + tracks["id"],
    )
    return PairPositionFeatures().apply(moving)


def _scene_rows(frames: int = FRAMES) -> pd.DataFrame:
    """One row per frame, naming no individual."""
    return pd.DataFrame(
        {"group": ENTRY[0], "sequence": ENTRY[1], "frame": np.arange(frames)}
    )


def _feature(
    files: dict[tuple[str, str], Path],
    run_root: Path,
    params: dict[str, object] | None = None,
) -> tuple[LabelPredictions, bool]:
    """The feature over *files*, with what its ``load_state`` answered."""
    feature = LabelPredictions(params=params or {})
    loaded = feature.load_state(run_root, {}, {"predictions": files})
    return feature, loaded


def _applied(
    files: dict[tuple[str, str], Path],
    rows: pd.DataFrame,
    tmp_path: Path,
    params: dict[str, object] | None = None,
) -> pd.DataFrame:
    """The output of *rows* through the feature over *files*."""
    feature, _ = _feature(files, tmp_path / "run", params)
    return feature.apply(rows)


def _predicted(out: pd.DataFrame) -> list[object]:
    """``predicted_label`` as a list, ``None`` for a null."""
    return [None if pd.isna(v) else int(v) for v in out["predicted_label"]]


# --- the rows ------------------------------------------------------------------


def test_individual_predictions_on_track_rows(tmp_path: Path) -> None:
    events = [(f, i, -1, (f + i) % 2) for f in range(FRAMES) for i in (0, 1)]
    probabilities = [[0.25, 0.75] if label else [0.75, 0.25] for *_, label in events]
    path = write_predictions(tmp_path / "p.npz", events, probabilities=probabilities)

    out = _applied({ENTRY: path}, _track_rows(), tmp_path)

    assert list(out.columns) == [
        "frame",
        "group",
        "id",
        "sequence",
        "predicted_label",
        "prob_0",
        "prob_1",
    ]
    assert str(out["predicted_label"].dtype) == "Int64"
    assert _predicted(out) == [label for *_, label in events]
    assert out["prob_1"].tolist() == [p[1] for p in probabilities]


def test_individual_predictions_on_pair_rows_take_the_focal_individual(
    tmp_path: Path,
) -> None:
    events = [(f, i, -1, i) for f in range(FRAMES) for i in (0, 1)]
    path = write_predictions(tmp_path / "p.npz", events)

    rows = _pair_rows()
    out = _applied({ENTRY: path}, rows, tmp_path)

    assert len(out) == 2 * FRAMES
    assert _predicted(out) == rows["id1"].tolist()
    assert not [c for c in out.columns if c.startswith("prob_")]
    assert {"id1", "id2", "perspective"} <= set(out.columns)


def test_pair_predictions_on_pair_rows_take_their_own_direction(
    tmp_path: Path,
) -> None:
    events = [(f, 0, 1, 1) for f in range(FRAMES)] + [
        (f, 1, 0, 0) for f in range(FRAMES)
    ]
    path = write_predictions(tmp_path / "p.npz", events)

    rows = _pair_rows()
    out = _applied({ENTRY: path}, rows, tmp_path)

    by_direction = {(0, 1): 1, (1, 0): 0}
    expected = [
        by_direction[(int(a), int(b))]
        for a, b in zip(rows["id1"], rows["id2"], strict=True)
    ]
    assert len(out) == 2 * FRAMES
    assert _predicted(out) == expected


def test_calms21_pair_predictions_on_track_rows_predict_the_resident_only(
    tmp_path: Path,
) -> None:
    events = [(f, 0, 1, f % 2) for f in range(FRAMES)]
    probabilities = [[0.5, 0.5]] * FRAMES
    path = write_predictions(tmp_path / "p.npz", events, probabilities=probabilities)

    out = _applied({ENTRY: path}, _track_rows(), tmp_path)

    resident = out[out["id"] == 0]
    intruder = out[out["id"] == 1]
    assert _predicted(resident) == [f % 2 for f in range(FRAMES)]
    assert _predicted(intruder) == [None] * FRAMES
    assert intruder["prob_0"].isna().all()
    assert resident["prob_0"].notna().all()


def test_scene_predictions_label_every_row_of_their_frame(tmp_path: Path) -> None:
    events = [(f, -1, -1, f % 2) for f in range(FRAMES)]
    path = write_predictions(tmp_path / "p.npz", events)

    on_tracks = _applied({ENTRY: path}, _track_rows(), tmp_path)
    on_scene = _applied({ENTRY: path}, _scene_rows(), tmp_path)

    assert _predicted(on_tracks) == [f % 2 for f in range(FRAMES) for _ in (0, 1)]
    assert _predicted(on_scene) == [f % 2 for f in range(FRAMES)]


def test_a_frame_without_an_event_is_null_with_nan_probabilities(
    tmp_path: Path,
) -> None:
    path = write_predictions(
        tmp_path / "p.npz", [(0, -1, -1, 1)], probabilities=[[0.2, 0.8]]
    )

    out = _applied({ENTRY: path}, _scene_rows(3), tmp_path)

    assert _predicted(out) == [1, None, None]
    assert out["prob_1"].tolist()[0] == 0.8
    assert out[["prob_0", "prob_1"]].iloc[1:].isna().all().all()


# --- refusals of an entry ----------------------------------------------------------


def test_a_row_two_events_match_is_refused_naming_its_entry_frame_and_ids(
    tmp_path: Path,
) -> None:
    # Two pairs focal on individual 0 at frame 2, read onto individual rows.
    events = [(2, 0, 1, 0), (2, 0, 2, 1)]
    path = write_predictions(tmp_path / "p.npz", events)
    feature, _ = _feature({ENTRY: path}, tmp_path / "run")

    with pytest.raises(ValueError, match=r"'g'.*'s'.*frame 2.*id 0.*2 events"):
        _ = feature.apply(_track_rows(ids=(0, 1, 2)))


def test_two_individual_events_on_rows_without_ids_are_refused(
    tmp_path: Path,
) -> None:
    path = write_predictions(tmp_path / "p.npz", [(1, 0, -1, 0), (1, 1, -1, 1)])
    feature, _ = _feature({ENTRY: path}, tmp_path / "run")

    with pytest.raises(ValueError, match="frame 1"):
        _ = feature.apply(_scene_rows())


def test_an_entry_without_a_file_writes_null_predictions(tmp_path: Path) -> None:
    other = write_predictions(
        tmp_path / "p.npz", [(0, -1, -1, 0)], probabilities=[[1.0, 0.0]]
    )

    out = _applied({("g", "other"): other}, _track_rows(), tmp_path)

    assert _predicted(out) == [None] * (2 * FRAMES)
    assert out[["prob_0", "prob_1"]].isna().all().all()


def test_an_entry_without_a_file_is_refused_under_error(tmp_path: Path) -> None:
    other = write_predictions(tmp_path / "p.npz", [(0, -1, -1, 0)])
    feature, _ = _feature(
        {("g", "other"): other},
        tmp_path / "run",
        {"missing_predictions": "error"},
    )

    with pytest.raises(ValueError, match=r"'g'.*'s'.*missing_predictions"):
        _ = feature.apply(_track_rows())


# --- refusals of a variant ---------------------------------------------------------


def test_files_listing_different_classes_are_refused(tmp_path: Path) -> None:
    first = write_predictions(
        tmp_path / "a.npz", [(0, -1, -1, 0)], probabilities=[[1.0, 0.0]]
    )
    second = write_predictions(
        tmp_path / "b.npz",
        [(0, -1, -1, 0)],
        classes=(0, 1, 2),
        probabilities=[[1.0, 0.0, 0.0]],
    )

    with pytest.raises(ValueError, match=r"\[0, 1\].*\[0, 1, 2\]"):
        _ = _feature({("g", "a"): first, ("g", "b"): second}, tmp_path / "run")


def test_probabilities_in_some_files_only_are_refused(tmp_path: Path) -> None:
    first = write_predictions(
        tmp_path / "a.npz", [(0, -1, -1, 0)], probabilities=[[1.0, 0.0]]
    )
    second = write_predictions(tmp_path / "b.npz", [(0, -1, -1, 0)])

    with pytest.raises(ValueError, match="probabilities"):
        _ = _feature({("g", "a"): first, ("g", "b"): second}, tmp_path / "run")


def test_folds_in_some_files_only_are_accepted_and_list_only_those_entries(
    tmp_path: Path,
) -> None:
    folded = write_predictions(
        tmp_path / "a.npz", [(0, -1, -1, 0), (1, -1, -1, 1)], folds=[1, 1]
    )
    plain = write_predictions(tmp_path / "b.npz", [(0, -1, -1, 0)])
    run_root = tmp_path / "run"

    feature, loaded = _feature({("g", "a"): folded, ("g", "b"): plain}, run_root)
    assert not loaded
    feature.save_state(run_root)

    table = pd.read_parquet(run_root / FOLD_TABLE_FILE)
    assert table.to_dict("list") == {
        "group": ["g"],
        "sequence": ["a"],
        "fold": [1],
        "n_rows": [2],
        "rows_0": [1],
        "rows_1": [1],
    }
    again, loaded_again = _feature({("g", "a"): folded, ("g", "b"): plain}, run_root)
    assert loaded_again
    assert _predicted(again.apply(_scene_rows(2).assign(sequence="b"))) == [0, None]


def test_a_variant_without_folds_needs_no_fit(tmp_path: Path) -> None:
    path = write_predictions(tmp_path / "p.npz", [(0, -1, -1, 0)], folds=[-1])

    _, loaded = _feature({ENTRY: path}, tmp_path / "run")

    assert loaded
    assert not (tmp_path / "run" / FOLD_TABLE_FILE).exists()


# --- label controls ------------------------------------------------------------------


def test_a_mask_nulls_the_masked_rows_and_their_probabilities(tmp_path: Path) -> None:
    events = [(f, -1, -1, f % 2) for f in range(FRAMES)]
    probabilities = [[0.4, 0.6] if f % 2 else [0.6, 0.4] for f in range(FRAMES)]
    path = write_predictions(tmp_path / "p.npz", events, probabilities=probabilities)

    out = _applied(
        {ENTRY: path},
        _scene_rows(),
        tmp_path,
        {"predictions": {"kind": "predictions", "mask": [1]}},
    )

    assert _predicted(out) == [0, None] * (FRAMES // 2)
    assert out.loc[1::2, ["prob_0", "prob_1"]].isna().all().all()
    assert out.loc[0::2, "prob_0"].tolist() == [0.6] * (FRAMES // 2)


def test_a_remap_without_probabilities_remaps_the_predicted_label(
    tmp_path: Path,
) -> None:
    events = [(f, -1, -1, f % 2) for f in range(FRAMES)]
    path = write_predictions(tmp_path / "p.npz", events)

    out = _applied(
        {ENTRY: path},
        _scene_rows(),
        tmp_path,
        {"predictions": {"kind": "predictions", "remap": {1: 5}}},
    )

    assert _predicted(out) == [0, 5] * (FRAMES // 2)


def test_a_remap_with_probabilities_is_refused(tmp_path: Path) -> None:
    path = write_predictions(
        tmp_path / "p.npz", [(0, -1, -1, 0)], probabilities=[[1.0, 0.0]]
    )

    with pytest.raises(ValueError, match="remap"):
        _ = _feature(
            {ENTRY: path},
            tmp_path / "run",
            {"predictions": {"kind": "predictions", "remap": {1: 0}}},
        )


# --- the fold table ------------------------------------------------------------------


def _entry(
    sequence: str,
    frames: list[int],
    folds: list[int] | None,
    blocks: list[int] | None = None,
    labels: list[int] | None = None,
) -> PredictedEntry:
    return PredictedEntry(
        group="g",
        sequence=sequence,
        frames=np.asarray(frames, dtype=np.int64),
        labels=np.asarray(labels or [0] * len(frames), dtype=np.int64),
        label_ids=(0, 1),
        folds=None if folds is None else np.asarray(folds, dtype=np.int64),
        blocks=None if blocks is None else np.asarray(blocks, dtype=np.int64),
    )


def test_three_entries_of_one_fold_each_give_a_table_by_entry() -> None:
    table = prediction_fold_table(
        [
            _entry("a", [0, 1], [0, 0], labels=[0, 1]),
            _entry("b", [0, 1, 2], [1, 1, 1]),
            _entry("c", [4], [2], labels=[1]),
        ]
    )

    assert table is not None
    assert table.to_dict("list") == {
        "group": ["g", "g", "g"],
        "sequence": ["a", "b", "c"],
        "fold": [0, 1, 2],
        "n_rows": [2, 3, 1],
        "rows_0": [1, 3, 0],
        "rows_1": [1, 0, 1],
    }
    lookup = FoldLookup.from_table(table)
    assert lookup.folds(("g", "b"), [0, 9]).tolist() == [1, 1]


def test_two_runs_of_one_fold_without_blocks_are_two_blocks() -> None:
    table = prediction_fold_table(
        [_entry("a", [0, 1, 2, 3, 4, 5], [0, 0, -1, 0, 1, 1])]
    )

    assert table is not None
    assert table[["block", "start", "stop", "fold", "n_rows"]].to_dict("list") == {
        "block": [0, 1, 2],
        "start": [0, 3, 4],
        "stop": [2, 4, 6],
        "fold": [0, 0, 1],
        "n_rows": [2, 1, 2],
    }
    lookup = FoldLookup.from_table(table)
    assert lookup.folds(("g", "a"), [0, 1, 2, 3, 4, 5]).tolist() == [
        0,
        0,
        -1,
        0,
        1,
        1,
    ]


def test_a_file_with_blocks_keeps_its_block_values() -> None:
    table = prediction_fold_table(
        [_entry("a", [0, 1, 2, 3], [1, 1, 0, 0], blocks=[7, 7, 3, 3])]
    )

    assert table is not None
    assert table[["block", "start", "stop", "fold"]].to_dict("list") == {
        "block": [3, 7],
        "start": [2, 0],
        "stop": [4, 2],
        "fold": [0, 1],
    }
    assert FoldLookup.from_table(table).folds(("g", "a"), [1, 2]).tolist() == [1, 0]


def test_an_entry_of_one_fold_beside_one_that_needs_blocks_is_one_block() -> None:
    table = prediction_fold_table(
        [
            _entry("a", [2, 3, 5], [1, 1, 1]),
            _entry("b", [0, 1], [0, 1]),
        ]
    )

    assert table is not None
    assert table[["sequence", "block", "start", "stop", "fold"]].to_dict("list") == {
        "sequence": ["a", "b", "b"],
        "block": [0, 0, 1],
        "start": [2, 0, 1],
        "stop": [6, 1, 2],
        "fold": [1, 0, 1],
    }
    lookup = FoldLookup.from_table(table)
    assert lookup.folds(("g", "a"), [2, 4, 6]).tolist() == [1, 1, -1]


def test_one_block_value_given_to_two_runs_is_refused() -> None:
    with pytest.raises(ValueError, match="block 4 to two runs"):
        _ = prediction_fold_table(
            [_entry("a", [0, 1, 2], [0, -1, 0], blocks=[4, -1, 4])]
        )


def test_entries_without_an_out_of_fold_row_give_no_table() -> None:
    assert prediction_fold_table([_entry("a", [0], None), _entry("b", [0], [-1])]) is (
        None
    )


def test_an_entry_in_sample_beside_folds_is_not_listed() -> None:
    table = prediction_fold_table([_entry("a", [0], [-1]), _entry("b", [0], [2])])

    assert table is not None
    assert read_fold_table_rows(table) == [("g", "b", 2)]


def read_fold_table_rows(table: pd.DataFrame) -> list[tuple[str, str, int]]:
    """The ``(group, sequence, fold)`` of each row of a table by entry."""
    return [
        (str(g), str(s), int(f))
        for g, s, f in zip(
            table["group"], table["sequence"], table["fold"], strict=True
        )
    ]


# --- through run_feature -------------------------------------------------------------

TRACKS_VARIANT = "convert-pixels.0.1-aaaaaaaaaa"


def _dataset(tmp_path: Path, sequences: tuple[str, ...]) -> Dataset:
    """Tracks of individuals 0 and 1 over :data:`FRAMES` frames in each of *sequences*."""
    dataset = make_dataset(tmp_path / "dataset")
    for sequence in sequences:
        table = _track_rows().assign(
            sequence=sequence, X=1.0, Y=2.0, time=lambda d: d["frame"] / 10.0
        )
        add_tracks_table(dataset, TRACKS_VARIANT, table, std_format="mosaic_v1")
    return dataset


def _output(dataset: Dataset, result: Result) -> pd.DataFrame:
    """Every parquet output of the run *result*, concatenated by entry."""
    assert result.run_id is not None
    root = feature_run_root(dataset, result.feature, result.run_id)
    return pd.concat(
        [pd.read_parquet(path) for path in sorted(root.glob("g__*.parquet"))],
        ignore_index=True,
    )


def test_a_row_two_events_match_fails_its_entry_and_the_others_are_written(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, ("good", "bad"))
    _ = add_predictions(dataset, "g", "good", [(f, 0, -1, 1) for f in range(FRAMES)])
    _ = add_predictions(dataset, "g", "bad", [(3, 0, -1, 0), (3, -1, -1, 1)])

    result = run_feature(dataset, LabelPredictions())

    assert result.failed_entries == (make_entry_key("g", "bad"),)
    written = _output(dataset, result)
    assert set(written["sequence"]) == {"good"}
    assert _predicted(written[written["id"] == 0]) == [1] * FRAMES


def test_missing_predictions_null_writes_and_error_fails_the_entry(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path, ("present", "absent"))
    _ = add_predictions(dataset, "g", "present", [(0, -1, -1, 1)])

    nulls = run_feature(dataset, LabelPredictions())
    errors = run_feature(
        dataset, LabelPredictions(params={"missing_predictions": "error"})
    )

    assert not nulls.failed_entries
    absent = _output(dataset, nulls).query("sequence == 'absent'")
    assert _predicted(absent) == [None] * (2 * FRAMES)
    assert errors.failed_entries == (make_entry_key("g", "absent"),)
    assert errors.run_id != nulls.run_id


def _scored_bridge(tmp_path: Path, predictions: dict[str, object]) -> Path:
    """The run root of ``evaluate-labels`` over a bridge of classes 0 and 1.

    The labels hold class 2 on frames 4 and 5 of the resident.
    """
    dataset = _dataset(tmp_path, ("s",))
    events = [(f, 0, -1, f % 2) for f in range(FRAMES)]
    _ = add_predictions(
        dataset,
        "g",
        "s",
        events,
        probabilities=[[0.3, 0.7] if f % 2 else [0.7, 0.3] for f in range(FRAMES)],
    )
    _ = add_labels(
        dataset, "g", "s", [(f, 0, -1, f % 2 if f < 4 else 2) for f in range(FRAMES)]
    )
    bridge = run_feature(dataset, LabelPredictions(params={"predictions": predictions}))
    score = run_feature(
        dataset,
        EvaluateLabels(
            EvaluateLabels.Inputs(
                (Result(feature=bridge.feature, run_id=bridge.run_id),)
            ),
            {},
        ),
    )
    assert score.run_id is not None
    return feature_run_root(dataset, score.feature, score.run_id)


def test_a_masked_bridge_refuses_a_truth_class_it_has_no_probability_for(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match=r"classes \[2\], for which the input writes no prob_"
    ):
        _ = _scored_bridge(tmp_path, {"kind": "predictions", "mask": [1]})


def test_an_unmasked_bridge_scores_a_truth_class_it_lacks_as_misses(
    tmp_path: Path,
) -> None:
    root = _scored_bridge(tmp_path, {"kind": "predictions"})

    report = EvaluationReport.model_validate(
        json.loads((root / METRICS_FILE).read_text())
    )
    by_label = {score.label: score for score in report.per_class}
    assert by_label[2].recall == 0.0
    assert report.n_scored == FRAMES


def test_fit_reads_no_input(tmp_path: Path) -> None:
    path = write_predictions(tmp_path / "p.npz", [(0, -1, -1, 0)], folds=[0])
    feature, loaded = _feature({ENTRY: path}, tmp_path / "run")
    assert not loaded

    def refuse() -> Iterator[tuple[str, pd.DataFrame]]:
        raise AssertionError("fit read its input")

    feature.fit(InputStream(refuse, n_entries=1))
    feature.save_state(tmp_path / "run")

    assert read_fold_table(tmp_path / "run" / FOLD_TABLE_FILE).to_dict("list") == {
        "group": ["g"],
        "sequence": ["s"],
        "fold": [0],
    }


def test_the_default_kind_is_predictions_where_the_run_reads_it() -> None:
    params = LabelPredictions.Params()

    assert params.predictions.kind == "predictions"
    assert LabelPredictions(params={}).params.predictions.kind == "predictions"
    assert (
        LabelPredictions(
            params={"predictions": {"kind": "behavior"}}
        ).params.predictions.kind
        == "behavior"
    )


def test_the_step_detail_shows_predictions_as_the_default_kind() -> None:
    rows = {row.path: row for row in step_type_detail("label-predictions").params}

    assert rows["predictions"].constructed
    assert rows["predictions.kind"].default == "predictions"


def test_the_reference_page_shows_predictions_as_the_default_kind() -> None:
    page = (REPO_ROOT / "docs" / "reference" / "features.md").read_text("utf-8")
    section = page.split("### `label-predictions`", 1)[1].split("\n### ", 1)[0]
    model = section.split('??? note "`PredictionsLabelsSource`"', 1)[1]
    kind_row = next(
        line for line in model.splitlines() if line.startswith("    | `kind` |")
    )

    assert '| `string` | `"predictions"` |' in kind_row
