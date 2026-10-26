"""The select-frames parameters, the frames chosen for one entry, and the op's run.

The tables are synthetic tracks of three keypoints on a horizontal line ten
pixels long. Every present row therefore measures a body length of 10. The index
rows are ``pd.Series`` of text cells, as ``read_tracks_index`` returns them. The
op's tests publish such tables into a tracks variant and run the op over it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

import mosaic.tracking.frame_extraction.dataset_runs as dataset_runs
import mosaic.tracking.frame_extraction.selection as selection
from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry, make_entry_key
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.graph.resolve import describe_step_type
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.index_lock import lock_path_for
from mosaic.core.pipeline.markers import (
    INFLIGHT_MARKER_NAME,
    new_inflight,
    read_inflight,
    write_inflight,
)
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import OPS, OpIdentity, op_resource_class, run_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import CellRow, read_tracks_index
from mosaic.core.scope import Scope
from mosaic.core.track_quality import select
from mosaic.runlog import RunLogSnapshot, read_run, run_log_dir
from mosaic.tracking import register_ops
from mosaic.tracking.frame_extraction.dataset_runs import (
    EntryFrames,
    ExtractFramesParams,
    extract_frames,
    frames_run_id,
)
from mosaic.tracking.frame_extraction.selection import (
    SELECTION_COLUMNS,
    EntrySelection,
    LowScoreCriterion,
    MissingKeypointsCriterion,
    SelectFramesConflict,
    SelectFramesNotFound,
    SelectFramesParams,
    select_entry_frames,
)
from mosaic.tracking.frame_extraction.selection_index import (
    EXTRACT_PARAMS_FILE_NAME,
    SELECTION_FILE_NAME,
    frame_selection_index,
    selected_frames,
    selection_index_path,
    selections_kind_root,
)
from tests.helpers import (
    add_tracks_table,
    entry_error_lines,
    index_locked_out,
    make_dataset,
    record_entry_selection,
)

register_ops()

VARIANT = "trex.0.1-0123456789"
SPAN = 100
LENGTH = 10.0

type Position = Callable[[int], tuple[float, float]]
"""The center of one id at a frame."""


def _steady(x: float, y: float) -> Position:
    """Return a center that moves one pixel per frame along x from (*x*, *y*)."""
    return lambda frame: (x + frame, y)


def _pose_row(
    frame: int, id_: int, x: float, y: float, *, score: float = 0.9
) -> dict[str, object]:
    """Return a row with three keypoints on a horizontal line centered on (x, y)."""
    row: dict[str, object] = {"frame": frame, "id": id_, "X": x, "Y": y}
    for k, offset in enumerate((-LENGTH / 2, 0.0, LENGTH / 2)):
        row[f"poseX{k}"] = x + offset
        row[f"poseY{k}"] = y
        row[f"poseP{k}"] = score
    return row


def _table(
    positions: Mapping[int, Position] | None = None, *, span: int = SPAN
) -> pd.DataFrame:
    """Return a table of every id of *positions* at every frame of the span.

    The default is two ids far apart, id 1 at (100, 100) and id 2 at (300, 300),
    each moving one pixel per frame.
    """
    tracks = positions or {1: _steady(100.0, 100.0), 2: _steady(300.0, 300.0)}
    rows = [
        _pose_row(frame, id_, *position(frame))
        for frame in range(span)
        for id_, position in tracks.items()
    ]
    return pd.DataFrame(rows)


def _jumping_at(onset: int) -> Position:
    """Return a center that steps 50 pixels along x at *onset* and stays there."""
    return lambda frame: (100.0 + frame + (50.0 if frame >= onset else 0.0), 100.0)


def _row(**cells: str) -> pd.Series:
    """Return a tracks-index row of *VARIANT* for entry ``(g, s1)``, a trex table."""
    base = {
        "run_id": VARIANT,
        "group": "g",
        "sequence": "s1",
        "producer": "trex",
        "media_frames": str(SPAN),
        "frames_read": "",
        "known_tail_loss": "",
    }
    return pd.Series({**base, **cells})


def _params(**fields: object) -> SelectFramesParams:
    """Return the parameters over *VARIANT*, with *fields* set."""
    return SelectFramesParams.model_validate(
        {"source_tracks_run_id": VARIANT, **fields}
    )


def _frames(selection: pd.DataFrame) -> list[int]:
    """Return the chosen frames of *selection*, in order."""
    return [int(frame) for frame in selection["frame"]]


# --- parameters --------------------------------------------------------------


def test_the_defaults_match_the_documented_criteria() -> None:
    params = _params()
    assert (params.n_frames, params.min_separation_frames) == (20, 30)
    assert (params.min_event_frames, params.random_state) == (1, 0)
    assert params.n_expected is None
    assert params.body_length_px is None
    assert (params.jump.enabled, params.jump.body_lengths) == (True, 0.8)
    assert params.low_score.enabled
    assert (params.low_score.threshold, params.low_score.quantile) == (None, 0.05)
    assert (params.missing_keypoints.enabled, params.missing_keypoints.keypoints) == (
        True,
        (),
    )
    assert (params.duplicate.enabled, params.duplicate.body_lengths) == (True, 0.2)
    assert params.exclude == ()


def test_a_low_score_threshold_alone_clears_the_default_quantile() -> None:
    criterion = LowScoreCriterion.model_validate({"threshold": 0.3})
    assert (criterion.threshold, criterion.quantile) == (0.3, None)
    assert _params(low_score={"threshold": 0.3}).low_score.quantile is None


def test_a_low_score_threshold_survives_a_round_trip_through_its_dump() -> None:
    """The recorded parameters validate again, with the quantile cleared."""
    params = _params(low_score={"threshold": 0.3})
    again = SelectFramesParams.model_validate(params.model_dump())
    assert again == params


def test_a_low_score_threshold_and_quantile_together_are_refused() -> None:
    with pytest.raises(ValidationError, match="threshold or a quantile"):
        _ = LowScoreCriterion.model_validate({"threshold": 0.3, "quantile": 0.1})


def test_an_enabled_low_score_without_a_threshold_or_quantile_is_refused() -> None:
    with pytest.raises(ValidationError, match="neither a threshold nor a quantile"):
        _ = LowScoreCriterion.model_validate({"quantile": None})
    off = LowScoreCriterion.model_validate({"enabled": False, "quantile": None})
    assert not off.enabled


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"head": "nose"}, id="head"),
        pytest.param({"tail": "tail"}, id="tail"),
        pytest.param(
            {"missing_keypoints": {"keypoints": [0, "nose"]}}, id="missing-keypoints"
        ),
    ],
)
def test_a_keypoint_name_without_keypoint_names_is_refused(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="keypoint_names"):
        _ = _params(**fields)


def test_keypoint_names_without_a_name_reference_are_refused() -> None:
    with pytest.raises(ValidationError, match="keypoint_names"):
        _ = _params(keypoint_names=["nose", "mid", "tail"], head=0, tail=2)


def test_keypoint_names_with_a_name_reference_validate() -> None:
    params = _params(keypoint_names=["nose", "mid", "tail"], head="nose", tail=2)
    assert params.keypoint_names == ("nose", "mid", "tail")


def test_exclude_is_sorted_by_entry_and_frame() -> None:
    params = _params(
        exclude=[
            {"group": "g", "sequence": "s2", "indices": [9, 3, 3]},
            {"group": "g", "sequence": "s1", "indices": [5]},
        ]
    )
    assert params.exclude == (
        EntryFrames(group="g", sequence="s1", indices=(5,)),
        EntryFrames(group="g", sequence="s2", indices=(3, 9)),
    )


def test_an_entry_excluded_twice_is_refused() -> None:
    with pytest.raises(ValidationError, match="exclude names"):
        _ = _params(
            exclude=[
                {"group": "g", "sequence": "s1", "indices": [5]},
                {"group": "g", "sequence": "s1", "indices": [7]},
            ]
        )


# --- keypoint references -----------------------------------------------------


def test_keypoint_names_of_another_length_fail_the_entry() -> None:
    params = _params(keypoint_names=["nose", "tail"], head="nose")
    with pytest.raises(ValueError, match="2 keypoint names") as caught:
        _ = select_entry_frames(_table(), _row(), params)
    assert "3 keypoints" in str(caught.value)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param(
            {"keypoint_names": ["nose", "mid", "tail"], "head": "snout"},
            "no keypoint named 'snout'",
            id="unknown-name",
        ),
        pytest.param({"tail": 5}, "outside the table's 3 keypoints", id="past-index"),
        pytest.param(
            {"missing_keypoints": {"keypoints": [7]}},
            "outside the table's 3 keypoints",
            id="past-missing-keypoint",
        ),
        pytest.param(
            {"missing_keypoints": {"enabled": False, "keypoints": [7]}},
            "outside the table's 3 keypoints",
            id="past-missing-keypoint-disabled",
        ),
    ],
)
def test_an_unresolved_keypoint_reference_fails_the_entry(
    fields: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = select_entry_frames(_table(), _row(), _params(**fields))


def test_keypoints_named_or_indexed_choose_the_same_frames() -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    named = _params(keypoint_names=["nose", "mid", "tail"], head="nose", tail="tail")
    indexed = _params(head=0, tail=2)
    assert _frames(select_entry_frames(table, _row(), named).frames) == [40]
    assert _frames(select_entry_frames(table, _row(), indexed).frames) == [40]


# --- frame axis --------------------------------------------------------------


def test_a_blank_media_frames_fails_the_entry_naming_variant_and_producer() -> None:
    row = _row(producer="convert-calms21_npy", media_frames="")
    with pytest.raises(ValueError, match="does not record a frame axis") as caught:
        _ = select_entry_frames(_table(), row, _params())
    message = str(caught.value)
    assert VARIANT in message
    assert "convert-calms21_npy" in message


def test_a_mismatched_frame_axis_fails_the_entry_with_both_counts() -> None:
    row = _row(frames_read="90", known_tail_loss="2")
    with pytest.raises(ValueError, match="90") as caught:
        _ = select_entry_frames(_table(), row, _params())
    assert str(SPAN) in str(caught.value)


def test_a_known_tail_loss_is_accepted() -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    row = _row(frames_read="98", known_tail_loss="2")
    assert _frames(select_entry_frames(table, row, _params()).frames) == [40]


def test_a_table_past_the_media_fails_the_entry() -> None:
    with pytest.raises(ValueError, match="outside the span"):
        _ = select_entry_frames(_table(span=120), _row(), _params())


# --- identity producers ------------------------------------------------------


@pytest.mark.parametrize("producer", ["infer-pose", "infer-points", "infer-localizer"])
def test_an_infer_producer_skips_jump_and_lists_it(producer: str) -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    chosen = select_entry_frames(table, _row(producer=producer), _params())
    assert chosen.skipped_criteria == ("jump",)
    assert chosen.frames.empty
    assert chosen.n_candidates == 0


def test_an_infer_producer_forms_row_events_per_frame() -> None:
    """Missing keypoints at consecutive frames under changing ids are one event."""
    table = _table()
    gaps = ((table["frame"] == 10) & (table["id"] == 1)) | (
        (table["frame"] == 11) & (table["id"] == 2)
    )
    table.loc[gaps, "poseX1"] = np.nan
    chosen = select_entry_frames(table, _row(producer="infer-pose"), _params())
    assert _frames(chosen.frames) == [10]
    assert chosen.frames["event_frames"].tolist() == [2]


def test_a_tracker_evaluates_jump() -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    chosen = select_entry_frames(table, _row(), _params())
    assert chosen.skipped_criteria == ()
    assert chosen.frames["event_id"].tolist() == ["jump:1:40"]


# --- body length -------------------------------------------------------------


def test_the_measured_body_length_is_reported() -> None:
    assert select_entry_frames(_table(), _row(), _params()).body_length_px == LENGTH


def test_an_unknown_body_length_fails_an_entry_that_needs_one() -> None:
    table = _table()[["frame", "id", "X", "Y"]]
    with pytest.raises(ValueError, match="body_length_px"):
        _ = select_entry_frames(table, _row(), _params())


def test_an_unknown_body_length_passes_without_a_criterion_in_body_lengths() -> None:
    table = _table()[["frame", "id", "X", "Y"]]
    params = _params(jump={"enabled": False}, duplicate={"enabled": False})
    chosen = select_entry_frames(table, _row(), params)
    assert chosen.body_length_px is None
    assert chosen.frames.empty


def test_a_given_body_length_scales_a_table_without_keypoints() -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    centers = table[["frame", "id", "X", "Y"]]
    chosen = select_entry_frames(centers, _row(), _params(body_length_px=LENGTH))
    assert chosen.body_length_px == LENGTH
    assert _frames(chosen.frames) == [40]


def test_a_table_without_scores_skips_low_score_and_lists_it() -> None:
    table = _table().drop(columns=["poseP0", "poseP1", "poseP2"])
    chosen = select_entry_frames(table, _row(), _params())
    assert chosen.skipped_criteria == ("low_score",)


# --- criteria ----------------------------------------------------------------


def _flagged_table() -> pd.DataFrame:
    """Return a table with one event of each row criterion.

    Id 2 misses keypoint 1 at frame 10, id 1 jumps at frame 40 and id 2 scores
    0.1 at frame 70.
    """
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    missing = (table["frame"] == 10) & (table["id"] == 2)
    table.loc[missing, "poseX1"] = np.nan
    low = (table["frame"] == 70) & (table["id"] == 2)
    table.loc[low, ["poseP0", "poseP1", "poseP2"]] = 0.1
    return table


def test_the_selection_has_its_columns_sorted_by_frame() -> None:
    params = _params(
        n_expected=3, min_separation_frames=5, low_score={"threshold": 0.5}
    )
    chosen = select_entry_frames(_flagged_table(), _row(), params)
    selection = chosen.frames
    assert tuple(selection.columns) == SELECTION_COLUMNS
    assert _frames(selection) == [0, 10, 40, 70]
    assert selection["cause"].tolist() == [
        "fewer_than_expected",
        "missing_keypoints",
        "jump",
        "low_score",
    ]
    assert selection["event_id"].tolist() == [
        "fewer_than_expected:*:0",
        "missing_keypoints:2:10",
        "jump:1:40",
        "low_score:2:70",
    ]
    assert selection["id"].tolist() == ["", "2", "1", "2"]
    assert selection["event_frames"].tolist() == [SPAN, 1, 1, 1]
    assert selection["group"].tolist() == ["g"] * 4
    assert selection["sequence"].tolist() == ["s1"] * 4
    assert np.isnan(selection["x"].iloc[0])
    assert selection["x"].iloc[2] == 190.0
    assert selection["score"].iloc[3] == pytest.approx(0.1)
    assert selection["frame"].dtype == np.int64
    assert selection["event_frames"].dtype == np.int64
    for column in ("x", "y", "score"):
        assert selection[column].dtype == np.float64
    assert chosen.n_candidates == 4


@pytest.mark.parametrize("criterion", ["jump", "low_score", "missing_keypoints"])
def test_a_disabled_criterion_flags_nothing(criterion: str) -> None:
    fields: dict[str, object] = {
        "min_separation_frames": 5,
        "low_score": {"threshold": 0.5},
        criterion: {"enabled": False},
    }
    chosen = select_entry_frames(_flagged_table(), _row(), _params(**fields))
    assert criterion not in chosen.frames["cause"].tolist()
    assert len(chosen.frames) == 2


def test_missing_keypoints_reads_only_the_keypoints_named() -> None:
    params = _params(missing_keypoints=MissingKeypointsCriterion(keypoints=(0, 2)))
    chosen = select_entry_frames(_flagged_table(), _row(), params)
    assert "missing_keypoints" not in chosen.frames["cause"].tolist()


def test_duplicate_rows_flag_their_frame() -> None:
    table = _table({1: _steady(100.0, 100.0), 2: _steady(300.0, 300.0)})
    close = (table["frame"] == 50) & (table["id"] == 2)
    table.loc[close, ["X", "Y"]] = [150.5, 100.0]
    chosen = select_entry_frames(table, _row(), _params(jump={"enabled": False}))
    assert chosen.frames["event_id"].tolist() == ["duplicate:*:50"]


def test_short_events_are_dropped() -> None:
    params = _params(min_event_frames=2, low_score={"threshold": 0.5})
    chosen = select_entry_frames(_flagged_table(), _row(), params)
    assert chosen.frames.empty


# --- selection ---------------------------------------------------------------


def _many_jumps() -> pd.DataFrame:
    """Return a table whose id 1 jumps every fifth frame, 19 jump events in all."""
    return _table(
        {
            1: lambda frame: (100.0 + 50.0 * (frame // 5 % 2), 100.0),
            2: _steady(300.0, 300.0),
        }
    )


def _seed(random_state: int, group: str, sequence: str) -> int:
    """Return the documented seed of an entry's permutation."""
    unit = f"{random_state}\0{make_entry_key(group, sequence)}"
    digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def test_the_permutation_is_seeded_from_the_random_state_and_entry_key() -> None:
    table = _many_jumps()
    every = select_entry_frames(
        table, _row(), _params(n_frames=100, min_separation_frames=0)
    ).frames
    assert len(every) == 19
    params = _params(n_frames=3, min_separation_frames=0, random_state=7)
    chosen = select_entry_frames(table, _row(), params).frames
    expected = select(
        every.drop(columns=["group", "sequence"]),
        n_frames=3,
        min_separation=0,
        exclude=(),
        seed=_seed(7, "g", "s1"),
    )
    assert _frames(chosen) == _frames(expected)


def test_an_entrys_picks_do_not_depend_on_the_other_entries() -> None:
    table = _many_jumps()
    alone = select_entry_frames(
        table, _row(), _params(n_frames=3, min_separation_frames=0)
    ).frames
    beside = _params(
        n_frames=3,
        min_separation_frames=0,
        exclude=[{"group": "g", "sequence": "s2", "indices": [5, 10, 15]}],
    )
    _ = select_entry_frames(table, _row(sequence="s2"), beside)
    again = select_entry_frames(table, _row(), beside).frames
    pd.testing.assert_frame_equal(alone, again)


def test_the_entrys_excluded_frames_block_nearby_candidates() -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    near = [{"group": "g", "sequence": "s1", "indices": [43]}]
    elsewhere = [{"group": "g", "sequence": "s2", "indices": [43]}]
    blocked = select_entry_frames(
        table, _row(), _params(min_separation_frames=5, exclude=near)
    )
    kept = select_entry_frames(
        table, _row(), _params(min_separation_frames=5, exclude=elsewhere)
    )
    assert blocked.frames.empty
    assert blocked.n_candidates == 1
    assert _frames(kept.frames) == [40]


def test_an_excluded_frame_is_never_chosen_without_a_separation() -> None:
    table = _table({1: _jumping_at(40), 2: _steady(300.0, 300.0)})
    exclude = [{"group": "g", "sequence": "s1", "indices": [40]}]
    params = _params(min_separation_frames=0, exclude=exclude)
    assert select_entry_frames(table, _row(), params).frames.empty


# --- the op ------------------------------------------------------------------

KIND: Final = "select-frames"
S1: Final[Entry] = ("g", "s1")
S2: Final[Entry] = ("g", "s2")
S3: Final[Entry] = ("g", "s3")
CONVERTED: Final = "convert-calms21_npy.0.1-0123456789"
INFERRED: Final = "infer-pose.0.5-0123456789"
PEER: Final = "01JPEER000000000000000000"

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def _publish(
    ds: Dataset,
    entry: Entry,
    *,
    onset: int | None = None,
    variant: str = VARIANT,
    media_frames: int | None = SPAN,
) -> None:
    """Publish *entry*'s table into *variant*, with id 1 jumping at *onset*.

    An *onset* of None publishes two steady ids, a table without an event.
    """
    group, sequence = entry
    first = _steady(100.0, 100.0) if onset is None else _jumping_at(onset)
    table = _table({1: first, 2: _steady(300.0, 300.0)})
    add_tracks_table(
        ds,
        variant,
        table.assign(group=group, sequence=sequence),
        media_frames=media_frames,
    )


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """A dataset whose variant ``VARIANT`` holds s1 jumping at 40 and s2 at 60."""
    dataset = make_dataset(tmp_path / "ds")
    _publish(dataset, S1, onset=40)
    _publish(dataset, S2, onset=60)
    return dataset


@pytest.fixture
def selected(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the sequence of each entry whose frames a run selects, in order."""
    calls: list[str] = []
    original = selection.select_entry_frames

    def recording(
        table: pd.DataFrame, row: CellRow, params: SelectFramesParams
    ) -> EntrySelection:
        calls.append(text_cell(row.get("sequence")))
        return original(table, row, params)

    monkeypatch.setattr(selection, "select_entry_frames", recording)
    return calls


def _run(
    ds: Dataset,
    *,
    entries: Sequence[Entry] | None = None,
    overwrite: bool = False,
    execution_id: str | None = None,
    **fields: object,
) -> str:
    return run_op(
        ds,
        KIND,
        {"source_tracks_run_id": VARIANT, **fields},
        scope=None if entries is None else Scope(entries=list(entries)),
        overwrite=overwrite,
        execution_id=execution_id,
    )


def _planned_run_id(ds: Dataset) -> str:
    return OPS[KIND]().plan_identity(ds, _params(), ds.resolve_scope(None)).run_id


def _run_root(ds: Dataset, run_id: str) -> Path:
    return selections_kind_root(ds) / run_id


def _selection_file(ds: Dataset, run_id: str, entry: Entry) -> Path:
    return _run_root(ds, run_id) / make_entry_key(*entry) / SELECTION_FILE_NAME


def _claims(ds: Dataset) -> list[Path]:
    """Every claim marker under the selections root."""
    return sorted(ds.get_root("selections").rglob(INFLIGHT_MARKER_NAME))


def _rows(ds: Dataset, run_id: str) -> list[dict[str, str]]:
    """The selections-index rows of *run_id*, in entry order."""
    path = selection_index_path(ds)
    if not path.exists():
        return []
    records = index_records(frame_selection_index(path).read())
    chosen = [record for record in records if record["run_id"] == run_id]
    return sorted(chosen, key=lambda record: (record["group"], record["sequence"]))


def _snapshot(ds: Dataset, execution_id: str) -> RunLogSnapshot:
    found = read_run(run_log_dir(ds.base_dir), execution_id)
    assert found is not None
    return found


def _extract_params(ds: Dataset, run_id: str) -> dict[str, object]:
    """Return the run's ``extract-frames.params.json`` as the JSON object written."""
    path = _run_root(ds, run_id) / EXTRACT_PARAMS_FILE_NAME
    return _JSON_OBJECT.validate_json(path.read_bytes())


def _listed(entry: Entry, *indices: int) -> dict[str, object]:
    """Return one entry of a ``list`` extraction's ``frames``, as the JSON names it."""
    group, sequence = entry
    return {"group": group, "sequence": sequence, "indices": list(indices)}


def _hold(ds: Dataset, run_id: str, entry: Entry) -> Path:
    """Claim *entry*'s directory of *run_id* for another live execution."""
    held = _run_root(ds, run_id) / make_entry_key(*entry)
    held.mkdir(parents=True, exist_ok=True)
    write_inflight(
        held,
        new_inflight(
            execution_id=PEER, host="peer", pid=1, phase=None, idle_seconds=3600
        ),
    )
    return held


def test_the_op_declares_a_cpu_extraction_over_any_scope() -> None:
    op = OPS[KIND]

    assert op.category == "extract"
    assert op.domain == "tracking"
    assert op.version == "0.1"
    assert op.writes_tracks is False
    assert op.scope_takes == "any"
    assert op.scope_dependent is False
    assert op.model_reference is None
    assert op_resource_class(KIND) == "cpu"


def test_the_run_id_digests_the_params_alone(ds: Dataset) -> None:
    exclude = [{"group": "g", "sequence": "s1", "indices": [3]}]
    params = _params(exclude=exclude)
    expected = op_run_id(KIND, "0.1", {"params": params.identity_dump()})
    op = OPS[KIND]()

    for scope in (None, Scope(entries=[S1]), Scope(entries=[S1, S2, S3])):
        identity = op.plan_identity(ds, params, ds.resolve_scope(scope))
        assert identity == OpIdentity(run_id=expected)
    assert expected != _planned_run_id(ds)
    assert _run(ds, exclude=exclude) == expected


def test_the_published_description_names_the_variants_that_it_reads() -> None:
    entry = describe_step_type(KIND)
    schema = entry.params_schema
    assert schema is not None
    properties = schema["properties"]
    assert isinstance(properties, dict)
    source = properties["source_tracks_run_id"]
    assert isinstance(source, dict)

    assert "a tracker or infer op" in entry.summary
    assert "of the full media" in entry.summary
    assert "a converted, resampled or trimmed variant does not record a frame " in (
        str(source["description"])
    )


# --- refusals before any claim -----------------------------------------------


def test_an_unknown_variant_is_not_found_before_any_claim(
    ds: Dataset, selected: list[str]
) -> None:
    missing = "trex.0.1-ffffffffff"

    with pytest.raises(SelectFramesNotFound) as caught:
        _ = _run(ds, source_tracks_run_id=missing)

    assert caught.value.error_kind == "not_found"
    assert missing in str(caught.value)
    assert _claims(ds) == []
    assert selected == []
    assert not selections_kind_root(ds).exists()


def test_a_scope_without_rows_in_the_variant_is_a_conflict_before_any_claim(
    ds: Dataset, selected: list[str]
) -> None:
    with pytest.raises(SelectFramesConflict) as caught:
        _ = _run(ds, entries=[("g", "elsewhere")])

    assert caught.value.error_kind == "conflict"
    message = str(caught.value)
    assert VARIANT in message
    assert "g__elsewhere" in message
    assert _claims(ds) == []
    assert selected == []
    assert not selections_kind_root(ds).exists()


# --- what a run writes -------------------------------------------------------


def test_each_entry_gets_its_selection_and_a_finished_row(ds: Dataset) -> None:
    run_id = _run(ds)

    tracks = read_tracks_index(ds)
    rows = _rows(ds, run_id)
    assert [row["sequence"] for row in rows] == ["s1", "s2"]
    for row, entry in zip(rows, (S1, S2), strict=True):
        path = _selection_file(ds, run_id, entry)
        assert row["abs_path"] == ds.relative_to_root(path)
        assert row["source_tracks_run_id"] == VARIANT
        assert row["producer"] == "trex"
        assert (row["n_candidates"], row["n_selected"]) == ("1", "1")
        assert row["skipped_criteria"] == "[]"
        assert row["body_length_px"] == "10.0"
        assert run_id.endswith(f"-{row['params_hash']}")
        assert row["finished_at"]
        tracks_row = tracks[tracks["sequence"] == entry[1]].iloc[0]
        table = pd.read_parquet(ds.resolve_path(str(tracks_row["abs_path"])))
        expected = select_entry_frames(table, tracks_row, _params()).frames
        pd.testing.assert_frame_equal(pd.read_parquet(path), expected)
    assert _claims(ds) == []


def test_scoped_entries_without_rows_in_the_variant_are_skipped_and_counted(
    ds: Dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = _run(ds, entries=[S1, S3])

    assert "1 scoped entry has no row in the tracks variant" in capsys.readouterr().out
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]


def test_an_infer_variant_skips_jump_and_records_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _publish(dataset, S1, onset=40, variant=INFERRED)

    run_id = _run(dataset, source_tracks_run_id=INFERRED)

    (row,) = _rows(dataset, run_id)
    assert row["producer"] == "infer-pose"
    assert json.loads(row["skipped_criteria"]) == ["jump"]
    assert row["n_selected"] == "0"
    assert "g__s1 skips jump" in capsys.readouterr().out


# --- reuse and the cache hit -------------------------------------------------


def test_a_rerun_reuses_every_entry_and_is_a_cache_hit(
    ds: Dataset, selected: list[str]
) -> None:
    first = _run(ds, execution_id="exec-first")
    second = _run(ds, execution_id="exec-second")

    assert second == first
    assert selected == ["s1", "s2"]
    assert _snapshot(ds, "exec-first")["cache_hit"] is False
    snapshot = _snapshot(ds, "exec-second")
    assert snapshot["cache_hit"] is True
    assert snapshot["entries_written"] == 2
    assert _claims(ds) == []


def test_a_rerun_over_a_wider_scope_than_the_variant_is_a_cache_hit(
    ds: Dataset, selected: list[str]
) -> None:
    _ = _run(ds, entries=[S1, S2, S3])
    _ = _run(ds, entries=[S1, S2, S3], execution_id="exec-wider")

    assert selected == ["s1", "s2"]
    assert _snapshot(ds, "exec-wider")["cache_hit"] is True


def test_an_entry_without_its_selection_file_is_selected_again(
    ds: Dataset, selected: list[str]
) -> None:
    run_id = _run(ds)
    _selection_file(ds, run_id, S2).unlink()

    _ = _run(ds, execution_id="exec-missing")

    assert selected == ["s1", "s2", "s2"]
    assert _snapshot(ds, "exec-missing")["cache_hit"] is False
    assert _selection_file(ds, run_id, S2).is_file()


def test_overwrite_selects_every_entry_again_under_the_same_run_id(
    ds: Dataset, selected: list[str]
) -> None:
    first = _run(ds)
    again = _run(ds, overwrite=True)

    assert again == first
    assert selected == ["s1", "s2", "s1", "s2"]
    assert [row["finished_at"] != "" for row in _rows(ds, first)] == [True, True]


def test_an_entry_held_elsewhere_is_not_attempted(
    ds: Dataset, selected: list[str]
) -> None:
    held = _hold(ds, _planned_run_id(ds), S2)

    run_id = _run(ds, execution_id="exec-contended")

    assert selected == ["s1"]
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]
    assert entry_error_lines(ds, "exec-contended") == []
    assert _snapshot(ds, "exec-contended")["entries_written"] == 1
    claim = read_inflight(held)
    assert claim is not None
    assert claim.execution_id == PEER


def test_an_entry_held_elsewhere_blocks_the_cache_hit(
    ds: Dataset, selected: list[str]
) -> None:
    run_id = _run(ds)
    _ = _hold(ds, run_id, S2)

    _ = _run(ds, execution_id="exec-held")

    assert selected == ["s1", "s2"]
    assert _snapshot(ds, "exec-held")["cache_hit"] is False


# --- losses and coverage -----------------------------------------------------


def test_the_run_log_records_the_variant_and_the_entries_written(ds: Dataset) -> None:
    _ = _run(ds, execution_id="exec-log")

    snapshot = _snapshot(ds, "exec-log")
    assert snapshot["tracks_variant"] == VARIANT
    assert snapshot["entries_written"] == 2
    assert snapshot["entries_failed"] == 0


def test_an_entry_without_a_frame_axis_is_lost_alone(ds: Dataset) -> None:
    _publish(ds, S2, onset=60, media_frames=None)

    run_id = _run(ds, execution_id="exec-partial")

    lines = entry_error_lines(ds, "exec-partial")
    assert len(lines) == 1
    assert _JSON_OBJECT.validate_json(lines[0])["key"] == "g__s2"
    assert "does not record a frame axis" in lines[0]
    snapshot = _snapshot(ds, "exec-partial")
    assert (snapshot["entries_written"], snapshot["entries_failed"]) == (1, 1)
    assert [row["sequence"] for row in _rows(ds, run_id)] == ["s1"]
    assert _claims(ds) == []


def test_a_converted_variant_fails_every_entry_naming_its_producer(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _publish(dataset, S1, onset=40, variant=CONVERTED, media_frames=None)
    _publish(dataset, S2, onset=60, variant=CONVERTED, media_frames=None)

    with pytest.raises(AllEntriesFailed) as caught:
        _ = _run(dataset, source_tracks_run_id=CONVERTED, execution_id="exec-converted")

    assert "g__s1" in str(caught.value)
    assert "g__s2" in str(caught.value)
    lines = entry_error_lines(dataset, "exec-converted")
    assert len(lines) == 2
    assert all("made by convert-calms21_npy" in line for line in lines)
    assert _snapshot(dataset, "exec-converted")["entries_written"] == 0
    assert _claims(dataset) == []


# --- the handoff to extraction -----------------------------------------------


def test_the_extract_frames_params_list_each_entry_with_a_frame(ds: Dataset) -> None:
    _publish(ds, S3)

    run_id = _run(ds)

    assert _extract_params(ds, run_id) == {
        "method": "list",
        "frames": [_listed(S1, 40), _listed(S2, 60)],
    }
    assert [row["n_selected"] for row in _rows(ds, run_id)] == ["1", "1", "0"]


def test_two_executions_over_different_entries_keep_both_in_the_params(
    ds: Dataset,
) -> None:
    first = _run(ds, entries=[S1])
    second = _run(ds, entries=[S2])

    assert second == first
    assert _extract_params(ds, first)["frames"] == [_listed(S1, 40), _listed(S2, 60)]


def test_the_params_are_rebuilt_from_the_finished_rows_of_the_run(
    ds: Dataset,
) -> None:
    run_id = _planned_run_id(ds)
    _ = record_entry_selection(
        ds,
        run_id=run_id,
        group="g",
        sequence="s0",
        frames=[7, 5],
        source_tracks_run_id=VARIANT,
    )
    _ = record_entry_selection(
        ds,
        run_id="select-frames.0.1-ffffffffff",
        group="g",
        sequence="s9",
        frames=[9],
        source_tracks_run_id=VARIANT,
    )

    _ = _run(ds, entries=[S1])

    assert _extract_params(ds, run_id)["frames"] == [
        _listed(("g", "s0"), 5, 7),
        _listed(S1, 40),
    ]


def test_a_finished_row_without_its_file_is_left_out_of_the_params(
    ds: Dataset,
) -> None:
    run_id = _run(ds, entries=[S1])
    _selection_file(ds, run_id, S1).unlink()

    _ = _run(ds, entries=[S2], execution_id="exec-orphan")

    assert entry_error_lines(ds, "exec-orphan") == []
    assert _extract_params(ds, run_id)["frames"] == [_listed(S2, 60)]


def test_the_params_are_written_under_the_selections_index_lock(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[bool] = []
    original = selection.atomic_write

    def probing(final_path: Path, write_fn: Callable[[Path], object]) -> None:
        if final_path.name == EXTRACT_PARAMS_FILE_NAME:
            held.append(index_locked_out(selection_index_path(ds)))
        original(final_path, write_fn)

    monkeypatch.setattr(selection, "atomic_write", probing)

    _ = _run(ds, entries=[S1])

    assert held == [True]
    assert not index_locked_out(selection_index_path(ds))


def test_a_rerun_rebuilds_the_params_when_their_file_is_gone(
    ds: Dataset, selected: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = _run(ds)
    (_run_root(ds, run_id) / EXTRACT_PARAMS_FILE_NAME).unlink()
    _ = capsys.readouterr()

    _ = _run(ds, execution_id="exec-rebuild")

    assert selected == ["s1", "s2"]
    assert _snapshot(ds, "exec-rebuild")["cache_hit"] is True
    assert _extract_params(ds, run_id)["frames"] == [_listed(S1, 40), _listed(S2, 60)]
    assert "is empty" not in capsys.readouterr().out
    assert not index_locked_out(selection_index_path(ds))


def test_an_empty_selection_writes_no_params_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _publish(dataset, S1)

    run_id = _run(dataset)

    assert not (_run_root(dataset, run_id) / EXTRACT_PARAMS_FILE_NAME).exists()
    assert f"the selection of run_id={run_id} is empty" in capsys.readouterr().out
    assert selected_frames(dataset, run_id) == {}


def test_a_first_run_over_entries_all_held_elsewhere_writes_no_index(
    ds: Dataset, selected: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = _planned_run_id(ds)
    for entry in (S1, S2):
        _ = _hold(ds, run_id, entry)

    _ = _run(ds)

    index = selection_index_path(ds)
    assert selected == []
    assert not index.exists()
    assert not lock_path_for(index.resolve()).exists()
    out = capsys.readouterr().out
    assert "is empty" not in out
    assert "another execution holds 2 scoped entries" in out


def test_a_selection_without_frames_and_a_held_entry_is_not_called_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    _publish(dataset, S1)
    _publish(dataset, S2)
    run_id = _planned_run_id(dataset)
    _ = _hold(dataset, run_id, S2)

    _ = _run(dataset)

    assert [row["n_selected"] for row in _rows(dataset, run_id)] == ["0"]
    assert not (_run_root(dataset, run_id) / EXTRACT_PARAMS_FILE_NAME).exists()
    out = capsys.readouterr().out
    assert "is empty" not in out
    message = f"the selection of run_id={run_id} has no frame to extract yet: "
    assert message + "another execution holds 1 scoped entry" in out


def _list_run_id(
    _ds: Dataset, kind: str, params: ExtractFramesParams, **_: object
) -> str:
    """Name the run that ``extract_frames`` would start, without extracting."""
    assert kind == "extract-frames"
    return frames_run_id(params.method, params)


def test_the_selected_frames_name_one_list_extraction_run(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dataset_runs, "run_op", _list_run_id)
    run_id = _run(ds)
    first = extract_frames(ds, method="list", frames=selected_frames(ds, run_id))

    _ = _run(ds, overwrite=True)
    second = extract_frames(ds, method="list", frames=selected_frames(ds, run_id))

    from_file = ExtractFramesParams.model_validate(_extract_params(ds, run_id))
    assert first == second == frames_run_id("list", from_file)
    assert selected_frames(ds, run_id) == {S1: (40,), S2: (60,)}
