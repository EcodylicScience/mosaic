"""An assign-identities run over an embedded tracks variant."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import mosaic.tracking.ops.assign_identities as assign_identities
from mosaic.core.dataset import Dataset
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.appearance import appearance_columns
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.graph import (
    Recipe,
    RecipeRefused,
    execute_step,
    plan_pipeline,
    submit_request,
)
from mosaic.core.pipeline.ops import run_op
from mosaic.core.pipeline.run import AllEntriesFailed
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.scope import Scope
from mosaic.runlog import read_run, run_log_dir
from mosaic.tracking.ops.assign_identities import AssignIdentitiesParams
from tests.helpers import add_tracks_table, add_tracks_variant, make_dataset

KIND = "assign-identities"

SOURCE = "sleap.1.6-abcdef0123"

DIM = 4

Vector = npt.NDArray[np.float64]


def _unit(*components: float) -> Vector:
    vector = np.asarray(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


LOOKS_A = _unit(1, 0, 0, 0)
LOOKS_B = _unit(0, 1, 0, 0)
LOOKS_C = _unit(0, 0, 1, 0)
LOOKS_D = _unit(0, 0, 0, 1)
BETWEEN = _unit(1, 1, 0, 0)


@dataclass(frozen=True)
class Track:
    """The rows of one source id in one entry, all with one vector.

    Attributes:
        source_id: The source track id.
        frames: The frames of its rows.
        vector: The appearance vector of every row, or ``None`` for rows
            without one.
    """

    source_id: int
    frames: Sequence[int]
    vector: Vector | None


def _tables(
    sequence: str, tracks: Sequence[Track]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One entry's tracks table and appearance table, frame-major."""
    rows = [
        (frame, track.source_id, track.vector)
        for track in tracks
        for frame in track.frames
    ]
    rows.sort(key=lambda row: (row[0], row[1]))
    frame = np.asarray([row[0] for row in rows], dtype=np.int64)
    ids = np.asarray([row[1] for row in rows], dtype=np.int64)
    table = pd.DataFrame(
        {
            "group": pd.Series([""] * len(rows), dtype=str),
            "sequence": pd.Series([sequence] * len(rows), dtype=str),
            "frame": frame,
            "time": frame / 30.0,
            "id": ids,
            "X": frame.astype(np.float64) + ids,
            "Y": frame.astype(np.float64) - ids,
        }
    )
    with_vector = [row for row in rows if row[2] is not None]
    appearance = pd.DataFrame(
        {
            "group": pd.Series([""] * len(with_vector), dtype=str),
            "sequence": pd.Series([sequence] * len(with_vector), dtype=str),
            "frame": np.asarray([row[0] for row in with_vector], dtype=np.int64),
            "id": np.asarray([row[1] for row in with_vector], dtype=np.int64),
        }
    )
    vectors = np.asarray([row[2] for row in with_vector], dtype=np.float32).reshape(
        len(with_vector), -1
    )
    for position, column in enumerate(appearance_columns(vectors.shape[1] or DIM)):
        appearance[column] = (
            vectors[:, position] if len(with_vector) else np.zeros(0, np.float32)
        )
    return table, appearance


def _seed(
    ds: Dataset,
    sequence: str,
    tracks: Sequence[Track],
    *,
    appearance: bool = True,
    std_format: str = "mosaic_v1",
    media_frames: int | None = None,
    frames_read: int | None = None,
    known_tail_loss: int | None = None,
    first_frame: int | None = None,
    consumed_composition: str | None = None,
) -> None:
    table, vectors = _tables(sequence, tracks)
    add_tracks_table(
        ds,
        SOURCE,
        table,
        std_format=std_format,
        media_frames=media_frames,
        frames_read=frames_read,
        known_tail_loss=known_tail_loss,
        first_frame=first_frame,
        consumed_composition=consumed_composition,
        appearance=vectors if appearance else None,
    )


REFERENCE_TRACKS = (
    Track(0, range(10), LOOKS_A),
    Track(0, (50, 51), LOOKS_D),
    Track(1, range(10), LOOKS_B),
)
"""Entry s1: the reference entry. Source id 0 is ``a`` and 1 is ``b``."""

SCORED_TRACKS = (
    Track(5, range(10), LOOKS_B),
    Track(7, range(5), LOOKS_A),
    Track(3, (20, 21, 22), BETWEEN),
    Track(7, range(30, 36), LOOKS_C),
    Track(9, range(40, 46), LOOKS_C),
)
"""Entry s2: two assigned tracklets, one abstaining by margin, two by score."""


def _seed_standard(ds: Dataset) -> None:
    _seed(ds, "s1", REFERENCE_TRACKS, media_frames=60, frames_read=59)
    _seed(
        ds,
        "s2",
        SCORED_TRACKS,
        media_frames=100,
        frames_read=99,
        known_tail_loss=1,
        first_frame=60,
        consumed_composition="tracks_raw=s2-v1",
    )


type Reference = tuple[str, str, int | None]
"""An identity, the sequence of its entry, and its source id."""

STANDARD: list[Reference] = [("a", "s1", 0), ("b", "s1", 1)]


def _params(
    references: Sequence[Reference] = STANDARD, **overrides: object
) -> AssignIdentitiesParams:
    return AssignIdentitiesParams.model_validate(
        {
            "source_tracks_run_id": SOURCE,
            "references": [
                {
                    "identity": identity,
                    "entry": {"group": "", "sequence": sequence},
                    "id": source_id,
                }
                for identity, sequence, source_id in references
            ],
            **overrides,
        }
    )


def _variant(params: AssignIdentitiesParams) -> str:
    return assign_identities.assign_identities_identity(params).tracks_variant


def _run(
    ds: Dataset,
    params: AssignIdentitiesParams,
    *,
    execution_id: str = "exec-1",
    scope: Scope | None = None,
) -> str:
    return run_op(ds, KIND, params, execution_id=execution_id, scope=scope)


def _output_rows(ds: Dataset, params: AssignIdentitiesParams) -> pd.DataFrame:
    index = read_tracks_index(ds)
    rows = index[index["run_id"] == _variant(params)]
    return rows.set_index("sequence")


def _output(ds: Dataset, params: AssignIdentitiesParams, sequence: str) -> pd.DataFrame:
    path = ds.resolve_path(str(_output_rows(ds, params).loc[sequence, "abs_path"]))
    return pd.read_parquet(path)


def _rows_of(
    table: pd.DataFrame, source_id: int, frames: Sequence[int]
) -> pd.DataFrame:
    hit = table[
        (table["identity_source_id"] == source_id) & table["frame"].isin(list(frames))
    ]
    assert len(hit) == len(frames)
    return hit


def _snapshot(ds: Dataset, execution_id: str) -> dict[str, object]:
    snapshot = read_run(run_log_dir(ds.base_dir), execution_id)
    assert snapshot is not None
    return dict(snapshot)


def _prototype(*vectors: tuple[Vector, int]) -> Vector:
    total = sum((vector * count for vector, count in vectors), np.zeros(DIM))
    return total / np.linalg.norm(total)


# --- the run's record --------------------------------------------------------------


def test_the_run_records_its_source_and_the_entries_it_wrote(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)

    run_id = _run(ds, _params())

    snapshot = _snapshot(ds, "exec-1")
    assert snapshot["run_id"] == run_id
    assert snapshot["tracks_variant"] == SOURCE
    assert snapshot["entries_written"] == 2
    assert snapshot["status"] == "finished"


# --- the relabeled tables ----------------------------------------------------------


def test_tracklets_take_the_identity_whose_prototype_they_resemble(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    table = _output(ds, params, "s2")
    assert set(_rows_of(table, 5, range(10))["identity"]) == {"b"}
    assert set(_rows_of(table, 5, range(10))["id"]) == {1}
    assert set(_rows_of(table, 7, range(5))["identity"]) == {"a"}
    assert set(_rows_of(table, 7, range(5))["id"]) == {0}


def test_scores_and_margins_are_recorded_for_assigned_and_abstaining_tracklets(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    table = _output(ds, params, "s2")
    proto_a = _prototype((LOOKS_A, 10), (LOOKS_D, 2))
    a_score = float(LOOKS_A @ proto_a)
    between = _rows_of(table, 3, (20, 21, 22))
    assert between["identity"].isna().all()
    between_a = float(BETWEEN @ proto_a)
    between_b = float(BETWEEN @ LOOKS_B)
    np.testing.assert_allclose(between["identity_score"], between_b, atol=1e-6)
    np.testing.assert_allclose(
        between["identity_margin"], between_b - between_a, atol=1e-6
    )
    assigned = _rows_of(table, 7, range(5))
    np.testing.assert_allclose(assigned["identity_score"], a_score, atol=1e-6)
    np.testing.assert_allclose(assigned["identity_margin"], a_score, atol=1e-6)


def test_assigned_ids_are_the_sorted_identity_positions(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    for sequence in ("s1", "s2"):
        table = _output(ds, params, sequence)
        assigned = table[table["identity"].notna()]
        assert dict(zip(assigned["identity"], assigned["id"], strict=True)) == {
            "a": 0,
            "b": 1,
        }


def test_unassigned_tracklets_count_down_from_minus_two_by_first_frame(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    table = _output(ds, params, "s2")
    assert set(_rows_of(table, 3, (20, 21, 22))["id"]) == {-2}
    assert set(_rows_of(table, 7, range(30, 36))["id"]) == {-3}
    assert set(_rows_of(table, 9, range(40, 46))["id"]) == {-4}
    assert -1 not in set(table["id"])
    assert table.loc[table["id"] < 0, "identity"].isna().all()


def test_the_relabeled_table_keeps_every_source_row_and_column(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    source, _vectors = _tables("s2", SCORED_TRACKS)
    table = _output(ds, params, "s2")
    assert len(table) == len(source)
    kept = table.sort_values(["frame", "identity_source_id"]).reset_index(drop=True)
    for column in ("frame", "time", "X", "Y"):
        np.testing.assert_array_equal(kept[column], source[column])
    np.testing.assert_array_equal(kept["identity_source_id"], source["id"])


def test_the_assignment_columns_have_their_dtypes(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    table = _output(ds, params, "s2")
    assert table["id"].dtype == np.int64
    assert pd.api.types.is_string_dtype(table["identity"])
    assert table["identity_score"].dtype == np.float64
    assert table["identity_margin"].dtype == np.float64
    assert table["identity_source_id"].dtype == np.int64


def test_a_reference_entry_takes_its_declared_identities(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    _ = _run(ds, params)

    table = _output(ds, params, "s1")
    late = _rows_of(table, 0, (50, 51))
    assert set(late["identity"]) == {"a"}
    assert set(late["id"]) == {0}
    proto_a = _prototype((LOOKS_A, 10), (LOOKS_D, 2))
    np.testing.assert_allclose(
        late["identity_score"], float(LOOKS_D @ proto_a), atol=1e-6
    )
    assert set(_rows_of(table, 1, range(10))["identity"]) == {"b"}


def test_a_whole_entry_reference_forces_a_tracklet_that_resembles_another(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, (0, 1), LOOKS_B),))
    params = _params([*STANDARD, ("a", "s3", None)])

    _ = _run(ds, params)

    table = _output(ds, params, "s3")
    assert set(table["identity"]) == {"a"}
    assert set(table["id"]) == {0}
    proto_a = _prototype((LOOKS_A, 10), (LOOKS_D, 2), (LOOKS_B, 2))
    np.testing.assert_allclose(table["identity_score"], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        table["identity_margin"], 1.0 - float(LOOKS_B @ proto_a), atol=1e-6
    )


def test_a_declared_tracklet_blocks_its_identity_where_it_is_present(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_B), Track(6, range(5), LOOKS_A)))
    params = _params([*STANDARD, ("a", "s3", 4)])

    _ = _run(ds, params)

    table = _output(ds, params, "s3")
    assert set(_rows_of(table, 4, range(5))["identity"]) == {"a"}
    blocked = _rows_of(table, 6, range(5))
    assert blocked["identity"].isna().all()
    assert set(blocked["id"]) == {-2}


def test_a_tracklet_without_vectors_abstains(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A), Track(6, range(5), None)))
    params = _params()

    _ = _run(ds, params)

    table = _output(ds, params, "s3")
    assert set(_rows_of(table, 4, range(5))["identity"]) == {"a"}
    vectorless = _rows_of(table, 6, range(5))
    assert vectorless["identity"].isna().all()
    assert set(vectorless["id"]) == {-2}
    assert vectorless["identity_score"].isna().all()


def test_prototypes_do_not_depend_on_the_scope(tmp_path: Path) -> None:
    whole = make_dataset(tmp_path / "whole")
    scoped = make_dataset(tmp_path / "scoped")
    for ds in (whole, scoped):
        _seed_standard(ds)
    params = _params()

    _ = _run(whole, params)
    _ = _run(scoped, params, scope=Scope(entries=[("", "s2")]))

    assert list(_output_rows(scoped, params).index) == ["s2"]
    pd.testing.assert_frame_equal(
        _output(scoped, params, "s2"), _output(whole, params, "s2")
    )


# --- the relabeled rows ------------------------------------------------------------


def test_the_rows_copy_the_source_cells_and_record_no_media_or_appearance(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()

    run_id = _run(ds, params)

    rows = _output_rows(ds, params)
    row = rows.loc["s2"]
    assert text_cell(row["producer"]) == KIND
    assert text_cell(row["producer_run_id"]) == run_id
    assert text_cell(row["std_format"]) == "mosaic_v1"
    assert text_cell(row["media_frames"]) == "100"
    assert text_cell(row["frames_read"]) == "99"
    assert text_cell(row["known_tail_loss"]) == "1"
    assert text_cell(row["first_frame"]) == "60"
    assert text_cell(row["consumed_composition"]).startswith(
        "tracks_raw=s2-v1;sources="
    )
    assert text_cell(rows.loc["s1", "known_tail_loss"]) == ""
    assert text_cell(rows.loc["s1", "first_frame"]) == ""
    for sequence in ("s1", "s2"):
        assert text_cell(rows.loc[sequence, "consumed_media_composition"]) == ""
        assert text_cell(rows.loc[sequence, "appearance_abs_path"]) == ""


def test_repeated_rows_are_refused_before_each_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    checked: list[str] = []

    def refuse(df: pd.DataFrame, who: str, order_col: str = "frame") -> None:
        assert "identity" in df.columns
        assert order_col == "frame"
        checked.append(who)
        if "s2" in who:
            raise ValueError(f"{who}: frame 0 of id 1 appears in more than one row")

    monkeypatch.setattr(assign_identities, "refuse_repeated_rows", refuse)
    params = _params()

    with pytest.raises(RuntimeError, match="s2"):
        _ = _run(ds, params)

    assert len(checked) == 2
    assert list(_output_rows(ds, params).index) == ["s1"]


# --- refusals ----------------------------------------------------------------------


def _assert_refused(
    ds: Dataset, params: AssignIdentitiesParams, kind: str, match: str
) -> None:
    with pytest.raises(ClassifiedError, match=match) as raised:
        _ = _run(ds, params, execution_id="exec-refused")
    assert raised.value.error_kind == kind
    assert _output_rows(ds, params).empty
    variant_root = ds.get_root("tracks") / _variant(params)
    assert not variant_root.exists() or not any(variant_root.glob("*.parquet"))
    snapshot = _snapshot(ds, "exec-refused")
    assert snapshot["tracks_variant"] == SOURCE
    assert snapshot["entries_written"] == 0


def test_a_scoped_entry_without_an_appearance_table_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A),), appearance=False)

    _assert_refused(ds, _params(), "conflict", r"\('', 's3'\).*sleap")


def test_a_reference_entry_without_an_appearance_table_is_refused_out_of_scope(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed(ds, "s1", REFERENCE_TRACKS, appearance=False)
    _seed(ds, "s2", SCORED_TRACKS)
    params = _params()

    with pytest.raises(ClassifiedError, match=r"\('', 's1'\).*sleap") as raised:
        _ = _run(ds, params, scope=Scope(entries=[("", "s2")]))

    assert raised.value.error_kind == "conflict"
    assert _output_rows(ds, params).empty


def test_a_reference_entry_absent_from_the_variant_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)

    _assert_refused(
        ds, _params([*STANDARD, ("a", "s9", 0)]), "not_found", r"\('', 's9'\)"
    )


def test_a_reference_id_without_a_row_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)

    _assert_refused(ds, _params([*STANDARD, ("a", "s2", 42)]), "not_found", "id 42")


def test_a_reference_without_a_vector_is_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A), Track(6, range(5), None)))

    _assert_refused(
        ds, _params([*STANDARD, ("a", "s3", 6)]), "not_found", "no appearance vector"
    )


def test_a_whole_entry_reference_with_two_rows_in_a_frame_is_refused(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)

    _assert_refused(ds, _params([*STANDARD, ("c", "s2", None)]), "conflict", "frame 0")


def test_two_references_of_one_identity_sharing_a_frame_are_refused(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)

    _assert_refused(
        ds,
        _params([*STANDARD, ("b", "s2", 5), ("b", "s2", 7)]),
        "conflict",
        "'b'.*frame 0",
    )


def test_appearance_tables_of_different_dimensions_are_refused(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), _unit(1, 0, 0)),))

    _assert_refused(ds, _params(), "conflict", "dimension")


@pytest.mark.parametrize(
    ("damage", "kind"),
    [("delete", "not_found"), ("corrupt", "conflict")],
)
def test_an_unreadable_appearance_table_is_refused(
    tmp_path: Path, damage: str, kind: str
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A),))
    row = read_tracks_index(ds).set_index("sequence").loc["s3"]
    appearance = ds.resolve_path(str(row["appearance_abs_path"]))
    if damage == "delete":
        appearance.unlink()
    else:
        _ = appearance.write_bytes(b"not a parquet")

    _assert_refused(ds, _params(), kind, r"\('', 's3'\)")


# --- a recipe over every variant of an entry ----------------------------------------


KEYPOINT_VARIANT = "sleap.1.6-1111111111"
"""A run of the same entries without an embedding model."""


def _assign_step(source: object) -> dict[str, object]:
    params = _params().model_dump(mode="json")
    return {
        "id": "assign",
        "type": "op",
        "kind": KIND,
        "params": {**params, "source_tracks_run_id": source},
    }


def _speed_step(tracks: dict[str, str] | None) -> dict[str, object]:
    return {
        "id": "speed",
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
        **({} if tracks is None else {"tracks": tracks}),
    }


def _recipe(*steps: Mapping[str, object]) -> Recipe:
    return Recipe.model_validate({"schema_version": 1, "steps": list(steps)})


def _two_runs_of_each_entry(ds: Dataset) -> None:
    _seed_standard(ds)
    add_tracks_variant(ds, KEYPOINT_VARIANT, "s1", "s2", std_format="mosaic_v1")


def test_a_recipe_relabels_and_reads_one_of_three_variants(tmp_path: Path) -> None:
    """The source and the pin answer which variant, and the rest stay on disk."""
    ds = make_dataset(tmp_path / "ds")
    _two_runs_of_each_entry(ds)
    recipe = _recipe(_assign_step(SOURCE), _speed_step({"step": "assign"}))
    request = submit_request(ds, recipe).request

    assert execute_step(ds, request, "assign").covered == 2
    assert execute_step(ds, request, "speed").covered == 2

    index = read_tracks_index(ds)
    assert set(index["run_id"]) == {SOURCE, KEYPOINT_VARIANT, _variant(_params())}
    plan = plan_pipeline(ds, recipe)
    assert not plan.problems
    assert plan.step("speed").spec.tracks_run_id == _variant(_params())


def test_an_unpinned_reader_after_a_relabeling_is_a_problem_at_its_step(
    tmp_path: Path,
) -> None:
    """Every variant is named, and the relabeling step is not refused with it."""
    ds = make_dataset(tmp_path / "ds")
    _two_runs_of_each_entry(ds)
    _ = _run(ds, _params())
    recipe = _recipe(_assign_step(SOURCE), _speed_step(None))

    with pytest.raises(RecipeRefused) as raised:
        _ = submit_request(ds, recipe)

    [problem] = raised.value.problems
    assert (problem.step, problem.where) == ("speed", "tracks")
    for variant in (SOURCE, KEYPOINT_VARIANT, _variant(_params())):
        assert variant in problem.message


def test_a_step_reference_names_the_variant_the_tracker_step_writes(
    tmp_path: Path,
) -> None:
    """A source that a step of the recipe writes resolves beside the other variants."""
    ds = make_dataset(tmp_path / "ds")
    _two_runs_of_each_entry(ds)
    track = {
        "id": "track",
        "type": "op",
        "kind": "trex",
        "params": {"track_max_individuals": 2},
    }

    plan = plan_pipeline(ds, _recipe(track, _assign_step({"step": "track"})))

    assert not plan.problems
    written = plan.step("track").tracks_variant
    assert written
    assert plan.step("assign").spec.params["source_tracks_run_id"] == written


# --- partial failure ---------------------------------------------------------------


def test_a_failed_entry_is_reported_and_a_rerun_resumes(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A),))
    s3 = ds.resolve_path(
        str(read_tracks_index(ds).set_index("sequence").loc["s3", "abs_path"])
    )
    intact = s3.read_bytes()
    _ = s3.write_bytes(b"not a parquet")
    params = _params()

    with pytest.raises(RuntimeError, match="s3"):
        _ = _run(ds, params, execution_id="exec-partial")

    snapshot = _snapshot(ds, "exec-partial")
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 2
    assert sorted(_output_rows(ds, params).index) == ["s1", "s2"]
    written = ds.resolve_path(str(_output_rows(ds, params).loc["s2", "abs_path"]))
    before = written.stat().st_mtime_ns

    _ = s3.write_bytes(intact)
    _ = _run(ds, params, execution_id="exec-resume")

    assert written.stat().st_mtime_ns == before
    assert sorted(_output_rows(ds, params).index) == ["s1", "s2", "s3"]
    resumed = _snapshot(ds, "exec-resume")
    assert resumed["entries_written"] == 3
    assert resumed["entries_failed"] == 0


@pytest.mark.parametrize(
    ("first_frame", "consumed_composition"),
    [(60, "tracks_raw=s2-v2"), (70, "tracks_raw=s2-v1")],
    ids=["composition", "first-frame"],
)
def test_a_source_table_written_again_differently_is_relabeled_again(
    tmp_path: Path, first_frame: int, consumed_composition: str
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()
    _ = _run(ds, params, execution_id="exec-first")
    unchanged = ds.resolve_path(str(_output_rows(ds, params).loc["s1", "abs_path"]))
    before = unchanged.stat().st_mtime_ns

    _seed(
        ds,
        "s2",
        SCORED_TRACKS,
        media_frames=100,
        frames_read=99,
        known_tail_loss=1,
        first_frame=first_frame,
        consumed_composition=consumed_composition,
    )
    _ = _run(ds, params, execution_id="exec-again")

    row = _output_rows(ds, params).loc["s2"]
    assert text_cell(row["consumed_composition"]).startswith(
        f"{consumed_composition};sources="
    )
    assert text_cell(row["first_frame"]) == str(first_frame)
    assert unchanged.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    ("first_frame", "consumed_composition"),
    [(None, "tracks_raw=s1-v2"), (30, None), (None, None)],
    ids=["composition", "first-frame", "vectors"],
)
def test_a_reference_source_written_again_relabels_every_entry(
    tmp_path: Path, first_frame: int | None, consumed_composition: str | None
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()
    _ = _run(ds, params, execution_id="exec-first")
    before = _output(ds, params, "s2")
    assert set(_rows_of(before, 5, range(10))["identity"]) == {"b"}

    swapped = (
        Track(0, range(10), LOOKS_B),
        Track(0, (50, 51), LOOKS_D),
        Track(1, range(10), LOOKS_A),
    )
    _seed(
        ds,
        "s1",
        swapped,
        media_frames=60,
        frames_read=59,
        first_frame=first_frame,
        consumed_composition=consumed_composition,
    )
    _ = _run(ds, params, execution_id="exec-again")

    after = _output(ds, params, "s2")
    assert set(_rows_of(after, 5, range(10))["identity"]) == {"a"}


def _rewrite_positions(ds: Dataset) -> None:
    """Move the rows of s2's source table, and leave its row and vectors as they are."""
    index = read_tracks_index(ds)
    source_rows = index[index["run_id"] == SOURCE].set_index("sequence")
    path = ds.resolve_path(str(source_rows.loc["s2", "abs_path"]))
    table = pd.read_parquet(path)
    table["X"] = table["X"] + 1.0
    _ = write_parquet_atomic(table, path)


def _rewrite_vectors(ds: Dataset) -> None:
    """Swap the appearance of s2's first tracklet, and leave its row as it is."""
    _seed(
        ds,
        "s2",
        (Track(5, range(10), LOOKS_A), *SCORED_TRACKS[1:]),
        media_frames=100,
        frames_read=99,
        known_tail_loss=1,
        first_frame=60,
        consumed_composition="tracks_raw=s2-v1",
    )


@pytest.mark.parametrize(
    "rewrite", [_rewrite_positions, _rewrite_vectors], ids=["tracks", "appearance"]
)
def test_a_source_table_changed_under_one_composition_is_relabeled_again(
    tmp_path: Path, rewrite: Callable[[Dataset], None]
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()
    _ = _run(ds, params, execution_id="exec-first")
    reference = ds.resolve_path(str(_output_rows(ds, params).loc["s1", "abs_path"]))
    relabeled = ds.resolve_path(str(_output_rows(ds, params).loc["s2", "abs_path"]))
    reference_before = reference.stat().st_mtime_ns
    relabeled_before = relabeled.stat().st_mtime_ns
    composition = text_cell(_output_rows(ds, params).loc["s2", "consumed_composition"])

    rewrite(ds)
    _ = _run(ds, params, execution_id="exec-again")

    assert relabeled.stat().st_mtime_ns != relabeled_before
    assert reference.stat().st_mtime_ns == reference_before
    assert (
        text_cell(_output_rows(ds, params).loc["s2", "consumed_composition"])
        != composition
    )
    assert _snapshot(ds, "exec-again")["cache_hit"] is False


def test_a_rerun_over_unchanged_sources_reuses_every_table_as_a_cache_hit(
    tmp_path: Path,
) -> None:
    """``entries_written`` counts the tables the scope holds, reused ones included."""
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    params = _params()
    _ = _run(ds, params, execution_id="exec-first")
    written = [
        ds.resolve_path(str(cell)) for cell in _output_rows(ds, params)["abs_path"]
    ]
    before = [path.stat().st_mtime_ns for path in written]
    rows = _output_rows(ds, params)

    _ = _run(ds, params, execution_id="exec-again")

    assert [path.stat().st_mtime_ns for path in written] == before
    pd.testing.assert_frame_equal(_output_rows(ds, params), rows)
    first, again = _snapshot(ds, "exec-first"), _snapshot(ds, "exec-again")
    assert first["cache_hit"] is False
    assert again["cache_hit"] is True
    assert first["entries_written"] == again["entries_written"] == 2


def test_a_missing_source_table_fails_its_entry_alone(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A),))
    params = _params()
    ds.resolve_path(
        str(read_tracks_index(ds).set_index("sequence").loc["s3", "abs_path"])
    ).unlink()

    with pytest.raises(RuntimeError, match="s3"):
        _ = _run(ds, params, execution_id="exec-partial")

    assert sorted(_output_rows(ds, params).index) == ["s1", "s2"]
    snapshot = _snapshot(ds, "exec-partial")
    assert snapshot["entries_failed"] == 1
    assert snapshot["entries_written"] == 2


def test_every_failed_entry_is_named(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    s2 = ds.resolve_path(
        str(read_tracks_index(ds).set_index("sequence").loc["s2", "abs_path"])
    )
    _ = s2.write_bytes(b"not a parquet")

    with pytest.raises(RuntimeError, match=r"1 of 2 entries.*s2"):
        _ = _run(ds, _params())


def test_a_run_that_fails_every_scoped_entry_fails(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _seed_standard(ds)
    _seed(ds, "s3", (Track(4, range(5), LOOKS_A),))
    paths = read_tracks_index(ds).set_index("sequence")["abs_path"]
    for sequence in ("s2", "s3"):
        _ = ds.resolve_path(str(paths[sequence])).write_bytes(b"not a parquet")

    with pytest.raises(AllEntriesFailed, match=r"2 of 2 entries"):
        _ = _run(ds, _params(), scope=Scope(entries=[("", "s2"), ("", "s3")]))
