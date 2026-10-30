"""Tracklets assigned to identity prototypes by appearance.

Every case uses synthetic vectors. A tracklet takes its best identity when the
score and margin pass both thresholds and no co-present tracklet of the entry
already holds that identity. Sorted keys fix every order, and shuffled rows
give the same result.
"""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.core.pipeline.appearance_assignment import (
    Assignment,
    Prototypes,
    Tracklet,
    assign,
    prototypes,
    score,
    tracklets,
)

_FloatArray = npt.NDArray[np.float64]


def _unit(*components: float) -> _FloatArray:
    vector = np.asarray(components, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _int_array(*values: int) -> npt.NDArray[np.int64]:
    return np.asarray(values, dtype=np.int64)


def _tracklet(source_id: int, frames: list[int]) -> Tracklet:
    frame_array = np.asarray(frames, dtype=np.int64)
    return Tracklet(
        source_id=source_id,
        first_frame=frames[0],
        last_frame=frames[-1],
        rows=np.arange(len(frames), dtype=np.intp),
        frames=frame_array,
    )


def _identities(assignments: list[Assignment]) -> list[int | None]:
    return [assignment.identity for assignment in assignments]


class TestPrototypes:
    def test_a_prototype_is_the_normalized_mean_of_normalized_vectors(self) -> None:
        result = prototypes(
            {
                "bob": np.array([[2.0, 0.0], [0.0, 3.0]]),
                "alice": np.array([[0.0, 5.0]]),
            }
        )
        assert result.names == ("alice", "bob")
        assert result.matrix.dtype == np.float32
        np.testing.assert_allclose(result.matrix[0], [0.0, 1.0])
        np.testing.assert_allclose(result.matrix[1], _unit(1.0, 1.0), rtol=1e-6)

    def test_a_prototype_does_not_depend_on_the_reference_row_order(self) -> None:
        # 1 + 1e-17 rounds to 1. An unsorted x sum is then 0 in some orders and
        # 1e-17 in others, and float32 holds 1e-17.
        rows = np.array([[1.0, 0.0], [1e-17, 1.0], [-1.0, 0.0]])
        results = [
            prototypes({"a": rows[list(order)], "b": rows[:1]}).matrix
            for order in itertools.permutations(range(len(rows)))
        ]
        for result in results[1:]:
            assert np.array_equal(results[0], result)

    @pytest.mark.parametrize(
        "vectors",
        [
            pytest.param({"a": np.zeros((0, 2))}, id="no-rows"),
            pytest.param({"": np.ones((1, 2))}, id="empty-name"),
            pytest.param({"a": np.ones((1, 2)), "b": np.ones((1, 3))}, id="mixed-D"),
            pytest.param({"a": np.zeros((1, 2))}, id="zero-vector"),
            pytest.param({"a": np.array([[np.nan, 1.0]])}, id="not-finite"),
            pytest.param({}, id="no-identities"),
        ],
    )
    def test_malformed_vectors_are_refused(
        self, vectors: dict[str, _FloatArray]
    ) -> None:
        with pytest.raises(ValueError):
            prototypes(vectors)


class TestTracklets:
    def test_a_track_splits_where_frames_differ_by_more_than_max_gap(self) -> None:
        frame = _int_array(0, 1, 3, 9, 10, 0, 20)
        source = _int_array(4, 4, 4, 4, 4, 7, 7)
        result = tracklets(frame, source, max_gap=2)
        assert [(t.source_id, t.first_frame, t.last_frame) for t in result] == [
            (4, 0, 3),
            (4, 9, 10),
            (7, 0, 0),
            (7, 20, 20),
        ]
        assert [t.rows.tolist() for t in result] == [[0, 1, 2], [3, 4], [5], [6]]
        assert result[0].frames.tolist() == [0, 1, 3]
        assert result[0].length == 3

    def test_a_gap_of_exactly_max_gap_does_not_split(self) -> None:
        result = tracklets(_int_array(0, 5), _int_array(1, 1), max_gap=5)
        assert len(result) == 1

    def test_max_gap_zero_splits_every_row(self) -> None:
        result = tracklets(_int_array(0, 1, 2), _int_array(1, 1, 1), max_gap=0)
        assert len(result) == 3

    def test_rows_are_ordered_by_frame_whatever_the_input_order(self) -> None:
        frame = _int_array(10, 0, 11, 1)
        source = _int_array(2, 2, 2, 2)
        result = tracklets(frame, source, max_gap=3)
        assert [t.rows.tolist() for t in result] == [[1, 3], [0, 2]]

    def test_a_repeated_frame_of_one_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame 3"):
            tracklets(_int_array(3, 3), _int_array(1, 1), max_gap=1)

    def test_a_negative_max_gap_is_refused(self) -> None:
        with pytest.raises(ValueError):
            tracklets(_int_array(0), _int_array(0), max_gap=-1)


class TestScore:
    def test_a_score_is_the_mean_cosine_to_each_prototype(self) -> None:
        protos = prototypes({"a": np.array([[1.0, 0.0]]), "b": np.array([[0.0, 1.0]])})
        vectors = np.array([[1.0, 0.0], [3.0, 3.0]])
        result = score(vectors, protos)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [(1.0 + 2**-0.5) / 2, 2**-0.5 / 2])

    def test_a_vector_of_another_dimension_is_refused(self) -> None:
        protos = prototypes({"a": np.array([[1.0, 0.0]])})
        with pytest.raises(ValueError):
            score(np.ones((1, 3)), protos)


class TestAssign:
    def test_a_tracklet_takes_its_best_identity_with_its_score_and_margin(
        self,
    ) -> None:
        tracklet = _tracklet(1, [0, 1])
        (result,) = assign(
            [tracklet], np.array([[0.9, 0.2]]), min_similarity=0.5, min_margin=0.05
        )
        assert result.tracklet is tracklet
        assert result.identity == 0
        assert result.best == 0
        assert result.score == pytest.approx(0.9)
        assert result.margin == pytest.approx(0.7)

    def test_a_tracklet_below_min_similarity_abstains(self) -> None:
        (result,) = assign(
            [_tracklet(1, [0])],
            np.array([[0.4, 0.1]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert result.identity is None
        assert result.best == 0
        assert result.score == pytest.approx(0.4)

    def test_a_tracklet_below_min_margin_abstains(self) -> None:
        (result,) = assign(
            [_tracklet(1, [0])],
            np.array([[0.9, 0.88]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert result.identity is None
        assert result.margin == pytest.approx(0.02)

    def test_equal_scores_rank_by_identity_name(self) -> None:
        (result,) = assign(
            [_tracklet(1, [0])],
            np.array([[0.2, 0.8, 0.8]]),
            min_similarity=0.5,
            min_margin=0.0,
        )
        assert result.best == 1
        assert result.identity == 1
        assert result.margin == 0.0

    def test_a_co_present_tracklet_cannot_take_an_assigned_identity(self) -> None:
        confident = _tracklet(1, [0, 1, 2])
        blocked = _tracklet(2, [2, 3])
        result = assign(
            [blocked, confident],
            np.array([[0.8, 0.6], [0.95, 0.1]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert _identities(result) == [None, 0]
        assert result[0].best == 0

    def test_a_blocked_tracklet_does_not_fall_back_to_its_second_identity(
        self,
    ) -> None:
        result = assign(
            [_tracklet(1, [0, 1]), _tracklet(2, [1, 2])],
            np.array([[0.95, 0.0], [0.9, 0.85]]),
            min_similarity=0.5,
            min_margin=0.0,
        )
        assert _identities(result) == [0, None]

    def test_tracklets_without_a_shared_frame_take_one_identity(self) -> None:
        result = assign(
            [_tracklet(1, [0, 1]), _tracklet(2, [2, 3])],
            np.array([[0.9, 0.1], [0.8, 0.2]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert _identities(result) == [0, 0]

    def test_overlapping_frame_ranges_without_a_shared_frame_take_one_identity(
        self,
    ) -> None:
        result = assign(
            [_tracklet(1, [0, 2, 4]), _tracklet(2, [1, 3])],
            np.array([[0.9, 0.1], [0.8, 0.2]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert _identities(result) == [0, 0]

    def test_a_longer_tracklet_wins_an_equal_margin(self) -> None:
        result = assign(
            [_tracklet(1, [0]), _tracklet(2, [0, 1])],
            np.array([[0.9, 0.5], [0.9, 0.5]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert _identities(result) == [None, 0]

    def test_the_lower_source_id_wins_an_equal_margin_and_length(self) -> None:
        result = assign(
            [_tracklet(5, [0]), _tracklet(3, [0])],
            np.array([[0.9, 0.5], [0.9, 0.5]]),
            min_similarity=0.5,
            min_margin=0.05,
        )
        assert _identities(result) == [None, 0]

    def test_the_earlier_first_frame_wins_within_one_source_id(self) -> None:
        later = _tracklet(3, [5])
        earlier = _tracklet(3, [0])
        other = _tracklet(4, [0, 5])
        result = assign(
            [later, earlier, other],
            np.array([[0.9, 0.5], [0.9, 0.5], [0.8, 0.79]]),
            min_similarity=0.5,
            min_margin=0.0,
        )
        assert _identities(result) == [0, 0, None]

    def test_a_declared_tracklet_takes_its_identity_and_blocks_others(self) -> None:
        declared = _tracklet(1, [0, 1])
        other = _tracklet(2, [1, 2])
        result = assign(
            [declared, other],
            np.array([[0.3, 0.2], [0.0, 0.9]]),
            min_similarity=0.5,
            min_margin=0.05,
            declared={0: 1},
        )
        assert _identities(result) == [1, None]
        assert result[0].best == 0
        assert result[0].score == pytest.approx(0.3)
        assert result[0].margin == pytest.approx(0.1)

    def test_a_tracklet_without_a_score_abstains_with_nan(self) -> None:
        unscored, scored = _tracklet(1, [0]), _tracklet(2, [1])
        result = assign(
            [unscored, scored],
            np.array([[np.nan, np.nan], [0.9, 0.1]]),
            min_similarity=-1.0,
            min_margin=0.0,
        )
        assert _identities(result) == [None, 0]
        assert np.isnan(result[0].score)
        assert np.isnan(result[0].margin)

    def test_a_declared_tracklet_without_a_score_takes_its_identity(self) -> None:
        declared, other = _tracklet(1, [0]), _tracklet(2, [0])
        result = assign(
            [declared, other],
            np.array([[np.nan, np.nan], [0.0, 0.9]]),
            min_similarity=0.5,
            min_margin=0.05,
            declared={0: 1},
        )
        assert _identities(result) == [1, None]
        assert np.isnan(result[0].score)

    def test_two_co_present_tracklets_declared_one_identity_are_refused(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="frame 1"):
            assign(
                [_tracklet(1, [0, 1]), _tracklet(2, [1])],
                np.array([[0.9, 0.1], [0.9, 0.1]]),
                min_similarity=0.5,
                min_margin=0.05,
                declared={0: 0, 1: 0},
            )

    @pytest.mark.parametrize(
        "declared",
        [
            pytest.param({1: 0}, id="position"),
            pytest.param({-1: 0}, id="negative-position"),
            pytest.param({0: 2}, id="identity"),
            pytest.param({0: -1}, id="negative-identity"),
        ],
    )
    def test_a_declared_value_out_of_range_is_refused(
        self, declared: dict[int, int]
    ) -> None:
        with pytest.raises(ValueError, match="out of range"):
            assign(
                [_tracklet(1, [0])],
                np.array([[0.9, 0.1]]),
                min_similarity=0.5,
                min_margin=0.05,
                declared=declared,
            )

    @pytest.mark.parametrize(
        "scores",
        [
            pytest.param(np.array([[0.9]]), id="one-identity"),
            pytest.param(np.array([[0.9, 0.1], [0.9, 0.1]]), id="row-count"),
        ],
    )
    def test_a_malformed_score_matrix_is_refused(self, scores: _FloatArray) -> None:
        with pytest.raises(ValueError):
            assign([_tracklet(1, [0])], scores, min_similarity=0.5, min_margin=0.05)


def _assign_entry(
    frame: npt.NDArray[np.int64],
    source: npt.NDArray[np.int64],
    vectors: _FloatArray,
    protos: Prototypes,
) -> list[tuple[int, int, int | None, float]]:
    entry_tracklets = tracklets(frame, source, max_gap=2)
    scores = np.stack([score(vectors[t.rows], protos) for t in entry_tracklets])
    return [
        (a.tracklet.source_id, a.tracklet.first_frame, a.identity, a.score)
        for a in assign(entry_tracklets, scores, min_similarity=0.5, min_margin=0.05)
    ]


def test_assignments_do_not_depend_on_row_order() -> None:
    generator = np.random.default_rng(7)
    directions = np.eye(4)[:3]
    protos = prototypes(
        {"a": directions[[0]], "b": directions[[1]], "c": directions[[2]]}
    )
    frames = np.arange(40, dtype=np.int64)
    frame = np.concatenate([frames, frames, frames[::2]])
    source = np.concatenate([np.full(40, 10), np.full(40, 11), np.full(20, 12)]).astype(
        np.int64
    )
    truth = np.concatenate([np.zeros(40), np.ones(40), np.full(20, 2)]).astype(int)
    vectors = directions[truth] + generator.normal(scale=0.2, size=(100, 4))

    expected = _assign_entry(frame, source, vectors, protos)
    order = generator.permutation(len(frame))
    assert _assign_entry(frame[order], source[order], vectors[order], protos) == (
        expected
    )
    assert {identity for _, _, identity, _ in expected} == {0, 1, 2}
