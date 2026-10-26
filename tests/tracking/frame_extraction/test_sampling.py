"""``select_frame_indices`` over stub sources, the fills and the exclusion mask."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pytest

from mosaic.core.json_value import JsonValue
from mosaic.tracking.frame_extraction import sampling
from mosaic.tracking.frame_extraction.sampling import (
    Exclusion,
    KmeansSettings,
    SamplingMethod,
    excluded_mask,
    fill_by_farthest_point,
    fill_largest_gaps,
    select_frame_indices,
    select_kmeans_frames,
)

LINE = np.array([[0.0], [1.0], [2.0], [3.0], [9.0]], dtype=np.float32)
"""Five one-dimensional features, at 0, 1, 2, 3 and 9."""

FIRST_DRAWS = [4, 2, 4, 4, 3]
"""The first picks that seeds 0 to 4 draw over the rows of ``LINE``."""

SETTINGS = KmeansSettings(
    resize=(16, 8),
    grayscale=False,
    max_candidates=None,
    batch_size=64,
    max_iter=20,
    n_init=3,
)


def _feature_rows(indices: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    """Return two features per frame that fall into four groups of frames."""
    return np.stack([indices % 4, indices // 25], axis=1).astype(np.float32)


@dataclass
class StubFeatures:
    """A candidate source whose features are a function of the frame index.

    Attributes:
        reads: The ``(start, end, step)`` of each candidate read.
        exact_reads: The indices of each read at exact indices.
        features_of: The features of an array of frame indices.
    """

    reads: list[tuple[int, int, int]] = field(
        default_factory=list[tuple[int, int, int]]
    )
    exact_reads: list[list[int]] = field(default_factory=list[list[int]])
    features_of: Callable[[npt.NDArray[np.int32]], npt.NDArray[np.float32]] = (
        _feature_rows
    )

    def candidates(
        self, start: int, end: int, step: int
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        self.reads.append((start, end, step))
        indices = np.arange(start, end + 1, step, dtype=np.int32)
        return indices, self.features_of(indices)

    def at(self, indices: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        self.exact_reads.append(indices.tolist())
        return self.features_of(np.asarray(indices, dtype=np.int32))


def _frame_line(indices: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    """Return one feature per frame, its index."""
    return indices.astype(np.float32).reshape(-1, 1)


def _three_levels(indices: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
    """Return one feature per frame: 0 below frame 30, 10 below frame 50, else 20."""
    levels = np.select([indices < 30, indices < 50], [0.0, 10.0], default=20.0)
    return levels.astype(np.float32).reshape(-1, 1)


def test_uniform_spaces_its_picks_over_the_candidate_range() -> None:
    source = StubFeatures()
    selected, details = select_frame_indices(
        "uniform",
        start=10,
        end=30,
        candidate_step=5,
        n_frames=3,
        source=source,
        kmeans=SETTINGS,
        random_state=0,
        excluded=None,
    )
    assert selected.tolist() == [10, 20, 30]
    assert details == {}
    assert source.reads == []


def test_kmeans_picks_from_the_candidates_that_the_source_reads() -> None:
    source = StubFeatures()
    selected, _ = select_frame_indices(
        "kmeans",
        start=3,
        end=60,
        candidate_step=3,
        n_frames=4,
        source=source,
        kmeans=SETTINGS,
        random_state=5,
        excluded=None,
    )
    assert source.reads == [(3, 60, 3)]
    indices = np.arange(3, 61, 3, dtype=np.int32)
    expected = select_kmeans_frames(
        indices,
        _feature_rows(indices),
        4,
        random_state=5,
        batch_size=64,
        max_iter=20,
        n_init=3,
    )
    assert selected.tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("candidate_step", "max_candidates", "effective_step"),
    [
        (1, 10, 10),
        (3, 10, 12),
        (2, 50, 2),
        (2, None, 2),
        (2, 0, 2),
    ],
)
def test_kmeans_widens_the_step_to_keep_the_candidates_under_the_cap(
    candidate_step: int, max_candidates: int | None, effective_step: int
) -> None:
    """From frames 0 to 99, a cap of 10 reads every 10th frame, or every 12th at 3.

    At step 3 the range holds 34 candidates, so the step is 3 times ceil(34 / 10).
    A cap that the range fits under, no cap and a cap of 0 keep the run's step.
    """
    source = StubFeatures()
    kmeans = KmeansSettings(
        resize=(16, 8),
        grayscale=False,
        max_candidates=max_candidates,
        batch_size=64,
        max_iter=20,
        n_init=3,
    )
    _, details = select_frame_indices(
        "kmeans",
        start=0,
        end=99,
        candidate_step=candidate_step,
        n_frames=4,
        source=source,
        kmeans=kmeans,
        random_state=0,
        excluded=None,
    )
    assert source.reads == [(0, 99, effective_step)]
    assert details["kmeans_effective_candidate_step"] == effective_step
    assert details["candidate_count"] == len(range(0, 100, effective_step))


def test_kmeans_records_its_settings_in_order() -> None:
    _, details = select_frame_indices(
        "kmeans",
        start=0,
        end=39,
        candidate_step=1,
        n_frames=4,
        source=StubFeatures(),
        kmeans=SETTINGS,
        random_state=0,
        excluded=None,
    )
    assert list(details.items()) == [
        ("kmeans_resize", [16, 8]),
        ("kmeans_grayscale", False),
        ("kmeans_max_candidates", None),
        ("kmeans_effective_candidate_step", 1),
        ("kmeans_batch_size", 64),
        ("kmeans_max_iter", 20),
        ("kmeans_n_init", 3),
        ("candidate_count", 40),
    ]


def test_fill_starts_from_the_anchors_without_a_random_pick() -> None:
    """Row 0 is farthest from the anchor at 10, then row 3 from row 0 and the anchor.

    Rows 1, 2 and 4 then tie at distance 1, and row 1 is the first of them. Without the
    anchor, these seeds draw the first picks of ``FIRST_DRAWS``.
    """
    anchors = np.array([[10.0]], dtype=np.float32)
    for seed in range(5):
        assert fill_by_farthest_point(LINE, [], 3, seed, anchors=anchors) == [0, 3, 1]


def test_fill_measures_from_the_anchors_and_the_selected_rows() -> None:
    """Row 3 is farthest from the anchor at -1 and the selected row 4.

    From row 4 alone, row 0 is farthest.
    """
    anchors = np.array([[-1.0]], dtype=np.float32)
    assert fill_by_farthest_point(LINE, [4], 3, 0, anchors=anchors) == [4, 3, 1]


def test_fill_ties_go_to_the_first_remaining_row() -> None:
    """Each pick takes the first of the tied rows.

    Every row is at distance 1 from the anchor. Rows 1 and 3 then tie, and then rows 2
    and 3.
    """
    features = np.array([[1.0], [-1.0], [1.0], [-1.0]], dtype=np.float32)
    anchors = np.array([[0.0]], dtype=np.float32)
    assert fill_by_farthest_point(features, [], 3, 0, anchors=anchors) == [0, 1, 2]


@pytest.mark.parametrize(
    "anchors",
    [None, np.empty(0, dtype=np.float32), np.empty((0, 1), dtype=np.float32)],
    ids=["none", "empty-vector", "empty-rows"],
)
def test_fill_without_anchors_or_selected_rows_draws_its_first_pick(
    anchors: npt.NDArray[np.float32] | None,
) -> None:
    """An anchor array without elements draws the first pick as ``None`` does."""
    first_picks = [
        fill_by_farthest_point(LINE, [], 1, seed, anchors=anchors)[0]
        for seed in range(5)
    ]
    assert first_picks == FIRST_DRAWS


def _exclude_record(
    *, count: int, spacing: int, removed: int, ignored: int
) -> list[tuple[str, JsonValue]]:
    """Return the exclude keys of a ``sampling`` record, in order."""
    return [
        ("exclude_count", count),
        ("exclude_spacing", spacing),
        ("candidates_removed", removed),
        ("exclude_entries_ignored", ignored),
    ]


@pytest.mark.parametrize(
    ("candidates", "excluded", "spacing", "marked"),
    [
        ([0, 4, 8, 12], [4, 6], 0, [False, True, False, False]),
        ([0, 4, 8, 12], [6], 2, [False, True, True, False]),
        ([0, 1, 29], [2, 27], 1, [False, True, False]),
        ([3, 9, 10, 16, 17], [5, 13], 3, [True, False, True, True, False]),
        ([0, 4, 8], [], 5, [False, False, False]),
    ],
    ids=[
        "equal-at-zero-spacing",
        "at-the-spacing-on-either-side",
        "before-the-first-and-after-the-last",
        "nearest-of-two",
        "no-excluded-frame",
    ],
)
def test_the_mask_marks_candidates_within_the_spacing_of_an_excluded_frame(
    candidates: list[int], excluded: list[int], spacing: int, marked: list[bool]
) -> None:
    mask = excluded_mask(
        np.array(candidates, dtype=np.int32),
        np.array(excluded, dtype=np.int64),
        spacing,
    )
    assert mask.tolist() == marked


@pytest.mark.parametrize(
    ("anchors", "n", "picks"),
    [
        ([60], 1, [29]),
        ([20, 80], 1, [50]),
        ([20, 80], 3, [35, 50, 65]),
        ([30, 69], 2, [14, 49]),
        ([30, 69], 3, [14, 49, 84]),
    ],
    ids=[
        "edge-gap-midpoint",
        "interior-gap-midpoint",
        "gaps-split-in-order",
        "equal-gaps-go-to-the-lower-start",
        "then-to-the-next-equal-gap",
    ],
)
def test_gap_filling_takes_the_midpoint_of_the_largest_gap(
    anchors: list[int], n: int, picks: list[int]
) -> None:
    """Frames 0 to 99 are candidates, with virtual anchors at -1 and 100.

    An edge gap is measured to its virtual anchor, so the anchor at 60 leaves the gap
    from -1 to 60, and its midpoint 29.5 takes frame 29 of the tied 29 and 30. Between
    20 and 80 the gap of 60 comes first, then the halves from 20 to 50 and from 50
    to 80, each of 30. Between 30 and 69 the gap of 39 is split at 49. The two edge
    gaps then tie at 31, and the gap from -1 is split at 14 before the gap to 100
    at 84.
    """
    candidates = np.arange(100, dtype=np.int32)
    remaining = candidates[~np.isin(candidates, anchors)]
    first_call = fill_largest_gaps(
        remaining, np.array(anchors, dtype=np.int64), n, first=0, last=99
    )
    second_call = fill_largest_gaps(
        remaining, np.array(anchors, dtype=np.int64), n, first=0, last=99
    )
    assert first_call.tolist() == picks
    assert second_call.tolist() == picks


def test_gap_filling_breaks_a_midpoint_tie_at_the_lower_frame() -> None:
    """Frames 3 and 6 are 1.5 from the midpoint 4.5 of the gap from -1 to 10.

    The pick of frame 3 leaves no candidate between -1 and 3, so frame 6 comes next.
    """
    remaining = np.array([3, 6], dtype=np.int32)
    no_anchors = np.empty(0, dtype=np.int64)
    one = fill_largest_gaps(remaining, no_anchors, 1, first=0, last=9)
    both = fill_largest_gaps(remaining, no_anchors, 5, first=0, last=9)
    assert one.tolist() == [3]
    assert both.tolist() == [3, 6]


def test_uniform_sampling_fills_the_gaps_around_the_excluded_frames() -> None:
    """Frames 13 to 17 are within 2 of frame 15, and the gaps then split as follows.

    From -1 to 15 (16) the pick is 7, from 15 to 30 (15) it is 22 of the tied 22
    and 23, and of the three gaps of 8 the one from -1 is split at 3.
    """
    source = StubFeatures()
    selected, details = select_frame_indices(
        "uniform",
        start=0,
        end=29,
        candidate_step=1,
        n_frames=3,
        source=source,
        kmeans=SETTINGS,
        random_state=0,
        excluded=Exclusion(indices=(15,), spacing=2, entries_ignored=3),
    )
    assert selected.tolist() == [3, 7, 22]
    assert list(details.items()) == _exclude_record(
        count=1, spacing=2, removed=5, ignored=3
    )
    assert source.reads == []
    assert source.exact_reads == []


@pytest.mark.parametrize("method", ["uniform", "kmeans"])
@pytest.mark.parametrize(
    ("excluded", "removed"),
    [((), 0), ((0, 99), 0), ((8,), 2)],
    ids=["no-excluded-frame", "outside-the-range", "outside-the-range-within-spacing"],
)
def test_without_an_anchor_the_method_picks_from_the_remaining_candidates(
    method: SamplingMethod, excluded: tuple[int, ...], removed: int
) -> None:
    """Candidates run from 10 to 60. An excluded frame outside them is no anchor.

    Frame 8 removes frames 10 and 11, which lie within 3 of it, and the method then
    runs over the candidates that remain.
    """
    plain, plain_details = select_frame_indices(
        method,
        start=10 + removed,
        end=60,
        candidate_step=1,
        n_frames=5,
        source=StubFeatures(),
        kmeans=SETTINGS,
        random_state=4,
        excluded=None,
    )
    source = StubFeatures()
    selected, details = select_frame_indices(
        method,
        start=10,
        end=60,
        candidate_step=1,
        n_frames=5,
        source=source,
        kmeans=SETTINGS,
        random_state=4,
        excluded=Exclusion(indices=excluded, spacing=3, entries_ignored=0),
    )
    assert selected.tolist() == plain.tolist()
    assert source.exact_reads == []
    record = _exclude_record(count=0, spacing=3, removed=removed, ignored=0)
    if method == "uniform":
        assert list(details.items()) == record
    else:
        settings = list(plain_details.items())[:-1]
        candidate_count: list[tuple[str, JsonValue]] = [("candidate_count", 51)]
        covered: list[tuple[str, JsonValue]] = [("clusters_covered", 0)]
        assert list(details.items()) == settings + candidate_count + record + covered


def test_kmeans_takes_the_clusters_that_no_excluded_frame_covers() -> None:
    """Frames 0 to 59 take three levels, and the excluded frame 5 covers level 0.

    The picks are the first frames of levels 10 and 20, which tie with the rest of
    their level at the centroid.
    """
    source = StubFeatures(features_of=_three_levels)
    selected, details = select_frame_indices(
        "kmeans",
        start=0,
        end=59,
        candidate_step=1,
        n_frames=2,
        source=source,
        kmeans=SETTINGS,
        random_state=0,
        excluded=Exclusion(indices=(5,), spacing=0, entries_ignored=0),
    )
    assert selected.tolist() == [30, 50]
    assert source.exact_reads == [[5]]
    assert list(details.items())[-6:] == [
        ("candidate_count", 60),
        *_exclude_record(count=1, spacing=0, removed=1, ignored=0),
        ("clusters_covered", 1),
    ]


type KmeansFit = tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]


@dataclass
class FakeFit:
    """A k-means fit that returns fixed labels and centers, and records its input.

    Attributes:
        labels: The label of each row of the features.
        centers: The center of each cluster.
        fits: The ``(features, n_clusters)`` of each fit.
    """

    labels: list[int]
    centers: list[float]
    fits: list[tuple[npt.NDArray[np.float32], int]] = field(
        default_factory=list[tuple[npt.NDArray[np.float32], int]]
    )

    def __call__(
        self,
        features: npt.NDArray[np.float32],
        n_clusters: int,
        *,
        random_state: int,
        batch_size: int,
        max_iter: int,
        n_init: int | str,
    ) -> KmeansFit:
        self.fits.append((features, n_clusters))
        return (
            np.array(self.labels, dtype=np.intp),
            np.array(self.centers, dtype=np.float32).reshape(-1, 1),
        )


def _select_on_the_even_frames(
    fake: FakeFit, n_frames: int, monkeypatch: pytest.MonkeyPatch
) -> npt.NDArray[np.int32]:
    """Select from the even frames 0 to 38, each its own index, around frames 1 and 3.

    Row ``i`` of the fit is frame ``2 * i``, and rows 20 and 21 are the anchors.
    """
    monkeypatch.setattr(sampling, "_fit_kmeans", fake)
    selected, _ = select_frame_indices(
        "kmeans",
        start=0,
        end=38,
        candidate_step=2,
        n_frames=n_frames,
        source=StubFeatures(features_of=_frame_line),
        kmeans=SETTINGS,
        random_state=0,
        excluded=Exclusion(indices=(1, 3), spacing=0, entries_ignored=0),
    )
    return selected


@pytest.mark.parametrize(
    ("first_of_cluster_two", "picks"),
    [(12, [14]), (11, [30])],
    ids=["equal-sizes-go-to-the-lower-cluster", "the-larger-cluster-goes-first"],
)
def test_kmeans_orders_the_uncovered_clusters_by_size_then_cluster(
    first_of_cluster_two: int, picks: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anchors and rows 0 to 3 form cluster 0. Clusters 1 and 2 split the rest.

    With eight rows each, cluster 1 comes first. Its center 15 ties frames 14 and 16,
    and frame 14 is taken. With nine rows in cluster 2, its frame 30 is taken.
    """
    labels = [0] * 4 + [1] * (first_of_cluster_two - 4)
    labels += [2] * (20 - first_of_cluster_two) + [0, 0]
    fake = FakeFit(labels=labels, centers=[2.0, 15.0, 31.0])
    assert _select_on_the_even_frames(fake, 1, monkeypatch).tolist() == picks
    [(features, n_clusters)] = fake.fits
    assert n_clusters == 3
    assert features.ravel().tolist() == [*range(0, 40, 2), 1, 3]


def test_kmeans_fills_a_shortfall_from_the_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cluster is uncovered, and its center 37 ties frames 36 and 38.

    The fill measures from the anchors 1 and 3 and from frame 36. Frame 20 is farthest
    from them (16), then frames 12 and 28 tie at 8, and frame 12 is the lower. From
    frame 36 alone, frame 0 would be farthest.
    """
    labels = [0] * 18 + [1, 1] + [0, 0]
    fake = FakeFit(labels=labels, centers=[9.0, 37.0, 0.0, 0.0, 0.0])
    assert _select_on_the_even_frames(fake, 3, monkeypatch).tolist() == [12, 20, 36]
    [(_, n_clusters)] = fake.fits
    assert n_clusters == 5


@pytest.mark.parametrize("method", ["uniform", "kmeans"])
def test_a_range_whose_every_candidate_is_excluded_is_refused(
    method: SamplingMethod,
) -> None:
    with pytest.raises(ValueError, match="all 6 candidate frames are excluded"):
        _ = select_frame_indices(
            method,
            start=0,
            end=10,
            candidate_step=2,
            n_frames=2,
            source=StubFeatures(),
            kmeans=SETTINGS,
            random_state=0,
            excluded=Exclusion(indices=(1, 5, 9), spacing=1, entries_ignored=0),
        )


class UndecodableFrames(StubFeatures):
    """A candidate source whose every frame fails to decode."""

    def candidates(
        self, start: int, end: int, step: int
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        self.reads.append((start, end, step))
        return np.empty(0, dtype=np.int32), np.empty((0, 2), dtype=np.float32)


@pytest.mark.parametrize(
    "excluded",
    [None, Exclusion(indices=(4,), spacing=0, entries_ignored=0)],
    ids=["without-exclude", "with-exclude"],
)
def test_a_range_without_a_decoded_candidate_is_refused_as_without_exclude(
    excluded: Exclusion | None,
) -> None:
    source = UndecodableFrames()
    with pytest.raises(ValueError, match="^No candidate frames to sample from"):
        _ = select_frame_indices(
            "kmeans",
            start=0,
            end=10,
            candidate_step=2,
            n_frames=2,
            source=source,
            kmeans=SETTINGS,
            random_state=0,
            excluded=excluded,
        )
    assert source.reads == [(0, 10, 2)]
    assert source.exact_reads == []
