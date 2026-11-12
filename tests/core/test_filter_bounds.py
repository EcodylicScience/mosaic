import pytest

from mosaic.core.filter_bounds import TimeWindow, check_one_filter_per_boundary


def test_a_frame_and_a_time_on_different_boundaries_are_accepted() -> None:
    check_one_filter_per_boundary(
        start_frame=100, end_frame=None, start_time=None, end_time=5.0
    )


def test_a_frame_and_a_time_on_the_start_are_refused() -> None:
    with pytest.raises(
        ValueError, match="set one of filter_start_frame and filter_start_time"
    ):
        check_one_filter_per_boundary(
            start_frame=100, end_frame=None, start_time=1.0, end_time=None
        )


def test_a_frame_and_a_time_on_the_end_are_refused() -> None:
    with pytest.raises(
        ValueError, match="set one of filter_end_frame and filter_end_time"
    ):
        check_one_filter_per_boundary(
            start_frame=None, end_frame=500, start_time=None, end_time=5.0
        )


def test_a_time_window_is_open_until_a_bound_is_set() -> None:
    assert not TimeWindow().is_set
    assert TimeWindow(start=1.0).is_set
    assert TimeWindow(end=0.0).is_set
