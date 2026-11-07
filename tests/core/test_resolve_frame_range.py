import re

import pytest
from mosaic.core.helpers import TIME_FILTER_NEEDS_FPS, resolve_frame_range


def test_frame_only():
    assert resolve_frame_range(fps=30, start_frame=100, end_frame=500) == (100, 500)


def test_time_only():
    assert resolve_frame_range(fps=30, start_time=1.0, end_time=5.0) == (30, 150)


def test_mixed():
    assert resolve_frame_range(fps=30, start_frame=100, end_time=5.0) == (100, 150)


def test_conflict_start():
    with pytest.raises(ValueError, match="Cannot set both start_frame and start_time"):
        resolve_frame_range(fps=30, start_frame=100, start_time=1.0)


def test_conflict_end():
    with pytest.raises(ValueError, match="Cannot set both end_frame and end_time"):
        resolve_frame_range(fps=30, end_frame=500, end_time=5.0)


def test_no_fps_with_time():
    with pytest.raises(ValueError, match=re.escape(TIME_FILTER_NEEDS_FPS)):
        resolve_frame_range(fps=None, start_time=1.0)


def test_the_missing_frame_rate_message_names_the_key_and_the_frame_filters():
    with pytest.raises(ValueError) as caught:
        resolve_frame_range(fps=None, end_time=5.0)

    message = str(caught.value)
    assert "meta.fps_default" in message
    assert "filter_start_frame" in message
    assert "filter_end_frame" in message


def test_all_none():
    assert resolve_frame_range(fps=30) == (None, None)


def test_none_fps_with_frames():
    assert resolve_frame_range(fps=None, start_frame=100) == (100, None)
