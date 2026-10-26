"""The phrases that mosaic's messages share."""

from __future__ import annotations

import pytest

from mosaic.core.wording import counted


@pytest.mark.parametrize(
    ("count", "phrase"),
    [(0, "0 keypoints"), (1, "1 keypoint"), (2, "2 keypoints")],
)
def test_a_count_is_singular_only_at_one(count: int, phrase: str) -> None:
    assert counted(count, "keypoint") == phrase
