"""The runs of consecutive labeled frames that a model of frame order trains on, and the entries it trains on.

A temporal model reads a window of neighboring frames, so a window must not span a gap
in the frame numbers or a frame that carries no label. `consecutive_runs` finds the
stretches of one series that are safe to cut windows from, and
`train_entries_of_templates` names the entries that an `extract-labeled-templates` run
kept for training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import (
    consecutive_runs,
    train_entries_of_templates,
)


@pytest.mark.parametrize(
    ("frames", "keep", "runs"),
    [
        pytest.param([0, 1, 2, 3], [1, 1, 1, 1], [(0, 4)], id="one-run"),
        pytest.param([], [], [], id="empty"),
        pytest.param([7], [1], [(0, 1)], id="one-frame"),
        pytest.param([0, 1, 2, 5, 6, 7], [1] * 6, [(0, 3), (3, 6)], id="frame-gap"),
        pytest.param(
            [0, 1, 2, 3, 4], [1, 1, 0, 1, 1], [(0, 2), (3, 5)], id="unlabeled"
        ),
        pytest.param([0, 1, 2, 3], [1, 0, 1, 1], [(0, 1), (2, 4)], id="lone-frame"),
        pytest.param([0, 1, 2], [0, 0, 0], [], id="nothing-kept"),
        pytest.param([0, 1, 3, 4], [1, 1, 1, 1], [(0, 2), (2, 4)], id="gap-of-one"),
        pytest.param([4, 5, 6, 8], [0, 1, 1, 1], [(1, 3), (3, 4)], id="gap-after-skip"),
    ],
)
def test_a_run_ends_at_a_frame_gap_and_at_a_frame_that_is_not_kept(
    frames: list[int], keep: list[int], runs: list[tuple[int, int]]
) -> None:
    assert consecutive_runs(frames, [bool(k) for k in keep]) == runs


def _provenance(root: Path, splits: dict[tuple[str, str], str]) -> Path:
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "group": [g for g, _ in splits],
            "sequence": [s for _, s in splits],
            "split": list(splits.values()),
            "count": 0,
            "proportion": 0.0,
        }
    ).to_parquet(root / "template_provenance.parquet")
    return root / "templates.parquet"


def test_the_train_entries_are_those_the_run_assigned_to_train(tmp_path: Path) -> None:
    templates = _provenance(
        tmp_path / "run",
        {("g", "a"): "train", ("g", "b"): "test", ("h", "c"): "train"},
    )

    assert train_entries_of_templates(templates) == [("g", "a"), ("h", "c")]


def test_a_run_without_provenance_has_no_known_train_entries(tmp_path: Path) -> None:
    assert train_entries_of_templates(tmp_path / "templates.parquet") is None
