"""Re-deriving the boxes of an emitted dataset, without relabelling.

The capability the old bbox rewriter existed for, kept: a dataset labelled with
midline keypoints gets a box that collapses to a line, and padding it is what
makes the dataset trainable. What changed is that padding is now a declared
policy rather than a separate rewriting pass, so these assert the outcome rather
than the mechanism.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from mosaic.core.annotations import KeypointSchema
from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.tracking.pose_training.repad import repad_yolo_pose

IMG_W, IMG_H = 1000, 1000
HEAD_IDX, TAIL_IDX = 0, 5
SCHEMA = KeypointSchema(names=tuple(f"kp{i}" for i in range(6)))


def _degenerate_dataset(root: Path) -> None:
    """One horizontal mouse: six colinear keypoints, so a tight box has no height."""
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (IMG_W, IMG_H), (128, 128, 128)).save(images / "frame000.png")

    xs = np.linspace(300.0, 700.0, 6)
    parts = ["0", "0.5", "0.5", "0.4", "0.001"]  # the collapsed box
    for x in xs:
        parts += [f"{x / IMG_W:.6f}", f"{500.0 / IMG_H:.6f}", "2"]
    _ = (labels / "frame000.txt").write_text(" ".join(parts) + "\n")


def _box(path: Path) -> tuple[float, float, float, float]:
    cx, cy, width, height = (float(v) for v in path.read_text().split()[1:5])
    return cx, cy, width, height


def test_padding_rescues_a_collapsed_box(tmp_path: Path) -> None:
    """The whole point: a box with no height becomes one that can be trained on."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _degenerate_dataset(src)

    written, skipped = repad_yolo_pose(
        src,
        dst,
        SCHEMA,
        policy=BboxPolicy(
            method="isotropic",
            head_index=HEAD_IDX,
            tail_index=TAIL_IDX,
            pad_frac_of_body=0.3,
            min_pad_px=20.0,
        ),
    )
    assert (written, skipped) == (1, 0)

    _, _, _, height = _box(dst / "train" / "labels" / "frame000.txt")
    assert height > 0.2, "was 0.001 before"


def test_the_pad_size_is_the_policy_the_caller_declared(tmp_path: Path) -> None:
    """Padding is a parameter, so two policies must give two different boxes."""
    src = tmp_path / "src"
    _degenerate_dataset(src)

    def height_for(pad: float) -> float:
        out = tmp_path / f"dst{pad}"
        _ = repad_yolo_pose(
            src,
            out,
            SCHEMA,
            policy=BboxPolicy(
                method="isotropic",
                head_index=HEAD_IDX,
                tail_index=TAIL_IDX,
                pad_frac_of_body=pad,
                min_pad_px=20.0,
            ),
        )
        return _box(out / "train" / "labels" / "frame000.txt")[3]

    # body length 400px; pad = max(20, 400 * frac), height = 2 * pad
    assert height_for(0.3) == pytest.approx(0.24, abs=0.01)
    assert height_for(0.5) == pytest.approx(0.40, abs=0.01)


def test_oriented_padding_needs_the_axis_it_is_defined_by(tmp_path: Path) -> None:
    src = tmp_path / "src"
    _degenerate_dataset(src)
    with pytest.raises(ValueError, match="head_index and tail_index"):
        _ = repad_yolo_pose(
            src, tmp_path / "dst", SCHEMA, policy=BboxPolicy(method="oriented")
        )


def test_the_keypoints_are_carried_through_untouched(tmp_path: Path) -> None:
    """Only the four box columns move; a re-pad that edits keypoints is a bug."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _degenerate_dataset(src)
    before = (src / "train" / "labels" / "frame000.txt").read_text().split()[5:]

    _ = repad_yolo_pose(
        src,
        dst,
        SCHEMA,
        policy=BboxPolicy(method="isotropic", head_index=HEAD_IDX, tail_index=TAIL_IDX),
    )
    after = (dst / "train" / "labels" / "frame000.txt").read_text().split()[5:]

    assert [round(float(v), 4) for v in after] == [round(float(v), 4) for v in before]


def test_a_schema_that_does_not_match_is_an_error(tmp_path: Path) -> None:
    """YOLO does not record the layout, so a wrong guess must not be silent."""
    src = tmp_path / "src"
    _degenerate_dataset(src)
    with pytest.raises(ValueError, match="schema declares"):
        _ = repad_yolo_pose(
            src,
            tmp_path / "dst",
            KeypointSchema(names=("a", "b", "c")),
            policy=BboxPolicy(method="isotropic"),
        )


def test_the_split_a_frame_was_in_is_preserved(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    _degenerate_dataset(src)
    for split in ("valid", "test"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
        Image.new("RGB", (IMG_W, IMG_H)).save(src / split / "images" / f"{split}.png")
        parts = ["0", "0.5", "0.5", "0.4", "0.001"]
        for x in np.linspace(300.0, 700.0, 6):
            parts += [f"{x / IMG_W:.6f}", "0.500000", "2"]
        _ = (src / split / "labels" / f"{split}.txt").write_text(" ".join(parts) + "\n")

    _ = repad_yolo_pose(
        src,
        dst,
        SCHEMA,
        policy=BboxPolicy(method="isotropic", head_index=HEAD_IDX, tail_index=TAIL_IDX),
    )
    assert (dst / "train" / "labels" / "frame000.txt").exists()
    assert (dst / "valid" / "labels" / "valid.txt").exists()
    assert (dst / "test" / "labels" / "test.txt").exists()
