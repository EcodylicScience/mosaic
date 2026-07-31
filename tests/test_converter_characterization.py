"""What the annotation converters actually write, snapshotted before they move.

The seven entry points under ``pose_training/converters/`` had no test of any
kind -- nothing in ``tests/`` imported one. They are also the most duplicated
code in the tree: the usable-images filter is written five times, the
write-label-and-place-image loop four times, negative-patch sampling twice
verbatim, COCO loading three times, and five modules reach into ``cvat_points``'
underscore namespace for the splitter. Deduplicating that is the point of the
work these tests exist to protect, and it cannot be done safely against nothing.

Characterization, not specification. The snapshot records what the converters do
today, bugs included -- it is not a claim that any of it is right. Its job is to
make every behavioral consequence of a refactor *visible*, so a change is either
intended and re-blessed in the same commit, or a defect.

Regenerate with ``MOSAIC_UPDATE_GOLDEN=1 pytest tests/test_converter_characterization.py``
and read the diff. A moved line is a question to answer, never a formality.

What is recorded, and why only this:

* **Label text, exactly.** The coordinates are the output; rounding them would
  hide the class of change most likely to slip through.
* **The full sorted path list.** Which split a file lands in, and the empty-test
  -split removal, are behavior.
* **Arrays by shape, dtype and digest.** The localizer converters emit ``.npy``;
  their negative sampling is seeded (``np.random.RandomState(seed)``), so the
  bytes are reproducible.
* **Images by presence and link kind only.** Their content is the unmodified
  input, and the symlink target is an absolute path into ``tmp_path``.
* **The returned schema.** It is what reaches ``make_data_yaml``, so it is part
  of the contract even though it never touches disk here.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mosaic.tracking.pose_training.converters.coco_keypoints import (
    convert_coco_keypoints,
)
from mosaic.tracking.pose_training.converters.coco_localizer import (
    convert_coco_localizer,
)
from mosaic.tracking.pose_training.converters.coco_points import convert_coco_points
from mosaic.tracking.pose_training.converters.cvat_localizer import (
    convert_cvat_localizer,
)
from mosaic.tracking.pose_training.converters.cvat_points import (
    convert_cvat_points,
    convert_cvat_points_polo,
)
from mosaic.tracking.pose_training.converters.lightning_pose import (
    convert_lightning_pose,
)

GOLDEN_PATH = Path(__file__).parent / "data" / "converter_characterization.json"

# Small enough to read, large enough that a 16px patch fits with room for the
# localizer's negative sampling to have somewhere to go.
_IMG_W = 160
_IMG_H = 120
_PATCH = 16

# Two "videos" of four frames, so ``split_by="group"`` is expressible and the
# default ``__frame`` group key finds something.
_FRAMES: list[tuple[str, int]] = [
    (video, index) for video in ("vidA", "vidB") for index in range(4)
]

# Every converter is called with this split rather than the (0.8, 0.15, 0.05)
# default: eight items under the default give a valid split of zero, which would
# make the snapshot silent about the split boundary that matters most.
_SPLIT: tuple[float, float, float] = (0.5, 0.25, 0.25)
_SEED = 20260731

_KEYPOINTS: tuple[str, ...] = ("nose", "thorax", "abdomen")
_SKELETON: tuple[tuple[int, int], ...] = ((1, 2), (2, 3))


def _image_name(video: str, index: int) -> str:
    return f"{video}__frame_{index:04d}.png"


def _write_images(images_dir: Path) -> None:
    """Deterministic greyscale PNGs -- content is irrelevant, readability is not.

    The localizer converters decode with ``cv2.imread``, so these must be real
    images rather than placeholder bytes.
    """
    import cv2

    images_dir.mkdir(parents=True, exist_ok=True)
    for order, (video, index) in enumerate(_FRAMES):
        frame = np.full((_IMG_H, _IMG_W), (order * 17) % 256, dtype=np.uint8)
        # A brighter block so patches at different points differ from each other.
        frame[20:60, 30:90] = (order * 53) % 256
        cv2.imwrite(str(images_dir / _image_name(video, index)), frame)


def _points_for(order: int) -> tuple[float, float]:
    """A per-frame point, kept well inside the border so no patch is clipped."""
    return 40.0 + order * 8.0, 50.0 + order * 4.0


def _write_coco(path: Path) -> None:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for order, (video, index) in enumerate(_FRAMES):
        image_id = order + 1
        images.append(
            {
                "id": image_id,
                "file_name": _image_name(video, index),
                "width": _IMG_W,
                "height": _IMG_H,
            }
        )
        x, y = _points_for(order)
        # Three keypoints, the last deliberately invisible (v=0) so the
        # zero-the-coordinates branch is exercised.
        keypoints = [x, y, 2, x + 12.0, y + 6.0, 2, 0.0, 0.0, 0]
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x - 15.0, y - 10.0, 40.0, 28.0],
                "num_keypoints": 2,
                "iscrowd": 0,
                "keypoints": keypoints,
            }
        )
        # Every second image carries a second instance, so the instance axis is
        # visible in the snapshot rather than being assumed absent.
        if order % 2 == 1:
            annotations.append(
                {
                    "id": 100 + image_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x + 20.0, y + 15.0, 30.0, 22.0],
                    "num_keypoints": 3,
                    "iscrowd": 0,
                    "keypoints": [
                        x + 25.0,
                        y + 20.0,
                        2,
                        x + 35.0,
                        y + 26.0,
                        1,
                        x + 30.0,
                        y + 30.0,
                        2,
                    ],
                }
            )
    path.write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {
                        "id": 1,
                        "name": "mouse",
                        "keypoints": list(_KEYPOINTS),
                        "skeleton": [list(edge) for edge in _SKELETON],
                    }
                ],
            },
            indent=2,
        )
    )


def _write_cvat(path: Path) -> None:
    parts = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>"]
    for order, (video, index) in enumerate(_FRAMES):
        x, y = _points_for(order)
        klass = "worker" if order % 2 == 0 else "queen"
        parts.append(
            f'  <image id="{order}" name="{_image_name(video, index)}" '
            f'width="{_IMG_W}" height="{_IMG_H}">'
        )
        parts.append(f'    <points label="animal" points="{x:.1f},{y:.1f}">')
        parts.append(f'      <attribute name="class">{klass}</attribute>')
        parts.append("    </points>")
        # A second element with two semicolon-joined points, so both the
        # multi-point split and the multi-element path are covered.
        parts.append(
            f'    <points label="animal" points="{x + 20:.1f},{y + 12:.1f};'
            f'{x + 30:.1f},{y + 18:.1f}">'
        )
        parts.append('      <attribute name="class">worker</attribute>')
        parts.append("    </points>")
        parts.append("  </image>")
    parts.append("</annotations>")
    path.write_text("\n".join(parts) + "\n")


def _write_lightning_pose_csv(path: Path) -> None:
    scorer = "heatmap_tracker"
    header_scorer = ["scorer"] + [scorer] * (len(_KEYPOINTS) * 3)
    header_bodypart = ["bodyparts"]
    header_coord = ["coords"]
    for name in _KEYPOINTS:
        header_bodypart.extend([name] * 3)
        header_coord.extend(["x", "y", "likelihood"])

    rows = [",".join(header_scorer), ",".join(header_bodypart), ",".join(header_coord)]
    for order, _ in enumerate(_FRAMES):
        x, y = _points_for(order)
        cells: list[str] = [str(order)]
        for offset, name in enumerate(_KEYPOINTS):
            # The last keypoint of every third frame falls below the default
            # 0.5 confidence threshold, exercising the vis=0 branch.
            likelihood = 0.2 if (offset == 2 and order % 3 == 0) else 0.9
            cells.extend(
                [
                    f"{x + offset * 9.0:.2f}",
                    f"{y + offset * 5.0:.2f}",
                    f"{likelihood:.2f}",
                ]
            )
        rows.append(",".join(cells))
    path.write_text("\n".join(rows) + "\n")


def _extracted_frame_records(images_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "frame_index": order,
            "path": str(images_dir / _image_name(video, index)),
            "width": _IMG_W,
            "height": _IMG_H,
        }
        for order, (video, index) in enumerate(_FRAMES)
    ]


def _digest(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _describe_file(path: Path, root: Path) -> dict[str, Any]:
    if path.suffix == ".npy":
        array = np.load(path)
        return {
            "kind": "array",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "digest": _digest(np.ascontiguousarray(array).tobytes()),
        }
    if path.suffix in {".png", ".jpg", ".jpeg"}:
        # Content is the unmodified input and the symlink target is an absolute
        # path into tmp_path, so only presence and link kind are portable.
        return {"kind": "image", "symlink": path.is_symlink()}
    return {"kind": "text", "text": path.read_text()}


def _snapshot(output_dir: Path, schema: object) -> dict[str, Any]:
    """The comparable form of one converter run.

    Round-tripped through JSON so the comparison is against the same shape the
    golden holds: a schema's ``skeleton`` is a list of tuples in memory and a
    list of lists on disk, and ``radii`` is keyed by ``int`` in memory and by
    ``str`` on disk. Comparing the live object against the parsed file would
    fail on both, every time, for no behavioral reason.
    """
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_dir():
            continue
        files[path.relative_to(output_dir).as_posix()] = _describe_file(
            path, output_dir
        )
    snapshot = {
        "schema": {
            "type": type(schema).__name__,
            **{
                field: getattr(schema, field)
                for field in ("names", "skeleton", "radii", "thresholds")
                if hasattr(schema, field)
            },
        },
        "files": files,
    }
    parsed: dict[str, Any] = json.loads(json.dumps(snapshot, sort_keys=True))
    return parsed


# --- the seven cases --------------------------------------------------------


def _case_coco_keypoints(sources: dict[str, Any], out: Path) -> object:
    return convert_coco_keypoints(
        sources["coco"], sources["images"], out, split=_SPLIT, seed=_SEED
    )


def _case_coco_keypoints_bbox_from_keypoints(
    sources: dict[str, Any], out: Path
) -> object:
    """``bbox_source="keypoints"`` takes the other branch, and hardcodes tight."""
    return convert_coco_keypoints(
        sources["coco"],
        sources["images"],
        out,
        split=_SPLIT,
        seed=_SEED,
        bbox_source="keypoints",
    )


def _case_coco_points(sources: dict[str, Any], out: Path) -> object:
    return convert_coco_points(
        sources["coco"],
        sources["images"],
        out,
        radii={"mouse": 12.0},
        split=_SPLIT,
        seed=_SEED,
    )


def _case_coco_localizer(sources: dict[str, Any], out: Path) -> object:
    return convert_coco_localizer(
        sources["coco"],
        sources["images"],
        out,
        patch_size=_PATCH,
        min_negative_dist=24.0,
        split=_SPLIT,
        seed=_SEED,
    )


def _case_cvat_points(sources: dict[str, Any], out: Path) -> object:
    return convert_cvat_points(
        sources["cvat"], sources["images"], out, split=_SPLIT, seed=_SEED
    )


def _case_cvat_points_polo(sources: dict[str, Any], out: Path) -> object:
    return convert_cvat_points_polo(
        sources["cvat"],
        sources["images"],
        out,
        radii={"worker": 10.0, "queen": 18.0},
        split=_SPLIT,
        seed=_SEED,
    )


def _case_cvat_localizer(sources: dict[str, Any], out: Path) -> object:
    return convert_cvat_localizer(
        sources["cvat"],
        sources["images"],
        out,
        patch_size=_PATCH,
        min_negative_dist=24.0,
        split=_SPLIT,
        seed=_SEED,
    )


def _case_lightning_pose(sources: dict[str, Any], out: Path) -> object:
    return convert_lightning_pose(
        sources["lp_csv"],
        sources["lp_frames"],
        out,
        img_w=_IMG_W,
        img_h=_IMG_H,
        split=_SPLIT,
        seed=_SEED,
    )


CASES: dict[str, Callable[[dict[str, Any], Path], object]] = {
    "coco_keypoints": _case_coco_keypoints,
    "coco_keypoints_bbox_from_keypoints": _case_coco_keypoints_bbox_from_keypoints,
    "coco_points": _case_coco_points,
    "coco_localizer": _case_coco_localizer,
    "cvat_points": _case_cvat_points,
    "cvat_points_polo": _case_cvat_points_polo,
    "cvat_localizer": _case_cvat_localizer,
    "lightning_pose": _case_lightning_pose,
}


@pytest.fixture
def sources(tmp_path: Path) -> dict[str, Any]:
    images_dir = tmp_path / "images"
    _write_images(images_dir)
    coco = tmp_path / "annotations.json"
    _write_coco(coco)
    cvat = tmp_path / "annotations.xml"
    _write_cvat(cvat)
    lp_csv = tmp_path / "predictions.csv"
    _write_lightning_pose_csv(lp_csv)
    return {
        "images": images_dir,
        "coco": coco,
        "cvat": cvat,
        "lp_csv": lp_csv,
        "lp_frames": _extracted_frame_records(images_dir),
    }


def _load_golden() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():
        return {}
    return json.loads(GOLDEN_PATH.read_text())


@pytest.mark.parametrize("case", sorted(CASES))
def test_converter_output_matches_the_snapshot(
    case: str, sources: dict[str, Any], tmp_path: Path
) -> None:
    produced = _snapshot(
        tmp_path / "out" / case,
        CASES[case](sources, tmp_path / "out" / case),
    )

    if os.environ.get("MOSAIC_UPDATE_GOLDEN") == "1":
        golden = _load_golden()
        golden[case] = produced
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated golden for {case}")

    golden = _load_golden()
    assert case in golden, (
        f"No golden entry for {case!r}. Regenerate with "
        f"MOSAIC_UPDATE_GOLDEN=1 and read the diff before committing."
    )
    assert produced == golden[case]


def test_the_golden_covers_every_case() -> None:
    """A case added without a golden entry, or left behind after one is removed."""
    if os.environ.get("MOSAIC_UPDATE_GOLDEN") == "1":
        pytest.skip("regenerating")
    assert set(_load_golden()) == set(CASES)
