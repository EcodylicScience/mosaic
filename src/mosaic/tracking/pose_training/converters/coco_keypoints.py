"""Convert COCO Keypoints JSON to YOLO pose training labels."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

from mosaic.core.annotations import AnnotationFrame, KeypointSchema
from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.readers import read_coco_keypoints
from mosaic.core.annotations.split import (
    default_group_key,
    print_split_summary,
    split_filenames,
)

from .emit import usable_frames, write_split_tree, yolo_pose_line


def convert_coco_keypoints(
    coco_json_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    *,
    category_name: str | None = None,
    keypoint_indices: Sequence[int] | None = None,
    split: tuple[float, float, float] = (0.8, 0.15, 0.05),
    class_id: int = 0,
    bbox_source: str = "annotation",
    bbox_margin: float = 0.1,
    symlink_images: bool = True,
    seed: int = 42,
    split_by: str = "image",
    group_key: Callable[[str], str] | None = None,
) -> KeypointSchema:
    """Convert COCO Keypoints JSON to YOLO pose labels.

    Reads a COCO Keypoints 1.0 JSON file (e.g. exported from CVAT) and
    produces a YOLO pose dataset with train/valid/test splits.

    Parameters
    ----------
    coco_json_path : path
        Path to the COCO Keypoints JSON file.
    images_dir : path
        Directory containing the source images.  Image filenames must match
        the ``file_name`` field in the COCO JSON ``images`` array.
    output_dir : path
        Root directory for the YOLO dataset output.
    category_name : str, optional
        Which COCO category to convert.  None = use the first category.
    keypoint_indices : sequence of int, optional
        Subset of keypoint indices to include.  None = all keypoints.
    split : (train, valid, test) floats
        Fraction of images per split.  Must sum to ~1.0.
    class_id : int
        YOLO class ID (typically 0 for single-class pose).
    bbox_source : str
        ``"annotation"`` uses the COCO annotation bbox directly.
        ``"keypoints"`` recomputes the bbox from keypoint coordinates.
    bbox_margin : float
        Margin for keypoint-derived bbox (only used when
        ``bbox_source="keypoints"``).
    symlink_images : bool
        If True, create symlinks to source images.  If False, copy them.
    seed : int
        Random seed for train/valid/test assignment.
    split_by : ``"image"`` or ``"group"``
        When ``"group"``, all images sharing the same group key (e.g.
        frames from the same video) are kept together in one split.
    group_key : callable, optional
        Function ``filename -> group_name``.  Only used when
        ``split_by="group"``.  Defaults to splitting on ``__frame``.

    Returns
    -------
    KeypointSchema
        Keypoint schema with names and skeleton from the COCO categories.
    """
    coco_json_path = Path(coco_json_path)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    annotations = read_coco_keypoints(
        coco_json_path,
        images_dir,
        category_name=category_name,
        keypoint_indices=keypoint_indices,
    )
    schema = annotations.schema

    usable = usable_frames(annotations)
    if not usable:
        print(
            f"[coco_keypoints] WARNING: no usable images found. "
            f"{len(annotations.frames)} images in JSON, "
            f"{len(annotations.annotated_frames)} annotated. "
            f"Check that images_dir contains matching filenames."
        )
        return schema

    split_assignment, n_train, n_valid = split_filenames(
        [frame.image_path.name for frame, _ in usable],
        split,
        seed,
        split_by=split_by,
        group_key=group_key,
    )

    # ``bbox_source="annotation"`` is the source's own box; "keypoints" says to
    # derive one, so the source box is dropped before the emitter sees it.
    policy = BboxPolicy(margin=bbox_margin)
    keep_source_box = bbox_source == "annotation"

    def lines_for(frame: AnnotationFrame) -> list[str]:
        rows: list[str] = []
        for obj in frame.objects:
            instance = obj if keep_source_box else dataclasses.replace(obj, bbox=None)
            row = yolo_pose_line(
                instance,
                frame.width,
                frame.height,
                class_id=class_id,
                policy=policy,
            )
            if row is not None:
                rows.append(row)
        return rows

    written, skipped = write_split_tree(
        usable,
        output_dir,
        split_assignment,
        lines_for,
        symlink_images=symlink_images,
    )

    print(
        f"[coco_keypoints] Wrote {written} labels to {output_dir}"
        + (f"  (skipped {skipped} with no valid keypoints)" if skipped else "")
    )
    print(
        f"  Category: '{annotations.categories[0]}', keypoints: {schema.num_keypoints}"
    )
    print_split_summary(
        split_assignment,
        n_train,
        n_valid,
        len(usable),
        split_by,
        group_key or default_group_key,
    )

    return schema
