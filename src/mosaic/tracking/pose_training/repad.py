"""Re-deriving the boxes of an emitted YOLO-pose dataset.

The reason this exists has not changed: a dataset labelled with midline
keypoints gets a box that collapses to a line, and training a detector on it
wants a box around the animal instead -- without relabelling anything.

What changed is that it is no longer its own machinery. The old rewriter parsed
label rows into arrays, recomputed four columns and re-emitted the text, because
the padding choice was not expressible at the point the dataset was built. Now
it is: :class:`~mosaic.core.annotations.bbox.BboxPolicy` says how a box is
derived, so re-padding is reading the dataset and emitting it again under a
different policy.

That is a smaller thing to own and a larger thing to do. The same two steps also
re-split a dataset, convert it to COCO, or export it as a ``.slp``, none of
which a bbox rewriter could reach.
"""

from __future__ import annotations

from pathlib import Path

from mosaic.core.annotations.bbox import BboxPolicy
from mosaic.core.annotations.model import AnnotationFrame, KeypointSchema

from .converters.emit import usable_frames, write_split_tree, yolo_pose_line

__all__ = ["repad_yolo_pose"]


def repad_yolo_pose(
    src_dir: str | Path,
    dst_dir: str | Path,
    schema: KeypointSchema,
    *,
    policy: BboxPolicy,
    class_id: int = 0,
    symlink_images: bool = True,
) -> tuple[int, int]:
    """Write *src_dir* to *dst_dir* with every box re-derived under *policy*.

    Keypoints, visibility, the instance axis and the split each frame was in are
    carried through; only the four box columns change.

    Args:
        src_dir: An emitted YOLO-pose dataset.
        dst_dir: Where to write the re-derived one.
        schema: The keypoint layout the labels were written against. YOLO does
            not record it, so a caller that guesses wrong gets an error rather
            than a silently misread dataset.
        policy: How to derive each box. ``method="isotropic"`` is the answer to
            a collapsed midline box; ``"oriented"`` also needs head and tail
            indices.
        class_id: The class id to write.
        symlink_images: Link rather than copy the images across.

    Returns:
        ``(written, skipped)``, where skipped counts instances no box could be
        derived for.

    Raises:
        FileNotFoundError: *src_dir* holds none of the expected splits.
        ValueError: A label row does not match *schema*, or ``"oriented"`` was
            asked for without both indices.
    """
    from mosaic.core.annotations.readers import read_yolo_pose

    if policy.method == "oriented" and (
        policy.head_index is None or policy.tail_index is None
    ):
        raise ValueError(
            "an oriented box is defined by the head-tail axis, so the policy "
            "must name head_index and tail_index"
        )

    annotations = read_yolo_pose(src_dir, schema)

    def lines_for(frame: AnnotationFrame) -> list[str]:
        rows = [
            yolo_pose_line(
                obj, frame.width, frame.height, class_id=class_id, policy=policy
            )
            for obj in frame.objects
        ]
        return [row for row in rows if row is not None]

    return write_split_tree(
        usable_frames(annotations),
        Path(dst_dir),
        {f.image_path.name: f.split for f in annotations.frames},
        lines_for,
        symlink_images=symlink_images,
    )
