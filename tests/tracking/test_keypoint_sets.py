"""The revision reader: a saved keypoint set read back with its images anchored."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from mosaic.core.annotations.pose_annotations import PoseAnnotationSet, PoseFrame
from mosaic.core.annotations.projection import write_keypoint_set_revision
from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LabelsScanSource
from mosaic.tracking import register_ops
from mosaic.tracking.keypoint_sets import (
    KeypointSetRef,
    ResolvedSet,
    read_resolved_set,
    resolve_keypoint_sets,
    revision_image_root,
)
from mosaic.tracking.ops import prepare
from mosaic.tracking.ops.prepare import PrepareTrainingDataParams, check_preparation
from tests.helpers import (
    make_dataset,
    pose_frame,
    pose_object,
    pose_set,
    revision_file,
)

register_ops()

SET_KEY = "17-openfield"


@pytest.fixture
def claimed(tmp_path: Path) -> tuple[Dataset, Dataset]:
    """A project that saved a frame of two recordings, and a library that claimed its revision."""
    project = make_dataset(tmp_path / "52", name="mice")
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    frames: list[PoseFrame] = []
    for sequence in ("m01", "m02"):
        relative = Path(f"media/frames/{sequence}/frame_000000.png")
        image = project.base_dir / relative
        image.parent.mkdir(parents=True)
        _ = image.write_bytes(sequence.encode())
        frames.append(
            pose_frame(
                relative, pose_object((10.0, 12.0), (30.0, 20.0)), sequence=sequence
            )
        )
    saved = write_keypoint_set_revision(
        project,
        set_key=SET_KEY,
        annotations=pose_set(frames, image_root=project.base_dir),
        origin={"dolt_commit": "c1"},
    )
    library.add_scan_source(
        LabelsScanSource(
            id="mice-set",
            path=str(project.get_root("labels_raw") / "keypoints" / SET_KEY),
            files=(revision_file(saved.revision),),
            series="keypoints",
        )
    )
    _ = library.scan_labels()
    return project, library


def test_read_resolved_set_anchors_images_at_the_manifest_image_root(
    claimed: tuple[Dataset, Dataset],
) -> None:
    project, library = claimed
    (item,) = resolve_keypoint_sets(
        library, (KeypointSetRef(set_key=SET_KEY),), kind="any-op"
    )

    state = read_resolved_set(item)
    root = revision_image_root(state.image_root)

    assert root.resolve() == project.base_dir.resolve()
    for frame in state.frames:
        assert (root / frame.image_path).is_file()


def test_revision_image_root_refuses_a_set_without_one() -> None:
    with pytest.raises(ValueError, match="lacks an image root"):
        _ = revision_image_root(None)


def test_resolve_keypoint_sets_requires_kind_as_a_keyword() -> None:
    parameter = inspect.signature(resolve_keypoint_sets).parameters["kind"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_prepare_reads_each_set_through_read_resolved_set(
    claimed: tuple[Dataset, Dataset], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, library = claimed
    read: list[str] = []
    real = prepare.read_resolved_set

    def spy(item: ResolvedSet) -> PoseAnnotationSet:
        read.append(item.set_key)
        return real(item)

    monkeypatch.setattr(prepare, "read_resolved_set", spy)
    check_preparation(
        library,
        PrepareTrainingDataParams(sets=(KeypointSetRef(set_key=SET_KEY),)),
    )
    assert read == [SET_KEY]
