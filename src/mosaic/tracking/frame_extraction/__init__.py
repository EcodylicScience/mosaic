"""Frame extraction / sampling for annotation → pose/detection training.

Uniform and k-means sampling of representative video frames, or the frames a
caller lists, saved as PNGs for annotation. This is a tracking-domain concern:
it reads media indexed by the dataset and writes to the dataset's ``frames``
root via ``ds.get_root("frames")``. Low-level frame decode/encode lives in
:mod:`mosaic.core.media.video_io`.

The headline ``extract_frames(ds, ...)`` is the dataset-wide orchestrator; the
per-video workflow function is exported as ``extract_frames_single``.

This file binds every public name on access, so ``frame_runs`` imports without
the extraction code and the media modules that it imports.
"""

from typing import TYPE_CHECKING, Final

from mosaic.lazy_exports import lazy_export

if TYPE_CHECKING:
    from .dataset_runs import (
        EntryFrames as EntryFrames,
        extract_frames as extract_frames,
        get_frame_manifests as get_frame_manifests,
        get_frame_paths as get_frame_paths,
    )
    from .extraction import (
        FrameExtractionResult as FrameExtractionResult,
        extract_frames as extract_frames_single,
        extract_frames_multi as extract_frames_multi,
        load_extraction_manifest as load_extraction_manifest,
    )
    from .frame_runs import (
        FramesIndexRow as FramesIndexRow,
        frames_index as frames_index,
        list_frame_runs as list_frame_runs,
    )
    from .sampling import (
        select_kmeans_frames as select_kmeans_frames,
        select_uniform_frames as select_uniform_frames,
    )

_MODULE_OF: Final[dict[str, str]] = {
    "EntryFrames": ".dataset_runs",
    "extract_frames": ".dataset_runs",
    "get_frame_manifests": ".dataset_runs",
    "get_frame_paths": ".dataset_runs",
    "FrameExtractionResult": ".extraction",
    "extract_frames_single": ".extraction:extract_frames",
    "extract_frames_multi": ".extraction",
    "load_extraction_manifest": ".extraction",
    "FramesIndexRow": ".frame_runs",
    "frames_index": ".frame_runs",
    "list_frame_runs": ".frame_runs",
    "select_kmeans_frames": ".sampling",
    "select_uniform_frames": ".sampling",
}
"""The submodule that defines each public name."""


def __getattr__(name: str) -> object:
    """Import the submodule that defines *name* on first access."""
    return lazy_export(__name__, _MODULE_OF, name)


__all__ = [
    "EntryFrames",
    "FrameExtractionResult",
    "FramesIndexRow",
    "extract_frames",
    "extract_frames_multi",
    "extract_frames_single",
    "frames_index",
    "get_frame_manifests",
    "get_frame_paths",
    "list_frame_runs",
    "load_extraction_manifest",
    "select_kmeans_frames",
    "select_uniform_frames",
]
