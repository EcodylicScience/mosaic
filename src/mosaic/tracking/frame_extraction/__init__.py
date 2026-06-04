"""Frame extraction / sampling for annotation → pose/detection training.

Uniform and k-means sampling of representative video frames, saved as PNGs for
annotation. This is a tracking-domain concern: it reads media indexed by the
dataset and writes to the dataset's ``frames`` root via ``ds.get_root("frames")``.
Low-level frame decode/encode lives in :mod:`mosaic.core.media.video_io`.

The headline ``extract_frames(ds, ...)`` is the dataset-wide orchestrator; the
per-video workflow function is exported as ``extract_frames_single``.
"""

from .dataset_runs import (
    FramesIndexRow,
    extract_frames,
    frames_index,
    get_frame_manifests,
    get_frame_paths,
    list_frame_runs,
    list_media_pairs,
)
from .extraction import (
    FrameExtractionResult,
    extract_frames_multi,
    load_extraction_manifest,
)
from .extraction import extract_frames as extract_frames_single
from .sampling import select_kmeans_frames, select_uniform_frames

__all__ = [
    "FrameExtractionResult",
    "FramesIndexRow",
    "extract_frames",
    "extract_frames_multi",
    "extract_frames_single",
    "frames_index",
    "get_frame_manifests",
    "get_frame_paths",
    "list_frame_runs",
    "list_media_pairs",
    "load_extraction_manifest",
    "select_kmeans_frames",
    "select_uniform_frames",
]
