"""Low-level media I/O: video/imgstore reading, decoding, and encoding.

This file binds every public name on access. Python executes it before any
``mosaic.core.media.*`` submodule, and the reader and writer modules import cv2
and av. A dataset read that needs only the media index or a placement imports a
leaf below this package and loads neither
(``tests/meta/test_media_lazy_exports.py``).

Each name is also declared under ``TYPE_CHECKING`` for type checkers and editors.
"""

from typing import TYPE_CHECKING, Final

from mosaic.lazy_exports import lazy_export

if TYPE_CHECKING:
    from mosaic_media import (
        MediaFacts as MediaFacts,
        MediaProbeError as MediaProbeError,
        probe_media as probe_media,
    )
    from mosaic_media.io import FFmpegVideoWriter as FFmpegVideoWriter

    from .imgstore_io import (
        ImgStoreCapture as ImgStoreCapture,
        ImgStoreFrameReader as ImgStoreFrameReader,
        imgstore_metadata as imgstore_metadata,
        imgstore_probe as imgstore_probe,
        is_imgstore as is_imgstore,
    )
    from .read_target import ReadTarget as ReadTarget
    from .timeline import (
        ConcatenatedTimeline as ConcatenatedTimeline,
        TimelineSegment as TimelineSegment,
        concatenated_timeline as concatenated_timeline,
    )
    from .video_io import (
        FrameReader as FrameReader,
        MultiVideoReader as MultiVideoReader,
        MultiVideoReaderLike as MultiVideoReaderLike,
        SupportsCapture as SupportsCapture,
        VideoMetadata as VideoMetadata,
        VideoSegment as VideoSegment,
        open_frame_reader as open_frame_reader,
    )

_MODULE_OF: Final[dict[str, str]] = {
    "MediaFacts": "mosaic_media",
    "MediaProbeError": "mosaic_media",
    "probe_media": "mosaic_media",
    "FFmpegVideoWriter": "mosaic_media.io",
    "ImgStoreCapture": ".imgstore_io",
    "ImgStoreFrameReader": ".imgstore_io",
    "imgstore_metadata": ".imgstore_io",
    "imgstore_probe": ".imgstore_io",
    "is_imgstore": ".imgstore_io",
    "ReadTarget": ".read_target",
    "ConcatenatedTimeline": ".timeline",
    "TimelineSegment": ".timeline",
    "concatenated_timeline": ".timeline",
    "FrameReader": ".video_io",
    "MultiVideoReader": ".video_io",
    "MultiVideoReaderLike": ".video_io",
    "SupportsCapture": ".video_io",
    "VideoMetadata": ".video_io",
    "VideoSegment": ".video_io",
    "open_frame_reader": ".video_io",
}
"""The module that defines each public name."""


def __getattr__(name: str) -> object:
    """Import the module that defines *name* on first access."""
    return lazy_export(__name__, _MODULE_OF, name)


__all__ = [
    "ConcatenatedTimeline",
    "concatenated_timeline",
    "FFmpegVideoWriter",
    "FrameReader",
    "ImgStoreCapture",
    "ImgStoreFrameReader",
    "imgstore_metadata",
    "imgstore_probe",
    "is_imgstore",
    "MediaFacts",
    "MediaProbeError",
    "MultiVideoReader",
    "MultiVideoReaderLike",
    "open_frame_reader",
    "probe_media",
    "ReadTarget",
    "SupportsCapture",
    "TimelineSegment",
    "VideoMetadata",
    "VideoSegment",
]
