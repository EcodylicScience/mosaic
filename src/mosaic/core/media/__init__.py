"""Low-level media I/O: video/imgstore reading, decoding, and encoding."""

from mosaic_media import MediaFacts, MediaProbeError, probe_media
from mosaic_media.io import FFmpegVideoWriter

from .imgstore_io import (
    ImgStoreCapture,
    ImgStoreFrameReader,
    imgstore_metadata,
    imgstore_probe,
    is_imgstore,
)
from .video_io import (
    FrameReader,
    MultiVideoReader,
    MultiVideoReaderLike,
    SupportsCapture,
    VideoMetadata,
    VideoSegment,
    open_frame_reader,
)

__all__ = [
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
    "SupportsCapture",
    "VideoMetadata",
    "VideoSegment",
]
