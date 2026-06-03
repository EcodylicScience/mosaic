"""Media utilities for behavior datasets."""

from .extraction import extract_frames, extract_frames_multi, load_extraction_manifest
from .imgstore_io import (
    ImgStoreCapture,
    ImgStoreFrameReader,
    imgstore_metadata,
    imgstore_probe,
    is_imgstore,
)
from .video_io import (
    FFmpegFrameReader,
    FFmpegVideoWriter,
    FrameReader,
    MultiVideoReader,
    SupportsCapture,
    VideoMetadata,
    VideoSegment,
    open_capture,
    open_frame_reader,
)

__all__ = [
    "extract_frames",
    "extract_frames_multi",
    "FFmpegFrameReader",
    "FFmpegVideoWriter",
    "FrameReader",
    "ImgStoreCapture",
    "ImgStoreFrameReader",
    "imgstore_metadata",
    "imgstore_probe",
    "is_imgstore",
    "load_extraction_manifest",
    "MultiVideoReader",
    "open_capture",
    "open_frame_reader",
    "SupportsCapture",
    "VideoMetadata",
    "VideoSegment",
]
