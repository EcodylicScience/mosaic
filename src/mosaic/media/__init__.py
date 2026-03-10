"""Media utilities for behavior datasets."""

from .extraction import extract_frames, extract_frames_multi, load_extraction_manifest
from .video_io import FFmpegFrameReader, FFmpegVideoWriter, MultiVideoReader, VideoMetadata, VideoSegment

__all__ = [
    "extract_frames",
    "extract_frames_multi",
    "FFmpegFrameReader",
    "FFmpegVideoWriter",
    "load_extraction_manifest",
    "MultiVideoReader",
    "VideoMetadata",
    "VideoSegment",
]
