# Video I/O

Video reading utilities backed by the `mosaic-media` package.
`get_video_metadata` probes a file via `mosaic_media.probe_media` (which
shells out to system `ffprobe`) and normalizes rotated dimensions;
`open_frame_reader` returns a high-throughput sequential reader --
`mosaic_media.io.VideoReader` for plain video files, decoding in-process via
`av` with no `ffmpeg` binary required, or `ImgStoreFrameReader` for imgstore
directories.

The module also keeps the imgstore adapters (`ImgStoreCapture`,
`ImgStoreFrameReader`), the dispatchers that route on path type
(`open_frame_reader`, `MultiVideoReader`), and pure helpers for frame-range
and crop-rect normalization, candidate-frame extraction, and PNG writing.

::: mosaic.core.media.video_io
    options:
      show_source: true
