# Video I/O

Video reading utilities with support for raw H.264 elementary streams
(common from Raspberry Pi cameras) where OpenCV metadata is unreliable.

Handles frame counting, FPS detection via ffprobe, and sequential reading
for non-seekable streams.

::: mosaic.media.video_io
    options:
      show_source: true
