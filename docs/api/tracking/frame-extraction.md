# Frame Extraction

Uniform and k-means sampling of representative video frames, saved as PNGs for
annotation and pose/detection-model training. The dataset-wide entry point is
`mosaic.tracking.extract_frames(ds, ...)`; low-level frame decode/encode lives in
`mosaic.core.media.video_io`.

::: mosaic.tracking.frame_extraction
    options:
      show_source: true
