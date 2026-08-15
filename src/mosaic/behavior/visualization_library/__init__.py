"""Rendering a sequence's tracks over its video, plus the crop features.

Two unrelated things live here, and the distinction matters:

- **The overlay renderer** -- ``load_tracks_and_labels`` -> ``prepare_overlay``
  -> ``render_stream`` -> ``play_video``. Plain functions, and the one
  presentational thing the toolkit keeps: a headless "draw this sequence's
  tracks on its video" is how you check a tracker actually worked, and it is
  what a CLI or queue context cannot get from a web interface.
- **The media features** -- ``overlay``, ``egocentric-crop`` and
  ``interaction-crop-pipeline``. Despite living here, these write *artifacts*
  something else reads: egocentric crops are the input all three identity models
  take, and the annotated video is the thing a biologist looks at. They are
  categorized ``media`` rather than as visualization for that reason, and
  ``overlay`` is a feature so that a graph can end on the deliverable rather
  than one step short of it.

Static plotting is deliberately absent. ``viz-timeline`` and
``viz-global-colored`` wrote matplotlib PNGs from a compute backend and were
retired; ``load_values()`` gets the same numbers out for a caller to plot however
it likes.

Example usage:
    from mosaic.behavior.visualization_library import playback
    playback.play_video(dataset, group="hex", sequence="hex_3", ...)

    from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
    crop_feat = EgocentricCrop(params={"target_id": 0, "crop_size": (256, 256)})
    dataset.run_feature(crop_feat, sequences=["hex_3"])
"""

from . import (
    data_loading,
    egocentric_crop,
    helpers,
    interaction_crop,
    overlay,
    overlay_feature,
    playback,
    video_stream,
    visual_spec,
)

# Re-export common functions for convenience
from .data_loading import (
    demo_load_visual_inputs,
    load_ground_truth_labels,
    load_tracks_and_labels,
)
from .egocentric_crop import (
    EgocentricCrop,
)
from .interaction_crop import (
    InteractionCropPipeline,
)
from .overlay import (
    draw_frame,
    prepare_overlay,
)
from .overlay_feature import (
    Overlay,
)
from .playback import (
    build_overlay,
    play_video,
    play_video_with_spec,
)
from .video_stream import (
    render_stream,
)
from .visual_spec import (
    apply_visualization_spec,
    list_visual_adapters,
    normalize_visualization_spec,
)

__all__ = [
    # Modules
    "helpers",
    "data_loading",
    "overlay",
    "video_stream",
    "playback",
    "visual_spec",
    "egocentric_crop",
    "interaction_crop",
    "overlay_feature",
    # Functions
    "load_tracks_and_labels",
    "load_ground_truth_labels",
    "demo_load_visual_inputs",
    "prepare_overlay",
    "draw_frame",
    "render_stream",
    "play_video",
    "play_video_with_spec",
    "build_overlay",
    "normalize_visualization_spec",
    "apply_visualization_spec",
    "list_visual_adapters",
    # Classes
    "EgocentricCrop",
    "InteractionCropPipeline",
    "Overlay",
]
