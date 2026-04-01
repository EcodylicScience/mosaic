"""Visualization library for behavior datasets.

This library provides modular visualization components:
- Data loading (tracks, labels, ground truth)
- Overlay preparation and frame drawing
- Video streaming with overlays
- Interactive video playback
- Egocentric crop generation

Example usage:
    from mosaic.behavior.visualization_library import playback
    playback.play_video(dataset, group="hex", sequence="hex_3", ...)

    from mosaic.behavior.visualization_library.egocentric_crop import EgocentricCrop
    crop_feat = EgocentricCrop(params={"target_id": 0, "crop_size": (256, 256)})
    dataset.run_feature(crop_feat, sequences=["hex_3"])
"""

# Import helpers module
# Import visualization modules
# Import egocentric crop feature
# Import visualization features (registered via @register_feature)
from . import (
    data_loading,
    egocentric_crop,
    helpers,
    interaction_crop,
    overlay,
    playback,
    video_stream,
    visual_spec,
    viz_global_colored,
    viz_timeline,
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
from .viz_global_colored import (
    VizGlobalColored,
)
from .viz_timeline import (
    TimelinePlot,
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
    "viz_global_colored",
    "viz_timeline",
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
    "VizGlobalColored",
    "TimelinePlot",
]
