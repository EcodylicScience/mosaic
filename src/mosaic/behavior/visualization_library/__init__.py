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
from . import helpers

# Import visualization modules
from . import data_loading
from . import overlay
from . import video_stream
from . import playback
from . import visual_spec

# Import egocentric crop feature
from . import egocentric_crop

# Import visualization features (registered via @register_feature)
from . import viz_global_colored

# Re-export common functions for convenience
from .data_loading import (
    load_tracks_and_labels,
    load_ground_truth_labels,
    demo_load_visual_inputs,
)
from .overlay import (
    prepare_overlay,
    draw_frame,
)
from .video_stream import (
    render_stream,
)
from .playback import (
    play_video,
    play_video_with_spec,
    build_overlay,
)
from .visual_spec import (
    normalize_visualization_spec,
    apply_visualization_spec,
    list_visual_adapters,
)
from .egocentric_crop import (
    EgocentricCrop,
)
from .viz_global_colored import (
    VizGlobalColored,
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
    "viz_global_colored",
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
    "VizGlobalColored",
]
