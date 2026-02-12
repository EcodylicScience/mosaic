"""Behavior analysis package — re-exports core contracts for backward compatibility."""
import warnings
from mosaic.core import Dataset, register_feature, to_safe_name, from_safe_name
from . import feature_library, label_library, model_library, visualization_library

def __getattr__(name):
    if name == "media":
        warnings.warn("behavior.media is deprecated. Use 'from mosaic import media' directly.",
                       DeprecationWarning, stacklevel=2)
        from mosaic import media
        return media
    if name == "tracking":
        warnings.warn("behavior.tracking is deprecated. Use 'from mosaic import tracking' directly.",
                       DeprecationWarning, stacklevel=2)
        from mosaic import tracking
        return tracking
    raise AttributeError(f"module 'behavior' has no attribute {name!r}")

__all__ = ["Dataset", "register_feature", "to_safe_name", "from_safe_name",
           "feature_library", "label_library", "model_library", "visualization_library"]
