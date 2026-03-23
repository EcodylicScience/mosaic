"""Core data contracts, schemas, and dataset orchestration."""

from . import track_library  # triggers track converter registration
from .dataset import Dataset
from .helpers import from_safe_name, to_safe_name


def __getattr__(name: str):
    if name == "register_feature":
        from mosaic.behavior.feature_library.spec import register_feature

        return register_feature
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Dataset", "register_feature", "to_safe_name", "from_safe_name"]
