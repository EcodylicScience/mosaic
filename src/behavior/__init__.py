"""Public package surface for behavior utilities."""

from .dataset import Dataset, register_feature
from .helpers import from_safe_name, to_safe_name

# Import sub-packages
from . import media
from . import tracking

# Import label_library to register converters
from . import label_library

# Import feature_library to register features
from . import feature_library

# Import model_library for machine learning models
from . import model_library

__all__ = [
    "Dataset",
    "register_feature",
    "to_safe_name",
    "from_safe_name",
    "media",
    "tracking",
    "feature_library",
    "label_library",
    "model_library",
]
