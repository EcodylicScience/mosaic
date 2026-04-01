"""
Model library for behavior datasets.

This module provides a collection of machine learning models for behavioral analysis.

Usage
-----
>>> from behavior import model_library
>>>
>>> # Use models by class
>>> model = model_library.behavior_xgboost.BehaviorXGBoostModel(params={...})
>>> model.bind_dataset(dataset)
>>> model.configure(config, run_root)
>>> metrics = model.train()
>>>
>>> # Access helpers
>>> from mosaic.behavior.model_library.helpers import XGB_PARAM_PRESETS
"""

# Import shared helpers
from . import helpers

# Import all models
from . import behavior_xgboost
from .behavior_xgboost import BehaviorXGBoostModel

# Optional: FERAL video behavior classifier (requires `feral` package)
try:
    from . import feral_model
    from .feral_model import FeralModel
except ImportError:
    pass

__all__ = [
    "helpers",
    "behavior_xgboost",
    "BehaviorXGBoostModel",
    "feral_model",
    "FeralModel",
]
