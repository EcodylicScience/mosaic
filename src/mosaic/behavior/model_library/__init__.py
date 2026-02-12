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

__all__ = [
    "helpers",
    "behavior_xgboost",
]
