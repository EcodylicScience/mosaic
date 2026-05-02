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

# T-Rex-compatible V200 CNN identity classifier (requires PyTorch)
from . import trex_identity_network
from .trex_identity_network import TRexIdentityNetwork

# T-Rex *native* V200 (3-conv variant) — matches T-Rex's actual saved .pth layout
from . import trex_native_identity
from .trex_native_identity import TRexNativeIdentityNetwork

# FERAL video behavior classifier -- DEPRECATED, use FeralFeature instead
from . import feral_model
from .feral_model import FeralModel

__all__ = [
    "helpers",
    "behavior_xgboost",
    "BehaviorXGBoostModel",
    "trex_identity_network",
    "TRexIdentityNetwork",
    "trex_native_identity",
    "TRexNativeIdentityNetwork",
    "feral_model",
    "FeralModel",
]
