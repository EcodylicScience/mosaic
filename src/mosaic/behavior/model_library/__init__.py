"""
Model library for behavior datasets.

This module provides machine learning models for behavioral analysis.

Usage
-----
>>> from mosaic.behavior.model_library.trex_identity_network import TRexIdentityNetwork
"""

# T-Rex-compatible V200 CNN identity classifier (requires PyTorch)
from . import trex_identity_network
from .trex_identity_network import TRexIdentityNetwork

# T-Rex *native* V200 (3-conv variant) — matches T-Rex's actual saved .pth layout
from . import trex_native_identity
from .trex_native_identity import TRexNativeIdentityNetwork

__all__ = [
    "trex_identity_network",
    "TRexIdentityNetwork",
    "trex_native_identity",
    "TRexNativeIdentityNetwork",
]
