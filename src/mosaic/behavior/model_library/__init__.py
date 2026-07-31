"""
Model library for behavior datasets.

This module provides machine learning models for behavioral analysis.

Usage
-----
>>> from mosaic.behavior.model_library.trex_identity_network import TRexIdentityNetwork
"""

# Shared architectures + checkpoint machinery for the T-Rex identity networks
from . import trex_identity_architectures

# T-Rex-compatible V200 CNN identity classifier (requires PyTorch)
from . import trex_identity_network
from .trex_identity_network import TRexIdentityNetwork

# T-Rex V118_3 — the compact 3-conv variant most real T-Rex checkpoints use.
# Called `TRexNativeIdentityNetwork` before 0.8, when it was mislabelled "V200".
from . import trex_v118_3_identity
from .trex_v118_3_identity import TRexV118_3IdentityNetwork

__all__ = [
    "trex_identity_architectures",
    "trex_identity_network",
    "TRexIdentityNetwork",
    "trex_v118_3_identity",
    "TRexV118_3IdentityNetwork",
]
