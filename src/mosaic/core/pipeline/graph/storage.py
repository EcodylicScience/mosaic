"""Where a feature step's outputs live, computed without the feature registry.

``run_feature`` derives a run's directory from the feature's slug and what its
inputs are -- ``speed-angvel`` reading tracks stores under
``speed-angvel__from__tracks``. Planning has to reach the same answer, because
that name is half of what coverage is looked up by, and it has to reach it
without importing the feature library.

It can, because the rule takes no feature: the suffix is the ``+``-joined names
of the inputs, and a planner already knows those -- each is either the ``tracks``
literal or an upstream step's storage name, resolved earlier in the same
topological walk. So this is the same two functions ``run_feature`` composes,
called with the values rather than with an instance.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._utils import derive_storage_name

__all__ = ["storage_name_of"]


def storage_name_of(feature_slug: str, inputs: Sequence[str]) -> str:
    """The directory name a run of *feature_slug* over *inputs* writes into.

    Args:
        feature_slug: The feature's ``name``, e.g. ``speed-angvel``.
        inputs: What it reads, in order -- the ``tracks`` literal, or the
            **storage name** of an upstream feature. Storage names rather than
            slugs, because that is what ``Inputs.storage_suffix()`` joins: a
            ``Result`` carries the producing run's storage name, so a chain
            three deep nests its suffixes and this must reproduce that exactly.
            Empty for a feature that takes no pipeline inputs, which stores under
            its bare slug.

    Returns:
        The storage name, e.g. ``speed-angvel__from__tracks``.
    """
    suffix = "+".join(inputs) if inputs else None
    return derive_storage_name(feature_slug, suffix)
