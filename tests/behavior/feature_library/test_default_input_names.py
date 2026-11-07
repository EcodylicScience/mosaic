"""A default input or params reference names a storage directory, not a slug.

A feature that reads inputs stores under ``<slug>__from__<inputs>``, so its
bare slug names no stored run, and ``run_feature`` refuses it with
``UnrunFeatureInput``. A default that names one makes the feature fail when it
is built with its defaults.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.core.pipeline.storage_names import feature_slug_of
from mosaic.core.pipeline.types import Feature, Result

_KNOWN_BARE_DEFAULTS = {("orientation-rel", "scale")}
"""Defaults that still name a bare slug, as ``(feature, params field)``.

``orientation-rel``'s ``scale`` defaults to ``body-scale``, which reads inputs.
A run with that default fails with ``UnrunFeatureInput`` until the default
changes.
"""


def _built_with_defaults(cls: type[Feature]) -> Feature | None:
    """*cls* built with no arguments, or None when its defaults do not validate."""
    try:
        return cls()
    except (TypeError, ValidationError):
        return None


_DEFAULT_BUILT = sorted(
    (
        (cls.__name__, feature)
        for cls in FEATURES.values()
        if (feature := _built_with_defaults(cls)) is not None
    ),
    key=lambda pair: pair[0],
)


def _requires_an_input(cls: type[Feature]) -> bool:
    inputs_cls: type | None = getattr(cls, "Inputs", None)
    require: object = getattr(inputs_cls, "_require", "nonempty")
    return require == "nonempty"


_READS_INPUTS = frozenset(
    cls.name for cls in FEATURES.values() if _requires_an_input(cls)
)
"""The slugs of the features that require an input, so never store under the slug."""


def _referenced_names(feature: Feature) -> list[tuple[str, str]]:
    """Each ``(where, name)`` a default input or top-level params field names."""
    names = [("inputs", result.feature) for result in feature.inputs.feature_inputs]
    params = feature.params
    for field_name in type(params).model_fields:
        value = getattr(params, field_name)
        # ``ArtifactSpec`` and ``NNResult`` are ``Result`` subclasses.
        match value:
            case Result(feature=str(name)):
                names.append((field_name, name))
            case _:
                pass
    return names


@pytest.mark.parametrize(
    "feature",
    [feature for _, feature in _DEFAULT_BUILT],
    ids=[name for name, _ in _DEFAULT_BUILT],
)
def test_no_default_names_the_bare_slug_of_a_feature_that_reads_inputs(
    feature: Feature,
) -> None:
    bare = [
        (where, name)
        for where, name in _referenced_names(feature)
        if name in _READS_INPUTS
        and feature_slug_of(name) == name
        and (feature.name, where) not in _KNOWN_BARE_DEFAULTS
    ]
    assert not bare, (
        f"{feature.name} defaults to the bare slug of a feature that reads inputs: "
        f"{bare}. Name its storage directory, such as '<slug>__from__tracks'."
    )
