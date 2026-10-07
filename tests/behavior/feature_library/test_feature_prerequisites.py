"""Which features need more than the base install, and what each one needs.

A feature declares its optional packages and the setups of its run in
``prerequisites``. The pipeline catalog offers it as unavailable while one is
unmet, and validation refuses a step running it.
The table names every feature that declares one.
"""

from __future__ import annotations

import pytest

from mosaic.behavior.feature_library.feral_feature import FERAL
from mosaic.behavior.feature_library.kpms import KpmsEnvironment, KpmsLicense
from mosaic.behavior.feature_library.lightning_action_feature import (
    LIGHTNING_ACTION,
)
from mosaic.behavior.feature_library.movement.convert import MOVEMENT
from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.behavior.model_library.timm_backbone import TIMM, TORCH
from mosaic.core.narrowing import is_object_tuple
from mosaic.optional_dependency import Prerequisite

EXPECTED: dict[str, tuple[Prerequisite, ...]] = {
    "feral": (FERAL,),
    "global-identity-dinov2-temporal": (TORCH,),
    "global-identity-embedding": (TIMM,),
    "global-identity-model": (TIMM,),
    "kpms": (KpmsLicense(), KpmsEnvironment()),
    "lightning-action": (LIGHTNING_ACTION,),
    "movement-filter-interpolate": (MOVEMENT,),
    "movement-smooth": (MOVEMENT,),
}

FEATURE_CLASSES = sorted(FEATURES.values(), key=lambda cls: cls.__name__)


def test_the_features_that_need_more_than_the_base_install() -> None:
    declared: dict[str, tuple[Prerequisite, ...]] = {
        cls.name: getattr(cls, "prerequisites", ()) for cls in FEATURE_CLASSES
    }

    assert {name: needs for name, needs in declared.items() if needs} == EXPECTED


@pytest.mark.parametrize(
    "cls", FEATURE_CLASSES, ids=[cls.__name__ for cls in FEATURE_CLASSES]
)
def test_every_declared_prerequisite_answers_missing(cls: type) -> None:
    """Each declaration is a tuple of ``Prerequisite``, the shape the catalog reads."""
    declared: object = getattr(cls, "prerequisites", ())

    assert is_object_tuple(declared)
    assert all(isinstance(item, Prerequisite) for item in declared)
