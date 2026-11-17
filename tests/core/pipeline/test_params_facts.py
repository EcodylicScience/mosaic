"""The identity digest of a params model and whether it holds its defaults."""

from __future__ import annotations

from typing import Annotated

from mosaic.core.params import HASH_EXCLUDE, Params
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.params_facts import (
    params_equal_defaults,
    params_identity_digest,
)
from mosaic.core.pipeline.run import identity_terms
from mosaic.core.pipeline.types import Inputs


class _Knobs(Params):
    """Params with a field that enters the identity and one that does not."""

    real_field: int = 1
    batch_size: Annotated[int, HASH_EXCLUDE] = 4


class _Required(Params):
    """Params that cannot be built without an argument."""

    count: int


def test_the_digest_is_the_params_term_of_the_run_id() -> None:
    params = _Knobs(real_field=3, batch_size=16)

    digest = params_identity_digest(params)

    term = identity_terms(params, Inputs(("tracks",)))["_params"]
    assert digest == hash_params(term)
    assert len(digest) == 10
    assert digest == params_identity_digest(_Knobs(real_field=3, batch_size=2))
    assert digest != params_identity_digest(_Knobs(real_field=4))


def test_a_params_class_with_a_required_field_is_not_at_defaults() -> None:
    assert params_equal_defaults(_Required(count=1)) is False


def test_params_at_defaults_ignore_a_hash_excluded_field() -> None:
    assert params_equal_defaults(_Knobs()) is True
    assert params_equal_defaults(_Knobs(batch_size=99)) is True
    assert params_equal_defaults(_Knobs(real_field=2)) is False
