"""The sleap-nn release floor that an embedding model needs."""

from __future__ import annotations

import pytest

from mosaic.tracking.sleap.probe import SleapProbeResponse
from mosaic.tracking.sleap.version import (
    SLEAP_NN_EMBEDDING_MIN,
    require_sleap_nn_at_least,
)


def _probe(version: str) -> SleapProbeResponse:
    return SleapProbeResponse(has_sleap_nn=bool(version), sleap_nn_version=version)


def test_the_embedding_floor_is_the_release_that_ran_the_trial() -> None:
    assert SLEAP_NN_EMBEDDING_MIN == "0.3.5"


@pytest.mark.parametrize("version", ["0.3.5", "0.4.0", "0.3.5rc1", "1.0"])
def test_a_release_from_the_floor_on_is_accepted(version: str) -> None:
    require_sleap_nn_at_least(_probe(version), SLEAP_NN_EMBEDDING_MIN)


@pytest.mark.parametrize("version", ["0.3.4", "0.3", "0.2.9", ""])
def test_an_older_or_unreadable_release_is_refused_naming_both(version: str) -> None:
    with pytest.raises(ValueError) as caught:
        require_sleap_nn_at_least(_probe(version), SLEAP_NN_EMBEDDING_MIN)

    message = str(caught.value)
    assert repr(version) in message
    assert SLEAP_NN_EMBEDDING_MIN in message
