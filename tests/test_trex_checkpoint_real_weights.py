"""Parity against real, deployed T-Rex identity checkpoints.

``test_trex_checkpoint_interop`` validates mosaic against T-Rex's *current*
source. This file validates it against models that were actually trained and
deployed, which is not the same claim: T-Rex's input preprocessing has changed
across builds, and the checkpoints in circulation predate the version now
checked out. Only a real checkpoint exercises that path.

The oracle is the TorchScript sidecar T-Rex writes next to every checkpoint
(``<base>_model.pth``). It carries the preprocessing *code*, not just the
weights, so agreeing with it is agreement with what T-Rex actually computed --
no assumption about the build required.

The weights are several megabytes each and live outside the repo, so this is
marked slow and skips when they are absent. Point ``MOSAIC_TREX_MODELS_DIR`` at
a directory of ``<name>/weights.pth`` + ``<name>/weights_model.pth`` pairs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mosaic.behavior.model_library.trex_identity_architectures import (  # noqa: E402
    load_trex_torchscript,
)
from mosaic.behavior.model_library.trex_identity_network import (  # noqa: E402
    TRexIdentityNetwork,
)
from mosaic.behavior.model_library.trex_v118_3_identity import (  # noqa: E402
    TRexV118_3IdentityNetwork,
)

_MODELS_DIR = Path(
    os.environ.get(
        "MOSAIC_TREX_MODELS_DIR",
        "/Volumes/JD-SSD/ESI-mice/identitymodels-comparing/moad-models",
    )
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _MODELS_DIR.is_dir(),
        reason=(
            f"real T-Rex checkpoints not found at {_MODELS_DIR}; set "
            f"MOSAIC_TREX_MODELS_DIR to a directory of <name>/weights.pth pairs"
        ),
    ),
]

# name -> (loader, crop size). These are cage-ehav00016 mouse identity models.
_CASES = {
    "v118_3": (TRexV118_3IdentityNetwork, 80),
    "v200": (TRexIdentityNetwork, 80),
    "v200_sz128_full": (TRexIdentityNetwork, 128),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_matches_trex_torchscript(name: str) -> None:
    """mosaic reproduces what T-Rex itself computes for this checkpoint.

    Covers the whole chain at once -- preprocessing, architecture, norm layers,
    key layout -- against a model whose predictions were used for real work.
    Before 0.8 none of these files loaded at all: the V200 pair raised on
    positional-vs-named key mismatch, and the V118_3 raised on the
    ``normalize.*`` buffers.
    """
    loader, size = _CASES[name]
    directory = _MODELS_DIR / name
    weights = directory / "weights.pth"
    sidecar = directory / "weights_model.pth"
    if not weights.is_file() or not sidecar.is_file():
        pytest.skip(f"{name}: weights.pth / weights_model.pth not both present")

    net = loader.from_trex_checkpoint(weights)
    reference, _ = load_trex_torchscript(sidecar)
    reference.eval()

    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (8, size, size, 1), dtype=np.uint8)

    with torch.no_grad():
        expected = torch.softmax(
            reference(torch.from_numpy(images).float()), dim=1
        ).numpy()
    actual = net.predict(images)

    np.testing.assert_allclose(actual, expected, atol=1e-4)
    assert (actual.argmax(1) == expected.argmax(1)).all()


@pytest.mark.parametrize("name", sorted(_CASES))
def test_real_checkpoint_normalization_is_read_from_the_file(name: str) -> None:
    """These builds shipped their statistics in the weights; we must use them.

    Their ``normalize.mean`` / ``normalize.std`` buffers are what make the
    contract knowable rather than guessed, so loading one must neither warn
    about ambiguity nor fall back to a default.
    """
    loader, _ = _CASES[name]
    weights = _MODELS_DIR / name / "weights.pth"
    if not weights.is_file():
        pytest.skip(f"{name}: weights.pth not present")

    state = torch.load(weights, map_location="cpu", weights_only=False)["state_dict"]
    if "normalize.mean" not in state:
        pytest.skip(f"{name}: checkpoint carries no normalize buffers")

    net = loader.from_trex_checkpoint(weights)
    assert net.input_normalization == "imagenet_scaled"
    np.testing.assert_allclose(
        net._model.normalize.mean.flatten().numpy(),
        state["normalize.mean"].flatten().numpy(),
        atol=0,
    )


@pytest.mark.parametrize("name", sorted(_CASES))
def test_real_checkpoint_predictions_are_batch_invariant(name: str) -> None:
    """The ``bn4`` regression, measured on the checkpoint that exposed it."""
    loader, size = _CASES[name]
    weights = _MODELS_DIR / name / "weights.pth"
    if not weights.is_file():
        pytest.skip(f"{name}: weights.pth not present")

    net = loader.from_trex_checkpoint(weights)
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, (6, size, size, 1), dtype=np.uint8)

    batched = net.predict(images)
    individually = np.concatenate([net.predict(images[i : i + 1]) for i in range(6)])

    np.testing.assert_allclose(individually, batched, atol=1e-5)
