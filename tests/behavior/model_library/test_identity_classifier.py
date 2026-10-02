"""``ClassifierIdentityNetwork``: assembly, checkpoint contract, round trip.

Two tiers, because the backbone is the expensive part and most of what can go
wrong does not need it. The first tier runs without ``torch`` or ``timm`` and
covers the checkpoint's *contract*: which keys a frozen export keeps, what
metadata it must carry for a reload to reproduce the fit, and that a stale
format version is refused rather than loaded. The second constructs a real
small backbone and trains on synthetic crops.

The frozen-backbone case is the one worth pinning hardest. It saves the head
alone and refetches the backbone by name, so a reload that quietly tolerated
missing weights would produce a network with a random backbone -- which predicts
happily and is wrong, the exact failure mode a permissive ``strict=False``
invites.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mosaic.behavior.model_library.identity_classifier import (
    CHECKPOINT_FORMAT_VERSION,
    ClassifierIdentityNetwork,
)
from mosaic.behavior.model_library.timm_backbone import data_config_from_metadata

# Selected by CI's `identity` job with `-m identity` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.identity

# A tiny timm model, so the torch-backed tests stay seconds rather than minutes.
_SMALL_BACKBONE = "resnet18"
_SMALL_SIZE = (64, 64)


# --- The checkpoint contract, without torch -------------------------------


def test_metadata_round_trips_as_a_data_config() -> None:
    """Exported metadata must be readable back as a preprocessing recipe.

    ``export_checkpoint`` writes ``image_size`` / ``mean`` / ``std`` as flat
    primitives and ``from_checkpoint`` hands the whole metadata mapping to
    ``data_config_from_metadata``. If the two ever spell a key differently the
    reload silently falls back to the backbone's *current* declared config
    rather than the one the fit used, so pin the pairing directly.
    """
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": "timm/whatever",
        "image_size": [96, 128],
        "mean": [0.1, 0.2, 0.3],
        "std": [0.4, 0.5, 0.6],
        "num_classes": 3,
    }
    resolved = data_config_from_metadata(metadata)
    assert resolved is not None
    assert resolved.image_size == (96, 128)
    assert resolved.mean == (0.1, 0.2, 0.3)
    assert resolved.std == (0.4, 0.5, 0.6)


def test_a_stale_format_version_is_refused(tmp_path: Path) -> None:
    """A checkpoint from another layout must raise, not load partially."""
    torch = pytest.importorskip("torch")

    path = tmp_path / "stale.pth"
    torch.save(
        {
            "state_dict": {},
            "metadata": {
                "format_version": CHECKPOINT_FORMAT_VERSION + 1,
                "model_name": _SMALL_BACKBONE,
                "num_classes": 2,
            },
        },
        path,
    )
    with pytest.raises(ValueError, match="format v"):
        _ = ClassifierIdentityNetwork.from_checkpoint(path)


# --- The real backbone ----------------------------------------------------


@pytest.fixture
def trained() -> ClassifierIdentityNetwork:
    """A two-class network trained for two epochs on synthetic crops."""
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    rng = np.random.default_rng(0)
    # Two visually separable classes: dark crops and bright ones. Enough signal
    # that two epochs move the head, which is all these tests need.
    dark = rng.integers(0, 80, size=(8, *_SMALL_SIZE, 3), dtype=np.uint8)
    bright = rng.integers(175, 256, size=(8, *_SMALL_SIZE, 3), dtype=np.uint8)
    images = np.concatenate([dark, bright], axis=0)
    labels = np.array([0] * 8 + [1] * 8, dtype=np.int64)

    net = ClassifierIdentityNetwork(
        num_classes=2,
        model_name=_SMALL_BACKBONE,
        image_size=_SMALL_SIZE,
        device="cpu",
    )
    _ = net.fit(images, labels, epochs=2, batch_size=4)
    return net


@pytest.mark.slow
def test_fit_reports_every_history_key(trained: ClassifierIdentityNetwork) -> None:
    """The history keys are a cross-model contract, not this model's choice.

    All three identity features write the same ``training_history.joblib``, so a
    renamed key here breaks anything reading another model's history.
    """
    rng = np.random.default_rng(1)
    images = rng.integers(0, 256, size=(6, *_SMALL_SIZE, 3), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    history = trained.fit(images, labels, epochs=2, batch_size=3)

    assert set(history) == {"train_loss", "train_acc", "val_loss", "val_acc"}
    assert all(len(v) == 2 for v in history.values())


@pytest.mark.slow
def test_two_identical_runs_produce_identical_results() -> None:
    """Identical params and inputs must give identical outputs.

    Three things would break this and each is seeded separately: the shuffle,
    dropout, and the head's starting weights. The last is the easy one to lose --
    ``nn.Linear`` initializes from torch's *global* RNG, so a head built the
    stock way starts somewhere different on every run, and the whole fit lands
    somewhere different with it. Nothing downstream would notice: the run_id
    would match, the cache would hit, and the numbers would have moved.
    """
    pytest.importorskip("torch")
    pytest.importorskip("timm")

    rng = np.random.default_rng(7)
    images = rng.integers(0, 256, size=(6, *_SMALL_SIZE, 3), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    probe = rng.integers(0, 256, size=(3, *_SMALL_SIZE, 3), dtype=np.uint8)

    def run() -> tuple[dict[str, list[float]], np.ndarray]:
        net = ClassifierIdentityNetwork(
            num_classes=2,
            model_name=_SMALL_BACKBONE,
            image_size=_SMALL_SIZE,
            device="cpu",
        )
        history = net.fit(images, labels, epochs=2, batch_size=3)
        return history, net.predict(probe)

    first_history, first_probs = run()
    second_history, second_probs = run()

    assert first_history == second_history
    np.testing.assert_array_equal(first_probs, second_probs)


@pytest.mark.slow
def test_predict_returns_a_probability_per_identity(
    trained: ClassifierIdentityNetwork,
) -> None:
    """``(N, num_classes)`` rows summing to 1 -- the shape every model shares."""
    rng = np.random.default_rng(2)
    images = rng.integers(0, 256, size=(5, *_SMALL_SIZE, 3), dtype=np.uint8)

    probs = trained.predict(images)

    assert probs.shape == (5, 2)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(5), rtol=1e-5)


@pytest.mark.slow
def test_grayscale_crops_are_accepted(trained: ClassifierIdentityNetwork) -> None:
    """A 1-channel crop must reach a 3-channel backbone, not raise."""
    rng = np.random.default_rng(3)
    images = rng.integers(0, 256, size=(4, *_SMALL_SIZE, 1), dtype=np.uint8)

    assert trained.predict(images).shape == (4, 2)


@pytest.mark.slow
def test_a_frozen_export_keeps_only_the_head(
    trained: ClassifierIdentityNetwork, tmp_path: Path
) -> None:
    """The point of freezing: the backbone is refetched, never written."""
    torch = pytest.importorskip("torch")

    path = trained.export_checkpoint(tmp_path / "net.pth", class_labels=["a", "b"])
    stored = torch.load(path, map_location="cpu", weights_only=False)

    assert set(stored["state_dict"]) == {"head.weight", "head.bias"}
    assert stored["metadata"]["class_labels"] == ["a", "b"]
    assert stored["metadata"]["freeze_backbone"] is True


@pytest.mark.slow
def test_reload_reproduces_predictions(
    trained: ClassifierIdentityNetwork, tmp_path: Path
) -> None:
    """A round trip must be numerically identical, not merely close in shape.

    This is what a partially-loaded network fails: it returns the right shape
    from a random backbone.
    """
    rng = np.random.default_rng(4)
    images = rng.integers(0, 256, size=(5, *_SMALL_SIZE, 3), dtype=np.uint8)

    before = trained.predict(images)
    path = trained.export_checkpoint(tmp_path / "net.pth", class_labels=["a", "b"])
    reloaded = ClassifierIdentityNetwork.from_checkpoint(path)
    after = reloaded.predict(images)

    np.testing.assert_allclose(before, after, atol=1e-5)
    assert reloaded.num_classes == 2
    assert reloaded.image_size == _SMALL_SIZE


@pytest.mark.slow
def test_a_checkpoint_missing_head_weights_is_refused(tmp_path: Path) -> None:
    """Missing ``head.*`` must raise, where missing ``backbone.*`` is normal.

    ``from_checkpoint`` loads with ``strict=False`` so a frozen file's absent
    backbone keys are tolerated, then judges the result strictly. Without that
    second step every malformed file would load into a random head.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")

    torch.save(
        {
            "state_dict": {},
            "metadata": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_name": _SMALL_BACKBONE,
                "image_size": list(_SMALL_SIZE),
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "num_classes": 2,
                "freeze_backbone": True,
            },
        },
        tmp_path / "headless.pth",
    )
    with pytest.raises(ValueError, match="head"):
        _ = ClassifierIdentityNetwork.from_checkpoint(tmp_path / "headless.pth")
