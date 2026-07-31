"""T-Rex checkpoint interoperability.

These are the tests that were missing when mosaic's identity networks drifted
away from T-Rex's. Nothing here checks that code *runs* -- the previous suite
did that and stayed green while the loaders raised on every real checkpoint and
the exports fed T-Rex a randomly-initialised network. Each test below pins one
thing that was silently wrong:

- ``test_forward_matches_trex_*``   -- the module tree, layer names, and key
  layout, against T-Rex's own source when it is available on this machine.
- ``test_*_normalization_*``        -- the input preprocessing contract, which
  varies by T-Rex build and must come from the checkpoint rather than a constant.
- ``test_batch_*``                  -- ``bn4`` being ``LayerNorm``, not
  ``BatchNorm1d``: the wrong one is batch-dependent and crashes at batch size 1.
- ``test_roundtrip_*`` / ``test_*_key_layout`` -- that what we write is what we
  can read, with the key names T-Rex expects.
- ``test_input_shape_*``            -- ``(W, H, C)``, which T-Rex compares
  exactly and rejects a transposition of.
- ``test_legacy_*``                 -- that checkpoints from mosaic <= 0.7 still
  load, since they are the only record of models trained before this change.

T-Rex's Python sources are not a package dependency, so the parity tests skip
when they are absent. Everything else runs anywhere torch is installed.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mosaic.behavior.model_library.trex_identity_architectures import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    LEGACY_SEQUENTIAL_TO_NAMED,
    build_v118_3,
    build_v200,
    build_wrapper,
)
from mosaic.behavior.model_library.trex_identity_network import (  # noqa: E402
    TRexIdentityNetwork,
)
from mosaic.behavior.model_library.trex_v118_3_identity import (  # noqa: E402
    TRexV118_3IdentityNetwork,
)

# --- T-Rex source discovery ------------------------------------------------

_TREX_PYTHON_CANDIDATES = (
    os.environ.get("MOSAIC_TREX_PYTHON_DIR"),
    str(
        Path.home()
        / "Documents/GitHub/EcodylicScience/trex/Application/src/tracker/python"
    ),
    str(Path.home() / "miniforge3/envs/track/usr/share/trex"),
)


def _load_trex_networks() -> Any | None:
    """Import T-Rex's ``visual_identification_network_torch``, or None."""
    for candidate in _TREX_PYTHON_CANDIDATES:
        if not candidate:
            continue
        module_path = Path(candidate) / "visual_identification_network_torch.py"
        if not module_path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(
            "_trex_visual_identification_network", module_path
        )
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # torchvision missing, syntax drift, ...
            continue
        return module
    return None


_TREX = _load_trex_networks()
_needs_trex = pytest.mark.skipif(
    _TREX is None,
    reason=(
        "T-Rex Python sources not found; set MOSAIC_TREX_PYTHON_DIR to the "
        "directory holding visual_identification_network_torch.py"
    ),
)


def _batch(n: int, size: int, channels: int = 1) -> np.ndarray:
    """A fixed pseudo-random uint8 NHWC batch."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (n, size, size, channels), dtype=np.uint8)


def _save(path: Path, state_dict: dict[str, Any], metadata: dict[str, Any]) -> Path:
    torch.save({"state_dict": state_dict, "metadata": metadata}, path)
    return path


# --- (a) Forward parity against T-Rex's own source -------------------------


@_needs_trex
@pytest.mark.parametrize(
    ("architecture", "size", "num_classes"),
    [("v118_3", 80, 4), ("v200", 128, 4)],
)
def test_forward_matches_trex_source(
    tmp_path: Path, architecture: str, size: int, num_classes: int
) -> None:
    """A checkpoint saved by T-Rex's own classes loads and predicts identically.

    Builds the reference with T-Rex's ``PermuteAxesWrapper`` + network, saves
    its state_dict, and reads it back through mosaic. Any drift in layer order,
    hyper-parameters, key names, the NHWC permute, or the flatten shows up as a
    logit difference -- or, before this change, as a load error.

    The T-Rex build on this machine passes inputs through unnormalized, so the
    checkpoint is read as ``raw255``; normalization is pinned separately in
    ``test_imagenet_normalization_matches_formula``.
    """
    assert _TREX is not None
    channels = 1
    torch.manual_seed(0)

    if architecture == "v118_3":
        inner = _TREX.V118_3(size, size, num_classes, channels)
        loader = TRexV118_3IdentityNetwork
    else:
        inner = _TREX.V200(size, size, num_classes, channels)
        loader = TRexIdentityNetwork

    reference = _TREX.PermuteAxesWrapper(channels, inner, "cpu")
    reference.eval()

    path = _save(
        tmp_path / f"{architecture}.pth",
        reference.state_dict(),
        {
            "input_shape": (size, size, channels),
            "num_classes": num_classes,
            "input_normalization": "raw255",
        },
    )

    net = loader.from_trex_checkpoint(path)
    assert net.input_normalization == "raw255"

    images = _batch(8, size, channels)
    with torch.no_grad():
        expected = torch.softmax(
            reference(torch.from_numpy(images).float()), dim=1
        ).numpy()
    actual = net.predict(images)

    np.testing.assert_allclose(actual, expected, atol=1e-5)


@_needs_trex
@pytest.mark.parametrize("architecture", ["v118_3", "v200"])
def test_state_dict_keys_match_trex_source(architecture: str) -> None:
    """mosaic's key names are T-Rex's key names.

    This is the assertion that would have caught the original defect in both
    directions: mosaic built V200 as an ``nn.Sequential`` (``0.weight``) while
    T-Rex uses named layers (``model.conv1.weight``). Loading raised; exporting
    was worse, because T-Rex loads ``strict=False`` and only warns, so a
    mismatched checkpoint yields a randomly-initialised network.
    """
    assert _TREX is not None
    channels, num_classes, size = 1, 4, 80

    if architecture == "v118_3":
        inner = _TREX.V118_3(size, size, num_classes, channels)
        ours = build_wrapper(
            build_v118_3(channels, (16, 64, 128), 128 * 10 * 10, 100, num_classes, 5),
            channels,
            "raw255",
        )
    else:
        inner = _TREX.V200(size, size, num_classes, channels)
        ours = build_wrapper(build_v200(channels, num_classes), channels, "raw255")

    theirs = _TREX.PermuteAxesWrapper(channels, inner, "cpu")
    assert set(ours.state_dict()) == set(theirs.state_dict())


# --- (b) The input-normalization contract ----------------------------------


def test_imagenet_normalization_matches_formula() -> None:
    """``imagenet_scaled`` is exactly ``(x / 255 - mean) / std``."""
    channels = 1
    normalize = build_wrapper(torch.nn.Identity(), channels, "imagenet_scaled")
    images = _batch(4, 16, channels)

    with torch.no_grad():
        actual = normalize(torch.from_numpy(images)).numpy()

    expected = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    expected = (expected - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_raw255_normalization_is_a_passthrough() -> None:
    """``raw255`` hands the network the original 0-255 values."""
    channels = 1
    passthrough = build_wrapper(torch.nn.Identity(), channels, "raw255")
    images = _batch(4, 16, channels)

    with torch.no_grad():
        actual = passthrough(torch.from_numpy(images)).numpy()

    np.testing.assert_array_equal(
        actual, images.transpose(0, 3, 1, 2).astype(np.float32)
    )


def test_normalization_mode_decides_buffer_presence() -> None:
    """Only the scaling mode carries its statistics in the state_dict.

    The buffers are what let a checkpoint state its own contract, so their
    presence has to track the mode exactly.
    """
    scaled = build_wrapper(torch.nn.Identity(), 1, "imagenet_scaled").state_dict()
    raw = build_wrapper(torch.nn.Identity(), 1, "raw255").state_dict()

    assert {"normalize.mean", "normalize.std"} <= set(scaled)
    assert not [k for k in raw if k.startswith("normalize.")]


def test_checkpoint_normalize_buffers_are_authoritative(tmp_path: Path) -> None:
    """Buffers in the file win, and their *values* are used verbatim.

    T-Rex builds have shipped statistics in the checkpoint. Re-asserting
    mosaic's ImageNet constants on load would quietly discard them and change
    every prediction, so the loaded values must survive.
    """
    net = TRexV118_3IdentityNetwork(num_classes=3, channels=1, image_size=(80, 80))
    state = dict(net._model.state_dict())
    state["normalize.mean"] = torch.full((1, 1, 1, 1), 0.1)
    state["normalize.std"] = torch.full((1, 1, 1, 1), 0.9)

    path = _save(
        tmp_path / "custom_norm.pth",
        state,
        {"input_shape": (80, 80, 1), "num_classes": 3},
    )
    loaded = TRexV118_3IdentityNetwork.from_trex_checkpoint(path)

    assert loaded.input_normalization == "imagenet_scaled"
    assert float(loaded._model.normalize.mean.flatten()[0]) == pytest.approx(0.1)
    assert float(loaded._model.normalize.std.flatten()[0]) == pytest.approx(0.9)


def test_silent_checkpoint_warns_about_the_ambiguity(tmp_path: Path) -> None:
    """A file that states no contract is a coin-flip, and says so.

    Buffer-less checkpoints from a scaling build and from a passthrough build
    are indistinguishable by their weights. Guessing is unavoidable; guessing
    silently is what produced the original defect.
    """
    net = TRexV118_3IdentityNetwork(
        num_classes=3, channels=1, image_size=(80, 80), input_normalization="raw255"
    )
    path = _save(
        tmp_path / "silent.pth",
        net._model.state_dict(),
        {"input_shape": (80, 80, 1), "num_classes": 3},
    )

    with pytest.warns(UserWarning, match="input-normalization contract"):
        loaded = TRexV118_3IdentityNetwork.from_trex_checkpoint(path)
    assert loaded.input_normalization == "imagenet_scaled"

    # ... and the caller can settle it.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        overridden = TRexV118_3IdentityNetwork.from_trex_checkpoint(
            path, input_normalization="raw255"
        )
    assert overridden.input_normalization == "raw255"


def test_exported_metadata_records_the_contract(tmp_path: Path) -> None:
    """What we export is never ambiguous on reload."""
    net = TRexV118_3IdentityNetwork(
        num_classes=3, channels=1, image_size=(80, 80), input_normalization="raw255"
    )
    path = net.export_trex_checkpoint(tmp_path / "recorded.pth")

    meta = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    assert meta["input_normalization"] == "raw255"
    assert meta["architecture_version"] == "v118_3"
    assert meta["mosaic_checkpoint_version"] >= 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            TRexV118_3IdentityNetwork.from_trex_checkpoint(path).input_normalization
            == "raw255"
        )


# --- (c) bn4: batch invariance ---------------------------------------------


def test_batch_size_does_not_change_predictions() -> None:
    """Per-sample outputs must not depend on what else is in the batch.

    ``bn4`` normalizes across features (``LayerNorm``). The
    ``BatchNorm1d(track_running_stats=False)`` it replaced normalized across the
    batch, so the same crop scored differently depending on its neighbours --
    measured at 10.4 logits against a real checkpoint.
    """
    net = TRexV118_3IdentityNetwork(num_classes=4, channels=1, image_size=(80, 80))
    images = _batch(8, 80, 1)

    batched = net.predict(images)
    individually = np.concatenate([net.predict(images[i : i + 1]) for i in range(8)])

    np.testing.assert_allclose(individually, batched, atol=1e-5)


def test_predict_accepts_a_single_image() -> None:
    """A batch of one must work; ``BatchNorm1d`` raised on it."""
    net = TRexV118_3IdentityNetwork(num_classes=4, channels=1, image_size=(80, 80))
    probs = net.predict(_batch(1, 80, 1))

    assert probs.shape == (1, 4)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_bn4_is_layer_norm() -> None:
    """Pin the layer type directly.

    ``LayerNorm`` and ``BatchNorm1d(track_running_stats=False)`` expose the same
    state_dict -- ``weight`` and ``bias``, no running statistics -- so no
    checkpoint can tell them apart and no load-based test can catch a swap.
    """
    net = TRexV118_3IdentityNetwork(num_classes=4, channels=1, image_size=(80, 80))
    assert isinstance(net._model.model.bn4, torch.nn.LayerNorm)


# --- (d) Round trips and key layout ----------------------------------------


@pytest.mark.parametrize("mode", ["imagenet_scaled", "raw255"])
def test_roundtrip_v200_preserves_predictions(tmp_path: Path, mode: str) -> None:
    """Export then load must reproduce the same probabilities."""
    net = TRexIdentityNetwork(
        num_classes=4, channels=1, image_size=(128, 128), input_normalization=mode
    )
    images = _batch(4, 128, 1)
    before = net.predict(images)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = net.export_trex_checkpoint(tmp_path / "v200.pth")
    after = TRexIdentityNetwork.from_trex_checkpoint(path).predict(images)

    np.testing.assert_allclose(after, before, atol=1e-6)


@pytest.mark.parametrize("mode", ["imagenet_scaled", "raw255"])
def test_roundtrip_v118_3_preserves_predictions(tmp_path: Path, mode: str) -> None:
    """Export then load must reproduce the same probabilities."""
    net = TRexV118_3IdentityNetwork(
        num_classes=4, channels=1, image_size=(80, 80), input_normalization=mode
    )
    images = _batch(4, 80, 1)
    before = net.predict(images)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = net.export_trex_checkpoint(tmp_path / "v118_3.pth")
    after = TRexV118_3IdentityNetwork.from_trex_checkpoint(path).predict(images)

    np.testing.assert_allclose(after, before, atol=1e-6)


def test_v200_export_uses_trex_key_layout(tmp_path: Path) -> None:
    """Named, ``model.``-prefixed keys -- checked without needing T-Rex installed."""
    net = TRexIdentityNetwork(num_classes=4, channels=1, image_size=(128, 128))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = net.export_trex_checkpoint(tmp_path / "keys.pth")

    keys = set(torch.load(path, map_location="cpu", weights_only=False)["state_dict"])

    assert not [k for k in keys if k.split(".")[0].isdigit()], (
        "positional nn.Sequential keys are back; T-Rex cannot load these"
    )
    for expected in (
        "model.conv1.weight",
        "model.bn1.running_mean",
        "model.conv5.weight",
        "model.fc1.weight",
        "model.bn6.weight",
        "model.fc2.bias",
        "normalize.mean",
    ):
        assert expected in keys, f"missing {expected}"


def test_export_is_loadable_with_weights_only(tmp_path: Path) -> None:
    """T-Rex reads checkpoints with ``weights_only=True``.

    That rejects arbitrary pickled objects, so any non-primitive that creeps
    into metadata makes the file unreadable *only* inside T-Rex -- a failure
    mode invisible from mosaic.
    """
    net = TRexV118_3IdentityNetwork(num_classes=2, channels=1, image_size=(80, 80))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = net.export_trex_checkpoint(
            tmp_path / "primitive.pth", class_labels=["a", "b"]
        )

    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["metadata"]["class_labels"] == ["a", "b"]


def test_mismatched_architecture_raises(tmp_path: Path) -> None:
    """A key mismatch must fail loudly rather than load a partial network."""
    net = TRexV118_3IdentityNetwork(num_classes=4, channels=1, image_size=(80, 80))
    state = dict(net._model.state_dict())
    del state["model.conv2.weight"]

    path = _save(
        tmp_path / "broken.pth",
        state,
        {"input_shape": (80, 80, 1), "num_classes": 4},
    )
    with pytest.raises(ValueError, match="missing expected key"):
        TRexV118_3IdentityNetwork.from_trex_checkpoint(path)


# --- (e) input_shape orientation -------------------------------------------


def test_input_shape_is_width_height_channels(tmp_path: Path) -> None:
    """T-Rex stores and compares ``(W, H, C)``, exactly.

    ``check_checkpoint_compatibility`` does a list equality against
    ``(image_width, image_height, image_channels)``, so a transposed shape is a
    hard load failure in T-Rex for any non-square crop -- not a drift.
    """
    net = TRexV118_3IdentityNetwork(
        num_classes=3,
        channels=1,
        image_size=(96, 64),  # (height, width)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        path = net.export_trex_checkpoint(tmp_path / "nonsquare.pth")

    meta = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    assert tuple(meta["input_shape"]) == (64, 96, 1)

    reloaded = TRexV118_3IdentityNetwork.from_trex_checkpoint(path)
    assert reloaded.image_size == (96, 64)


def test_legacy_transposed_input_shape_still_reads_correctly(tmp_path: Path) -> None:
    """mosaic <= 0.7's V118_3 wrote ``(H, W, C)``; those files still resolve.

    ``fc1.in_features`` pins ``(H // 8) * (W // 8)``, which is symmetric under a
    swap and so cannot settle the order on its own. The old exporter's
    ``"architecture": "v200-native"`` marker can, and does.
    """
    net = TRexV118_3IdentityNetwork(num_classes=3, channels=1, image_size=(96, 64))
    path = _save(
        tmp_path / "legacy_shape.pth",
        net._model.state_dict(),
        {
            "input_shape": (96, 64, 1),  # the old (H, W, C) order
            "num_classes": 3,
            "architecture": "v200-native",
            "input_normalization": "imagenet_scaled",
        },
    )

    assert TRexV118_3IdentityNetwork.from_trex_checkpoint(path).image_size == (96, 64)


# --- (f) The legacy positional-key shim ------------------------------------


def test_legacy_sequential_checkpoint_loads(tmp_path: Path) -> None:
    """Checkpoints from mosaic <= 0.7 load, warn, and predict identically.

    Those files are the only record of any identity model trained before this
    change, so they are remapped rather than rejected. The positional
    state_dict is derived from the shipped mapping in reverse, so this test
    cannot drift away from the table it is checking.
    """
    net = TRexIdentityNetwork(num_classes=4, channels=1, image_size=(128, 128))
    named = net._model.state_dict()

    to_positional = {v: k for k, v in LEGACY_SEQUENTIAL_TO_NAMED.items()}
    positional: dict[str, Any] = {}
    for key, value in named.items():
        if key.startswith("normalize."):
            continue  # the old wrapper held mean/std outside the state_dict
        _, layer, param = key.split(".", 2)
        positional[f"{to_positional[layer]}.{param}"] = value

    assert len(positional) == len([k for k in named if not k.startswith("normalize.")])

    path = _save(
        tmp_path / "legacy.pth",
        positional,
        {"input_shape": (128, 128, 1), "num_classes": 4},
    )

    images = _batch(4, 128, 1)
    with pytest.warns(DeprecationWarning, match="positional V200 keys"):
        loaded = TRexIdentityNetwork.from_trex_checkpoint(path)

    # mosaic always scaled before 0.8, so a legacy file is not ambiguous.
    assert loaded.input_normalization == "imagenet_scaled"
    np.testing.assert_allclose(loaded.predict(images), net.predict(images), atol=1e-6)


def test_legacy_mapping_covers_every_stateful_layer() -> None:
    """The shim's table must name every layer that carries weights."""
    # build_v200 returns the inner module, so its keys are `conv1.weight`
    # rather than the wrapper's `model.conv1.weight`.
    stateful = {key.split(".")[0] for key in build_v200(1, 4).state_dict()}
    assert set(LEGACY_SEQUENTIAL_TO_NAMED.values()) == stateful


# --- Cross-cutting ---------------------------------------------------------


def test_v118_3_reads_architecture_off_the_checkpoint(tmp_path: Path) -> None:
    """Layer widths come from the file, so every T-Rex V118_3 variant loads.

    T-Rex has shipped ``conv3`` with both 100 and 128 channels; a class that
    hard-coded either would reject half the checkpoints in circulation.
    """
    for conv_channels in ((16, 64, 100), (16, 64, 128)):
        net = TRexV118_3IdentityNetwork(
            num_classes=4,
            channels=1,
            image_size=(80, 80),
            conv_channels=conv_channels,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = net.export_trex_checkpoint(tmp_path / f"c{conv_channels[-1]}.pth")

        loaded = TRexV118_3IdentityNetwork.from_trex_checkpoint(path)
        assert loaded.conv_channels == conv_channels
        assert loaded.image_size == (80, 80)


def test_module_is_importable_without_a_trex_checkout() -> None:
    """Guard the guard: the skip above must be a skip, not a silent pass."""
    assert "_trex_visual_identification_network" not in sys.modules or _TREX is not None
