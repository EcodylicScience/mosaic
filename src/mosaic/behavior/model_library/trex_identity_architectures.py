"""Shared building blocks for the T-Rex-compatible identity networks.

Both :mod:`trex_identity_network` (V200) and :mod:`trex_v118_3_identity`
(V118_3) mirror architectures that live in T-Rex's own Python sources
(``Application/src/tracker/python/visual_identification_network_torch.py``).
This module holds everything they share so the two stay diffable against
upstream: the module tree, the input-normalization contract, and the
checkpoint metadata helpers.

**The module tree is the contract.** T-Rex saves a wrapper whose children are
named ``normalize`` and ``model``, so a real checkpoint's keys look like::

    normalize.mean          normalize.std
    model.conv1.weight      model.bn1.weight       …

Mirroring that tree — rather than reproducing the layer *sequence* in some
other container — is what makes the state_dict keys line up in both
directions. Anything that flattens it (an ``nn.Sequential``, a hand-applied
``"model."`` string prefix) reintroduces the mismatch.

**Input normalization is per-checkpoint, not a constant.** T-Rex's
``Normalize.forward`` has changed across builds:

- builds that scale and standardize: ``(x / 255 - mean) / std``, with ImageNet
  statistics truncated to the channel count. Some of these register ``mean`` /
  ``std`` as buffers, so the checkpoint states its own contract.
- builds that pass through: ``return x``, i.e. the network sees raw ``[0, 255]``.

A checkpoint carrying ``normalize.*`` buffers answers the question itself. One
that does not is genuinely ambiguous, so the mode is an explicit parameter that
is recorded in exported metadata rather than an assumption baked into a wrapper.

All ``nn.Module`` subclasses are defined *inside* factory functions: this module
is imported eagerly by ``mosaic.behavior`` and must stay importable without
PyTorch installed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal, cast

InputNormalization = Literal["imagenet_scaled", "raw255"]

#: What mosaic assumes when a checkpoint does not state its own contract.
#: See :func:`detect_input_normalization` for why this, and not ``"raw255"``.
DEFAULT_INPUT_NORMALIZATION: InputNormalization = "imagenet_scaled"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Maps the stateful positions of mosaic's pre-0.8 ``nn.Sequential`` V200 onto
#: the named layers it is being replaced by. Verified against the shipped
#: architecture, not derived by eye — the stateless ReLU/pool/dropout entries
#: occupy positions too, so the numbering is not contiguous.
LEGACY_SEQUENTIAL_TO_NAMED: dict[str, str] = {
    "0": "conv1",
    "1": "bn1",
    "3": "conv2",
    "4": "bn2",
    "8": "conv3",
    "9": "bn3",
    "11": "conv4",
    "12": "bn4",
    "16": "conv5",
    "17": "bn5",
    "23": "fc1",
    "24": "bn6",
    "27": "fc2",
}


def import_torch() -> Any:
    """Lazily import torch with a helpful error message."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for the T-Rex identity networks. "
            "Install it with: pip install 'mosaic-behavior[identity]'\n"
            "See https://pytorch.org/get-started/locally/ for platform-specific "
            "instructions."
        ) from None
    return torch


# --- Module tree ----------------------------------------------------------


def build_normalize(channels: int, mode: InputNormalization) -> Any:
    """Build the input-normalization submodule.

    Args:
        channels: Number of input channels; ImageNet statistics are truncated
            to this many, matching T-Rex's ``[:channels]`` slicing.
        mode: ``"imagenet_scaled"`` computes ``(x / 255 - mean) / std`` and
            registers ``mean`` / ``std`` as persistent buffers, so the exported
            checkpoint carries its own contract. ``"raw255"`` is a passthrough,
            matching T-Rex builds that feed the network raw ``[0, 255]``.

    Returns:
        An ``nn.Module`` whose ``state_dict`` keys are ``mean`` / ``std`` under
        ``"imagenet_scaled"`` and empty under ``"raw255"``.
    """
    torch = import_torch()
    nn = torch.nn

    class Normalize(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mode = mode
            if mode == "imagenet_scaled":
                mean = torch.tensor(IMAGENET_MEAN[:channels]).reshape(1, -1, 1, 1)
                std = torch.tensor(IMAGENET_STD[:channels]).reshape(1, -1, 1, 1)
                self.register_buffer("mean", mean)
                self.register_buffer("std", std)

        def forward(self, x: Any) -> Any:
            if self.mode == "raw255":
                return x
            return (x / 255.0 - self.mean) / self.std

    return Normalize()


def build_wrapper(model: Any, channels: int, mode: InputNormalization) -> Any:
    """Wrap a network so it accepts ``(N, H, W, C)`` uint8 input.

    Mirrors T-Rex's ``PermuteAxesWrapper``: permute to NCHW, normalize, forward.
    The child names (``normalize``, ``model``) are the reason a mosaic
    state_dict and a T-Rex state_dict share key names.

    Args:
        model: The inner network (from :func:`build_v200` or
            :func:`build_v118_3`).
        channels: Number of input channels.
        mode: Input-normalization contract, see :func:`build_normalize`.

    Returns:
        An ``nn.Module`` taking ``(N, H, W, C)`` uint8 and returning logits.
    """
    torch = import_torch()
    nn = torch.nn

    class TRexWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalize = build_normalize(channels, mode)
            self.model = model

        def forward(self, x: Any) -> Any:
            # (N, H, W, C) -> (N, C, H, W). `.float()` is mosaic's addition:
            # T-Rex's own caller has already cast to float32, ours passes uint8.
            x = x.permute(0, 3, 1, 2).float()
            x = x.contiguous(memory_format=torch.channels_last)
            x = self.normalize(x)
            return self.model(x)

    return TRexWrapper()


def build_v200(channels: int, num_classes: int) -> Any:
    """Build T-Rex's ``V200``: five conv blocks, global average pool, 2-layer head.

    Layer *names* are copied verbatim from T-Rex's ``V200`` so the state_dict
    keys match. Stateless layers (``relu*``) are kept as named attributes even
    though they contribute no keys — being able to diff this class line-for-line
    against upstream is the property whose absence let the keys drift.

    Args:
        channels: Input channels (1 = grayscale, 3 = RGB).
        num_classes: Number of identities.

    Returns:
        An ``nn.Module`` expecting NCHW float input.
    """
    torch = import_torch()
    nn = torch.nn

    class V200(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding="same")
            self.bn1 = nn.BatchNorm2d(64)
            self.relu1 = nn.ReLU()

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding="same")
            self.bn2 = nn.BatchNorm2d(128)
            self.relu2 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=3)
            self.dropout1 = nn.Dropout2d(0.05)

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding="same")
            self.bn3 = nn.BatchNorm2d(256)
            self.relu3 = nn.ReLU()

            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding="same")
            self.bn4 = nn.BatchNorm2d(512)
            self.relu4 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=3)
            self.dropout2 = nn.Dropout2d(0.25)

            self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding="same")
            self.bn5 = nn.BatchNorm2d(512)
            self.relu5 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=3)
            self.dropout3 = nn.Dropout2d(0.05)

            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc1 = nn.Linear(512, 1024)
            self.bn6 = nn.BatchNorm1d(1024)
            self.relu6 = nn.ReLU()
            self.dropout4 = nn.Dropout(0.05)

            self.fc2 = nn.Linear(1024, num_classes)

        def forward(self, x: Any) -> Any:
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.pool1(self.relu2(self.bn2(self.conv2(x))))
            x = self.dropout1(x)

            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.pool2(self.relu4(self.bn4(self.conv4(x))))
            x = self.dropout2(x)

            x = self.relu5(self.bn5(self.conv5(x)))
            x = self.pool3(x)
            x = self.dropout3(x)

            x = self.global_avg_pool(x)
            x = x.reshape(x.size(0), -1)

            x = self.relu6(self.bn6(self.fc1(x)))
            x = self.dropout4(x)
            return self.fc2(x)

    return V200()


def build_v118_3(
    channels: int,
    conv_channels: tuple[int, int, int],
    flatten_dim: int,
    fc_hidden: int,
    num_classes: int,
    kernel_size: int = 5,
) -> Any:
    """Build T-Rex's ``V118_3``: three conv blocks, direct flatten, 2-layer head.

    Layer names are copied verbatim from T-Rex's ``V118_3``. Two details are
    load-bearing and easy to get wrong:

    - ``bn4`` is ``nn.LayerNorm``, **not** ``nn.BatchNorm1d``. The two expose an
      identical state_dict (``weight`` and ``bias``, no running statistics), so
      a checkpoint cannot tell you which it was and the wrong choice loads clean
      while computing different math.
    - There is no global average pool. ``fc1`` consumes the flattened conv
      output, so ``flatten_dim`` is fixed at construction and inputs must be
      resized to the trained image size before inference.

    The channel counts are parameters rather than constants because T-Rex has
    shipped more than one ``V118_3`` (``conv3`` has been both 100 and 128
    channels); :func:`infer_v118_3_dims` reads them off the checkpoint, so one
    class covers every variant.

    Args:
        channels: Input channels.
        conv_channels: ``(C1, C2, C3)`` output channels of the three blocks.
        flatten_dim: ``fc1.in_features``, i.e. ``C3 x (H/8) x (W/8)``.
        fc_hidden: ``fc1`` output width.
        num_classes: Number of identities.
        kernel_size: Conv kernel size, square and shared across blocks.

    Returns:
        An ``nn.Module`` expecting NCHW float input.
    """
    torch = import_torch()
    nn = torch.nn

    class V118_3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3 = conv_channels
            k = kernel_size

            self.conv1 = nn.Conv2d(channels, c1, kernel_size=k, padding="same")
            self.bn1 = nn.BatchNorm2d(c1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)
            self.dropout1 = nn.Dropout2d(0.05)

            self.conv2 = nn.Conv2d(c1, c2, kernel_size=k, padding="same")
            self.bn2 = nn.BatchNorm2d(c2)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.dropout2 = nn.Dropout2d(0.05)

            self.conv3 = nn.Conv2d(c2, c3, kernel_size=k, padding="same")
            self.bn3 = nn.BatchNorm2d(c3)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(kernel_size=2)
            self.dropout3 = nn.Dropout2d(0.05)

            self.fc1 = nn.Linear(flatten_dim, fc_hidden)
            self.bn4 = nn.LayerNorm(fc_hidden)
            self.relu4 = nn.ReLU()
            self.dropout4 = nn.Dropout(0.05)
            self.fc2 = nn.Linear(fc_hidden, num_classes)

        def forward(self, x: Any) -> Any:
            x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
            x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
            x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))

            x = x.reshape(x.size(0), -1)

            x = self.fc1(x)
            x = self.bn4(x.contiguous())
            x = self.relu4(x)
            x = self.dropout4(x)
            return self.fc2(x)

    return V118_3()


# --- Reading architecture off a checkpoint --------------------------------


def _shapes(
    state_dict: dict[str, Any], keys: dict[str, str]
) -> dict[str, tuple[int, ...]]:
    """Fetch the shapes of required keys, or raise naming the missing one."""
    out: dict[str, tuple[int, ...]] = {}
    for key, described in keys.items():
        if key not in state_dict:
            raise ValueError(
                f"checkpoint missing expected key {key!r} {described}; "
                f"present keys (first 8): {list(state_dict)[:8]} ..."
            )
        out[key] = tuple(int(x) for x in state_dict[key].shape)
    return out


def infer_v200_dims(state_dict: dict[str, Any]) -> tuple[int, int]:
    """Read ``(channels, num_classes)`` off a V200 state_dict.

    V200 ends in a global average pool, so the input image size leaves no trace
    in the weights and cannot be recovered here — it has to come from metadata.

    Args:
        state_dict: Keys in the named, ``model.``-prefixed layout.

    Returns:
        ``(channels, num_classes)``.

    Raises:
        ValueError: when a required key is missing or misshapen.
    """
    shapes = _shapes(
        state_dict,
        {
            "model.conv1.weight": "(64, channels, 3, 3)",
            "model.fc2.weight": "(num_classes, 1024)",
        },
    )
    conv1 = shapes["model.conv1.weight"]
    fc2 = shapes["model.fc2.weight"]
    if len(conv1) != 4:
        raise ValueError(f"model.conv1.weight must be 4D, got {conv1}")
    if len(fc2) != 2:
        raise ValueError(f"model.fc2.weight must be 2D, got {fc2}")
    return conv1[1], fc2[0]


def infer_v118_3_dims(
    state_dict: dict[str, Any],
) -> tuple[int, tuple[int, int, int], int, int, int, int, int]:
    """Read the full V118_3 architecture off a state_dict.

    T-Rex has shipped more than one V118_3 (``conv3`` with 100 and with 128
    channels), so the shapes are the only reliable description of a given file.

    Args:
        state_dict: Keys in the named, ``model.``-prefixed layout.

    Returns:
        ``(channels, (C1, C2, C3), flatten_dim, fc_hidden, num_classes,
        spatial_pixels, kernel_size)``, where ``flatten_dim ==
        fc1.in_features == C3 * spatial_pixels`` and ``spatial_pixels`` is the
        post-conv ``H * W`` (not a side length -- the map need not be square).

    Raises:
        ValueError: when keys are missing, misshapen, or mutually inconsistent.
    """
    shapes = _shapes(
        state_dict,
        {
            "model.conv1.weight": "(C1, channels, k, k)",
            "model.conv2.weight": "(C2, C1, k, k)",
            "model.conv3.weight": "(C3, C2, k, k)",
            "model.fc1.weight": "(fc_hidden, flatten_dim)",
            "model.fc2.weight": "(num_classes, fc_hidden)",
        },
    )
    s1, s2, s3 = (shapes[f"model.conv{i}.weight"] for i in (1, 2, 3))
    sf1 = shapes["model.fc1.weight"]
    sf2 = shapes["model.fc2.weight"]

    if not all(len(s) == 4 for s in (s1, s2, s3)):
        raise ValueError(
            f"conv weights must be 4D, got conv1={s1} conv2={s2} conv3={s3}"
        )
    if len(sf1) != 2 or len(sf2) != 2:
        raise ValueError(f"fc weights must be 2D, got fc1={sf1} fc2={sf2}")

    c1_out, channels = s1[0], s1[1]
    c2_out, c2_in = s2[0], s2[1]
    c3_out, c3_in = s3[0], s3[1]
    fc_hidden, flatten_dim = sf1
    num_classes, fc2_in = sf2

    if c2_in != c1_out:
        raise ValueError(f"conv2 in_channels ({c2_in}) != conv1 out ({c1_out})")
    if c3_in != c2_out:
        raise ValueError(f"conv3 in_channels ({c3_in}) != conv2 out ({c2_out})")
    if fc2_in != fc_hidden:
        raise ValueError(f"fc2 in_features ({fc2_in}) != fc1 out ({fc_hidden})")

    if flatten_dim % c3_out != 0:
        raise ValueError(
            f"fc1 in_features ({flatten_dim}) is not divisible by conv3 "
            f"out_channels ({c3_out}); state_dict layout is unexpected"
        )
    spatial_pixels = flatten_dim // c3_out

    kernels = {s[-1] for s in (s1, s2, s3)}
    if len(kernels) != 1:
        raise ValueError(
            f"inconsistent conv kernel sizes: conv1={s1[-1]}, conv2={s2[-1]}, "
            f"conv3={s3[-1]}"
        )

    return (
        channels,
        (c1_out, c2_out, c3_out),
        flatten_dim,
        fc_hidden,
        num_classes,
        spatial_pixels,
        kernels.pop(),
    )


# --- Checkpoint helpers ---------------------------------------------------


def split_checkpoint(ckpt: Any, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a loaded ``.pth`` into ``(state_dict, metadata)``.

    Tolerates the three shapes seen in the wild: a ``{"state_dict", "metadata"}``
    wrapper (what T-Rex and mosaic write), the same wrapper without metadata,
    and a bare state_dict.

    Args:
        ckpt: The object returned by ``torch.load``.
        path: Source path, used only for error messages.

    Returns:
        ``(state_dict, metadata)``; metadata is ``{}`` when absent.

    Raises:
        ValueError: when the object is not a recognisable checkpoint.
    """
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"unrecognised checkpoint format at {path}: expected a "
            f"{{'state_dict': ...}} wrapper or a bare state_dict, found "
            f"{type(ckpt).__name__}"
        )

    loaded = cast(dict[str, Any], ckpt)

    if loaded.get("state_dict") is not None:
        meta = cast("dict[str, Any] | None", loaded.get("metadata")) or {}
        return dict(cast(dict[str, Any], loaded["state_dict"])), dict(meta)

    if any(k.startswith("model.") or k.split(".")[0].isdigit() for k in loaded):
        return dict(loaded), {}

    raise ValueError(
        f"unrecognised checkpoint format at {path}: expected a "
        f"{{'state_dict': ...}} wrapper or a bare state_dict, found keys "
        f"{list(loaded)[:8]}"
    )


def remap_legacy_sequential_keys(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a pre-0.8 mosaic V200 state_dict onto the named layout.

    mosaic <= 0.7 built V200 as an ``nn.Sequential``, so its exported keys were
    positional (``0.weight``) rather than named (``model.conv1.weight``). Such a
    file is still the only record of any model trained before 0.8, so it is
    remapped rather than rejected.

    Args:
        state_dict: Keys in the positional layout.

    Returns:
        Keys in the named layout, ``model.``-prefixed.

    Raises:
        ValueError: when a positional key has no counterpart, which means the
            file did not come from mosaic's V200.
    """
    out: dict[str, Any] = {}
    for key, value in state_dict.items():
        head, _, tail = key.partition(".")
        if not head.isdigit():
            out[key] = value
            continue
        named = LEGACY_SEQUENTIAL_TO_NAMED.get(head)
        if named is None:
            raise ValueError(
                f"positional key {key!r} has no counterpart in mosaic's V200 "
                f"layout; this checkpoint was not written by mosaic <= 0.7"
            )
        out[f"model.{named}.{tail}"] = value
    return out


def is_legacy_sequential(state_dict: dict[str, Any]) -> bool:
    """True when a state_dict uses mosaic's pre-0.8 positional V200 keys."""
    return any(k.split(".")[0].isdigit() for k in state_dict)


def align_normalize_buffers(
    state_dict: dict[str, Any], channels: int
) -> dict[str, Any]:
    """Reshape checkpoint ``normalize.*`` buffers to this model's channel count.

    T-Rex builds have stored these as a scalar ``(1, 1, 1, 1)`` even for
    multi-channel input, relying on broadcasting; mosaic builds them as
    ``(1, C, 1, 1)``. ``load_state_dict`` is strict about shape, so align first.

    Args:
        state_dict: Checkpoint keys, possibly containing ``normalize.mean`` and
            ``normalize.std``.
        channels: The model's channel count.

    Returns:
        A copy with any normalize buffers shaped ``(1, channels, 1, 1)``.

    Raises:
        ValueError: when a buffer is neither per-channel nor scalar.
    """
    out = dict(state_dict)
    for key in ("normalize.mean", "normalize.std"):
        if key not in out:
            continue
        tensor = out[key]
        n = int(tensor.numel())
        if n == channels:
            out[key] = tensor.reshape(1, channels, 1, 1)
        elif n == 1:
            out[key] = tensor.reshape(1, 1, 1, 1).expand(1, channels, 1, 1).clone()
        else:
            raise ValueError(
                f"{key} has {n} elements, which is neither 1 (scalar, "
                f"broadcast) nor {channels} (per-channel); cannot align"
            )
    return out


def load_into_wrapper(
    wrapper: Any, state_dict: dict[str, Any], *, source: str = "checkpoint"
) -> None:
    """Load a state_dict into a wrapper, tolerating only absent normalize buffers.

    A checkpoint from a T-Rex build that computed normalization in Python
    carries no ``normalize.*`` buffers. Those are the one safe omission: the
    wrapper initialises them to the same statistics the build used. Every other
    missing or unexpected key means the architectures disagree, which is the
    failure this whole module exists to make loud rather than silent -- T-Rex
    itself loads ``strict=False`` and merely logs, which is how an entirely
    randomly-initialised network can look like a successful load.

    Args:
        wrapper: The module from :func:`build_wrapper`.
        state_dict: Keys in the named layout.
        source: Name used in the error message.

    Raises:
        ValueError: on any key mismatch beyond the normalize buffers.
    """
    result = wrapper.load_state_dict(state_dict, strict=False)
    tolerated = {"normalize.mean", "normalize.std"}
    missing = [k for k in result.missing_keys if k not in tolerated]
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        raise ValueError(
            f"{source} does not match this architecture: "
            f"missing={missing[:8]}{' ...' if len(missing) > 8 else ''} "
            f"unexpected={unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
        )


def detect_input_normalization(
    state_dict: dict[str, Any],
    metadata: dict[str, Any],
    override: InputNormalization | None = None,
    *,
    source: str = "checkpoint",
) -> InputNormalization:
    """Decide which input-normalization contract a checkpoint was trained under.

    Resolution order:

    1. ``normalize.mean`` / ``normalize.std`` in the state_dict. These are
       authoritative — the file states its own contract, and the *values* are
       taken from the file rather than re-asserted, because a build may ship
       statistics that are not ImageNet's.
    2. An explicit ``override``.
    3. ``metadata["input_normalization"]``, which mosaic writes and T-Rex does not.
    4. :data:`DEFAULT_INPUT_NORMALIZATION`, with a warning.

    Step 4 is a genuine coin-flip: a checkpoint from a scaling build and one
    from a passthrough build are indistinguishable by their weights alone. The
    default is the scaling contract because it is what mosaic has always
    assumed, so existing mosaic exports keep their meaning. The warning is the
    honest half — an unstated assumption here is what this machinery exists to
    prevent.

    Args:
        state_dict: Checkpoint state_dict.
        metadata: Checkpoint metadata, possibly empty.
        override: Caller's explicit choice, used only when the file is silent.
        source: Name used in warnings.

    Returns:
        The resolved mode.
    """
    if "normalize.mean" in state_dict or "normalize.std" in state_dict:
        if override is not None and override != "imagenet_scaled":
            warnings.warn(
                f"{source} carries normalize.* buffers, which state a scaling "
                f"contract; ignoring input_normalization={override!r}",
                UserWarning,
                stacklevel=3,
            )
        return "imagenet_scaled"

    if override is not None:
        return override

    declared = metadata.get("input_normalization")
    if declared in ("imagenet_scaled", "raw255"):
        return declared

    warnings.warn(
        f"{source} does not state its input-normalization contract: it carries "
        f"no normalize.* buffers and no input_normalization metadata. Assuming "
        f"{DEFAULT_INPUT_NORMALIZATION!r} ((x/255 - mean)/std). If this model was "
        f"trained by a T-Rex build whose Normalize is a passthrough, predictions "
        f"will be wrong -- pass input_normalization='raw255' to override.",
        UserWarning,
        stacklevel=3,
    )
    return DEFAULT_INPUT_NORMALIZATION


def pack_input_shape(
    image_size: tuple[int, int], channels: int
) -> tuple[int, int, int]:
    """Render ``(height, width)`` as T-Rex's metadata ``input_shape``.

    T-Rex writes and validates ``(width, height, channels)``
    (``visual_recognition_torch.save_model_files`` and
    ``trex_utils.check_checkpoint_compatibility``). The comparison there is
    exact, so a transposed shape is a hard load failure for any non-square crop.

    Args:
        image_size: ``(height, width)``, mosaic's internal order.
        channels: Number of channels.

    Returns:
        ``(width, height, channels)``.
    """
    h, w = image_size
    return (w, h, channels)


def unpack_input_shape(
    shape: tuple[int, ...],
    channels: int,
    *,
    square_hint: int | None = None,
    trust_orientation: bool = True,
) -> tuple[int, int]:
    """Read a metadata ``input_shape`` back as ``(height, width)``.

    Args:
        shape: The stored tuple.
        channels: The model's channel count, used to locate the channel axis.
        square_hint: Fallback side length when the shape is unusable.
        trust_orientation: True when the file is known to use T-Rex's
            ``(W, H, C)`` order. False for files written by mosaic <= 0.7's
            V118_3 exporter, which stored ``(H, W, C)``.

    Returns:
        ``(height, width)``.
    """
    dims = tuple(int(x) for x in shape)

    if len(dims) == 3:
        a, b, c = dims
        if c != channels:
            warnings.warn(
                f"input_shape {dims} has a trailing {c} that is not the channel "
                f"count ({channels}); reading the first two entries as width, height",
                UserWarning,
                stacklevel=3,
            )
        if a == b:
            return (a, b)
        return (b, a) if trust_orientation else (a, b)

    if len(dims) == 2:
        return (dims[0], dims[1])

    if square_hint is not None:
        return (square_hint, square_hint)

    raise ValueError(f"cannot read image size from input_shape {dims}")


def load_trex_torchscript(path: Path) -> tuple[Any, dict[str, Any]]:
    """Load a T-Rex TorchScript sidecar and its embedded metadata.

    T-Rex writes two files per model: ``<base>_dict.pth`` (weights + metadata)
    and ``<base>_model.pth`` (a TorchScript archive with the metadata JSON in
    its extra-files table). The TorchScript form is fully self-describing —
    it carries the preprocessing *code*, not just the weights — which makes it
    the reference oracle for checkpoint-parity tests.

    Args:
        path: Path to a ``*_model.pth`` TorchScript archive.

    Returns:
        ``(module, metadata)``; metadata is ``{}`` when absent or unparsable.
    """
    torch = import_torch()

    extra: dict[str, Any] = {"metadata": ""}
    module = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)

    raw = extra.get("metadata") or ""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    if not raw:
        return module, {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return module, {}
    if not isinstance(parsed, dict):
        return module, {}
    return module, dict(cast(dict[str, Any], parsed))


def torchscript_sidecar_path(path: Path) -> Path | None:
    """Return the ``*_model.pth`` sibling of a weights file, if it exists.

    Mirrors T-Rex's own fallback lookup: a ``<base>.pth`` or ``<base>_dict.pth``
    weights file is accompanied by ``<base>_model.pth``.

    Args:
        path: Path to the weights ``.pth``.

    Returns:
        The sidecar path, or None when there isn't one.
    """
    name = path.name
    if name.endswith("_model.pth"):
        return path if path.exists() else None
    stem = name[: -len(".pth")] if name.endswith(".pth") else name
    if stem.endswith("_dict"):
        stem = stem[: -len("_dict")]
    candidate = path.with_name(f"{stem}_model.pth")
    return candidate if candidate.exists() else None
