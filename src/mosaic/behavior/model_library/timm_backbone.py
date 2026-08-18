"""Shared timm-backbone plumbing for the identity networks.

Every identity network in this package that stands on a pretrained image
backbone needs the same four things before it can do anything of its own:
resolve a user-typed model name into something ``timm.create_model`` accepts,
discover the preprocessing recipe that backbone was trained under, turn a batch
of crops into a normalized tensor, and pick a device. This module owns all four
so that a network module holds only what makes it that network.

**A backbone's recipe travels with it.** Input size and normalization statistics
are read from the loaded model's own ``pretrained_cfg`` rather than assumed, so
naming a different backbone brings its own recipe rather than inheriting the
caller's. :data:`FALLBACK_DATA_CONFIG` covers a model that declares nothing, and
:func:`data_config_from_metadata` replays a recipe a checkpoint recorded -- which
is what lets a refetched backbone reproduce the numbers a fit produced even if
its upstream repository has since edited its declared config.

Nothing here imports torch or timm at module scope: :func:`import_torch` and
:func:`import_timm` are lazy, so ``mosaic.behavior`` stays importable without the
``deep-learning`` extra installed.

Mosaic distributes no model weights. Whatever ``model_name`` a caller passes is
fetched at run time under its own license, independent of mosaic's own AGPLv3+.
See ``docs/licensing.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeGuard

import numpy as np

DEFAULT_MODEL_NAME: Final = "timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k"
"""The backbone loaded when ``model_name`` is not set.

MIT-licensed, and the same Swin architecture MegaDescriptor fine-tuned, so
naming ``BVRA/MegaDescriptor-L-384`` instead is a weights swap and nothing
else. Changing this line is a licensing decision, not a tuning decision.
"""

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneDataConfig:
    """The input recipe a backbone declares for itself.

    Attributes:
        image_size: ``(height, width)`` the backbone expects.
        mean: Per-channel normalization mean, in ``[0, 1]`` units.
        std: Per-channel normalization standard deviation.
    """

    image_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


FALLBACK_DATA_CONFIG: Final = BackboneDataConfig(
    image_size=(384, 384), mean=IMAGENET_MEAN, std=IMAGENET_STD
)
"""What to use when the backbone declares nothing.

Deliberately identical to what the embedding network hardcoded before
resolution existed, so the no-information path is the previous behavior rather
than a third one.
"""

_TIMM_SOURCE_PREFIXES: Final = ("hf-hub:", "hf_hub:", "local-dir:")


def import_torch() -> Any:
    """Lazily import torch with a helpful error message."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for the identity networks. "
            "Install it with: pip install 'mosaic-behavior[deep-learning]'\n"
            "See https://pytorch.org/get-started/locally/ for platform-specific "
            "instructions."
        ) from None
    return torch


def import_timm() -> Any:
    """Lazily import timm with a helpful error message."""
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for the backbone-based identity networks. "
            "Install it with: pip install 'mosaic-behavior[deep-learning]'"
        ) from None
    return timm


def resolve_timm_model_id(model_name: str) -> str:
    """The string ``timm.create_model`` should be handed for *model_name*.

    Three spellings must all work, because all three are what a user copies:

    * a bare timm architecture tag --
      ``swin_large_patch4_window12_384.ms_in22k_ft_in1k`` -- resolved through
      timm's own registry;
    * a Hugging Face hub id -- ``BVRA/MegaDescriptor-L-384``, ``timm/swin_...``
      -- which timm loads only under an explicit ``hf-hub:`` prefix;
    * either of the above already carrying its prefix.

    A hub id is exactly a name with an owner, so the ``/`` is the whole test.
    ``timm/<tag>`` carries one and is a real hub repository, so it takes the
    hub path and resolves to the same weights the bare tag does.

    Args:
        model_name: Architecture tag, hub id, or already-prefixed source.

    Returns:
        The identifier to hand ``timm.create_model``.
    """
    if model_name.startswith(_TIMM_SOURCE_PREFIXES):
        return model_name
    if "/" in model_name:
        return f"hf-hub:{model_name}"
    return model_name


def _is_object_sequence(value: object) -> TypeGuard[Sequence[object]]:
    """Whether *value* is a sequence whose items may be read as objects.

    Strings and bytes are sequences and would iterate into characters, so they
    are excluded. Every other ``Sequence`` yields items that are at least
    ``object``, which is all any caller here needs.
    """
    return not isinstance(value, (str, bytes)) and isinstance(value, Sequence)


def _is_string_keyed_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    """Whether *value* may be read as a mapping from string keys to objects.

    The key type is asserted rather than checked: every caller reads through
    ``.get()`` with a string literal, which returns the default on a mapping
    keyed some other way rather than failing.
    """
    return isinstance(value, Mapping)


def as_sequence(value: object) -> Sequence[object] | None:
    """*value* as a non-string sequence, or None if it is not one.

    Shared with the networks, which read the same untyped checkpoint
    metadata and need the same narrowing to walk it under strict typing.
    """
    return value if _is_object_sequence(value) else None


def _triple(value: object) -> tuple[float, float, float] | None:
    """The three floats in *value*, or None if it does not hold exactly three."""
    items = as_sequence(value)
    if items is None or len(items) != 3:
        return None
    numbers: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        numbers.append(float(item))
    return (numbers[0], numbers[1], numbers[2])


def _pair_of_ints(value: object) -> tuple[int, int] | None:
    """The two ints in *value*, or None if it does not hold exactly two."""
    items = as_sequence(value)
    if items is None or len(items) != 2:
        return None
    sizes: list[int] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        sizes.append(item)
    return (sizes[0], sizes[1])


def _hw_from_input_size(value: object) -> tuple[int, int] | None:
    """``(height, width)`` from a timm ``input_size``, or None if malformed.

    timm spells it ``(channels, height, width)``, so the leading channel count
    is dropped.
    """
    items = as_sequence(value)
    if items is None or len(items) != 3:
        return None
    return _pair_of_ints(items[1:])


def data_config_from_mapping(raw: Mapping[str, object]) -> BackboneDataConfig:
    """Read a timm data config, falling back per key rather than wholesale.

    A partial ``pretrained_cfg`` is a real shape -- a repository may declare
    ``mean`` and ``std`` and omit ``input_size`` -- so a missing or malformed
    key takes the fallback for that key alone and leaves the rest resolved.

    Args:
        raw: A ``timm.data.resolve_model_data_config`` result, or any mapping
            spelled the same way.

    Returns:
        The resolved config, with :data:`FALLBACK_DATA_CONFIG` supplying every
        key *raw* did not answer.
    """
    image_size = _hw_from_input_size(raw.get("input_size"))
    mean = _triple(raw.get("mean"))
    std = _triple(raw.get("std"))
    return BackboneDataConfig(
        image_size=image_size or FALLBACK_DATA_CONFIG.image_size,
        mean=mean or FALLBACK_DATA_CONFIG.mean,
        std=std or FALLBACK_DATA_CONFIG.std,
    )


def resolve_backbone_data_config(backbone: object) -> BackboneDataConfig:
    """The config *backbone* declares, or the fallback if it declares none.

    ``timm.data.resolve_model_data_config`` reads ``model.pretrained_cfg``,
    which timm's hub loader populates from the repository's ``config.json``. A
    model carrying no such config -- a locally defined architecture, say -- has
    nothing to read, and that check happens before timm is imported, so the
    fallback path costs no import.

    Args:
        backbone: A constructed timm model.

    Returns:
        The backbone's declared input size and normalization statistics.
    """
    declared: object = getattr(backbone, "pretrained_cfg", None)
    if not _is_string_keyed_mapping(declared) or not declared:
        return FALLBACK_DATA_CONFIG
    timm = import_timm()
    resolved: object = timm.data.resolve_model_data_config(backbone)
    if not _is_string_keyed_mapping(resolved):
        return FALLBACK_DATA_CONFIG
    return data_config_from_mapping(resolved)


def data_config_from_metadata(stored: object) -> BackboneDataConfig | None:
    """A checkpoint's recorded recipe, or None if it recorded none usable."""
    if not _is_string_keyed_mapping(stored):
        return None
    image_size = _pair_of_ints(stored.get("image_size"))
    mean = _triple(stored.get("mean"))
    std = _triple(stored.get("std"))
    if image_size is None or mean is None or std is None:
        return None
    return BackboneDataConfig(image_size=image_size, mean=mean, std=std)


def normalization_tensors(config: BackboneDataConfig) -> tuple[Any, Any]:
    """``(mean, std)`` as broadcastable ``(1, 3, 1, 1)`` CPU tensors.

    CPU, not the compute device: :func:`preprocess_batch` runs entirely on CPU
    tensors and the caller moves each batch afterwards. Building these on the
    device means every call pulls them back, a host round-trip per batch.

    Args:
        config: The resolved recipe whose statistics to materialize.

    Returns:
        The mean and standard deviation tensors, in that order.
    """
    torch = import_torch()
    mean = torch.tensor(config.mean, dtype=torch.float32).reshape(1, 3, 1, 1)
    std = torch.tensor(config.std, dtype=torch.float32).reshape(1, 3, 1, 1)
    return mean, std


def preprocess_batch(
    images: np.ndarray,
    image_size: tuple[int, int],
    mean: Any,
    std: Any,
) -> Any:
    """Convert ``(N, H, W, C)`` uint8 crops to a normalized ``(N, 3, H', W')`` tensor.

    A backbone's declared ``interpolation`` and ``crop_pct`` are deliberately
    not honored. Those describe timm's evaluation transform for a *full image*
    being center-cropped; the input here is already a tight egocentric crop, so
    a 0.9 crop ratio would discard the border a discriminative marking may sit
    in. Only the input size and the normalization statistics follow the
    backbone.

    Args:
        images: ``(N, H, W, C)`` uint8 array. Grayscale (C=1) is replicated to
            3 channels, because every backbone here expects 3.
        image_size: Target ``(height, width)``; the batch is bilinear-resized
            to it when it differs.
        mean: Broadcastable mean tensor, from :func:`normalization_tensors`.
        std: Broadcastable standard-deviation tensor, likewise.

    Returns:
        The normalized float32 tensor, on CPU.

    Raises:
        ValueError: If *images* is not 4-dimensional, or carries a channel
            count other than 1 or 3.
    """
    torch = import_torch()

    if images.ndim != 4:
        msg = f"[timm-backbone] expected (N, H, W, C), got {images.shape}"
        raise ValueError(msg)
    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.shape[-1] != 3:
        msg = f"[timm-backbone] expected 1 or 3 channels, got {images.shape[-1]}"
        raise ValueError(msg)

    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0

    target_h, target_w = image_size
    if x.shape[-2] != target_h or x.shape[-1] != target_w:
        x = torch.nn.functional.interpolate(
            x, size=(target_h, target_w), mode="bilinear", align_corners=False
        )

    return (x - mean) / std


def resolve_device(device: str) -> Any:
    """The torch device named by *device*, or the best available under ``"auto"``.

    Args:
        device: ``"auto"``, or any string ``torch.device`` accepts.

    Returns:
        The resolved ``torch.device``. Under ``"auto"``, CUDA is preferred,
        then MPS, then CPU.
    """
    torch = import_torch()
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
