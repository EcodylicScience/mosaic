"""Frozen image-backbone identity recognition.

Loads any timm-resolvable image model as a frozen embedding extractor for
individual animal re-identification. Identity is decided at inference by
cosine-similarity k-NN against per-identity prototype embeddings computed at
fit time; nothing is trained.

``model_name`` accepts a bare timm architecture tag
(``swin_large_patch4_window12_384.ms_in22k_ft_in1k``) or a Hugging Face hub id
(``BVRA/MegaDescriptor-L-384``). Normalization statistics and input size are
read from the loaded backbone's own ``pretrained_cfg``, so naming a different
backbone brings its own recipe with it rather than inheriting this module's
assumptions.

This is a sibling implementation to
:class:`~mosaic.behavior.model_library.trex_identity_network.TRexIdentityNetwork`
(V200), with two structural differences:

* Embedding-based: ``predict()`` returns k-NN probabilities over identities
  rather than classifier logits.
* No training loop: ``fit()`` computes prototype embeddings; the backbone
  itself is frozen and never updated.

Weights and licensing
---------------------
Mosaic distributes no model weights. The backbone you name is fetched at run
time and carries its own license, independent of mosaic's own AGPLv3+.

The default, ``timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k``, is MIT:
permissive, commercially usable, and pretrained on ImageNet-22k.

``BVRA/MegaDescriptor-L-384`` (Cermak et al., WACV 2024, `WildlifeDatasets`_)
is the same Swin architecture pretrained on a metadataset of 53 wildlife
re-identification datasets, and substantially outperforms generic ImageNet
backbones on animal individuals. It is released under **CC-BY-NC-4.0**, which
does not permit commercial use. It is the right choice for academic wildlife
re-identification and is one parameter away; selecting it is your decision and
your license to comply with.

.. _WildlifeDatasets: https://github.com/WildlifeDatasets/wildlife-tools
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Final, TypeGuard

import numpy as np

from mosaic.behavior.model_library.identity_common import (
    compute_prototypes,
    knn_predict,
)

DEFAULT_MODEL_NAME: Final = "timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k"
"""The backbone loaded when ``model_name`` is not set.

MIT-licensed, and the same Swin architecture MegaDescriptor fine-tuned, so
naming ``BVRA/MegaDescriptor-L-384`` instead is a weights swap and nothing
else. Changing this line is a licensing decision, not a tuning decision -- see
the module docstring.
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

Deliberately identical to what this class hardcoded before resolution existed,
so the no-information path is the previous behavior rather than a third one.
"""

_TIMM_SOURCE_PREFIXES: Final = ("hf-hub:", "hf_hub:", "local-dir:")

CHECKPOINT_FORMAT_VERSION: Final = 2
"""Layout of the exported ``.pth``.

v2 records the resolved :class:`BackboneDataConfig` alongside the prototypes.
v1 was written by ``global-identity-megadescriptor``, which asserted its
preprocessing instead of recording it, so a v1 file cannot be replayed.
"""


def _import_torch() -> Any:
    """Lazily import torch with a helpful error message."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for EmbeddingIdentityNetwork. "
            "Install it with: pip install torch torchvision"
        ) from None
    return torch


def _import_timm() -> Any:
    """Lazily import timm with a helpful error message."""
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for EmbeddingIdentityNetwork. "
            "Install it with: pip install timm"
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


def _as_sequence(value: object) -> Sequence[object] | None:
    """*value* as a non-string sequence, or None if it is not one."""
    return value if _is_object_sequence(value) else None


def _triple(value: object) -> tuple[float, float, float] | None:
    """The three floats in *value*, or None if it does not hold exactly three."""
    items = _as_sequence(value)
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
    items = _as_sequence(value)
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
    items = _as_sequence(value)
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
    timm = _import_timm()
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


class EmbeddingIdentityNetwork:
    """Frozen image backbone + per-identity prototype k-NN.

    Args:
        model_name: A bare timm architecture tag or a Hugging Face hub id.
            Defaults to :data:`DEFAULT_MODEL_NAME`. Mosaic ships no weights;
            whatever is named here is downloaded at run time under its own
            license -- see the module docstring.
        image_size: Input ``(height, width)`` override. Defaults to None,
            meaning follow the backbone's declared input size, which is the
            only value correct for every backbone.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        data_config: Preprocessing recipe to use instead of resolving one from
            the backbone. Set when reloading a checkpoint, so a fitted model
            reproduces the preprocessing it was fitted with even if the
            upstream repository has changed underneath it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        image_size: tuple[int, int] | None = None,
        device: str = "auto",
        data_config: BackboneDataConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self._device: Any = self._resolve_device(device)

        timm = _import_timm()
        torch = _import_torch()

        backbone = timm.create_model(
            resolve_timm_model_id(model_name),
            pretrained=True,
            num_classes=0,  # remove classifier; we want pooled embeddings
        )
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._backbone: Any = backbone.to(self._device)

        resolved = data_config or resolve_backbone_data_config(backbone)
        if image_size is not None:
            resolved = replace(resolved, image_size=image_size)
        self._data_config: BackboneDataConfig = resolved
        self.image_size: tuple[int, int] = resolved.image_size

        # CPU, not device: _preprocess runs entirely on CPU tensors and embed()
        # moves each batch afterwards. Building these on the device meant every
        # call pulled them back, a host round-trip per batch.
        self._mean = torch.tensor(resolved.mean, dtype=torch.float32).reshape(
            1, 3, 1, 1
        )
        self._std = torch.tensor(resolved.std, dtype=torch.float32).reshape(1, 3, 1, 1)

        # A real self-check, not incidental: a backbone declaring
        # ``fixed_input_size`` raises here rather than at the first real batch
        # if resolution produced a size it cannot accept.
        with torch.no_grad():
            probe = torch.zeros(
                1, 3, self.image_size[0], self.image_size[1], device=self._device
            )
            feat = self._backbone(probe)
        self.embedding_dim: int = int(feat.shape[-1])

        self._prototypes: np.ndarray | None = None
        self._identity_names: list[str] | None = None
        self._best_accuracy: float = 0.0

    @property
    def num_classes(self) -> int:
        if self._prototypes is None:
            return 0
        return self._prototypes.shape[0]

    def embed(self, images: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        """Extract pooled feature embeddings.

        Args:
            images: ``(N, H, W, C)`` uint8 array. Grayscale (C=1) is
                replicated to 3 channels. Spatial size is bilinear-resized
                to the backbone's input size if it differs.
            batch_size: Inference batch size.

        Returns:
            ``(N, embedding_dim)`` float32 array.
        """
        torch = _import_torch()

        x = self._preprocess(images)
        out: list[np.ndarray] = []
        self._backbone.eval()
        with torch.no_grad():
            for i in range(0, x.shape[0], batch_size):
                batch = x[i : i + batch_size].to(self._device)
                feats = self._backbone(batch)
                out.append(feats.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        val_images: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        num_classes: int | None = None,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        """Compute per-identity prototype embeddings.

        Mirrors the V200 ``fit()`` signature so the feature plugin can
        delegate to either model identically. There is no training loop --
        this is a one-pass embedding + mean-pool.

        Args:
            images: ``(N, H, W, C)`` uint8 training crops.
            labels: ``(N,)`` integer class labels.
            val_images: Optional validation crops, used to report top-1
                k-NN accuracy.
            val_labels: Optional validation labels.
            num_classes: Total identities. Defaults to ``labels.max() + 1``.
            batch_size: Embedding batch size.

        Returns:
            History dict matching V200's keys (single-entry lists since
            there's no epoch loop).
        """
        if num_classes is None:
            num_classes = int(labels.max()) + 1

        print(
            f"[identity-embedding] embedding {len(images)} training images "
            f"for {num_classes} identities ({self.model_name})",
            file=sys.stderr,
        )
        train_emb = self.embed(images, batch_size=batch_size)
        self._prototypes = compute_prototypes(train_emb, labels, num_classes)

        train_probs = knn_predict(train_emb, self._prototypes)
        train_acc = float((train_probs.argmax(axis=1) == labels).mean())
        self._best_accuracy = train_acc

        val_acc = 0.0
        if val_images is not None and val_labels is not None and len(val_images) > 0:
            val_emb = self.embed(val_images, batch_size=batch_size)
            val_probs = knn_predict(val_emb, self._prototypes)
            val_acc = float((val_probs.argmax(axis=1) == val_labels).mean())

        print(
            f"[identity-embedding] train top-1={train_acc:.4f}  "
            f"val top-1={val_acc:.4f}",
            file=sys.stderr,
        )

        return {
            "train_loss": [0.0],
            "train_acc": [train_acc],
            "val_loss": [0.0],
            "val_acc": [val_acc],
        }

    def predict(self, images: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        """Return per-class probabilities via cosine k-NN against prototypes.

        Args:
            images: ``(N, H, W, C)`` uint8 array.
            batch_size: Embedding batch size.

        Returns:
            ``(N, num_classes)`` float32 probability array.
        """
        if self._prototypes is None:
            msg = (
                "[identity-embedding] No prototypes; call fit() or load a "
                "checkpoint first."
            )
            raise RuntimeError(msg)
        emb = self.embed(images, batch_size=batch_size)
        return knn_predict(emb, self._prototypes)

    def export_checkpoint(self, path: Path) -> Path:
        """Save prototypes + config to a ``.pth`` file.

        Unlike :meth:`TRexIdentityNetwork.export_trex_checkpoint`, this is
        not a T-Rex-loadable checkpoint -- it stores the prototype matrix
        and minimal config needed to reconstruct the network and predict.

        The resolved preprocessing recipe travels with it, so a reload
        reproduces the numbers this fit produced even if the backbone's
        upstream repository later edits its declared config.

        Args:
            path: Output file path. ``.pth`` is appended if missing.

        Returns:
            The resolved path the checkpoint was saved to.
        """
        torch = _import_torch()
        if self._prototypes is None:
            msg = "[identity-embedding] No prototypes to export; call fit() first."
            raise RuntimeError(msg)

        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "prototypes": self._prototypes,
            "identity_names": self._identity_names,
            "metadata": {
                "model_name": self.model_name,
                "image_size": self.image_size,
                "data_config": asdict(self._data_config),
                "embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
                "uniqueness": self._best_accuracy,
                "format_version": CHECKPOINT_FORMAT_VERSION,
            },
        }
        torch.save(checkpoint, path)
        print(
            f"[identity-embedding] exported checkpoint: {path}  "
            f"({self.num_classes} identities, dim={self.embedding_dim}, "
            f"acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: Path) -> EmbeddingIdentityNetwork:
        """Load a saved checkpoint and rebuild the backbone.

        Args:
            path: Path to ``.pth`` checkpoint file.

        Raises:
            ValueError: If the file is not
                :data:`CHECKPOINT_FORMAT_VERSION`.
        """
        torch = _import_torch()
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["metadata"]

        found = int(meta.get("format_version", 1))
        if found != CHECKPOINT_FORMAT_VERSION:
            msg = (
                f"[identity-embedding] checkpoint at {path} is format v{found}; "
                f"this network reads v{CHECKPOINT_FORMAT_VERSION}. A v1 file was "
                "written by global-identity-megadescriptor, which asserted its "
                "preprocessing rather than recording it, so it cannot be "
                "replayed. Refit under global-identity-embedding."
            )
            raise ValueError(msg)

        net = cls(
            model_name=meta["model_name"],
            data_config=data_config_from_metadata(meta.get("data_config")),
        )
        net._prototypes = np.asarray(checkpoint["prototypes"], dtype=np.float32)
        net._identity_names = checkpoint.get("identity_names")
        net._best_accuracy = float(meta.get("uniqueness", 0.0))
        return net

    # --- Internal ---

    def _preprocess(self, images: np.ndarray) -> Any:
        """Convert ``(N, H, W, C)`` uint8 to normalized ``(N, 3, H', W')`` tensor.

        The backbone's declared ``interpolation`` and ``crop_pct`` are
        deliberately not honored. Those describe timm's evaluation transform
        for a *full image* being center-cropped; the input here is already a
        tight egocentric crop, so a 0.9 crop ratio would discard the border a
        discriminative marking may sit in. Only the input size and the
        normalization statistics follow the backbone.
        """
        torch = _import_torch()

        if images.ndim != 4:
            msg = f"[identity-embedding] expected (N, H, W, C), got {images.shape}"
            raise ValueError(msg)
        if images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        elif images.shape[-1] != 3:
            msg = (
                f"[identity-embedding] expected 1 or 3 channels, got {images.shape[-1]}"
            )
            raise ValueError(msg)

        x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0

        target_h, target_w = self.image_size
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = torch.nn.functional.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        return (x - self._mean) / self._std

    @staticmethod
    def _resolve_device(device: str) -> Any:
        torch = _import_torch()
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
