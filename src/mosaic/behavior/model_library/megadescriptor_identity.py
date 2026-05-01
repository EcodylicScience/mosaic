"""MegaDescriptor-based identity recognition.

Wraps the MegaDescriptor foundation model (Cermak et al., WACV 2024,
`WildlifeDatasets`_) as a frozen embedding extractor for individual animal
re-identification. Identity is decided at inference by cosine-similarity
k-NN against per-identity prototype embeddings computed at fit time.

MegaDescriptor is pretrained on a metadataset of 53 wildlife re-ID datasets
and outperforms generic foundation models (DINOv2, CLIP) on animal re-ID by
a significant margin -- making it a strong zero-shot baseline that requires
no per-mouse training cycle.

This is a sibling implementation to
:class:`~mosaic.behavior.model_library.trex_identity_network.TRexIdentityNetwork`
(V200), with two structural differences:

* Embedding-based: ``predict()`` returns k-NN probabilities over identities
  rather than classifier logits.
* No training loop: ``fit()`` computes prototype embeddings; the backbone
  itself is frozen and never updated.

.. _WildlifeDatasets: https://github.com/WildlifeDatasets/wildlife-tools
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from mosaic.behavior.model_library.identity_common import (
    compute_prototypes,
    knn_predict,
)


def _import_torch() -> Any:
    """Lazily import torch with a helpful error message."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for MegaDescriptorNetwork. "
            "Install it with: pip install torch torchvision"
        ) from None
    return torch


def _import_timm() -> Any:
    """Lazily import timm with a helpful error message."""
    try:
        import timm
    except ImportError:
        raise ImportError(
            "timm is required for MegaDescriptorNetwork. "
            "Install it with: pip install timm"
        ) from None
    return timm


class MegaDescriptorNetwork:
    """Frozen MegaDescriptor backbone + per-identity prototype k-NN.

    Args:
        model_name: HuggingFace hub id. Defaults to the largest variant
            (``MegaDescriptor-L-384``, SwinV2-L at 384x384). Use
            ``MegaDescriptor-T-224`` for a much smaller / faster model.
        image_size: Input ``(height, width)``. Should match the model
            variant's expected size. Defaults to ``(384, 384)``.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """

    IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "BVRA/MegaDescriptor-L-384",
        image_size: tuple[int, int] = (384, 384),
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.image_size = image_size
        self._device: Any = self._resolve_device(device)

        timm = _import_timm()
        torch = _import_torch()

        backbone = timm.create_model(
            f"hf-hub:{model_name}",
            pretrained=True,
            num_classes=0,  # remove classifier; we want pooled embeddings
        )
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self._backbone: Any = backbone.to(self._device)

        self._mean = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).reshape(
            1, 3, 1, 1
        ).to(self._device)
        self._std = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).reshape(
            1, 3, 1, 1
        ).to(self._device)

        with torch.no_grad():
            probe = torch.zeros(
                1, 3, image_size[0], image_size[1], device=self._device
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
                to ``image_size`` if it differs.
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
            f"[megadescriptor] embedding {len(images)} training images "
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
            f"[megadescriptor] train top-1={train_acc:.4f}  "
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
            msg = "[megadescriptor] No prototypes; call fit() or load a checkpoint first."
            raise RuntimeError(msg)
        emb = self.embed(images, batch_size=batch_size)
        return knn_predict(emb, self._prototypes)

    def export_checkpoint(self, path: Path) -> Path:
        """Save prototypes + config to a ``.pth`` file.

        Unlike :meth:`TRexIdentityNetwork.export_trex_checkpoint`, this is
        not a T-Rex-loadable checkpoint -- it stores the prototype matrix
        and minimal config needed to reconstruct the network and predict.

        Args:
            path: Output file path. ``.pth`` is appended if missing.

        Returns:
            The resolved path the checkpoint was saved to.
        """
        torch = _import_torch()
        if self._prototypes is None:
            msg = "[megadescriptor] No prototypes to export; call fit() first."
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
                "embedding_dim": self.embedding_dim,
                "num_classes": self.num_classes,
                "uniqueness": self._best_accuracy,
                "format_version": 1,
            },
        }
        torch.save(checkpoint, path)
        print(
            f"[megadescriptor] exported checkpoint: {path}  "
            f"({self.num_classes} identities, dim={self.embedding_dim}, "
            f"acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: Path) -> MegaDescriptorNetwork:
        """Load a saved checkpoint and rebuild the backbone.

        Args:
            path: Path to ``.pth`` checkpoint file.
        """
        torch = _import_torch()
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["metadata"]

        net = cls(
            model_name=meta["model_name"],
            image_size=tuple(meta["image_size"]),
        )
        net._prototypes = np.asarray(checkpoint["prototypes"], dtype=np.float32)
        net._identity_names = checkpoint.get("identity_names")
        net._best_accuracy = float(meta.get("uniqueness", 0.0))
        return net

    # --- Internal ---

    def _preprocess(self, images: np.ndarray) -> Any:
        """Convert ``(N, H, W, C)`` uint8 to normalized ``(N, 3, H', W')`` tensor."""
        torch = _import_torch()

        if images.ndim != 4:
            msg = f"[megadescriptor] expected (N, H, W, C), got {images.shape}"
            raise ValueError(msg)
        if images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        elif images.shape[-1] != 3:
            msg = (
                f"[megadescriptor] expected 1 or 3 channels, "
                f"got {images.shape[-1]}"
            )
            raise ValueError(msg)

        x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0

        target_h, target_w = self.image_size
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = torch.nn.functional.interpolate(
                x, size=(target_h, target_w), mode="bilinear", align_corners=False
            )

        x = (x - self._mean.cpu()) / self._std.cpu()
        return x

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
