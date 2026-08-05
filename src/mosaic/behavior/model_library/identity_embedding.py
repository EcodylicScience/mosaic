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
:class:`~mosaic.behavior.model_library.identity_classifier.ClassifierIdentityNetwork`,
with two structural differences:

* Embedding-based: ``predict()`` returns k-NN probabilities over identities
  rather than classifier logits.
* No training loop: ``fit()`` computes prototype embeddings; the backbone
  itself is frozen and never updated. Reach for the classifier instead when
  there are enough crops per animal to train a head on.

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
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from mosaic.behavior.model_library.identity_common import (
    compute_prototypes,
    knn_predict,
)
from mosaic.behavior.model_library.timm_backbone import (
    DEFAULT_MODEL_NAME,
    BackboneDataConfig,
    data_config_from_metadata,
    import_timm,
    import_torch,
    normalization_tensors,
    preprocess_batch,
    resolve_backbone_data_config,
    resolve_device,
    resolve_timm_model_id,
)

CHECKPOINT_FORMAT_VERSION: Final = 2
"""Layout of the exported ``.pth``.

v2 records the resolved :class:`BackboneDataConfig` alongside the prototypes.
v1 was written by ``global-identity-megadescriptor``, which asserted its
preprocessing instead of recording it, so a v1 file cannot be replayed.
"""


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
        self._device: Any = resolve_device(device)

        timm = import_timm()
        torch = import_torch()

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

        self._mean, self._std = normalization_tensors(resolved)

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
        torch = import_torch()

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

        Mirrors the trained classifier's ``fit()`` signature so the feature
        plugin can delegate to either model identically. There is no
        training loop -- this is a one-pass embedding + mean-pool.

        Args:
            images: ``(N, H, W, C)`` uint8 training crops.
            labels: ``(N,)`` integer class labels.
            val_images: Optional validation crops, used to report top-1
                k-NN accuracy.
            val_labels: Optional validation labels.
            num_classes: Total identities. Defaults to ``labels.max() + 1``.
            batch_size: Embedding batch size.

        Returns:
            History dict matching the trained classifier's keys (single-entry
            lists since
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

        Stores the prototype matrix and the minimal config needed to
        reconstruct the network and predict; the backbone itself is refetched
        by name, never written here.

        The resolved preprocessing recipe travels with it, so a reload
        reproduces the numbers this fit produced even if the backbone's
        upstream repository later edits its declared config.

        Args:
            path: Output file path. ``.pth`` is appended if missing.

        Returns:
            The resolved path the checkpoint was saved to.
        """
        torch = import_torch()
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
        torch = import_torch()
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
        """Convert ``(N, H, W, C)`` uint8 to a normalized ``(N, 3, H', W')`` tensor."""
        return preprocess_batch(images, self.image_size, self._mean, self._std)
