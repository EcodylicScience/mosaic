"""DINOv2 + temporal aggregator identity recognition.

Frozen DINOv2 ViT as a per-frame embedder + a small temporal head that
aggregates frame embeddings within a clip into a single video embedding.
The temporal head is trained with ArcFace loss; identity is decided at
inference by cosine k-NN against per-identity prototype embeddings.

Three temporal heads are selectable via ``temporal_head``:

* ``"gru"``: single-layer GRU; last hidden state is the clip embedding.
  Closest match if the colleague's "DINOv2 + GRU" recipe was literal.
* ``"perceiver"``: a learned latent vector cross-attends to frame
  embeddings then self-attends; closer to RoVF (Rogers et al., IJCV 2025).
* ``"pool"``: mean-pool over time + linear projection. Sanity-check
  baseline -- if it matches the recurrent heads, the recurrence is not
  earning its keep on this dataset.

Sibling implementation to
:class:`~mosaic.behavior.model_library.trex_identity_network.TRexIdentityNetwork`
(V200) and
:class:`~mosaic.behavior.model_library.megadescriptor_identity.MegaDescriptorNetwork`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mosaic.behavior.model_library.identity_common import (
    compute_prototypes,
    knn_predict,
)

TemporalHead = Literal["gru", "perceiver", "pool"]


def _import_torch() -> Any:
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for DinoV2TemporalNetwork. "
            "Install it with: pip install torch torchvision"
        ) from None
    return torch


def _load_dinov2_backbone(name: str, device: Any) -> Any:
    """Load a DINOv2 ViT via torch.hub, cached locally on first use."""
    torch = _import_torch()
    backbone = torch.hub.load("facebookresearch/dinov2", name, trust_repo=True)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone.to(device)


def _build_temporal_head(
    head_type: TemporalHead, *, in_dim: int, out_dim: int
) -> Any:
    torch = _import_torch()
    nn = torch.nn

    if head_type == "gru":
        class GRUHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(in_dim, out_dim, batch_first=True)

            def forward(self, x: Any) -> Any:  # x: (B, T, in_dim)
                _, h = self.gru(x)
                return h.squeeze(0)  # (B, out_dim)

        return GRUHead()

    if head_type == "perceiver":
        class PerceiverHead(nn.Module):
            """Single-block Perceiver-IO-lite: one learned latent attends
            to all frames, followed by an MLP."""

            def __init__(self) -> None:
                super().__init__()
                self.latent = nn.Parameter(torch.randn(1, 1, out_dim) * 0.02)
                self.kv_proj = nn.Linear(in_dim, out_dim)
                self.attn = nn.MultiheadAttention(
                    out_dim, num_heads=4, batch_first=True
                )
                self.norm1 = nn.LayerNorm(out_dim)
                self.mlp = nn.Sequential(
                    nn.Linear(out_dim, out_dim * 2),
                    nn.GELU(),
                    nn.Linear(out_dim * 2, out_dim),
                )
                self.norm2 = nn.LayerNorm(out_dim)

            def forward(self, x: Any) -> Any:  # x: (B, T, in_dim)
                b = x.shape[0]
                kv = self.kv_proj(x)
                latent = self.latent.expand(b, -1, -1)
                attn_out, _ = self.attn(latent, kv, kv)
                latent = self.norm1(latent + attn_out)
                latent = self.norm2(latent + self.mlp(latent))
                return latent.squeeze(1)  # (B, out_dim)

        return PerceiverHead()

    if head_type == "pool":
        class PoolHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(in_dim, out_dim)

            def forward(self, x: Any) -> Any:
                return self.proj(x.mean(dim=1))

        return PoolHead()

    msg = f"[dinov2-temporal] unknown temporal_head: {head_type}"
    raise ValueError(msg)


def _build_arcface_head(embedding_dim: int, num_classes: int) -> Any:
    torch = _import_torch()
    nn = torch.nn

    class ArcFaceHead(nn.Module):
        """ArcFace classifier head.

        Projects L2-normalized embeddings against L2-normalized class
        weights, applies an additive angular margin to the true class,
        scales, and returns logits ready for CrossEntropyLoss.
        """

        def __init__(self, embedding_dim: int, num_classes: int,
                     margin: float = 0.3, scale: float = 30.0) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(num_classes, embedding_dim))
            nn.init.xavier_uniform_(self.weight)
            self.margin = margin
            self.scale = scale

        def forward(self, embeddings: Any, labels: Any | None = None) -> Any:
            w = torch.nn.functional.normalize(self.weight, dim=1)
            e = torch.nn.functional.normalize(embeddings, dim=1)
            cos = e @ w.t()  # (B, C)
            if labels is None:
                return cos * self.scale
            # Add angular margin to true class only.
            cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            target = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
            theta_m = theta + target * self.margin
            return torch.cos(theta_m) * self.scale

    return ArcFaceHead(embedding_dim, num_classes)


class DinoV2TemporalNetwork:
    """Frozen DINOv2 backbone + trainable temporal head + ArcFace.

    Args:
        backbone: DINOv2 hub model name. Default ``"dinov2_vits14"`` (21M
            params, 384-d embedding). ``"dinov2_vitb14"`` for the larger
            base variant.
        temporal_head: Aggregator over a clip's frame embeddings. One of
            ``"gru"``, ``"perceiver"``, ``"pool"``.
        embedding_dim: Output dimension of the temporal head. Default 128.
        clip_len: Frames per clip. Default 8.
        image_size: Input ``(height, width)``. Must be a multiple of 14
            for DINOv2's patch size. Default ``(224, 224)``.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """

    IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone: str = "dinov2_vits14",
        temporal_head: TemporalHead = "gru",
        embedding_dim: int = 128,
        clip_len: int = 8,
        image_size: tuple[int, int] = (224, 224),
        device: str = "auto",
    ) -> None:
        self.backbone_name = backbone
        self.temporal_head_name: TemporalHead = temporal_head
        self.embedding_dim = embedding_dim
        self.clip_len = clip_len
        self.image_size = image_size
        self._device: Any = self._resolve_device(device)

        if image_size[0] % 14 != 0 or image_size[1] % 14 != 0:
            msg = (
                f"[dinov2-temporal] image_size must be a multiple of 14 "
                f"for DINOv2, got {image_size}"
            )
            raise ValueError(msg)

        torch = _import_torch()
        self._backbone = _load_dinov2_backbone(backbone, self._device)
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
        self.frame_dim: int = int(feat.shape[-1])

        self._temporal: Any = _build_temporal_head(
            temporal_head, in_dim=self.frame_dim, out_dim=embedding_dim
        ).to(self._device)
        self._arcface: Any | None = None  # built in fit() once num_classes known

        self._prototypes: np.ndarray | None = None
        self._identity_names: list[str] | None = None
        self._best_accuracy: float = 0.0
        self._epoch: int = 0

    @property
    def num_classes(self) -> int:
        if self._prototypes is None:
            return 0
        return self._prototypes.shape[0]

    def fit(
        self,
        clips: np.ndarray,
        labels: np.ndarray,
        *,
        val_clips: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        num_classes: int | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        """Train the temporal head + ArcFace classifier.

        The DINOv2 backbone is frozen and its per-frame embeddings are
        cached in memory across epochs to keep training fast.

        Args:
            clips: ``(N, T, H, W, C)`` uint8 array. ``T`` must equal
                ``self.clip_len``.
            labels: ``(N,)`` integer class labels.
            val_clips: Optional validation clips.
            val_labels: Optional validation labels.
            num_classes: Defaults to ``labels.max() + 1``.
            epochs: Number of optimization passes over the cached frame
                features. Default 50.
            lr: Adam learning rate. Default 1e-3.
            batch_size: Training batch size. Default 32.

        Returns:
            Per-epoch history dict matching V200's keys.
        """
        torch = _import_torch()

        if clips.ndim != 5 or clips.shape[1] != self.clip_len:
            msg = (
                f"[dinov2-temporal] expected (N, {self.clip_len}, H, W, C), "
                f"got {clips.shape}"
            )
            raise ValueError(msg)

        if num_classes is None:
            num_classes = int(labels.max()) + 1

        self._arcface = _build_arcface_head(
            self.embedding_dim, num_classes
        ).to(self._device)

        print(
            f"[dinov2-temporal] caching frame features for "
            f"{len(clips)} train clips...",
            file=sys.stderr,
        )
        train_feats = self._embed_frames_batched(clips, batch_size=batch_size)
        train_feats_t = torch.from_numpy(train_feats).to(self._device)
        labels_t = torch.from_numpy(labels.astype(np.int64)).to(self._device)

        val_feats_t: Any = None
        val_labels_t: Any = None
        has_val = val_clips is not None and val_labels is not None and len(val_clips) > 0
        if has_val:
            print(
                f"[dinov2-temporal] caching frame features for "
                f"{len(val_clips)} val clips...",
                file=sys.stderr,
            )
            val_feats = self._embed_frames_batched(val_clips, batch_size=batch_size)
            val_feats_t = torch.from_numpy(val_feats).to(self._device)
            val_labels_t = torch.from_numpy(val_labels.astype(np.int64)).to(
                self._device
            )

        params = list(self._temporal.parameters()) + list(self._arcface.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        n = train_feats_t.shape[0]
        for epoch in range(1, epochs + 1):
            self._temporal.train()
            self._arcface.train()
            perm = torch.randperm(n, device=self._device)
            running_loss = 0.0
            correct = 0
            total = 0

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                batch_feats = train_feats_t[idx]
                batch_labels = labels_t[idx]

                optimizer.zero_grad()
                emb = self._temporal(batch_feats)
                logits = self._arcface(emb, batch_labels)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_labels.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            val_loss = 0.0
            val_acc = 0.0
            if has_val:
                self._temporal.eval()
                self._arcface.eval()
                with torch.no_grad():
                    emb = self._temporal(val_feats_t)
                    logits = self._arcface(emb, val_labels_t)
                    v_loss = criterion(logits, val_labels_t).item()
                    v_correct = (logits.argmax(dim=1) == val_labels_t).sum().item()
                val_loss = v_loss
                val_acc = v_correct / val_labels_t.shape[0]
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if train_acc > self._best_accuracy:
                self._best_accuracy = train_acc

            if epoch % 10 == 0 or epoch == 1:
                msg = (
                    f"[dinov2-temporal] epoch {epoch}/{epochs}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )
                if has_val:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                print(msg, file=sys.stderr)

        self._epoch = epochs

        # Compute prototypes for k-NN inference path (so predict() works
        # even if the ArcFace head was trained on a subset).
        self._temporal.eval()
        with torch.no_grad():
            train_emb = self._temporal(train_feats_t).cpu().numpy()
        self._prototypes = compute_prototypes(train_emb, labels, num_classes)

        return history

    def predict(self, clips: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        """Per-class probabilities via cosine k-NN against prototypes.

        Args:
            clips: ``(N, T, H, W, C)`` uint8 array.
            batch_size: Inference batch size.

        Returns:
            ``(N, num_classes)`` float32 probability array.
        """
        if self._prototypes is None:
            msg = "[dinov2-temporal] No prototypes; call fit() or load a checkpoint."
            raise RuntimeError(msg)
        emb = self.embed(clips, batch_size=batch_size)
        return knn_predict(emb, self._prototypes)

    def embed(self, clips: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        """Return clip-level embeddings ``(N, embedding_dim)``."""
        torch = _import_torch()

        feats = self._embed_frames_batched(clips, batch_size=batch_size)
        feats_t = torch.from_numpy(feats).to(self._device)
        self._temporal.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, feats_t.shape[0], batch_size):
                e = self._temporal(feats_t[i : i + batch_size])
                out.append(e.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def export_checkpoint(self, path: Path) -> Path:
        torch = _import_torch()
        if self._prototypes is None:
            msg = "[dinov2-temporal] No prototypes to export; call fit() first."
            raise RuntimeError(msg)

        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "temporal_state_dict": self._temporal.state_dict(),
            "arcface_state_dict": (
                self._arcface.state_dict() if self._arcface is not None else None
            ),
            "prototypes": self._prototypes,
            "identity_names": self._identity_names,
            "metadata": {
                "backbone": self.backbone_name,
                "temporal_head": self.temporal_head_name,
                "embedding_dim": self.embedding_dim,
                "clip_len": self.clip_len,
                "image_size": self.image_size,
                "num_classes": self.num_classes,
                "epoch": self._epoch,
                "uniqueness": self._best_accuracy,
                "format_version": 1,
            },
        }
        torch.save(checkpoint, path)
        print(
            f"[dinov2-temporal] exported checkpoint: {path}  "
            f"({self.num_classes} identities, head={self.temporal_head_name}, "
            f"acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: Path) -> DinoV2TemporalNetwork:
        torch = _import_torch()
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["metadata"]

        net = cls(
            backbone=meta["backbone"],
            temporal_head=meta["temporal_head"],
            embedding_dim=meta["embedding_dim"],
            clip_len=meta["clip_len"],
            image_size=tuple(meta["image_size"]),
        )
        net._temporal.load_state_dict(checkpoint["temporal_state_dict"])
        if checkpoint.get("arcface_state_dict") is not None:
            from mosaic.behavior.model_library.dinov2_temporal_identity import (
                _build_arcface_head,
            )
            net._arcface = _build_arcface_head(
                meta["embedding_dim"], meta["num_classes"]
            ).to(net._device)
            net._arcface.load_state_dict(checkpoint["arcface_state_dict"])
        net._prototypes = np.asarray(checkpoint["prototypes"], dtype=np.float32)
        net._identity_names = checkpoint.get("identity_names")
        net._epoch = int(meta.get("epoch", 0))
        net._best_accuracy = float(meta.get("uniqueness", 0.0))
        return net

    # --- Internal ---

    def _embed_frames_batched(
        self, clips: np.ndarray, *, batch_size: int
    ) -> np.ndarray:
        """Run DINOv2 over every frame and reshape back to clip format.

        Args:
            clips: ``(N, T, H, W, C)`` uint8.

        Returns:
            ``(N, T, frame_dim)`` float32.
        """
        torch = _import_torch()

        n, t, h, w, c = clips.shape
        flat = clips.reshape(n * t, h, w, c)
        x = self._preprocess(flat)

        out: list[np.ndarray] = []
        # The flatten makes effective batch n*t; clamp by frames per pass.
        frames_per_pass = max(1, batch_size * t)
        with torch.no_grad():
            for i in range(0, x.shape[0], frames_per_pass):
                batch = x[i : i + frames_per_pass].to(self._device)
                feats = self._backbone(batch)
                out.append(feats.detach().cpu().float().numpy())
        all_feats = np.concatenate(out, axis=0)  # (N*T, frame_dim)
        return all_feats.reshape(n, t, all_feats.shape[-1]).astype(np.float32)

    def _preprocess(self, images: np.ndarray) -> Any:
        torch = _import_torch()

        if images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        elif images.shape[-1] != 3:
            msg = (
                f"[dinov2-temporal] expected 1 or 3 channels, "
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
