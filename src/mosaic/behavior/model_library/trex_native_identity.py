"""T-Rex *native* V200 identity network (3-conv + 2-fc variant).

The existing :class:`TRexIdentityNetwork` ships with a deeper 5-conv-block
``nn.Sequential`` whose state_dict keys (``0.weight``, ``1.weight`` …) do not
match T-Rex's native ``.pth`` files, which are saved from a smaller named-layer
architecture wrapped under ``model.``::

    model.conv1, model.bn1
    model.conv2, model.bn2
    model.conv3, model.bn3
    model.fc1,   model.bn4
    model.fc2

This module provides :class:`TRexNativeIdentityNetwork`, a sibling of the
existing class with the smaller architecture and named layers, so T-Rex's saved
weights load directly via :meth:`from_trex_checkpoint`. Channel counts and the
FC1 hidden size are auto-detected from the loaded state_dict, so the same
class works regardless of which T-Rex training variant produced the file.

The wrapper class :class:`_PermuteAxesWrapper` is reused from
:mod:`trex_identity_network` (input contract is identical: ``(N, H, W, C)``
``uint8`` arrays, ImageNet-normalized internally).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from .trex_identity_network import _PermuteAxesWrapper, _import_torch


class _V200NativeNet:
    """Lazy nn.Module factory matching T-Rex's saved-checkpoint architecture.

    The class itself is not an nn.Module; calling
    :func:`_build_v200_native` with concrete dimensions returns one. This
    indirection keeps the module importable when PyTorch is not installed.
    """

    pass


def _build_v200_native(
    channels: int,
    conv_channels: tuple[int, int, int],
    fc_hidden: int,
    num_classes: int,
) -> Any:
    """Build the native V200 architecture as a named-layer ``nn.Module``.

    Layout matches the saved-checkpoint keys exactly:

    - ``conv1`` → ``bn1`` → ReLU → MaxPool2d(2)
    - ``conv2`` → ``bn2`` → ReLU → MaxPool2d(2)
    - ``conv3`` → ``bn3`` → ReLU → MaxPool2d(2)
    - AdaptiveAvgPool2d((1,1)) → Flatten
    - ``fc1`` → ``bn4`` → ReLU → Dropout(0.05)
    - ``fc2`` (no batchnorm; final classifier)

    Returns:
        An ``nn.Module`` with attributes named exactly as above so its
        ``state_dict()`` keys match the user's saved ``.pth``.
    """
    torch = _import_torch()
    nn = torch.nn

    class V200NativeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3 = conv_channels
            # Block 1
            self.conv1 = nn.Conv2d(channels, c1, kernel_size=3, padding="same")
            self.bn1 = nn.BatchNorm2d(c1)
            # Block 2
            self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding="same")
            self.bn2 = nn.BatchNorm2d(c2)
            # Block 3
            self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, padding="same")
            self.bn3 = nn.BatchNorm2d(c3)
            # Head
            self.pool = nn.MaxPool2d(2)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(c3, fc_hidden)
            self.bn4 = nn.BatchNorm1d(fc_hidden)
            self.dropout = nn.Dropout(0.05)
            self.fc2 = nn.Linear(fc_hidden, num_classes)

        def forward(self, x: Any) -> Any:
            relu = nn.functional.relu
            x = self.pool(relu(self.bn1(self.conv1(x))))
            x = self.pool(relu(self.bn2(self.conv2(x))))
            x = self.pool(relu(self.bn3(self.conv3(x))))
            x = self.gap(x)
            x = x.flatten(1)
            x = self.dropout(relu(self.bn4(self.fc1(x))))
            return self.fc2(x)

    return V200NativeNet()


def _infer_dims_from_state_dict(
    state_dict: dict[str, Any],
) -> tuple[int, tuple[int, int, int], int, int]:
    """Inspect a state_dict (with ``model.`` prefix) and read off layer dims.

    Returns:
        ``(channels, (C1, C2, C3), fc_hidden, num_classes)``.

    Raises:
        ValueError: when an expected key is missing or has an unexpected shape.
    """
    expected = {
        "model.conv1.weight": "(C1, channels, 3, 3)",
        "model.conv2.weight": "(C2, C1, 3, 3)",
        "model.conv3.weight": "(C3, C2, 3, 3)",
        "model.fc1.weight":   "(fc_hidden, C3)",
        "model.fc2.weight":   "(num_classes, fc_hidden)",
    }
    for key in expected:
        if key not in state_dict:
            raise ValueError(
                f"checkpoint missing expected key {key!r}; "
                f"present keys (first 8): {list(state_dict.keys())[:8]} …"
            )

    def _shape(key: str) -> tuple[int, ...]:
        s = state_dict[key].shape
        return tuple(int(x) for x in s)

    s1 = _shape("model.conv1.weight")
    s2 = _shape("model.conv2.weight")
    s3 = _shape("model.conv3.weight")
    sf1 = _shape("model.fc1.weight")
    sf2 = _shape("model.fc2.weight")

    if len(s1) != 4 or len(s2) != 4 or len(s3) != 4:
        raise ValueError(
            f"conv weights must be 4D, got shapes "
            f"conv1={s1} conv2={s2} conv3={s3}"
        )
    if len(sf1) != 2 or len(sf2) != 2:
        raise ValueError(
            f"fc weights must be 2D, got shapes fc1={sf1} fc2={sf2}"
        )

    c1_out, channels = s1[0], s1[1]
    c2_out, c2_in = s2[0], s2[1]
    c3_out, c3_in = s3[0], s3[1]
    fc_hidden, fc1_in = sf1
    num_classes, fc2_in = sf2

    if c2_in != c1_out:
        raise ValueError(f"conv2 in_channels ({c2_in}) ≠ conv1 out ({c1_out})")
    if c3_in != c2_out:
        raise ValueError(f"conv3 in_channels ({c3_in}) ≠ conv2 out ({c2_out})")
    if fc1_in != c3_out:
        raise ValueError(f"fc1 in_features ({fc1_in}) ≠ conv3 out ({c3_out})")
    if fc2_in != fc_hidden:
        raise ValueError(f"fc2 in_features ({fc2_in}) ≠ fc1 out ({fc_hidden})")

    return channels, (c1_out, c2_out, c3_out), fc_hidden, num_classes


class TRexNativeIdentityNetwork:
    """V200 identity classifier matching T-Rex's actual saved checkpoint layout.

    Use :meth:`from_trex_checkpoint` to load a T-Rex ``.pth`` directly. The
    architecture (conv channel counts, FC1 hidden size, num_classes, channels)
    is auto-detected from the file's state_dict shapes — no manual config
    needed.

    Public API mirrors :class:`TRexIdentityNetwork` so the two are
    drop-in-compatible at the call site:

    - :meth:`predict` returns ``(N, num_classes)`` softmax probabilities.
    - :meth:`fit` trains with Adam + CrossEntropyLoss.
    - :meth:`export_trex_checkpoint` writes a ``{state_dict, metadata}``
      ``.pth`` round-trippable through :meth:`from_trex_checkpoint`.

    Args:
        num_classes: Number of output classes (identities).
        channels: Number of input channels (1 = grayscale, 3 = RGB).
        image_size: ``(height, width)``. Default ``(80, 80)`` matches T-Rex's
            common training crop size; the network is spatial-dim-agnostic
            thanks to AdaptiveAvgPool, so any size works at inference.
        conv_channels: ``(C1, C2, C3)`` for the three conv blocks.
        fc_hidden: Hidden size of the FC1 layer.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 1,
        image_size: tuple[int, int] = (80, 80),
        conv_channels: tuple[int, int, int] = (16, 32, 64),
        fc_hidden: int = 256,
    ) -> None:
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size  # (height, width)
        self.conv_channels = conv_channels
        self.fc_hidden = fc_hidden

        raw_model = _build_v200_native(
            channels=channels,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            num_classes=num_classes,
        )
        self._model = _PermuteAxesWrapper(raw_model, channels)
        self._device: Any = None
        self._epoch: int = 0
        self._best_accuracy: float = 0.0

    # --- Training (mirror of TRexIdentityNetwork.fit) ---------------------

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        val_images: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        epochs: int = 150,
        lr: float = 0.0001,
        batch_size: int = 64,
        device: str = "auto",
    ) -> dict[str, list[float]]:
        """Train the identity classifier.

        Args:
            images: ``(N, H, W, C)`` ``uint8`` array, values 0–255.
            labels: ``(N,)`` integer class labels.
            val_images: Optional validation set.
            val_labels: Optional validation labels.
            epochs: Training epochs.
            lr: Adam learning rate.
            batch_size: Batch size.
            device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

        Returns:
            Training history dict with keys ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc`` (per epoch).
        """
        torch = _import_torch()

        self._device = self._resolve_device(device)
        self._model.to(self._device)

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(images),
            torch.from_numpy(labels.astype(np.int64)),
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        )

        has_val = val_images is not None and val_labels is not None
        val_loader = None
        if has_val:
            val_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(val_images),
                torch.from_numpy(val_labels.astype(np.int64)),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
            )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc":  [],
            "val_loss":   [],
            "val_acc":    [],
        }

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0
            self._model.train()

            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self._device)
                batch_labels = batch_labels.to(self._device)

                optimizer.zero_grad()
                logits = self._model(batch_images)
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
            if has_val and val_loader is not None:
                self._model.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for batch_images, batch_labels in val_loader:
                        batch_images = batch_images.to(self._device)
                        batch_labels = batch_labels.to(self._device)
                        logits = self._model(batch_images)
                        loss = criterion(logits, batch_labels)
                        v_loss += loss.item() * batch_labels.size(0)
                        preds = logits.argmax(dim=1)
                        v_correct += (preds == batch_labels).sum().item()
                        v_total += batch_labels.size(0)
                val_loss = v_loss / v_total if v_total > 0 else 0.0
                val_acc = v_correct / v_total if v_total > 0 else 0.0

            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if train_acc > self._best_accuracy:
                self._best_accuracy = train_acc

            if epoch % 10 == 0 or epoch == 1:
                msg = (
                    f"[trex-native] epoch {epoch}/{epochs}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )
                if has_val:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                print(msg, file=sys.stderr)

        self._epoch = epochs
        return history

    # --- Inference --------------------------------------------------------

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return per-class probabilities.

        Args:
            images: ``(N, H, W, C)`` ``uint8`` array.

        Returns:
            ``(N, num_classes)`` ``float32`` probability array.
        """
        torch = _import_torch()

        if self._device is None:
            self._device = self._resolve_device("auto")
            self._model.to(self._device)

        self._model.eval()
        tensor = torch.from_numpy(images).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy().astype(np.float32)

    # --- Persistence ------------------------------------------------------

    def export_trex_checkpoint(
        self, path: Path, video_name: str = "external"
    ) -> Path:
        """Save weights in T-Rex-compatible format.

        Round-trips through :meth:`from_trex_checkpoint`.

        Args:
            path: Output file path (``.pth`` extension auto-appended if
                missing).
            video_name: Stored in metadata for traceability.

        Returns:
            The resolved path the checkpoint was saved to.
        """
        torch = _import_torch()

        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)

        h, w = self.image_size
        checkpoint = {
            "state_dict": self._model.state_dict(),
            "metadata": {
                "input_shape": (w, h, self.channels),  # T-Rex order: width, height, channels
                "num_classes": self.num_classes,
                "video_name":  video_name,
                "epoch":       self._epoch,
                "uniqueness":  self._best_accuracy,
                "conv_channels": list(self.conv_channels),
                "fc_hidden":   self.fc_hidden,
                "architecture": "v200-native",
            },
        }
        torch.save(checkpoint, path)
        print(
            f"[trex-native] Exported T-Rex-native checkpoint: {path}  "
            f"({self.num_classes} classes, channels={self.channels}, "
            f"conv={self.conv_channels}, fc_hidden={self.fc_hidden}, "
            f"epoch={self._epoch}, acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        return path

    @classmethod
    def from_trex_checkpoint(cls, path: Path) -> TRexNativeIdentityNetwork:
        """Load a T-Rex ``.pth`` and auto-detect architecture from its shapes.

        Handles three checkpoint shapes:

        1. ``{"state_dict": ..., "metadata": ...}`` (Mosaic-exported, T-Rex's
           preferred wrapper).
        2. ``{"state_dict": ...}`` without metadata (raw T-Rex export).
        3. A bare ``state_dict`` (no top-level wrapper).

        Args:
            path: Path to the ``.pth`` file.

        Returns:
            A :class:`TRexNativeIdentityNetwork` with weights loaded.

        Raises:
            ValueError: when the architecture cannot be inferred from the
                state_dict shapes (e.g. wrong layer names, unexpected ranks).
        """
        torch = _import_torch()

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        # Resolve to the actual state_dict
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            meta = ckpt.get("metadata", {}) or {}
        elif isinstance(ckpt, dict) and any(
            k.startswith("model.") for k in ckpt.keys()
        ):
            sd = ckpt
            meta = {}
        else:
            raise ValueError(
                f"unrecognised checkpoint format at {path}: top-level keys = "
                f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}"
            )

        channels, conv_channels, fc_hidden, num_classes = _infer_dims_from_state_dict(sd)

        # image_size: prefer metadata; fall back to (80, 80) which is T-Rex's default
        if "input_shape" in meta:
            shape = tuple(int(x) for x in meta["input_shape"])
            if len(shape) == 3:
                w, h, _ = shape
                image_size: tuple[int, int] = (h, w)
            elif len(shape) == 2:
                image_size = (shape[0], shape[1])
            else:
                image_size = (80, 80)
        else:
            image_size = (80, 80)

        net = cls(
            num_classes=num_classes,
            channels=channels,
            image_size=image_size,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
        )
        net._model.load_state_dict(sd)
        net._epoch = int(meta.get("epoch", 0))
        net._best_accuracy = float(meta.get("uniqueness", 0.0))

        print(
            f"[trex-native] Loaded {path.name}  "
            f"channels={channels}  conv={conv_channels}  fc_hidden={fc_hidden}  "
            f"num_classes={num_classes}  image_size={image_size}",
            file=sys.stderr,
        )
        return net

    # --- Internals --------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> Any:
        """Resolve device string to a torch.device."""
        torch = _import_torch()

        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
