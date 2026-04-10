"""T-Rex-compatible V200 CNN identity classifier.

Provides a PyTorch CNN that exactly matches the V200 architecture used by
T-Rex for visual individual identification. Trained weights can be exported
as .pth checkpoints loadable via T-Rex's ``visual_identification_model_path``
setting.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _import_torch() -> Any:
    """Lazily import torch with a helpful error message."""
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for TRexIdentityNetwork. "
            "Install it with: pip install torch torchvision\n"
            "See https://pytorch.org/get-started/locally/ for platform-specific instructions."
        ) from None
    return torch


def _build_v200(channels: int, num_classes: int) -> Any:
    """Build the V200 CNN architecture matching T-Rex source exactly."""
    torch = _import_torch()
    nn = torch.nn

    return nn.Sequential(
        # Block 1
        nn.Conv2d(channels, 64, kernel_size=3, padding="same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(3),
        nn.Dropout2d(0.05),
        # Block 3
        nn.Conv2d(128, 256, kernel_size=3, padding="same"),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        # Block 4
        nn.Conv2d(256, 512, kernel_size=3, padding="same"),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(3),
        nn.Dropout2d(0.25),
        # Block 5
        nn.Conv2d(512, 512, kernel_size=3, padding="same"),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.MaxPool2d(3),
        nn.Dropout2d(0.05),
        # Head
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.05),
        nn.Linear(1024, num_classes),
    )


class _PermuteAxesWrapper:
    """Wraps V200 to accept (batch, H, W, C) uint8 input.

    Permutes to (batch, C, H, W), divides by 255, then applies ImageNet
    normalization (truncated to the number of input channels).
    """

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, model: Any, channels: int) -> None:
        self.model = model
        self.channels = channels
        torch = _import_torch()
        self.mean = torch.tensor(self.IMAGENET_MEAN[:channels]).reshape(1, -1, 1, 1)
        self.std = torch.tensor(self.IMAGENET_STD[:channels]).reshape(1, -1, 1, 1)

    def to(self, device: Any) -> _PermuteAxesWrapper:
        self.model = self.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def train(self) -> None:
        self.model.train()

    def eval(self) -> None:
        self.model.eval()

    def parameters(self) -> Any:
        return self.model.parameters()

    def state_dict(self) -> dict[str, Any]:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.model.load_state_dict(state_dict)

    def __call__(self, x: Any) -> Any:
        """Forward pass.

        Args:
            x: (batch, H, W, C) tensor with values 0-255.

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2).float()
        # Normalize
        x = x / 255.0
        x = (x - self.mean) / self.std
        return self.model(x)


class TRexIdentityNetwork:
    """V200 CNN classifier compatible with T-Rex visual identification.

    Wraps the exact V200 architecture used by T-Rex so that trained weights
    can be exported and loaded via ``visual_identification_model_path``.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 1,
        image_size: tuple[int, int] = (128, 128),
    ) -> None:
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size  # (height, width)

        raw_model = _build_v200(channels, num_classes)
        self._model = _PermuteAxesWrapper(raw_model, channels)
        self._device: Any = None
        self._epoch: int = 0
        self._best_accuracy: float = 0.0

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
            images: (N, H, W, C) uint8 array, values 0-255.
            labels: (N,) integer class labels.
            val_images: Optional validation set.
            val_labels: Optional validation labels.
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Batch size.
            device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

        Returns:
            Training history dict with keys ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc`` (per epoch).
        """
        torch = _import_torch()

        self._device = self._resolve_device(device)
        self._model.to(self._device)

        # Build datasets
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
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self._model.train()
        for epoch in range(1, epochs + 1):
            # --- Training ---
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

            # --- Validation ---
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

            # Progress
            if epoch % 10 == 0 or epoch == 1:
                msg = (
                    f"[identity-model] epoch {epoch}/{epochs}  "
                    f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                )
                if has_val:
                    msg += f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
                print(msg, file=sys.stderr)

        self._epoch = epochs
        return history

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Return per-class probabilities.

        Args:
            images: (N, H, W, C) uint8 array.

        Returns:
            (N, num_classes) float32 probability array.
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

    def export_trex_checkpoint(
        self, path: Path, video_name: str = "external"
    ) -> Path:
        """Save weights in T-Rex-compatible format.

        The checkpoint dict contains ``state_dict`` and ``metadata``.
        In T-Rex, load via::

            visual_identification_model_path = "/path/to/file"

        (without ``.pth`` extension -- T-Rex adds it automatically).

        Args:
            path: Output file path (will be ensured to end with ``.pth``).
            video_name: Video name stored in metadata. Default ``"external"``.

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
                "input_shape": (w, h, self.channels),  # width, height, channels
                "num_classes": self.num_classes,
                "video_name": video_name,
                "epoch": self._epoch,
                "uniqueness": self._best_accuracy,
            },
        }
        torch.save(checkpoint, path)
        print(
            f"[identity-model] Exported T-Rex checkpoint: {path}  "
            f"({self.num_classes} classes, epoch {self._epoch}, "
            f"acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        return path

    @classmethod
    def from_trex_checkpoint(cls, path: Path) -> TRexIdentityNetwork:
        """Load from a T-Rex-compatible .pth checkpoint.

        Args:
            path: Path to .pth checkpoint file.

        Returns:
            A ``TRexIdentityNetwork`` with loaded weights.
        """
        torch = _import_torch()

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["metadata"]

        w, h, channels = meta["input_shape"]
        num_classes = meta["num_classes"]

        net = cls(num_classes=num_classes, channels=channels, image_size=(h, w))
        net._model.load_state_dict(checkpoint["state_dict"])
        net._epoch = meta.get("epoch", 0)
        net._best_accuracy = meta.get("uniqueness", 0.0)
        return net

    # --- Private helpers ---

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
