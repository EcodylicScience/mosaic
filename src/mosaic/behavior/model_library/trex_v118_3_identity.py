"""T-Rex ``V118_3`` identity network (3 conv blocks + 2 fc layers).

This is the compact architecture T-Rex ships alongside the deeper ``V200``
(:mod:`trex_identity_network`): three ``conv -> bn -> relu -> pool(2) ->
dropout`` blocks, a direct flatten, then ``fc1 -> bn4 -> relu -> dropout ->
fc2``. It is what most real T-Rex identity checkpoints in circulation use.

T-Rex has shipped more than one ``V118_3`` -- ``conv3`` has had both 100 and
128 output channels -- so the layer widths are read off the checkpoint by
:func:`~.trex_identity_architectures.infer_v118_3_dims` rather than hard-coded.
One class therefore loads every variant.

Two properties are worth stating because getting them wrong is silent:

- ``bn4`` is ``nn.LayerNorm``. It and ``nn.BatchNorm1d(..., track_running_stats=False)``
  have identical state_dicts -- ``weight`` and ``bias``, no running statistics --
  so a checkpoint cannot distinguish them, and the wrong one loads cleanly while
  normalizing across the batch instead of across features.
- The network is **not** spatial-dimension-agnostic. There is no global average
  pool; ``fc1`` consumes the flattened conv output, so inputs must be resized to
  the trained ``image_size`` before inference.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .trex_identity_architectures import (
    DEFAULT_INPUT_NORMALIZATION,
    InputNormalization,
    import_torch,
    align_normalize_buffers,
    build_v118_3,
    build_wrapper,
    detect_input_normalization,
    infer_v118_3_dims,
    load_into_wrapper,
    pack_input_shape,
    split_checkpoint,
    unpack_input_shape,
)
from .trex_identity_network import CHECKPOINT_FORMAT_VERSION

__all__ = ["TRexV118_3IdentityNetwork"]


class TRexV118_3IdentityNetwork:
    """T-Rex ``V118_3`` identity classifier.

    Use :meth:`from_trex_checkpoint` to load a T-Rex ``.pth`` directly; the
    architecture (conv channel counts, kernel size, FC1 hidden size,
    num_classes, channels) is read off the file's shapes, so no manual
    configuration is needed and every ``V118_3`` variant is covered.

    Public API mirrors :class:`~.trex_identity_network.TRexIdentityNetwork`:

    - :meth:`predict` returns ``(N, num_classes)`` softmax probabilities.
    - :meth:`fit` trains with Adam + CrossEntropyLoss.
    - :meth:`export_trex_checkpoint` writes a ``{state_dict, metadata}`` ``.pth``.

    Args:
        num_classes: Number of identities.
        channels: Input channels (1 = grayscale, 3 = RGB).
        image_size: ``(height, width)`` of the training crops. Inputs must be
            resized to this before inference -- see the module docstring.
        conv_channels: ``(C1, C2, C3)`` for the three conv blocks.
        fc_hidden: Hidden width of ``fc1``.
        flatten_dim: ``fc1.in_features``. Derived from ``image_size`` when None.
        kernel_size: Conv kernel size, shared across the three blocks.
        input_normalization: Preprocessing contract; see
            :class:`~.trex_identity_network.TRexIdentityNetwork`.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 1,
        image_size: tuple[int, int] = (80, 80),
        conv_channels: tuple[int, int, int] = (16, 64, 100),
        fc_hidden: int = 100,
        flatten_dim: int | None = None,
        kernel_size: int = 5,
        input_normalization: InputNormalization = DEFAULT_INPUT_NORMALIZATION,
    ) -> None:
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size  # (height, width)
        self.conv_channels = conv_channels
        self.fc_hidden = fc_hidden
        self.kernel_size = kernel_size
        self.input_normalization: InputNormalization = input_normalization

        # Three MaxPool2d(2) layers => post-conv H = H_in // 8 (with floor).
        if flatten_dim is None:
            h, w = image_size
            flatten_dim = (h // 8) * (w // 8) * conv_channels[-1]
        self.flatten_dim = flatten_dim

        self._model = build_wrapper(
            build_v118_3(
                channels=channels,
                conv_channels=conv_channels,
                flatten_dim=flatten_dim,
                fc_hidden=fc_hidden,
                num_classes=num_classes,
                kernel_size=kernel_size,
            ),
            channels,
            input_normalization,
        )
        self._device: Any = None
        self._epoch: int = 0
        self._best_accuracy: float = 0.0

    # --- Training ---------------------------------------------------------

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
            images: ``(N, H, W, C)`` ``uint8`` array, values 0-255.
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
        torch = import_torch()

        self._device = self._resolve_device(device)
        self._model.to(self._device)

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(images),
            torch.from_numpy(labels.astype(np.int64)),
        )
        # `bn1`-`bn3` are BatchNorm2d and need more than one sample in training
        # mode, so drop a trailing batch of exactly one -- and only that.
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_dataset) % batch_size == 1,
        )

        has_val = val_images is not None and val_labels is not None
        val_loader = None
        if val_images is not None and val_labels is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(val_images),
                torch.from_numpy(val_labels.astype(np.int64)),
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
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
                    f"[v118_3] epoch {epoch}/{epochs}  "
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
            images: ``(N, H, W, C)`` ``uint8`` array, resized to the trained
                ``image_size``.

        Returns:
            ``(N, num_classes)`` ``float32`` probability array.
        """
        torch = import_torch()

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
        self,
        path: Path,
        video_name: str = "external",
        *,
        class_labels: list[str] | None = None,
    ) -> Path:
        """Save weights in T-Rex-compatible format.

        In T-Rex::

            visual_identification_version    = v118_3
            visual_identification_model_path = "/path/to/file"

        Args:
            path: Output file path (``.pth`` appended if missing).
            video_name: Stored in metadata for traceability.
            class_labels: Identity names in class order. T-Rex assigns
                identities by softmax index and does not preserve labels, so
                recording them here is the only link back to the animals.

        Returns:
            The resolved path the checkpoint was saved to.
        """
        torch = import_torch()

        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any] = {
            "input_shape": pack_input_shape(self.image_size, self.channels),
            "num_classes": self.num_classes,
            "video_name": video_name,
            "epoch": self._epoch,
            "uniqueness": self._best_accuracy,
            "conv_channels": list(self.conv_channels),
            "flatten_dim": self.flatten_dim,
            "fc_hidden": self.fc_hidden,
            "kernel_size": self.kernel_size,
            "model_type": "v118_3",
            "architecture_version": "v118_3",
            "input_normalization": self.input_normalization,
            "mosaic_checkpoint_version": CHECKPOINT_FORMAT_VERSION,
        }
        if class_labels is not None:
            metadata["class_labels"] = list(class_labels)

        # Metadata stays primitive: T-Rex reads these files with
        # `weights_only=True`, which rejects arbitrary pickled objects.
        checkpoint = {"state_dict": self._model.state_dict(), "metadata": metadata}
        torch.save(checkpoint, path)

        print(
            f"[v118_3] Exported T-Rex checkpoint: {path}  "
            f"({self.num_classes} classes, channels={self.channels}, "
            f"conv={self.conv_channels}, fc_hidden={self.fc_hidden}, "
            f"epoch={self._epoch}, acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        print(
            f"[v118_3] In T-Rex, set visual_identification_version = v118_3 "
            f"(the architecture is chosen by that setting, not by this file) and "
            f"visual_identification_model_path = {path.with_suffix('')}",
            file=sys.stderr,
        )
        if self.input_normalization == "imagenet_scaled":
            warnings.warn(
                "exported with input_normalization='imagenet_scaled' "
                "((x/255 - mean)/std). Some T-Rex builds pass inputs through "
                "unnormalized; against such a build this model will predict "
                "nonsense. Check that build's Normalize.forward and re-train "
                "with input_normalization='raw255' if it returns x unchanged.",
                UserWarning,
                stacklevel=2,
            )
        return path

    @classmethod
    def from_trex_checkpoint(
        cls,
        path: Path,
        *,
        input_normalization: InputNormalization | None = None,
    ) -> TRexV118_3IdentityNetwork:
        """Load a T-Rex ``.pth``, reading the architecture off its shapes.

        Handles the ``{"state_dict", "metadata"}`` wrapper T-Rex and mosaic
        write, the same wrapper without metadata, and a bare state_dict.

        Args:
            path: Path to the ``.pth`` file.
            input_normalization: Override for files that do not state their own
                preprocessing contract. Ignored when the file carries
                ``normalize.*`` buffers, which are authoritative.

        Returns:
            A :class:`TRexV118_3IdentityNetwork` with weights loaded.

        Raises:
            ValueError: when the architecture cannot be read from the
                state_dict, or the keys do not match it.
        """
        torch = import_torch()

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        sd, meta = split_checkpoint(
            torch.load(path, map_location="cpu", weights_only=False), path
        )

        (
            channels,
            conv_channels,
            flatten_dim,
            fc_hidden,
            num_classes,
            spatial_pixels,
            kernel_size,
        ) = infer_v118_3_dims(sd)

        image_size = cls._resolve_image_size(meta, channels, spatial_pixels)
        mode = detect_input_normalization(
            sd, meta, input_normalization, source=path.name
        )

        net = cls(
            num_classes=num_classes,
            channels=channels,
            image_size=image_size,
            conv_channels=conv_channels,
            fc_hidden=fc_hidden,
            flatten_dim=flatten_dim,
            kernel_size=kernel_size,
            input_normalization=mode,
        )
        load_into_wrapper(
            net._model, align_normalize_buffers(sd, channels), source=path.name
        )
        net._epoch = int(meta.get("epoch", 0) or 0)
        net._best_accuracy = float(meta.get("uniqueness", 0.0) or 0.0)

        print(
            f"[v118_3] Loaded {path.name}  "
            f"channels={channels}  conv={conv_channels}  k={kernel_size}  "
            f"flatten_dim={flatten_dim}  fc_hidden={fc_hidden}  "
            f"num_classes={num_classes}  image_size={image_size}  "
            f"normalization={mode}",
            file=sys.stderr,
        )
        return net

    # --- Internals --------------------------------------------------------

    @staticmethod
    def _resolve_image_size(
        meta: dict[str, Any], channels: int, spatial_pixels: int
    ) -> tuple[int, int]:
        """Recover ``(height, width)`` from metadata, cross-checked on the weights.

        Unlike V200, this architecture leaves a trace of the input size in the
        weights: ``fc1.in_features`` pins ``(H // 8) * (W // 8)``. That product
        validates the *pair* but not its order -- it is symmetric under a swap --
        so orientation comes from provenance instead:

        - ``"architecture": "v200-native"`` in the metadata is the fingerprint of
          mosaic <= 0.7's exporter, the one place ``(H, W, C)`` was written.
        - everything else follows T-Rex's ``(W, H, C)``.

        A shape that reproduces neither reading is reported and the square
        reading from the weights is used instead.
        """
        side = int(round(spatial_pixels**0.5))
        square = (side * 8, side * 8) if side * side == spatial_pixels else None

        raw = meta.get("input_shape")
        if raw is None:
            if square is None:
                raise ValueError(
                    f"checkpoint has no input_shape metadata and a non-square "
                    f"post-conv map ({spatial_pixels} px); cannot infer image_size"
                )
            return square

        shape = tuple(int(x) for x in raw)
        legacy_mosaic = meta.get("architecture") == "v200-native"
        if len(shape) == 3:
            a, b, _ = shape
            candidate = (a, b) if legacy_mosaic else (b, a)
        elif len(shape) == 2:
            candidate = (shape[0], shape[1])
        else:
            candidate = None

        if candidate is not None:
            h, w = candidate
            if (h // 8) * (w // 8) == spatial_pixels:
                return candidate
            warnings.warn(
                f"input_shape {shape} implies image_size {candidate}, which "
                f"does not reproduce the post-conv map ({spatial_pixels} px); "
                f"falling back to the weights. Crops may need resizing before "
                f"predict().",
                UserWarning,
                stacklevel=3,
            )

        if square is not None:
            return square
        return unpack_input_shape(shape, channels, trust_orientation=not legacy_mosaic)

    @staticmethod
    def _resolve_device(device: str) -> Any:
        """Resolve device string to a torch.device."""
        torch = import_torch()

        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
