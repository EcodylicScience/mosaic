"""T-Rex-compatible V200 CNN identity classifier.

Provides a PyTorch CNN matching the ``V200`` architecture used by T-Rex for
visual individual identification, including its module tree and layer names, so
trained weights can be exported as ``.pth`` checkpoints that T-Rex loads via
``visual_identification_model_path``.

Two things to know before exporting a model for T-Rex:

- T-Rex loads checkpoints with ``load_state_dict(strict=False)`` and only
  *warns* on a key mismatch. A checkpoint whose keys do not line up therefore
  produces a randomly-initialised network and a log line, not an error. The
  acceptance signal for a working export is the **absence** of that warning.
- T-Rex chooses which architecture to build from its own
  ``visual_identification_version`` setting, not from the checkpoint. Exporting
  a V200 and tracking with ``visual_identification_version = v118_3`` fails the
  same silent way. :meth:`TRexIdentityNetwork.export_trex_checkpoint` records
  the architecture in metadata and says which setting to use.

The shared architecture and checkpoint machinery lives in
:mod:`trex_identity_architectures`, which also carries the T-Rex attribution
this file inherits: the ``V200`` layout is T-Rex's, by Tristan Walter and
contributors, under AGPL-3.0-or-later, and company use of T-Rex itself requires
a paid commercial license from that project. This module reimplements a
published layer layout and contains no T-Rex source code.
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
    build_v200,
    build_wrapper,
    detect_input_normalization,
    infer_v200_dims,
    is_legacy_sequential,
    load_into_wrapper,
    pack_input_shape,
    remap_legacy_sequential_keys,
    split_checkpoint,
    unpack_input_shape,
)

__all__ = ["TRexIdentityNetwork"]

#: Bumped when the exported checkpoint layout changes in a way a reader must
#: know about. 1 = named ``model.*`` keys, optional ``normalize.*`` buffers,
#: and the ``input_normalization`` / ``architecture_version`` metadata.
CHECKPOINT_FORMAT_VERSION = 1


class TRexIdentityNetwork:
    """V200 CNN classifier compatible with T-Rex visual identification.

    Mirrors T-Rex's ``V200`` module tree exactly -- ``normalize`` and ``model``
    children, named conv/bn layers -- so state_dicts are interchangeable in both
    directions.

    Args:
        num_classes: Number of identities.
        channels: Input channels (1 = grayscale, 3 = RGB).
        image_size: ``(height, width)`` of the training crops.
        input_normalization: Which preprocessing contract this network expects.
            ``"imagenet_scaled"`` computes ``(x / 255 - mean) / std``;
            ``"raw255"`` passes ``[0, 255]`` through untouched. This must match
            the T-Rex build the weights will be used with -- see
            :func:`~.trex_identity_architectures.detect_input_normalization`.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int = 1,
        image_size: tuple[int, int] = (128, 128),
        input_normalization: InputNormalization = DEFAULT_INPUT_NORMALIZATION,
    ) -> None:
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size  # (height, width)
        self.input_normalization: InputNormalization = input_normalization

        self._model = build_wrapper(
            build_v200(channels, num_classes), channels, input_normalization
        )
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
        torch = import_torch()

        self._device = self._resolve_device(device)
        self._model.to(self._device)

        # Build datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(images),
            torch.from_numpy(labels.astype(np.int64)),
        )
        # A trailing batch of one would crash `bn6` (BatchNorm1d needs >1 sample
        # in training mode), so drop it -- and only it.
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

    def export_trex_checkpoint(
        self,
        path: Path,
        video_name: str = "external",
        *,
        class_labels: list[str] | None = None,
    ) -> Path:
        """Save weights in T-Rex-compatible format.

        The checkpoint dict contains ``state_dict`` and ``metadata``. In T-Rex::

            visual_identification_version    = v200
            visual_identification_model_path = "/path/to/file"

        Args:
            path: Output file path (will be ensured to end with ``.pth``).
            video_name: Video name stored in metadata. Default ``"external"``.
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
            "model_type": "v200",
            "architecture_version": "v200",
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
            f"[identity-model] Exported T-Rex checkpoint: {path}  "
            f"({self.num_classes} classes, epoch {self._epoch}, "
            f"acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        print(
            f"[identity-model] In T-Rex, set visual_identification_version = v200 "
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
    ) -> TRexIdentityNetwork:
        """Load from a T-Rex-compatible ``.pth`` checkpoint.

        Accepts T-Rex's own exports, mosaic's exports, and -- with a
        ``DeprecationWarning`` -- checkpoints written by mosaic <= 0.7, whose
        V200 was an ``nn.Sequential`` with positional keys.

        Args:
            path: Path to the ``.pth`` checkpoint.
            input_normalization: Override for files that do not state their own
                preprocessing contract. Ignored when the file carries
                ``normalize.*`` buffers, which are authoritative.

        Returns:
            A ``TRexIdentityNetwork`` with loaded weights.
        """
        torch = import_torch()

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        sd, meta = split_checkpoint(
            torch.load(path, map_location="cpu", weights_only=False), path
        )

        legacy = is_legacy_sequential(sd)
        if legacy:
            warnings.warn(
                f"{path.name} uses mosaic <= 0.7's positional V200 keys "
                f"('0.weight'); remapping onto the named layout. Re-export it "
                f"with mosaic >= 0.8 -- this shim is removed at 0.9.",
                DeprecationWarning,
                stacklevel=2,
            )
            sd = remap_legacy_sequential_keys(sd)

        channels, num_classes = infer_v200_dims(sd)

        square_hint = None
        if "input_shape" in meta:
            image_size = unpack_input_shape(
                tuple(meta["input_shape"]), channels, square_hint=square_hint
            )
        else:
            # V200's global average pool erases the input size, so a file with
            # no metadata cannot tell us; fall back to the constructor default.
            image_size = (128, 128)
            warnings.warn(
                f"{path.name} has no input_shape metadata and V200's global "
                f"average pool leaves no trace of it in the weights; assuming "
                f"{image_size}. Resize crops to the size this model was "
                f"trained at before calling predict().",
                UserWarning,
                stacklevel=2,
            )

        # A file mosaic wrote before 0.8 is not ambiguous: mosaic always scaled.
        mode = (
            "imagenet_scaled"
            if legacy
            else detect_input_normalization(
                sd, meta, input_normalization, source=path.name
            )
        )

        net = cls(
            num_classes=num_classes,
            channels=channels,
            image_size=image_size,
            input_normalization=mode,
        )
        load_into_wrapper(
            net._model, align_normalize_buffers(sd, channels), source=path.name
        )
        net._epoch = int(meta.get("epoch", 0) or 0)
        net._best_accuracy = float(meta.get("uniqueness", 0.0) or 0.0)
        return net

    # --- Private helpers ---

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
