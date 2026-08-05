"""Supervised identity classification over a pretrained image backbone.

A timm backbone with a linear classification head on top, trained with
cross-entropy against a closed set of individually-known animals. ``predict()``
returns a softmax over those identities.

This is the trained counterpart to
:class:`~mosaic.behavior.model_library.identity_embedding.EmbeddingIdentityNetwork`,
which freezes the same family of backbones and decides identity by k-NN against
per-identity prototypes without training anything. Which to reach for is a
question of how much labelled crop data there is per animal: the embedding model
answers in one pass and is the right first attempt, and a trained head starts to
win once each identity has enough crops to fit one.

**The backbone is frozen by default.** Training only the head is fast, needs far
less data than fine-tuning, and cannot damage a pretrained representation that is
already strong on animal appearance. ``freeze_backbone=False`` fine-tunes end to
end, which is worth trying when there is a lot of data and the animals differ in
ways ImageNet features do not capture.

That choice also decides what a checkpoint holds. A frozen run saves the head
alone and refetches the backbone by name on load, so the file is kilobytes rather
than hundreds of megabytes; a fine-tuned run saves everything, because its
backbone is no longer the one the hub would hand back.

Mosaic distributes no model weights. Whatever ``model_name`` names is fetched at
run time under its own license -- see
:mod:`mosaic.behavior.model_library.timm_backbone` and ``docs/licensing.md``.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy as np

from mosaic.behavior.model_library.timm_backbone import (
    DEFAULT_MODEL_NAME,
    BackboneDataConfig,
    as_sequence,
    data_config_from_metadata,
    import_timm,
    import_torch,
    normalization_tensors,
    preprocess_batch,
    resolve_backbone_data_config,
    resolve_device,
    resolve_timm_model_id,
)

__all__ = ["CHECKPOINT_FORMAT_VERSION", "ClassifierIdentityNetwork"]

CHECKPOINT_FORMAT_VERSION: Final = 1
"""Layout of the exported ``.pth``.

Bumped when the stored keys or metadata change shape, so a stale file is
rejected outright rather than loading into a network that reads it differently.
"""

_TRAINING_SEED: Final = 42
"""Seeds the shuffle and the training-time RNG.

Mosaic requires identical params and inputs to produce identical outputs, and a
training loop has three sources of randomness: which order the samples arrive
in, how the head starts, and dropout. This covers the first and the third;
:data:`_HEAD_INIT_SEED` covers the second.
"""

_HEAD_INIT_SEED: Final = 1234
"""Seeds the classification head's initial weights.

Distinct from :data:`_TRAINING_SEED` so that changing how training is shuffled
never silently changes where training starts from.
"""


def build_classifier(backbone: Any, embedding_dim: int, num_classes: int) -> Any:
    """Assemble the backbone and a fresh linear head into one module.

    Defined inside a function, like every other ``nn.Module`` in this package,
    so the module imports without PyTorch installed.

    The two children are named ``backbone`` and ``head``, which is what makes
    ``state_dict()`` separable: a frozen run filters to the ``head.`` keys and
    stores only those.

    The head's weights are drawn from a seeded generator rather than left to
    ``nn.Linear``'s own initialization, which reads torch's *global* RNG --
    unseeded, so two runs with identical params would otherwise start from
    different weights and end at different predictions. The distribution is
    unchanged: ``nn.Linear`` draws both weight and bias uniformly from
    ``+/- 1/sqrt(fan_in)``, and so does this.

    Args:
        backbone: A constructed timm model with its classifier removed, so it
            emits pooled features.
        embedding_dim: Width of those pooled features.
        num_classes: Number of identities to score.

    Returns:
        The assembled ``nn.Module``.
    """
    torch = import_torch()
    nn = torch.nn

    class IdentityClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(embedding_dim, num_classes)

        def forward(self, x: Any) -> Any:
            return self.head(self.backbone(x))

    model = IdentityClassifier()

    bound = 1.0 / math.sqrt(embedding_dim)
    generator = torch.Generator().manual_seed(_HEAD_INIT_SEED)
    with torch.no_grad():
        _ = model.head.weight.uniform_(-bound, bound, generator=generator)
        _ = model.head.bias.uniform_(-bound, bound, generator=generator)
    return model


class ClassifierIdentityNetwork:
    """Pretrained image backbone + a trained linear identity head.

    Args:
        num_classes: Number of identities to distinguish.
        model_name: A bare timm architecture tag or a Hugging Face hub id.
            Defaults to :data:`~mosaic.behavior.model_library.timm_backbone.DEFAULT_MODEL_NAME`.
            Mosaic ships no weights; whatever is named here is downloaded at
            run time under its own license.
        image_size: Input ``(height, width)`` override. Defaults to None,
            meaning follow the backbone's declared input size -- the only value
            correct for every backbone.
        freeze_backbone: Train the head alone. Default True.
        device: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
        data_config: Preprocessing recipe to use instead of resolving one from
            the backbone. Set when reloading a checkpoint, so a fitted model
            reproduces the preprocessing it was fitted with even if the
            upstream repository has changed underneath it.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = DEFAULT_MODEL_NAME,
        image_size: tuple[int, int] | None = None,
        *,
        freeze_backbone: bool = True,
        device: str = "auto",
        data_config: BackboneDataConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self._device: Any = resolve_device(device)

        timm = import_timm()
        torch = import_torch()

        backbone = timm.create_model(
            resolve_timm_model_id(model_name),
            pretrained=True,
            num_classes=0,  # remove timm's own classifier; the head replaces it
        )
        backbone.eval()

        resolved = data_config or resolve_backbone_data_config(backbone)
        if image_size is not None:
            resolved = replace(resolved, image_size=image_size)
        self._data_config: BackboneDataConfig = resolved
        self.image_size: tuple[int, int] = resolved.image_size
        self._mean, self._std = normalization_tensors(resolved)

        # A real self-check, not incidental: a backbone declaring
        # ``fixed_input_size`` raises here rather than at the first real batch
        # if resolution produced a size it cannot accept. Probed on CPU, before
        # the move, so it costs no device memory.
        with torch.no_grad():
            probe = torch.zeros(1, 3, self.image_size[0], self.image_size[1])
            feat = backbone(probe)
        self.embedding_dim: int = int(feat.shape[-1])

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        self._model: Any = build_classifier(
            backbone, self.embedding_dim, num_classes
        ).to(self._device)

        self._identity_names: list[str] | None = None
        self._best_accuracy: float = 0.0

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        *,
        val_images: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 32,
    ) -> dict[str, list[float]]:
        """Train the head (and the backbone, if unfrozen) on labelled crops.

        Crops are preprocessed one batch at a time rather than up front. The
        normalized float32 form of a full training set is orders of magnitude
        larger than its uint8 source, so materializing it would exhaust memory
        on exactly the datasets this is worth running on.

        Args:
            images: ``(N, H, W, C)`` uint8 training crops.
            labels: ``(N,)`` integer class labels.
            val_images: Optional validation crops.
            val_labels: Optional validation labels.
            epochs: Passes over the training set.
            lr: Adam learning rate.
            batch_size: Samples per step.

        Returns:
            Per-epoch history under the keys ``train_loss``, ``train_acc``,
            ``val_loss`` and ``val_acc``.
        """
        torch = import_torch()

        # Dropout in an unfrozen backbone reads torch's global RNG, so seed it:
        # without this a fine-tuning run is not reproducible even though the
        # shuffle and the head's starting weights both are.
        _ = torch.manual_seed(_TRAINING_SEED)

        labels = np.asarray(labels, dtype=np.int64)
        n = len(images)
        trainable = [p for p in self._model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        rng = np.random.default_rng(_TRAINING_SEED)

        # A trailing batch of exactly one sample puts any batch-norm layer in a
        # fine-tuned backbone into a state it cannot compute, so drop it. Only
        # the size-one case: dropping more would discard real training data.
        drop_last_single = n % batch_size == 1

        history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        for epoch in range(epochs):
            self._set_training_mode()
            order = rng.permutation(n)
            total_loss = 0.0
            correct = 0
            seen = 0

            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 1 and drop_last_single:
                    continue
                batch = self._to_device(images[idx])
                target = torch.from_numpy(labels[idx]).to(self._device)

                optimizer.zero_grad()
                logits = self._model(batch)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach()) * len(idx)
                correct += int((logits.detach().argmax(dim=1) == target).sum())
                seen += len(idx)

            train_loss = total_loss / max(seen, 1)
            train_acc = correct / max(seen, 1)

            val_loss = 0.0
            val_acc = 0.0
            if val_images is not None and val_labels is not None and len(val_images):
                val_loss, val_acc = self._evaluate(
                    val_images,
                    np.asarray(val_labels, dtype=np.int64),
                    criterion,
                    batch_size,
                )
                self._best_accuracy = max(self._best_accuracy, val_acc)
            else:
                self._best_accuracy = max(self._best_accuracy, train_acc)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(
                    f"[identity-classifier] epoch {epoch + 1}/{epochs}  "
                    f"loss={train_loss:.4f} acc={train_acc:.4f}  "
                    f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
                    file=sys.stderr,
                )

        return history

    def predict(self, images: np.ndarray, *, batch_size: int = 32) -> np.ndarray:
        """Score *images* against the trained identities.

        Args:
            images: ``(N, H, W, C)`` uint8 array. Grayscale is replicated to
                3 channels and the batch is resized to the trained input size.
            batch_size: Inference batch size.

        Returns:
            ``(N, num_classes)`` float32 probabilities, each row summing to 1.
        """
        torch = import_torch()

        self._model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                batch = self._to_device(images[start : start + batch_size])
                logits = self._model(batch)
                probs = torch.softmax(logits, dim=1)
                out.append(probs.detach().cpu().float().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def export_checkpoint(
        self, path: Path, *, class_labels: list[str] | None = None
    ) -> Path:
        """Save the trained weights and the recipe needed to replay them.

        Under a frozen backbone only the ``head.`` keys are written; the
        backbone is refetched by ``model_name`` on load. The resolved
        preprocessing recipe travels with the file either way, so a reload
        reproduces the numbers this fit produced even if the backbone's
        upstream repository later edits its declared config.

        Args:
            path: Output file path. ``.pth`` is appended if missing.
            class_labels: Identity names, in the class order the head was
                trained with. Stored because the head scores indices and
                nothing else records what animal an index means.

        Returns:
            The resolved path the checkpoint was saved to.
        """
        torch = import_torch()

        path = Path(path)
        if path.suffix != ".pth":
            path = path.with_suffix(".pth")
        path.parent.mkdir(parents=True, exist_ok=True)

        state: dict[str, Any] = dict(self._model.state_dict())
        if self.freeze_backbone:
            state = {k: v for k, v in state.items() if k.startswith("head.")}

        names = list(class_labels) if class_labels else list(self._identity_names or ())
        checkpoint = {
            "state_dict": state,
            # Primitives only: this is read back under ``weights_only=True``
            # elsewhere in the ecosystem, which refuses arbitrary objects.
            "metadata": {
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_name": self.model_name,
                "image_size": list(self.image_size),
                "mean": list(self._data_config.mean),
                "std": list(self._data_config.std),
                "num_classes": self.num_classes,
                "embedding_dim": self.embedding_dim,
                "freeze_backbone": self.freeze_backbone,
                "class_labels": names,
                "accuracy": self._best_accuracy,
            },
        }
        torch.save(checkpoint, path)
        print(
            f"[identity-classifier] exported checkpoint: {path}  "
            f"({self.num_classes} identities, backbone="
            f"{'frozen' if self.freeze_backbone else 'fine-tuned'}, "
            f"acc={self._best_accuracy:.4f})",
            file=sys.stderr,
        )
        return path

    @classmethod
    def from_checkpoint(cls, path: Path) -> ClassifierIdentityNetwork:
        """Rebuild a network from a checkpoint :meth:`export_checkpoint` wrote.

        Args:
            path: Path to the ``.pth`` file.

        Returns:
            The reconstructed network, ready to :meth:`predict`.

        Raises:
            ValueError: If the file is not :data:`CHECKPOINT_FORMAT_VERSION`,
                or if its weights do not match the architecture its own
                metadata describes.
        """
        torch = import_torch()

        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".pth")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint["metadata"]

        found = int(meta.get("format_version", 0))
        if found != CHECKPOINT_FORMAT_VERSION:
            msg = (
                f"[identity-classifier] checkpoint at {path} is format v{found}; "
                f"this network reads v{CHECKPOINT_FORMAT_VERSION}. Refit under "
                "the current version."
            )
            raise ValueError(msg)

        frozen = bool(meta.get("freeze_backbone", True))
        net = cls(
            num_classes=int(meta["num_classes"]),
            model_name=str(meta["model_name"]),
            freeze_backbone=frozen,
            data_config=data_config_from_metadata(meta),
        )

        # Loaded permissively, then judged strictly. A frozen checkpoint
        # legitimately omits every ``backbone.`` key, so ``strict=True`` would
        # reject a valid file -- but letting the rest slide is how a partially
        # loaded network passes for a working one.
        incompatible = net._model.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            k
            for k in incompatible.missing_keys
            if not (frozen and k.startswith("backbone."))
        ]
        if missing or unexpected:
            msg = (
                f"[identity-classifier] checkpoint at {path} does not match the "
                f"architecture its metadata describes "
                f"({meta['model_name']}, {meta['num_classes']} classes). "
                f"missing={missing} unexpected={unexpected}"
            )
            raise ValueError(msg)

        stored_names = as_sequence(meta.get("class_labels")) or ()
        net._identity_names = [str(n) for n in stored_names] or None
        net._best_accuracy = float(meta.get("accuracy", 0.0))
        return net

    # --- Internal ---

    def _set_training_mode(self) -> None:
        """Put the model in train mode, keeping a frozen backbone in eval.

        A frozen backbone left in train mode would still update its batch-norm
        running statistics and apply dropout, so it would not be frozen in the
        sense that matters -- its outputs would drift between epochs while its
        weights stayed put.
        """
        self._model.train()
        if self.freeze_backbone:
            self._model.backbone.eval()

    def _to_device(self, images: np.ndarray) -> Any:
        """Preprocess a uint8 crop batch and move it to the compute device."""
        batch = preprocess_batch(images, self.image_size, self._mean, self._std)
        return batch.to(self._device)

    def _evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        criterion: Any,
        batch_size: int,
    ) -> tuple[float, float]:
        """Mean loss and top-1 accuracy over *images*."""
        torch = import_torch()

        self._model.eval()
        total_loss = 0.0
        correct = 0
        n = len(images)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                stop = start + batch_size
                batch = self._to_device(images[start:stop])
                target = torch.from_numpy(labels[start:stop]).to(self._device)
                logits = self._model(batch)
                total_loss += float(criterion(logits, target)) * len(target)
                correct += int((logits.argmax(dim=1) == target).sum())
        return total_loss / max(n, 1), correct / max(n, 1)
