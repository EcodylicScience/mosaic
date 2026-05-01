"""GlobalIdentityMegaDescriptor feature.

Sibling of
:class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel`
(V200) that uses the MegaDescriptor SwinV2 foundation model as a frozen
embedding extractor and predicts identities by cosine-similarity k-NN
against per-identity prototypes computed at fit time.

MegaDescriptor (Cermak et al., WACV 2024) is pretrained on a metadataset of
53 wildlife re-identification datasets and outperforms generic foundation
models (DINOv2, CLIP) on animal re-ID. No per-mouse training cycle is
required for the zero-shot baseline -- ``fit()`` only computes prototypes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, final

import cv2
import joblib
import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    DependencyLookup,
    InputRequire,
    Inputs,
    InputStream,
    Result,
)
from mosaic.core.pipeline.types.params import Params

from .registry import register_feature


@final
@register_feature
class GlobalIdentityMegaDescriptor:
    """Train an identity model using MegaDescriptor embeddings + k-NN.

    Takes EgocentricCrop output as input. Each identity is specified as a
    mapping of identity names to lists of sequences containing that
    individual alone. Computes a prototype embedding per identity from the
    training crops and predicts at inference by cosine k-NN against those
    prototypes.

    Example::

        ego_result = dataset.run_feature(
            EgocentricCrop(params={"crop_size": (384, 384)})
        )

        identity_model = GlobalIdentityMegaDescriptor(
            Inputs((Result(feature="egocentric-crop"),)),
            params={
                "identities": {
                    "mouse_A": ["cage1/day1_mouseA_alone"],
                    "mouse_B": ["cage1/day1_mouseB_alone"],
                    "mouse_C": ["cage1/day2_mouseC_alone"],
                    "mouse_D": ["cage1/day1_mouseD_alone"],
                },
                "model_name": "BVRA/MegaDescriptor-L-384",
                "image_size": (384, 384),
            },
        )
        result = dataset.run_feature(identity_model)

    Params:
        identities: Explicit identity -> sequences mapping.
        group_as_identity: Treat each group name as one identity. Default
            False.
        model_name: HuggingFace hub id of the MegaDescriptor variant.
            Default ``"BVRA/MegaDescriptor-L-384"``.
        image_size: Crop resize target ``(height, width)``. Default
            ``(384, 384)`` to match ``MegaDescriptor-L-384``.
        channels: Number of channels read from disk (1 = grayscale,
            3 = RGB). Grayscale inputs are replicated to 3 channels for
            the backbone. Default 3.
        batch_size: Embedding batch size. Default 32.
        max_images_per_identity: Cap on training crops per identity.
            Default 2000.
        crop_root: Optional EgocentricCrop output root override.
        weights_name: Stem of the exported ``.pth`` checkpoint. Default
            ``"megadescriptor_identity"``.
    """

    category = "global"
    name: str = "global-identity-megadescriptor"
    version: str = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        """MegaDescriptor identity model parameters."""

        # Primary: explicit identity -> sequences mapping
        identities: dict[str, list[str]] | None = None
        # Convenience shortcut: treat each group as one identity
        group_as_identity: bool = False

        # Backbone selection
        model_name: str = "BVRA/MegaDescriptor-L-384"
        image_size: tuple[int, int] = (384, 384)
        channels: int = 3

        # Inference
        batch_size: int = Field(default=32, ge=1)

        # Sampling
        max_images_per_identity: int = Field(default=2000, ge=1)

        # Export
        weights_name: str = "megadescriptor_identity"

        # Path to EgocentricCrop output root.
        crop_root: str | None = None

    def __init__(
        self,
        inputs: GlobalIdentityMegaDescriptor.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._network: object | None = None
        self._history: dict[str, list[float]] | None = None
        self._identity_names: list[str] | None = None

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        from mosaic.behavior.model_library.megadescriptor_identity import (
            MegaDescriptorNetwork,
        )

        self._network = None
        self._history = None
        self._identity_names = None

        cached_path = run_root / f"{self.params.weights_name}.pth"
        if cached_path.exists():
            self._network = MegaDescriptorNetwork.from_checkpoint(cached_path)
            history_path = run_root / "training_history.joblib"
            if history_path.exists():
                self._history = joblib.load(history_path)
            names_path = run_root / "identity_names.joblib"
            if names_path.exists():
                self._identity_names = joblib.load(names_path)
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        from mosaic.behavior.model_library.identity_common import (
            build_label_mapping,
            load_crop_frames,
        )
        from mosaic.behavior.model_library.megadescriptor_identity import (
            MegaDescriptorNetwork,
        )

        p = self.params

        seq_to_label, identity_names = build_label_mapping(p, inputs)
        self._identity_names = identity_names
        num_classes = len(identity_names)

        if num_classes < 2:
            msg = (
                f"[megadescriptor] Need at least 2 identities, "
                f"got {num_classes}: {identity_names}"
            )
            raise ValueError(msg)

        print(
            f"[megadescriptor] training with {num_classes} identities: "
            f"{identity_names}",
            file=sys.stderr,
        )

        # Collect images per identity
        all_images: dict[int, list[np.ndarray]] = {i: [] for i in range(num_classes)}
        for entry_key, df in inputs():
            label = seq_to_label.get(entry_key)
            if label is None:
                continue
            frames = load_crop_frames(
                entry_key,
                df,
                crop_root=p.crop_root,
                channels=p.channels,
                max_frames=p.max_images_per_identity,
            )
            if frames:
                all_images[label].extend(frames)

        # Cap per-identity and report counts
        images_list: list[np.ndarray] = []
        labels_list: list[int] = []
        for label_idx in range(num_classes):
            imgs = all_images[label_idx]
            if not imgs:
                print(
                    f"[megadescriptor] WARNING: no images for "
                    f"{identity_names[label_idx]}",
                    file=sys.stderr,
                )
                continue
            if len(imgs) > p.max_images_per_identity:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(imgs), p.max_images_per_identity, replace=False)
                imgs = [imgs[i] for i in indices]
            print(
                f"[megadescriptor]   {identity_names[label_idx]}: {len(imgs)} images",
                file=sys.stderr,
            )
            images_list.extend(imgs)
            labels_list.extend([label_idx] * len(imgs))

        if not images_list:
            msg = (
                "[megadescriptor] No images collected. Check sequence keys "
                "and crop output."
            )
            raise RuntimeError(msg)

        images_arr = np.stack(images_list, axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)

        # Resize if needed (matches V200's behavior)
        h, w = p.image_size
        if images_arr.shape[1] != h or images_arr.shape[2] != w:
            resized = np.empty(
                (len(images_arr), h, w, images_arr.shape[3]), dtype=np.uint8
            )
            for i in range(len(images_arr)):
                resized[i] = cv2.resize(
                    images_arr[i], (w, h), interpolation=cv2.INTER_LINEAR
                ).reshape(h, w, images_arr.shape[3])
            images_arr = resized

        # Hold out a small validation slice for top-1 reporting
        val_images: np.ndarray | None = None
        val_labels: np.ndarray | None = None
        if len(images_arr) > 10 * num_classes:
            rng = np.random.default_rng(42)
            n = len(images_arr)
            n_val = max(num_classes, int(n * 0.1))
            perm = rng.permutation(n)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            val_images = images_arr[val_idx]
            val_labels = labels_arr[val_idx]
            images_arr = images_arr[train_idx]
            labels_arr = labels_arr[train_idx]

        self._network = MegaDescriptorNetwork(
            model_name=p.model_name,
            image_size=p.image_size,
        )
        self._history = self._network.fit(
            images_arr,
            labels_arr,
            val_images=val_images,
            val_labels=val_labels,
            num_classes=num_classes,
            batch_size=p.batch_size,
        )
        # Stash identity names on the network for checkpoint export.
        self._network._identity_names = identity_names  # pyright: ignore[reportPrivateUsage]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Passthrough -- identity predictions are consumed downstream."""
        return df

    def save_state(self, run_root: Path) -> None:
        from mosaic.behavior.model_library.megadescriptor_identity import (
            MegaDescriptorNetwork,
        )

        if self._network is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        if isinstance(self._network, MegaDescriptorNetwork):
            self._network.export_checkpoint(
                run_root / f"{self.params.weights_name}.pth"
            )

        if self._history is not None:
            joblib.dump(self._history, run_root / "training_history.joblib")

        if self._identity_names is not None:
            joblib.dump(self._identity_names, run_root / "identity_names.joblib")
