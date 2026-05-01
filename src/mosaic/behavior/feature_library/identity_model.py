"""GlobalIdentityModel feature.

Trains a T-Rex-compatible visual identification model from egocentric crop
images of individual animals. Uses the V200 CNN architecture to produce
weights loadable via T-Rex's ``visual_identification_model_path`` setting.
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

from mosaic.core.helpers import make_entry_key
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
class GlobalIdentityModel:
    """Train a visual identity model from individual animal sequences.

    Takes EgocentricCrop output as input. Each identity is specified as a
    mapping of identity names to lists of sequences containing that
    individual alone. Trains a V200 CNN classifier (T-Rex-compatible)
    and exports weights loadable via ``visual_identification_model_path``.

    Example::

        ego_result = dataset.run_feature(ego_crop)

        identity_model = GlobalIdentityModel(
            Inputs((Result(feature="egocentric-crop"),)),
            params={
                "identities": {
                    "mouse_A": ["cage1/day1_mouseA_alone", "cage1/day3_mouseA_alone"],
                    "mouse_B": ["cage1/day1_mouseB_alone"],
                    "mouse_C": ["cage1/day2_mouseC_alone"],
                    "mouse_D": ["cage1/day1_mouseD_alone"],
                },
                "image_size": (128, 128),
                "channels": 1,
            },
        )
        result = dataset.run_feature(identity_model)

    Params:
        identities: Explicit identity -> sequences mapping. Keys are
            identity names, values are lists of "group/sequence" strings.
        group_as_identity: Convenience shortcut -- treat each group name
            as one identity. Default False.
        image_size: Crop resize target (height, width). Default (128, 128).
        channels: Number of image channels (1=grayscale, 3=color).
            Default 1.
        epochs: Training epochs. Default 150.
        learning_rate: Adam learning rate. Default 0.0001.
        batch_size: Training batch size. Default 64.
        val_split: Fraction of data reserved for validation. Default 0.2.
        max_images_per_identity: Cap on images per identity to balance
            classes. Default 2000.
        export_trex_weights: Save a T-Rex-loadable .pth file. Default True.
        trex_weights_name: Stem of the exported .pth file. Default
            "identity_model".
    """

    category = "global"
    name: str = "global-identity-model"
    version: str = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        """Global identity model parameters."""

        # Primary: explicit identity -> sequences mapping
        identities: dict[str, list[str]] | None = None
        # Convenience shortcut: treat each group as one identity
        group_as_identity: bool = False

        # Network params
        image_size: tuple[int, int] = (128, 128)
        channels: int = 1
        epochs: int = 150
        learning_rate: float = 0.0001
        batch_size: int = 64
        val_split: float = Field(default=0.2, ge=0.0, lt=1.0)

        # Sampling
        max_images_per_identity: int = Field(default=2000, ge=1)

        # Export
        export_trex_weights: bool = True
        trex_weights_name: str = "identity_model"

        # Path to EgocentricCrop output root (contains group__sequence/ subdirs).
        # If None, the feature tries to resolve it from the input Result.
        crop_root: str | None = None

    def __init__(
        self,
        inputs: GlobalIdentityModel.Inputs,
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
        self._network = None
        self._history = None
        self._identity_names = None

        from mosaic.behavior.model_library.trex_identity_network import (
            TRexIdentityNetwork,
        )

        # Check for cached checkpoint
        cached_path = run_root / f"{self.params.trex_weights_name}.pth"
        if cached_path.exists():
            self._network = TRexIdentityNetwork.from_trex_checkpoint(cached_path)
            # Load history if available
            history_path = run_root / "training_history.joblib"
            if history_path.exists():
                self._history = joblib.load(history_path)
            # Load identity names if available
            names_path = run_root / "identity_names.joblib"
            if names_path.exists():
                self._identity_names = joblib.load(names_path)
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        from mosaic.behavior.model_library.trex_identity_network import (
            TRexIdentityNetwork,
        )

        p = self.params

        # Build sequence -> label mapping
        seq_to_label, identity_names = self._build_label_mapping(inputs)
        self._identity_names = identity_names
        num_classes = len(identity_names)

        if num_classes < 2:
            msg = (
                f"[identity-model] Need at least 2 identities, "
                f"got {num_classes}: {identity_names}"
            )
            raise ValueError(msg)

        print(
            f"[identity-model] Training with {num_classes} identities: "
            f"{identity_names}",
            file=sys.stderr,
        )

        # Collect images and labels from input sequences
        all_images: dict[int, list[np.ndarray]] = {i: [] for i in range(num_classes)}

        for entry_key, df in inputs():
            label = seq_to_label.get(entry_key)
            if label is None:
                continue

            # Load crop frames from egocentric crop output
            frames = self._load_crop_frames(entry_key, df)
            if frames:
                all_images[label].extend(frames)

        # Cap per-identity and report counts
        images_list: list[np.ndarray] = []
        labels_list: list[int] = []
        for label_idx in range(num_classes):
            imgs = all_images[label_idx]
            if not imgs:
                print(
                    f"[identity-model] WARNING: no images for "
                    f"{identity_names[label_idx]}",
                    file=sys.stderr,
                )
                continue
            if len(imgs) > p.max_images_per_identity:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(imgs), p.max_images_per_identity, replace=False)
                imgs = [imgs[i] for i in indices]
            print(
                f"[identity-model]   {identity_names[label_idx]}: {len(imgs)} images",
                file=sys.stderr,
            )
            images_list.extend(imgs)
            labels_list.extend([label_idx] * len(imgs))

        if not images_list:
            msg = "[identity-model] No images collected. Check sequence keys and crop output."
            raise RuntimeError(msg)

        images_arr = np.stack(images_list, axis=0)
        labels_arr = np.array(labels_list, dtype=np.int64)

        # Resize if needed
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

        # Train/val split
        val_images = None
        val_labels = None
        if p.val_split > 0:
            rng = np.random.default_rng(42)
            n = len(images_arr)
            n_val = max(1, int(n * p.val_split))
            perm = rng.permutation(n)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            val_images = images_arr[val_idx]
            val_labels = labels_arr[val_idx]
            images_arr = images_arr[train_idx]
            labels_arr = labels_arr[train_idx]

        # Train
        self._network = TRexIdentityNetwork(
            num_classes=num_classes,
            channels=p.channels,
            image_size=p.image_size,
        )
        self._history = self._network.fit(
            images_arr,
            labels_arr,
            val_images=val_images,
            val_labels=val_labels,
            epochs=p.epochs,
            lr=p.learning_rate,
            batch_size=p.batch_size,
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Passthrough -- identity predictions are applied by T-Rex, not Mosaic."""
        return df

    def save_state(self, run_root: Path) -> None:
        if self._network is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        from mosaic.behavior.model_library.trex_identity_network import (
            TRexIdentityNetwork,
        )

        # Export T-Rex checkpoint
        if self.params.export_trex_weights and isinstance(
            self._network, TRexIdentityNetwork
        ):
            self._network.export_trex_checkpoint(
                run_root / f"{self.params.trex_weights_name}.pth"
            )

        # Save training history
        if self._history is not None:
            joblib.dump(self._history, run_root / "training_history.joblib")

        # Save identity names for reference
        if self._identity_names is not None:
            joblib.dump(self._identity_names, run_root / "identity_names.joblib")

    # --- Private helpers ---

    def _build_label_mapping(
        self, inputs: InputStream
    ) -> tuple[dict[str, int], list[str]]:
        """Build a mapping from pipeline ``entry_key`` to integer label.

        The InputStream yields ``entry_key = make_entry_key(group, sequence)``
        (i.e. ``"safe_group__safe_seq"`` with URL-encoded names).  Users
        specify ``identities`` with the more readable ``"group/sequence"``
        form, so we translate every key into the canonical entry-key form
        before storing.

        Returns:
            Tuple of (entry_key -> label, sorted identity names).
        """
        p = self.params
        seq_to_label: dict[str, int] = {}

        def _normalize(seq_key: str) -> str:
            if "/" in seq_key:
                group, sequence = seq_key.split("/", 1)
                return make_entry_key(group, sequence)
            return seq_key

        if p.identities is not None:
            identity_names = sorted(p.identities.keys())
            name_to_label = {name: i for i, name in enumerate(identity_names)}
            for name, seqs in p.identities.items():
                label = name_to_label[name]
                for seq_key in seqs:
                    seq_to_label[_normalize(seq_key)] = label
        elif p.group_as_identity:
            # Discover groups from input stream entry keys.
            # entry_key is "safe_group__safe_seq"; split on "__" once.
            group_set: set[str] = set()
            entry_keys = self._iter_entry_keys(inputs)
            for entry_key in entry_keys:
                group = (
                    entry_key.split("__", 1)[0] if "__" in entry_key else entry_key
                )
                group_set.add(group)
            identity_names = sorted(group_set)
            name_to_label = {name: i for i, name in enumerate(identity_names)}
            for entry_key in entry_keys:
                group = (
                    entry_key.split("__", 1)[0] if "__" in entry_key else entry_key
                )
                if group in name_to_label:
                    seq_to_label[entry_key] = name_to_label[group]
        else:
            msg = (
                "[identity-model] Either 'identities' dict or "
                "'group_as_identity=True' must be provided."
            )
            raise ValueError(msg)

        return seq_to_label, identity_names

    @staticmethod
    def _iter_entry_keys(inputs: InputStream) -> list[str]:
        """Collect all entry keys from the input stream."""
        keys: list[str] = []
        for entry_key, _df in inputs():
            keys.append(entry_key)
        return keys

    def _load_crop_frames(
        self, entry_key: str, df: pd.DataFrame
    ) -> list[np.ndarray]:
        """Load egocentric crop frame images for a sequence.

        Resolves the EgocentricCrop output directory and reads every PNG
        under ``<crop_root>/<group>__<sequence>/frames_id*/frame_*.png``.

        Resolution order for the crop root:
        1. ``params.crop_root`` (explicit override -- recommended).
        2. ``df._source_dir`` if set by the pipeline runner.
        3. Skip if neither is available.

        Args:
            entry_key: ``"group/sequence"`` identifier.
            df: DataFrame from the input stream (may contain crop metadata).

        Returns:
            List of (H, W, C) uint8 arrays, capped at
            ``max_images_per_identity``.
        """
        p = self.params
        frames: list[np.ndarray] = []

        # Resolve the per-sequence directory containing frames_id*/ subdirs.
        # EgocentricCrop writes dirs as f"{group}__{sequence}" using RAW
        # names (not URL-encoded); the df from the InputStream carries those
        # raw names in its "group" / "sequence" columns.
        seq_dir: Path | None = None

        if p.crop_root is not None:
            crop_root = Path(p.crop_root)
            if (
                df is not None
                and not df.empty
                and "group" in df.columns
                and "sequence" in df.columns
            ):
                group = str(df["group"].iloc[0])
                sequence = str(df["sequence"].iloc[0])
                seq_dir = crop_root / f"{group}__{sequence}"
            else:
                # Fallback to entry_key (already in safe_group__safe_seq form;
                # works when raw names contain no special chars).
                seq_dir = crop_root / entry_key

        # Fallback: pipeline-provided _source_dir attribute on the df.
        if seq_dir is None or not seq_dir.is_dir():
            source_dir = getattr(df, "_source_dir", None)
            if source_dir is not None and Path(source_dir).is_dir():
                seq_dir = Path(source_dir)

        if seq_dir is None or not seq_dir.is_dir():
            return frames

        # Scan all frames_id* subdirectories in deterministic order.
        cap = self.params.max_images_per_identity
        for frames_subdir in sorted(seq_dir.glob("frames_id*")):
            if not frames_subdir.is_dir():
                continue
            for img_path in sorted(frames_subdir.glob("frame_*.png")):
                img = self._load_image(img_path)
                if img is not None:
                    frames.append(img)
                    if len(frames) >= cap:
                        return frames
        return frames

    def _scan_frames_from_df(self, df: pd.DataFrame) -> list[np.ndarray]:
        """Fallback: scan for frame images from metadata in the DataFrame.

        When the egocentric-crop output saved frames to disk, the run_root
        path is typically stored as a ``_source_dir`` attribute on the df
        or can be inferred from the pipeline.
        """
        frames: list[np.ndarray] = []

        # Check for source directory attribute (set by the pipeline runner)
        source_dir: Path | None = getattr(df, "_source_dir", None)
        if source_dir is None:
            # Try to infer from frame_path columns if present
            return frames

        if not source_dir.is_dir():
            return frames

        # Scan all frames_id* subdirectories
        for frames_dir in sorted(source_dir.glob("frames_id*")):
            if not frames_dir.is_dir():
                continue
            for img_path in sorted(frames_dir.glob("frame_*.png")):
                img = self._load_image(img_path)
                if img is not None:
                    frames.append(img)
                    if len(frames) >= self.params.max_images_per_identity:
                        return frames

        return frames

    def _load_image(self, path: Path) -> np.ndarray | None:
        """Load a single image and return as (H, W, C) uint8.

        Handles grayscale and color based on ``channels`` param.
        """
        p = self.params
        if p.channels == 1:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img[:, :, np.newaxis]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img
