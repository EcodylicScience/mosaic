"""GlobalIdentityDinoV2Temporal feature.

Sibling of
:class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel`
that uses a frozen DINOv2 backbone with a small trainable temporal
head (GRU / Perceiver / pool, selectable via Params) and an ArcFace loss
to learn identity-discriminative clip embeddings. Identity is decided at
inference by cosine k-NN against per-identity prototypes.

The pluggable temporal head is the "what did my colleague mean by 'DINOv2
+ GRU'?" answer in code: train all three on the same data and compare.

Choosing between this and ``global-identity-embedding``: this one trains a
small head on **clips**, so it can learn identity cues that only appear over
time (gait, posture dynamics) and needs a training cycle. That one trains
nothing and embeds **single frames**, so it is a zero-shot baseline that
returns an answer in one pass. Run the cheap one first.

Closest published reference: RoVF (Rogers et al., IJCV 2025), which uses
a Perceiver recurrent head on DINOv2 frame embeddings for animal re-ID.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, Literal, TypedDict, final

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
    JoblibArtifact,
    JoblibLoadSpec,
    Result,
)
from mosaic.core.pipeline.types.params import Params

from .registry import register_feature

TemporalHead = Literal["gru", "perceiver", "pool"]


# --- Model artifact ---

# The exported weights are a torch ``.pth``, and an ArtifactSpec can only load
# npz / parquet / joblib -- so the referencable artifact is this joblib sidecar,
# written beside the checkpoint and naming it. Same shape as
# ``lightning_action_feature``. The name is fixed rather than derived from
# ``weights_name`` because dependency resolution globs it, and the run root also
# holds ``identity_names.joblib`` and ``training_history.joblib``.
_BUNDLE_NAME = "dinov2_temporal_identity_model.joblib"


class DinoV2TemporalIdentityBundle(TypedDict):
    """Sidecar naming the exported DINOv2 + temporal checkpoint.

    Attributes:
        weights: Checkpoint filename, relative to the bundle's directory.
        identity_names: Class order the checkpoint was exported with.
        version: Feature version that wrote the bundle.
    """

    weights: str
    identity_names: list[str]
    version: str


class DinoV2TemporalIdentityArtifact(JoblibArtifact[DinoV2TemporalIdentityBundle]):
    """Fitted DINOv2 + temporal identity bundle (dinov2_temporal_identity_model.joblib)."""

    feature: str = "global-identity-dinov2-temporal"
    pattern: str = _BUNDLE_NAME
    load: JoblibLoadSpec = Field(default_factory=JoblibLoadSpec)


# --- Feature class ---


@final
@register_feature
class GlobalIdentityDinoV2Temporal:
    """Train a DINOv2 + temporal identity model from individual sequences.

    Takes EgocentricCrop output as input. Each identity is a mapping of
    name -> single-individual sequences. Builds clips of ``clip_len``
    consecutive frames per sequence, embeds each frame with a frozen
    DINOv2 backbone, aggregates frame embeddings with the chosen temporal
    head, and trains the head + ArcFace classifier. Predictions at
    inference are cosine k-NN against per-identity prototype embeddings.

    Example::

        ego_result = dataset.run_feature(
            EgocentricCrop(params={"crop_size": (224, 224)})
        )

        identity_model = GlobalIdentityDinoV2Temporal(
            Inputs((Result(feature="egocentric-crop"),)),
            params={
                "identities": {
                    "mouse_A": ["cage1/day1_mouseA_alone"],
                    "mouse_B": ["cage1/day1_mouseB_alone"],
                    "mouse_C": ["cage1/day2_mouseC_alone"],
                    "mouse_D": ["cage1/day1_mouseD_alone"],
                },
                "temporal_head": "gru",
                "clip_len": 8,
                "image_size": (224, 224),
            },
        )
        result = dataset.run_feature(identity_model)

    Re-running with ``temporal_head="perceiver"`` or ``"pool"`` produces a
    fresh ``run_id`` automatically -- ablations come for free from
    Mosaic's caching.

    Params:
        model: Pre-fitted DinoV2TemporalIdentityArtifact to load, skipping the
            fit. Default None (fit from scratch). Pinning one makes an
            inference run's identity carry its training run by reference, so
            the run needs no scope of its own.
        identities: Explicit identity -> sequences mapping.
        group_as_identity: Treat each group as one identity. Default False.
        backbone: DINOv2 hub model. Default ``"dinov2_vits14"``.
        temporal_head: ``"gru"``, ``"perceiver"``, or ``"pool"``.
        clip_len: Frames per clip. Default 8.
        clip_stride: Step between clip starts. Default ``clip_len`` (no
            overlap).
        embedding_dim: Output dim of the temporal head. Default 128.
        image_size: ``(height, width)`` resize target. Must be a multiple
            of 14. Default ``(224, 224)``.
        channels: 1 or 3. Default 3.
        epochs: Training epochs. Default 50.
        learning_rate: Adam learning rate. Default 1e-3.
        batch_size: Batch size. Default 32.
        val_split: Validation fraction. Default 0.2.
        max_clips_per_identity: Cap on training clips per identity.
            Default 500.
        crop_root: Optional EgocentricCrop output root override.
        weights_name: Stem of the exported ``.pth`` checkpoint. Default
            ``"dinov2_temporal_identity"``.
    """

    category = "global"
    name: str = "global-identity-dinov2-temporal"
    version: str = "0.1"
    parallelizable = False
    # fit() reads the ambient stream -- both to discover the label set under
    # group_as_identity and to collect the training clips -- so the scope IS the
    # training set and belongs in the identifier (P2f). An inference run pins
    # ``model`` instead and carries its training set by reference, so fit and
    # apply are two runs with two identifiers rather than one that silently
    # reuses a network fitted on a narrower scope.
    scope_dependent = True
    consumed_roots: tuple[str, ...] = ()

    ModelArtifact = DinoV2TemporalIdentityArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        """DINOv2 + temporal identity model parameters."""

        # Pre-fitted model reference: when set (and resolvable), fit is skipped.
        model: DinoV2TemporalIdentityArtifact | None = None

        # Primary: explicit identity -> sequences mapping
        identities: dict[str, list[str]] | None = None
        group_as_identity: bool = False

        # Backbone + temporal head
        backbone: str = "dinov2_vits14"
        temporal_head: TemporalHead = "gru"
        embedding_dim: int = Field(default=128, ge=16)
        clip_len: int = Field(default=8, ge=2)
        clip_stride: int | None = None  # defaults to clip_len at runtime

        # Image preprocessing
        image_size: tuple[int, int] = (224, 224)
        channels: int = 3

        # Training
        epochs: int = Field(default=50, ge=1)
        learning_rate: float = 1e-3
        batch_size: int = Field(default=32, ge=1)
        val_split: float = Field(default=0.2, ge=0.0, lt=1.0)

        # Sampling
        max_clips_per_identity: int = Field(default=500, ge=1)

        # Export
        weights_name: str = "dinov2_temporal_identity"

        # Path to EgocentricCrop output root.
        crop_root: str | None = None

    def __init__(
        self,
        inputs: GlobalIdentityDinoV2Temporal.Inputs,
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
        from mosaic.behavior.model_library.dinov2_temporal_identity import (
            DinoV2TemporalNetwork,
        )

        self._network = None
        self._history = None
        self._identity_names = None

        # Branch 1: this run's own cached checkpoint.
        cached_path = run_root / f"{self.params.weights_name}.pth"
        if cached_path.exists():
            self._network = DinoV2TemporalNetwork.from_checkpoint(cached_path)
            history_path = run_root / "training_history.joblib"
            if history_path.exists():
                self._history = joblib.load(history_path)
            names_path = run_root / "identity_names.joblib"
            if names_path.exists():
                self._identity_names = joblib.load(names_path)
            return True

        # Branch 2: a pre-fitted model pinned in params. The checkpoint name
        # comes from the bundle, never from self.params -- an inference run's
        # weights_name need not match the training run's.
        if self.params.model is not None and "model" in artifact_paths:
            bundle_path = artifact_paths["model"]
            bundle = self.params.model.from_path(bundle_path)
            self._network = DinoV2TemporalNetwork.from_checkpoint(
                bundle_path.parent / bundle["weights"]
            )
            self._identity_names = list(bundle["identity_names"])
            return True

        return False

    def fit(self, inputs: InputStream) -> None:
        from mosaic.behavior.model_library.dinov2_temporal_identity import (
            DinoV2TemporalNetwork,
        )
        from mosaic.behavior.model_library.identity_common import (
            build_label_mapping,
            load_crop_clips,
        )

        p = self.params

        seq_to_label, identity_names = build_label_mapping(p, inputs)
        self._identity_names = identity_names
        num_classes = len(identity_names)

        if num_classes < 2:
            msg = (
                f"[dinov2-temporal] Need at least 2 identities, "
                f"got {num_classes}: {identity_names}"
            )
            raise ValueError(msg)

        print(
            f"[dinov2-temporal] training with {num_classes} identities, "
            f"head={p.temporal_head}: {identity_names}",
            file=sys.stderr,
        )

        # Collect clips per identity.
        all_clips: dict[int, list[np.ndarray]] = {i: [] for i in range(num_classes)}
        for entry_key, df in inputs():
            label = seq_to_label.get(entry_key)
            if label is None:
                continue
            clips_for_seq = load_crop_clips(
                entry_key,
                df,
                crop_root=p.crop_root,
                channels=p.channels,
                clip_len=p.clip_len,
                max_clips=p.max_clips_per_identity,
                stride=p.clip_stride,
            )
            if clips_for_seq:
                all_clips[label].extend(clips_for_seq)

        clips_list: list[np.ndarray] = []
        labels_list: list[int] = []
        for label_idx in range(num_classes):
            cs = all_clips[label_idx]
            if not cs:
                print(
                    f"[dinov2-temporal] WARNING: no clips for "
                    f"{identity_names[label_idx]}",
                    file=sys.stderr,
                )
                continue
            if len(cs) > p.max_clips_per_identity:
                rng = np.random.default_rng(42)
                indices = rng.choice(len(cs), p.max_clips_per_identity, replace=False)
                cs = [cs[i] for i in indices]
            print(
                f"[dinov2-temporal]   {identity_names[label_idx]}: {len(cs)} clips",
                file=sys.stderr,
            )
            clips_list.extend(cs)
            labels_list.extend([label_idx] * len(cs))

        if not clips_list:
            msg = (
                "[dinov2-temporal] No clips collected. Check sequence keys, "
                "crop output, and that each sequence has at least clip_len "
                "consecutive frames per identity directory."
            )
            raise RuntimeError(msg)

        # All clips must have the same H/W, so resize to image_size up front.
        target_h, target_w = p.image_size
        clips_arr = self._stack_resized_clips(clips_list, target_h, target_w)
        labels_arr = np.array(labels_list, dtype=np.int64)

        # Train/val split
        val_clips: np.ndarray | None = None
        val_labels: np.ndarray | None = None
        if p.val_split > 0:
            rng = np.random.default_rng(42)
            n = len(clips_arr)
            n_val = max(1, int(n * p.val_split))
            perm = rng.permutation(n)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            val_clips = clips_arr[val_idx]
            val_labels = labels_arr[val_idx]
            clips_arr = clips_arr[train_idx]
            labels_arr = labels_arr[train_idx]

        self._network = DinoV2TemporalNetwork(
            backbone=p.backbone,
            temporal_head=p.temporal_head,
            embedding_dim=p.embedding_dim,
            clip_len=p.clip_len,
            image_size=p.image_size,
        )
        self._history = self._network.fit(
            clips_arr,
            labels_arr,
            val_clips=val_clips,
            val_labels=val_labels,
            num_classes=num_classes,
            epochs=p.epochs,
            lr=p.learning_rate,
            batch_size=p.batch_size,
        )
        self._network._identity_names = identity_names  # pyright: ignore[reportPrivateUsage]

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Passthrough -- identity predictions are consumed downstream."""
        return df

    def save_state(self, run_root: Path) -> None:
        from mosaic.behavior.model_library.dinov2_temporal_identity import (
            DinoV2TemporalNetwork,
        )

        if self._network is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        if isinstance(self._network, DinoV2TemporalNetwork):
            weights_name = f"{self.params.weights_name}.pth"
            self._network.export_checkpoint(run_root / weights_name)
            # The sidecar a later run references as ``model``. Written here so
            # this run's output is loadable as the next run's pre-fitted model.
            bundle: DinoV2TemporalIdentityBundle = {
                "weights": weights_name,
                "identity_names": list(self._identity_names or ()),
                "version": self.version,
            }
            joblib.dump(bundle, run_root / _BUNDLE_NAME)

        if self._history is not None:
            joblib.dump(self._history, run_root / "training_history.joblib")

        if self._identity_names is not None:
            joblib.dump(self._identity_names, run_root / "identity_names.joblib")

    # --- Private helpers ---

    @staticmethod
    def _stack_resized_clips(
        clips_list: list[np.ndarray], target_h: int, target_w: int
    ) -> np.ndarray:
        """Stack clips into a single ``(N, T, H, W, C)`` array, resizing if needed.

        Each clip is ``(T, H, W, C)``. Frames within a clip share H/W, but
        clips from different sources may not, so we resize to a common
        ``(target_h, target_w)`` here rather than relying on the model.
        """
        out = []
        for clip in clips_list:
            t, h, w, c = clip.shape
            if h == target_h and w == target_w:
                out.append(clip)
                continue
            resized = np.empty((t, target_h, target_w, c), dtype=np.uint8)
            for i in range(t):
                frame = cv2.resize(
                    clip[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
                if frame.ndim == 2:
                    frame = frame[:, :, np.newaxis]
                resized[i] = frame
            out.append(resized)
        return np.stack(out, axis=0)
