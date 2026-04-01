"""
FeralFeature -- FERAL vision-transformer inference as a Mosaic pipeline feature.

Loads a pre-trained FERAL model and runs frame-level behavior classification
on crop videos produced by InteractionCropPipeline or EgocentricCrop.

Output follows the same pattern as XgboostFeature: per-frame rows with
``prob_<class>`` probability columns and a ``predicted_label`` column.

Requires the ``feral`` package: ``pip install git+https://github.com/Skovorp/feral.git``
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
    DependencyLookup,
    InputRequire,
    Inputs,
    InputStream,
    Params,
    Result,
)

from .registry import register_feature


def _require_feral():
    """Raise a clear error if the feral package is not installed."""
    try:
        __import__("feral")
    except ImportError:
        raise ImportError(
            "The 'feral' package is required for FeralFeature. "
            "Install it with: pip install git+https://github.com/Skovorp/feral.git"
        ) from None


@final
@register_feature
class FeralFeature:
    """FERAL vision-transformer inference as a pipeline feature.

    Loads a pre-trained FERAL model checkpoint and runs per-frame behavior
    classification on crop videos.  Supports two input formats:

    1. **InteractionCropPipeline** output (pair-level):
       One row per crop video with ``video_path``, ``id_a``, ``id_b``,
       ``target_id``, ``interaction_id``, ``start_frame``, ``end_frame``.

    2. **EgocentricCrop** output (individual-level):
       One row per frame with ``target_id``, ``frame``.  Videos are
       derived as ``egocentric_id{target_id}.mp4``.

    The output adapts to the input type: pair inputs produce ``id_a``,
    ``id_b``, ``target_id``, ``interaction_id`` columns; individual
    inputs produce an ``id`` column.

    Params
    ------
    model_dir : Path
        Directory containing ``model_best.pt`` and ``config.json``
        from a trained FERAL model.
    chunk_length : int
        Frames per video chunk (default 64).
    chunk_shift : int
        Stride between chunks for overlapping inference (default 32).
    chunk_step : int
        Frame sampling step within chunks (default 1).
    resize_to : int
        Input resolution for ViT (default 256).
    device : str
        PyTorch device (default "cuda").
    class_names : dict | None
        Class index → name mapping. Auto-detected from model config
        if None (default).
    decision_threshold : float | None
        Probability threshold for positive class. None uses argmax.
    default_class : int
        Fallback class when no class exceeds threshold (default 0).
    """

    name = "feral"
    version = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        model_dir: Path
        # Inference hyperparameters (defaults from official FERAL default_vjepa.yaml)
        chunk_length: int = 64
        chunk_shift: int = 32
        chunk_step: int = 1
        resize_to: int = 256
        device: str = "cuda"
        # Class configuration (auto-detected from model config if None)
        class_names: dict[str, str] | None = None
        # Decision threshold (matches XgboostFeature pattern)
        decision_threshold: float | None = None
        default_class: int = 0

    def __init__(
        self,
        inputs: FeralFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._model = None  # torch.nn.Module
        self._config: dict = {}
        self._classes: list[int] = []
        self._class_names: dict[int, str] = {}
        self._ds = None
        self._video_dir: Path | None = None

    def bind_dataset(self, ds):
        """Store dataset reference for resolving media paths."""
        self._ds = ds

    # --- State management ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._model = None
        self._config = {}
        self._classes = []
        self._class_names = {}

        # Resolve crop video directory from dependency lookups
        # The lookup maps (group, sequence) -> parquet path; parent = video dir
        for _field_name, lookup in dependency_lookups.items():
            if lookup:
                some_path = next(iter(lookup.values()))
                self._video_dir = some_path.parent
                break

        # Branch 1: cached model in run_root
        cached_model = run_root / "feral_model.pt"
        cached_config = run_root / "feral_config.json"
        if cached_model.exists() and cached_config.exists():
            self._load_model(cached_model, cached_config)
            return True

        # Branch 2: pre-trained model from params.model_dir
        model_dir = self.params.model_dir
        if model_dir is not None:
            checkpoint = Path(model_dir) / "model_best.pt"
            config_path = Path(model_dir) / "config.json"
            if checkpoint.exists():
                self._load_model(
                    checkpoint,
                    config_path if config_path.exists() else None,
                )
                return True

        # Branch 3: no model available
        return False

    def _load_model(
        self,
        checkpoint_path: Path,
        config_path: Path | None,
    ) -> None:
        """Load FERAL model from checkpoint + config."""
        _require_feral()
        import torch
        from feral.model import HFModel

        # Load config if available
        if config_path is not None and config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

        p = self.params
        num_classes = 2  # default

        # Try to get class info from config -> label_json
        if "label_json" in self._config:
            label_json_path = Path(self._config["label_json"])
            if label_json_path.exists():
                with open(label_json_path) as f:
                    labels_json = json.load(f)
                class_names_raw = labels_json.get("class_names", {})
                num_classes = len(class_names_raw)
                self._class_names = {int(k): v for k, v in class_names_raw.items()}

        # Override from params if explicitly set
        if p.class_names is not None:
            self._class_names = {int(k): v for k, v in p.class_names.items()}
            num_classes = len(p.class_names)

        self._classes = list(range(num_classes))

        device = torch.device(p.device)

        model = HFModel(
            model_name=self._config.get("model_name", "facebook/vjepa2-vitl-fpc32-256-diving48"),
            num_classes=num_classes,
            predict_per_item=self._config.get("predict_per_item", 64),
            fc_drop_rate=self._config.get("fc_drop_rate", 0.5),
            freeze_encoder_layers=self._config.get("freeze_encoder_layers", 14),
        )

        state_dict = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        self._model = model

    def fit(self, inputs: InputStream) -> None:
        if self._model is not None:
            return
        raise RuntimeError(
            "FeralFeature requires a pre-trained model. "
            "Set model_dir to a directory containing model_best.pt + config.json. "
            "To train a model, use FeralModel via ds.train_model()."
        )

    def save_state(self, run_root: Path) -> None:
        if self._model is None:
            return

        import torch

        run_root.mkdir(parents=True, exist_ok=True)

        # Save model state dict (handle compiled models)
        model = self._model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        torch.save(model.state_dict(), run_root / "feral_model.pt")

        # Save config
        config_to_save = {
            **self._config,
            "num_classes": len(self._classes),
            "class_names": {str(k): v for k, v in self._class_names.items()},
            "model_name": self._config.get(
                "model_name", "facebook/vjepa2-vitl-fpc32-256-diving48"
            ),
            "predict_per_item": self._config.get("predict_per_item", 64),
            "fc_drop_rate": self._config.get("fc_drop_rate", 0.5),
            "freeze_encoder_layers": self._config.get("freeze_encoder_layers", 14),
            "version": self.version,
        }
        with open(run_root / "feral_config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("No model loaded — call load_state() first")
        if df.empty:
            return pd.DataFrame()

        _require_feral()
        import torch
        import torchvision
        from feral.dataset import get_frame_ids, read_range_video_decord, get_frame_count

        p = self.params
        device = next(self._model.parameters()).device

        resize = torchvision.transforms.v2.Resize(
            (p.resize_to, p.resize_to), antialias=True
        )
        norm = torchvision.transforms.v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        scale = 0.00392156862745098  # 1/255

        # Detect input format and normalize to per-video rows
        is_pair_input = "id_a" in df.columns and "id_b" in df.columns
        video_rows = self._normalize_input(df)

        # Resolve video directory
        video_dir = self._video_dir
        if video_dir is None:
            raise RuntimeError(
                "Could not resolve crop video directory. "
                "Ensure the upstream crop feature has been run."
            )

        num_classes = len(self._classes)
        all_predictions: list[dict] = []

        for _, row in video_rows.iterrows():
            video_path_str = row.get("video_path")
            if video_path_str is None:
                continue

            # Resolve absolute path
            abs_video_path = video_dir / video_path_str
            if not abs_video_path.exists():
                continue

            total_frames = get_frame_count(str(abs_video_path))
            if total_frames is None or total_frames == 0:
                continue

            frame_ids = get_frame_ids(
                total_frames, p.chunk_shift, p.chunk_length, p.chunk_step,
            )

            # Per-frame prediction accumulator
            frame_preds = np.zeros((total_frames, num_classes), dtype=np.float32)
            frame_counts = np.zeros(total_frames, dtype=np.float32)

            for frames in frame_ids:
                video_tensor = read_range_video_decord(str(abs_video_path), frames)
                video_tensor = resize(video_tensor)
                video_tensor = norm(video_tensor * scale)
                video_tensor = video_tensor.unsqueeze(0).to(device)

                device_type = str(device).split(":")[0]
                with torch.no_grad():
                    with torch.amp.autocast(
                        dtype=torch.bfloat16, device_type=device_type
                    ):
                        output = self._model(video_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        probs = probs.cpu().numpy()

                for i, frame_idx in enumerate(frames):
                    if i < len(probs):
                        frame_preds[frame_idx] += probs[i]
                        frame_counts[frame_idx] += 1

            # Average overlapping predictions
            valid = frame_counts > 0
            frame_preds[valid] /= frame_counts[valid, np.newaxis]

            # Build per-frame rows
            start_frame = int(row.get("start_frame", 0))
            for i in range(total_frames):
                if not valid[i]:
                    continue

                pred_row: dict = {C.frame_col: start_frame + i}

                # Identity columns — adapt to input type
                if is_pair_input:
                    pred_row["id_a"] = row.get("id_a")
                    pred_row["id_b"] = row.get("id_b")
                    pred_row["target_id"] = row.get("target_id")
                    pred_row["interaction_id"] = row.get("interaction_id")
                else:
                    pred_row[C.id_col] = row.get("target_id")

                # Probability columns (prob_0, prob_1, ...)
                for cls_idx, cls in enumerate(self._classes):
                    pred_row[f"prob_{cls}"] = float(frame_preds[i, cls_idx])

                # Predicted label (with optional threshold)
                pred_row["predicted_label"] = self._predict_label(frame_preds[i])

                all_predictions.append(pred_row)

        if not all_predictions:
            return pd.DataFrame()

        result = pd.DataFrame(all_predictions)

        # Carry over group/sequence from input
        for col in (C.group_col, C.seq_col):
            if col in df.columns and col not in result.columns:
                result[col] = df[col].iloc[0]

        return result

    def _normalize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize input to per-video rows with video_path + start_frame.

        Handles two formats:
        1. InteractionCropPipeline: already per-video, has video_path
        2. EgocentricCrop: per-frame, derive video_path from target_id
        """
        if "video_path" in df.columns:
            # InteractionCropPipeline format — already per-video rows
            return df

        # EgocentricCrop format — per-frame rows, aggregate to per-video
        if "target_id" not in df.columns:
            raise ValueError(
                "Input must have either 'video_path' (InteractionCropPipeline) "
                "or 'target_id' + 'frame' columns (EgocentricCrop)."
            )

        rows = []
        for target_id, sub in df.groupby("target_id"):
            rows.append({
                "video_path": f"egocentric_id{target_id}.mp4",
                "target_id": target_id,
                "start_frame": int(sub[C.frame_col].min()) if C.frame_col in sub.columns else 0,
                "end_frame": int(sub[C.frame_col].max()) if C.frame_col in sub.columns else 0,
                "n_frames": len(sub),
                C.group_col: sub[C.group_col].iloc[0] if C.group_col in sub.columns else "",
                C.seq_col: sub[C.seq_col].iloc[0] if C.seq_col in sub.columns else "",
            })
        return pd.DataFrame(rows)

    def _predict_label(self, frame_probs: np.ndarray) -> int:
        """Determine predicted class from frame probabilities."""
        threshold = self.params.decision_threshold
        if threshold is not None:
            masked = frame_probs.copy()
            for cls_idx in range(len(self._classes)):
                if masked[cls_idx] < threshold:
                    masked[cls_idx] = 0.0
            if masked.sum() == 0:
                return self.params.default_class
            return self._classes[int(np.argmax(masked))]
        return self._classes[int(np.argmax(frame_probs))]
