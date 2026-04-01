"""
FeralModel -- FERAL vision-transformer behavior classifier for Mosaic.

Follows the Mosaic model protocol (same as BehaviorXGBoostModel):
    bind_dataset() -> configure() -> train() -> returns metrics dict
    load_trained_model() -> predict_sequence()

FERAL uses a V-JEPA2 backbone with attention pooling for video chunk
classification.  Training consumes crop videos + per-frame labels,
producing per-frame behavior predictions.

Requires the ``feral`` package: ``pip install git+https://github.com/Skovorp/feral.git``

Config keys
-----------
video_dir : str
    Directory containing crop videos (from InteractionCropPipeline or EgocentricCrop).
label_json : str
    Path to FERAL-format label JSON (see ``feral_label_converter``).
model_name : str
    HuggingFace model name (default: "facebook/vjepa2-vitl-fpc32-256-diving48").
predict_per_item : int
    Predictions per chunk (default 64).
chunk_length : int
    Frames per video chunk (default 64).
chunk_shift : int
    Stride between chunks (default 32).
chunk_step : int
    Frame sampling step within chunks (default 1).
resize_to : int
    Input resolution for ViT (default 256).
epochs : int
    Training epochs (default 10).
train_bs : int
    Training batch size (default 4).
val_bs : int
    Validation batch size (default 8).
num_workers : int
    DataLoader workers (default 4).
lr : float
    Learning rate (default 4e-5).
weight_decay : float
    AdamW weight decay (default 0.1).
label_smoothing : float
    CrossEntropy label smoothing (default 0.1).
fc_drop_rate : float
    Dropout rate on classification head (default 0.5).
freeze_encoder_layers : int
    Number of ViT layers to freeze from bottom (default 14).
class_weights : str
    Class weighting strategy (default "inv_freq_sqrt").
ema_decay : float | None
    EMA model averaging decay (default 0.999). None to disable.
mixup_alpha : float | None
    MixUp augmentation alpha (default 0.8). None to disable.
part_warmup : float
    Fraction of total steps for LR warmup (default 0.2).
patience : int | None
    Early stopping patience (default None = disabled).
compile : bool
    Use torch.compile (default True).
do_aa : bool
    Use TrivialAugmentWide (default True).
seed : int
    Random seed (default 0).
device : str
    PyTorch device (default "cuda").
wandb_project : str | None
    W&B project name. None disables logging.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def _require_feral(module_name: str = "feral"):
    """Raise a clear error if the feral package is not installed."""
    try:
        __import__(module_name)
    except ImportError:
        raise ImportError(
            f"The '{module_name}' package is required for FeralModel. "
            f"Install it with: pip install git+https://github.com/Skovorp/feral.git"
        ) from None


class FeralModel:
    """FERAL vision-transformer model for frame-level video behavior classification.

    Follows the Mosaic model protocol:
      - ``bind_dataset(ds)``: attach dataset reference
      - ``configure(config, run_root)``: parse FERAL hyperparameters
      - ``train(progress_callback)``: run training loop, return metrics dict
      - ``load_trained_model(run_root)``: load trained checkpoint
      - ``predict_sequence(df_feat, meta)``: run inference on one sequence
    """

    name = "feral"
    version = "0.1"

    # Default hyperparameters (from official FERAL configs/default_vjepa.yaml)
    DEFAULTS: dict[str, Any] = {
        "model_name": "facebook/vjepa2-vitl-fpc32-256-diving48",
        "predict_per_item": 64,
        "chunk_length": 64,
        "chunk_shift": 32,
        "chunk_step": 1,
        "resize_to": 256,
        "epochs": 10,
        "train_bs": 4,
        "val_bs": 8,
        "num_workers": 4,
        "lr": 4e-5,
        "weight_decay": 0.1,
        "label_smoothing": 0.1,
        "fc_drop_rate": 0.5,
        "freeze_encoder_layers": 14,
        "class_weights": "inv_freq_sqrt",
        "ema_decay": 0.999,
        "mixup_alpha": 0.8,
        "part_warmup": 0.2,
        "patience": None,
        "compile": True,
        "do_aa": True,
        "seed": 0,
        "device": "cuda",
        "part_sample": 1.0,
        "wandb_project": None,
    }

    def __init__(self, params: Optional[dict] = None):
        self.params = {**self.DEFAULTS, **(params or {})}
        self._ds = None
        self._config: dict = {}
        self._run_root: Optional[Path] = None
        # Prediction state
        self._model = None
        self._predict_config: dict = {}

    def bind_dataset(self, ds):
        """Attach Mosaic dataset reference."""
        self._ds = ds

    def configure(self, config: dict, run_root: Path):
        """Merge user config with defaults and validate."""
        cfg = dict(self.params)
        cfg.update(config or {})

        required = ["video_dir", "label_json"]
        for key in required:
            if key not in cfg:
                raise ValueError(
                    f"FeralModel config requires '{key}'. "
                    f"Provide the path to the video directory and label JSON."
                )

        self._config = cfg
        self._run_root = Path(run_root)

    def train(self, progress_callback=None) -> dict:
        """Run FERAL training loop. Returns metrics dict.

        Saves best checkpoint to ``run_root/model_best.pt``.

        Parameters
        ----------
        progress_callback : optional
            If provided, ``on_epoch_end(epoch, total, metrics)`` is called
            after each epoch.
        """
        _require_feral()
        import torch

        from feral.model import HFModel
        from feral.dataset import ClsDataset, collate_fn_val
        from feral.utils import prep_for_answers, save_model, get_weights
        from feral.metrics import (
            calculate_multiclass_metrics,
            calc_frame_level_map,
            calculate_f1_metrics,
        )

        cfg = self._config
        run_root = self._run_root
        run_root.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(run_root / "config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

        # Load labels
        with open(cfg["label_json"]) as f:
            labels_json = json.load(f)
        class_names = {int(k): v for k, v in labels_json["class_names"].items()}
        num_classes = len(class_names)

        # Set seeds
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        device = torch.device(cfg["device"])

        # Build datasets
        data_kwargs = {
            "label_json": cfg["label_json"],
            "prefix": cfg["video_dir"],
            "chunk_shift": cfg["chunk_shift"],
            "chunk_length": cfg["chunk_length"],
            "chunk_step": cfg["chunk_step"],
            "resize_to": cfg["resize_to"],
        }

        train_dataset = ClsDataset(
            partition="train",
            do_aa=cfg["do_aa"],
            predict_per_item=cfg["predict_per_item"],
            num_classes=num_classes,
            part_sample=cfg["part_sample"],
            **data_kwargs,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=cfg["train_bs"],
            num_workers=cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            persistent_workers=cfg["num_workers"] > 0,
        )

        val_loader = None
        if "val" in labels_json["splits"] and labels_json["splits"]["val"]:
            val_dataset = ClsDataset(
                partition="val",
                do_aa=False,
                predict_per_item=cfg["predict_per_item"],
                num_classes=num_classes,
                **data_kwargs,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=cfg["val_bs"],
                num_workers=cfg["num_workers"],
                pin_memory=True,
                drop_last=False,
                persistent_workers=cfg["num_workers"] > 0,
                collate_fn=collate_fn_val,
            )

        # Build model
        model = HFModel(
            model_name=cfg["model_name"],
            num_classes=num_classes,
            predict_per_item=cfg["predict_per_item"],
            fc_drop_rate=cfg["fc_drop_rate"],
            freeze_encoder_layers=cfg["freeze_encoder_layers"],
        )
        model.to(device)

        if cfg["compile"]:
            model = torch.compile(model, mode="max-autotune", dynamic=True)

        # EMA
        model_ema = None
        if cfg["ema_decay"] is not None:
            from timm.utils import ModelEma

            model_ema = ModelEma(model, decay=cfg["ema_decay"], device=device)

        # Loss
        class_weights_tensor = get_weights(
            train_dataset.json_data, cfg["class_weights"], device
        )
        is_multilabel = labels_json.get("is_multilabel", False)
        if is_multilabel:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        else:
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=cfg["label_smoothing"], weight=class_weights_tensor
            )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )
        total_steps = len(train_loader) * cfg["epochs"]
        warmup_steps = round(total_steps * cfg["part_warmup"])
        from transformers import get_cosine_schedule_with_warmup

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # MixUp
        mixup = None
        if cfg["mixup_alpha"] is not None:
            from torchvision.transforms.v2 import MixUp

            mixup = MixUp(alpha=cfg["mixup_alpha"], num_classes=cfg["train_bs"])

        # Training loop
        best_checkpoint_path = run_root / "model_best.pt"
        best_map = -1.0
        epochs_without_improvement = 0
        training_metrics = []

        for epoch in range(cfg["epochs"]):
            # --- Train ---
            model.train()
            train_losses = []
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    if mixup is not None:
                        N, T, C, A, B = data.shape
                        data = data.reshape(N, T, C, A * B)
                        batch_size = data.shape[0]
                        eye = torch.eye(batch_size, device=device)
                        data, mix = mixup(data, eye)
                        data = data.reshape(N, T, C, A, B)
                        if cfg["predict_per_item"] != 1:
                            target = target.permute(1, 0, 2)
                            target = mix.unsqueeze(0) @ target
                            target = target.permute(1, 0, 2)
                        else:
                            target = mix @ target
                    target = target.reshape(-1, num_classes)
                    output = model(data)
                    loss = criterion(output, target)

                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if model_ema is not None:
                    model_ema.update(model)
                train_losses.append(loss.item())

            epoch_train_loss = sum(train_losses) / len(train_losses)
            epoch_metrics = {"epoch": epoch, "train_loss": epoch_train_loss}

            # --- Validate ---
            if val_loader is not None:
                model.eval()
                answers = []
                val_losses = []
                answers_ema = []
                with torch.no_grad():
                    for data, target, names in val_loader:
                        data = data.to(device)
                        target = target.to(device).view(-1, num_classes)
                        with torch.amp.autocast(
                            dtype=torch.bfloat16, device_type="cuda"
                        ):
                            output = model(data)
                            loss = criterion(output, target)
                            output_prob = (
                                torch.sigmoid(output)
                                if is_multilabel
                                else torch.nn.functional.softmax(output, 1)
                            )
                            answers.extend(prep_for_answers(output_prob, target, names))
                            val_losses.append(loss.item())

                            if model_ema is not None:
                                output_ema = model_ema.ema(data)
                                output_ema_prob = (
                                    torch.sigmoid(output_ema)
                                    if is_multilabel
                                    else torch.nn.functional.softmax(output_ema, 1)
                                )
                                answers_ema.extend(
                                    prep_for_answers(output_ema_prob, target, names)
                                )

                val_metrics = calculate_multiclass_metrics(answers, class_names, "val")
                val_f1 = calculate_f1_metrics(
                    answers, labels_json, "val", is_multilabel, "val"
                )
                val_map = calc_frame_level_map(answers, labels_json, "val")

                epoch_metrics.update(val_metrics)
                epoch_metrics.update(val_f1)
                epoch_metrics["val_frame_level_map"] = val_map
                epoch_metrics["val_loss"] = sum(val_losses) / len(val_losses)

                ema_map = -2.0
                if model_ema is not None and answers_ema:
                    ema_map = calc_frame_level_map(answers_ema, labels_json, "val")
                    epoch_metrics["ema_val_frame_level_map"] = ema_map

                # Checkpoint best model
                if val_map > ema_map and val_map > best_map:
                    save_model(model, str(best_checkpoint_path))
                    best_map = val_map
                    epochs_without_improvement = 0
                elif model_ema is not None and ema_map > val_map and ema_map > best_map:
                    save_model(model_ema.ema, str(best_checkpoint_path))
                    best_map = ema_map
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                save_model(model, str(best_checkpoint_path))

            training_metrics.append(epoch_metrics)
            print(f"Epoch {epoch}: {epoch_metrics}")

            if progress_callback is not None:
                progress_callback.on_epoch_end(epoch, cfg["epochs"], epoch_metrics)

            if cfg["patience"] is not None and epochs_without_improvement >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save EMA model separately
        if model_ema is not None:
            save_model(model_ema.ema, str(run_root / "model_ema.pt"))

        # Save training summary
        summary_df = pd.DataFrame(training_metrics)
        summary_df.to_csv(run_root / "summary.csv", index=False)

        with open(run_root / "reports.json", "w") as f:
            json.dump(
                {
                    "class_names": {str(k): v for k, v in class_names.items()},
                    "best_val_map": best_map,
                    "training_metrics": training_metrics,
                },
                f,
                indent=2,
                default=str,
            )

        return {
            "best_val_map": best_map,
            "epochs_trained": len(training_metrics),
            "model_path": str(best_checkpoint_path),
            "summary_csv": str(run_root / "summary.csv"),
            "reports_json": str(run_root / "reports.json"),
            "class_names": class_names,
        }

    def load_trained_model(self, run_root: Path):
        """Load a trained FERAL model for inference."""
        _require_feral()
        import torch
        from feral.model import HFModel

        run_root = Path(run_root)
        config_path = run_root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json in {run_root}")

        with open(config_path) as f:
            self._predict_config = json.load(f)

        cfg = self._predict_config

        # Load label info
        with open(cfg["label_json"]) as f:
            labels_json = json.load(f)
        num_classes = len(labels_json["class_names"])

        device = torch.device(cfg.get("device", "cuda"))

        model = HFModel(
            model_name=cfg["model_name"],
            num_classes=num_classes,
            predict_per_item=cfg["predict_per_item"],
            fc_drop_rate=cfg.get("fc_drop_rate", 0.5),
            freeze_encoder_layers=cfg.get("freeze_encoder_layers", 14),
        )

        checkpoint_path = run_root / "model_best.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No model_best.pt in {run_root}")

        state_dict = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        self._model = model
        self._predict_config = cfg

    def predict_sequence(self, df_feat: pd.DataFrame, meta: dict) -> pd.DataFrame:
        """Run inference on crop videos for one sequence.

        Parameters
        ----------
        df_feat : pd.DataFrame
            Metadata from InteractionCropPipeline or EgocentricCrop
            with ``video_path`` column.
        meta : dict
            Sequence metadata (group, sequence).

        Returns
        -------
        pd.DataFrame
            Per-frame predictions with columns:
            frame, id_a, id_b, target_id, interaction_id,
            predicted_class, predicted_label, trophallaxis_prob
        """
        _require_feral()
        import torch
        import torchvision
        from feral.dataset import get_frame_ids, read_range_video_decord, get_frame_count

        if self._model is None:
            raise RuntimeError("Call load_trained_model() first.")

        cfg = self._predict_config

        device = next(self._model.parameters()).device
        resize = torchvision.transforms.v2.Resize(
            (cfg["resize_to"], cfg["resize_to"]), antialias=True
        )
        norm = torchvision.transforms.v2.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )
        scale = 0.00392156862745098

        with open(cfg["label_json"]) as f:
            labels_json = json.load(f)
        num_classes = len(labels_json["class_names"])
        class_names = {int(k): v for k, v in labels_json["class_names"].items()}

        all_predictions = []

        for _, row in df_feat.iterrows():
            video_path = row.get("video_path")
            if video_path is not None and not Path(video_path).is_absolute():
                video_path = str(Path(cfg["video_dir"]) / video_path)
            if video_path is None or not Path(video_path).exists():
                continue

            total_frames = get_frame_count(video_path)
            if total_frames is None or total_frames == 0:
                continue

            frame_ids = get_frame_ids(
                total_frames,
                cfg["chunk_shift"],
                cfg["chunk_length"],
                cfg["chunk_step"],
            )

            # Per-frame prediction accumulator
            frame_preds = np.zeros((total_frames, num_classes), dtype=np.float32)
            frame_counts = np.zeros(total_frames, dtype=np.float32)

            for frames in frame_ids:
                video_tensor = read_range_video_decord(video_path, frames)
                video_tensor = resize(video_tensor)
                video_tensor = norm(video_tensor * scale)
                video_tensor = video_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        output = self._model(video_tensor)
                        probs = torch.nn.functional.softmax(output, dim=1)
                        probs = probs.cpu().numpy()

                # Aggregate predictions into frame-level
                for i, frame_idx in enumerate(frames):
                    if i < len(probs):
                        frame_preds[frame_idx] += probs[i]
                        frame_counts[frame_idx] += 1

            # Average overlapping predictions
            valid = frame_counts > 0
            frame_preds[valid] /= frame_counts[valid, np.newaxis]

            predicted_classes = frame_preds.argmax(axis=1)
            troph_prob = (
                frame_preds[:, 1] if num_classes > 1 else frame_preds[:, 0]
            )

            start_frame = int(row.get("start_frame", 0))
            for i in range(total_frames):
                if not valid[i]:
                    continue
                pred = {
                    "frame": start_frame + i,
                    "id_a": row.get("id_a"),
                    "id_b": row.get("id_b"),
                    "target_id": row.get("target_id"),
                    "interaction_id": row.get("interaction_id"),
                    "predicted_class": int(predicted_classes[i]),
                    "predicted_label": class_names.get(
                        int(predicted_classes[i]), str(predicted_classes[i])
                    ),
                    "trophallaxis_prob": float(troph_prob[i]),
                }
                all_predictions.append(pred)

        if not all_predictions:
            return pd.DataFrame()
        return pd.DataFrame(all_predictions)

    def get_prediction_input_signature(self):
        """Return None — FERAL consumes video files, not feature DataFrames."""
        return None
