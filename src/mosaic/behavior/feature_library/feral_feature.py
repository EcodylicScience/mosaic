"""
FeralFeature -- FERAL vision-transformer behavior classifier as a Mosaic pipeline feature.

Supports both **training** and **inference** in a single unified feature,
following the same global-feature pattern as XgboostFeature and KpmsFeature.

Training mode
-------------
Provide ``video_dir``, ``label_json``, and a ``training`` config dict.
The ``label_json`` file must contain ``class_names``, ``splits``
(with ``train`` and optionally ``val``/``test`` keys), and optionally
``is_multilabel``.  Training runs the full FERAL ViT fine-tuning loop
with intermediate checkpoints saved to disk for crash recovery.
After training, the test split (if present) is automatically evaluated.

Inference mode
--------------
Provide ``model_dir`` pointing to a directory with ``model_best.pt``
and ``config.json`` from a previous training run.

Output follows the same pattern as XgboostFeature: per-frame rows with
``prob_<class>`` probability columns and a ``predicted_label`` column.

Requires the FERAL code directory (https://github.com/Skovorp/feral).
Point ``feral_code_dir`` to a local clone of the repository.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path
from typing import ClassVar, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.pipeline._loaders import StrictModel
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

log = logging.getLogger(__name__)


def _import_feral(feral_code_dir: str | Path | None):
    """Ensure the FERAL code directory is importable and return its modules.

    FERAL is not an installable Python package — it's a flat directory of
    scripts.  This helper adds the directory to ``sys.path`` so that
    ``import model``, ``import dataset``, etc. work.

    Parameters
    ----------
    feral_code_dir : str or Path or None
        Path to the FERAL code directory (containing model.py, dataset.py).
        If None, assumes FERAL modules are already importable.
    """
    if feral_code_dir is not None:
        feral_dir = str(feral_code_dir)
        if feral_dir not in sys.path:
            sys.path.insert(0, feral_dir)
    # Verify we can import the key module
    try:
        importlib.import_module("model")
    except ImportError:
        raise ImportError(
            "Cannot import FERAL modules. Set feral_code_dir to a local clone "
            "of https://github.com/Skovorp/feral containing model.py, dataset.py, etc."
        ) from None


class FeralTrainingConfig(StrictModel):
    """Training hyperparameters for FERAL ViT fine-tuning.

    These mirror the FERAL default_vjepa.yaml configuration.
    """

    epochs: int = Field(default=10, ge=1)
    train_bs: int = Field(default=4, ge=1)
    val_bs: int = Field(default=8, ge=1)
    num_workers: int = Field(default=4, ge=0)
    lr: float = Field(default=4e-5, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    label_smoothing: float = Field(default=0.1, ge=0, le=1)
    fc_drop_rate: float = Field(default=0.5, ge=0, le=1)
    freeze_encoder_layers: int = Field(default=14, ge=0)
    class_weights: str = "inv_freq_sqrt"
    ema_decay: float | None = 0.999
    mixup_alpha: float | None = 0.8
    part_warmup: float = Field(default=0.2, ge=0, le=1)
    patience: int | None = None
    compile: bool = True
    do_aa: bool = True
    seed: int = 0
    part_sample: float = Field(default=1.0, gt=0, le=1)
    wandb_project: str | None = None


@final
@register_feature
class FeralFeature:
    """FERAL vision-transformer behavior classifier as a pipeline feature.

    Supports two operating modes:

    **Training mode** (``video_dir`` + ``label_json`` + ``training``):
        Runs the full FERAL ViT fine-tuning loop, saves checkpoints,
        evaluates the test split (if present), then applies to all
        sequences in the apply phase.

    **Inference mode** (``model_dir``):
        Loads a pre-trained FERAL model and runs per-frame behavior
        classification on crop videos.

    Supports two input formats for the apply phase:

    1. **InteractionCropPipeline** output (pair-level):
       One row per crop video with ``video_path``, ``id_a``, ``id_b``,
       ``target_id``, ``interaction_id``, ``start_frame``, ``end_frame``.

    2. **EgocentricCrop** output (individual-level):
       One row per frame with ``target_id``, ``frame``.  Videos are
       derived as ``egocentric_id{target_id}.mp4``.

    Params
    ------
    feral_code_dir : Path
        Path to a local clone of https://github.com/Skovorp/feral.
    model_name : str
        HuggingFace model name (default: V-JEPA2 ViT-L).
    predict_per_item : int
        Predictions per chunk (default 64).
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
        Class index -> name mapping. Auto-detected from model config.
    decision_threshold : float | None
        Probability threshold for positive class. None uses argmax.
    default_class : int
        Fallback class when no class exceeds threshold (default 0).
    model_dir : Path | None
        Directory with ``model_best.pt`` + ``config.json`` (inference mode).
    video_dir : Path | None
        Directory containing crop videos (training mode).
    label_json : Path | None
        Path to FERAL-format label JSON with splits (training mode).
    training : FeralTrainingConfig | None
        Training hyperparameters. None = inference-only mode.
    """

    name = "feral"
    version = "0.1"
    parallelizable = False
    scope_dependent = False

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        feral_code_dir: Path
        # Shared model config
        model_name: str = "facebook/vjepa2-vitl-fpc32-256-diving48"
        predict_per_item: int = 64
        # Inference hyperparameters
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
        # Inference mode: point to pre-trained model directory
        model_dir: Path | None = None
        # Training mode: video directory + labels + training config
        video_dir: Path | None = None
        label_json: Path | None = None
        training: FeralTrainingConfig | None = None

        @model_validator(mode="after")
        def _check_mode(self) -> Self:
            has_model = self.model_dir is not None
            has_training = self.video_dir is not None and self.label_json is not None
            if not has_model and not has_training:
                raise ValueError(
                    "Either model_dir (inference) or video_dir + label_json "
                    "(training) must be provided."
                )
            return self

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
        self._run_root: Path | None = None
        self._metrics: dict | None = None

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
        self._metrics = None
        self._run_root = run_root

        # Resolve crop video directory from dependency lookups
        for _field_name, lookup in dependency_lookups.items():
            if lookup:
                some_path = next(iter(lookup.values()))
                self._video_dir = some_path.parent
                break

        # Branch 1: cached model in run_root (from prior fit+save_state)
        cached_model = run_root / "feral_model.pt"
        cached_config = run_root / "feral_config.json"
        if cached_model.exists() and cached_config.exists():
            self._load_model(cached_model, cached_config)
            return True

        # Branch 2: pre-trained model from params.model_dir
        if self.params.model_dir is not None:
            model_dir = Path(self.params.model_dir)
            checkpoint = model_dir / "model_best.pt"
            config_path = model_dir / "config.json"
            if checkpoint.exists():
                self._load_model(
                    checkpoint,
                    config_path if config_path.exists() else None,
                )
                return True

        # Branch 3: training mode -- return False to trigger fit()
        return False

    def _load_model(
        self,
        checkpoint_path: Path,
        config_path: Path | None,
    ) -> None:
        """Load FERAL model from checkpoint + config."""
        _import_feral(self.params.feral_code_dir)
        import torch
        from model import HFModel  # type: ignore[import-not-found]

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

        # Try class_names from config itself (saved by save_state)
        if not self._class_names and "class_names" in self._config:
            raw = self._config["class_names"]
            self._class_names = {int(k): v for k, v in raw.items()}
            num_classes = len(self._class_names)

        # Override from params if explicitly set
        if p.class_names is not None:
            self._class_names = {int(k): v for k, v in p.class_names.items()}
            num_classes = len(p.class_names)

        self._classes = list(range(num_classes))

        device = torch.device(p.device)

        # Use config values for architecture params, falling back to self.params
        model_name = self._config.get("model_name", p.model_name)
        predict_per_item = self._config.get("predict_per_item", p.predict_per_item)
        fc_drop_rate = self._config.get("fc_drop_rate", 0.5)
        freeze_encoder_layers = self._config.get("freeze_encoder_layers", 14)

        model = HFModel(
            model_name=model_name,
            num_classes=num_classes,
            predict_per_item=predict_per_item,
            fc_drop_rate=fc_drop_rate,
            freeze_encoder_layers=freeze_encoder_layers,
        )

        state_dict = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        self._model = model

    # --- Fit (training) ---

    def fit(self, inputs: InputStream) -> None:
        """Train a FERAL model or verify pre-trained model is loaded.

        In training mode (``video_dir`` + ``label_json`` + ``training`` set),
        runs the full ViT fine-tuning loop with intermediate checkpoints.
        After training, evaluates the test split if present.

        In inference mode (``model_dir`` set), the model is already loaded
        by ``load_state()`` and this method is not called.

        The ``inputs`` argument is not consumed -- FERAL reads video files
        directly from ``params.video_dir``.
        """
        if self._model is not None:
            return

        p = self.params
        if p.video_dir is None or p.label_json is None or p.training is None:
            raise RuntimeError(
                "FeralFeature requires either a pre-trained model (model_dir) "
                "or training data (video_dir + label_json + training config)."
            )

        _import_feral(p.feral_code_dir)
        import torch
        from model import HFModel  # type: ignore[import-not-found]
        from dataset import ClsDataset, collate_fn_val  # type: ignore[import-not-found]
        from utils import prep_for_answers, save_model, get_weights  # type: ignore[import-not-found]

        tc = p.training  # FeralTrainingConfig
        run_root = self._run_root
        if run_root is not None:
            run_root.mkdir(parents=True, exist_ok=True)

        # Build a flat config dict for FERAL compatibility and saving
        cfg: dict = {
            "feral_code_dir": str(p.feral_code_dir),
            "video_dir": str(p.video_dir),
            "label_json": str(p.label_json),
            "model_name": p.model_name,
            "predict_per_item": p.predict_per_item,
            "chunk_length": p.chunk_length,
            "chunk_shift": p.chunk_shift,
            "chunk_step": p.chunk_step,
            "resize_to": p.resize_to,
            "device": p.device,
            "epochs": tc.epochs,
            "train_bs": tc.train_bs,
            "val_bs": tc.val_bs,
            "num_workers": tc.num_workers,
            "lr": tc.lr,
            "weight_decay": tc.weight_decay,
            "label_smoothing": tc.label_smoothing,
            "fc_drop_rate": tc.fc_drop_rate,
            "freeze_encoder_layers": tc.freeze_encoder_layers,
            "class_weights": tc.class_weights,
            "ema_decay": tc.ema_decay,
            "mixup_alpha": tc.mixup_alpha,
            "part_warmup": tc.part_warmup,
            "patience": tc.patience,
            "compile": tc.compile,
            "do_aa": tc.do_aa,
            "seed": tc.seed,
            "part_sample": tc.part_sample,
            "wandb_project": tc.wandb_project,
        }
        self._config = cfg

        # Save config for reproducibility / crash recovery
        if run_root is not None:
            with open(run_root / "config.json", "w") as f:
                json.dump(cfg, f, indent=2, default=str)

        # Load labels
        with open(str(p.label_json)) as f:
            labels_json = json.load(f)
        class_names = {int(k): v for k, v in labels_json["class_names"].items()}
        num_classes = len(class_names)
        self._class_names = class_names
        self._classes = list(range(num_classes))

        # Set seeds
        torch.manual_seed(tc.seed)
        np.random.seed(tc.seed)

        device = torch.device(cfg["device"])

        # Build datasets
        data_kwargs = {
            "label_json": str(p.label_json),
            "prefix": str(p.video_dir),
            "chunk_shift": cfg["chunk_shift"],
            "chunk_length": cfg["chunk_length"],
            "chunk_step": cfg["chunk_step"],
            "resize_to": cfg["resize_to"],
        }

        train_dataset = ClsDataset(
            partition="train",
            do_aa=tc.do_aa,
            predict_per_item=p.predict_per_item,
            num_classes=num_classes,
            part_sample=tc.part_sample,
            **data_kwargs,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=tc.train_bs,
            num_workers=tc.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=tc.num_workers > 0,
        )

        val_loader = None
        if "val" in labels_json["splits"] and labels_json["splits"]["val"]:
            val_dataset = ClsDataset(
                partition="val",
                do_aa=False,
                predict_per_item=p.predict_per_item,
                num_classes=num_classes,
                **data_kwargs,
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                shuffle=False,
                batch_size=tc.val_bs,
                num_workers=tc.num_workers,
                pin_memory=True,
                drop_last=False,
                persistent_workers=tc.num_workers > 0,
                collate_fn=collate_fn_val,
            )

        # Build model
        model = HFModel(
            model_name=cfg["model_name"],
            num_classes=num_classes,
            predict_per_item=p.predict_per_item,
            fc_drop_rate=tc.fc_drop_rate,
            freeze_encoder_layers=tc.freeze_encoder_layers,
        )
        model.to(device)

        if tc.compile:
            model = torch.compile(model, mode="max-autotune", dynamic=True)

        # EMA
        model_ema = None
        if tc.ema_decay is not None:
            from timm.utils import ModelEma

            model_ema = ModelEma(model, decay=tc.ema_decay, device=device)

        # Loss
        class_weights_tensor = get_weights(
            train_dataset.json_data, tc.class_weights, device
        )
        is_multilabel = labels_json.get("is_multilabel", False)
        if is_multilabel:
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)
        else:
            criterion = torch.nn.CrossEntropyLoss(
                label_smoothing=tc.label_smoothing, weight=class_weights_tensor
            )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        total_steps = len(train_loader) * tc.epochs
        warmup_steps = round(total_steps * tc.part_warmup)
        from transformers import get_cosine_schedule_with_warmup

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # MixUp
        mixup = None
        if tc.mixup_alpha is not None:
            from torchvision.transforms.v2 import MixUp

            mixup = MixUp(alpha=tc.mixup_alpha, num_classes=tc.train_bs)

        # Training loop
        best_checkpoint_path = run_root / "model_best.pt" if run_root else None
        best_map = -1.0
        epochs_without_improvement = 0
        training_metrics: list[dict] = []

        for epoch in range(tc.epochs):
            # --- Train ---
            model.train()
            train_losses: list[float] = []
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    if mixup is not None:
                        N, T, Ch, A, B = data.shape
                        data = data.reshape(N, T, Ch, A * B)
                        batch_size = data.shape[0]
                        eye = torch.eye(batch_size, device=device)
                        data, mix = mixup(data, eye)
                        data = data.reshape(N, T, Ch, A, B)
                        if p.predict_per_item != 1:
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
            epoch_metrics: dict = {"epoch": epoch, "train_loss": epoch_train_loss}

            # --- Validate ---
            if val_loader is not None:
                model.eval()
                answers = []
                val_losses: list[float] = []
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

                from metrics import (  # type: ignore[import-not-found]
                    calculate_multiclass_metrics,
                    calc_frame_level_map,
                    calculate_f1_metrics,
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
                    if best_checkpoint_path is not None:
                        save_model(model, str(best_checkpoint_path))
                    best_map = val_map
                    epochs_without_improvement = 0
                elif (
                    model_ema is not None
                    and ema_map > val_map
                    and ema_map > best_map
                ):
                    if best_checkpoint_path is not None:
                        save_model(model_ema.ema, str(best_checkpoint_path))
                    best_map = ema_map
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                if best_checkpoint_path is not None:
                    save_model(model, str(best_checkpoint_path))

            training_metrics.append(epoch_metrics)
            log.info("Epoch %d: %s", epoch, epoch_metrics)

            if (
                tc.patience is not None
                and epochs_without_improvement >= tc.patience
            ):
                log.info("Early stopping at epoch %d", epoch)
                break

        # Save EMA model separately
        if model_ema is not None and run_root is not None:
            save_model(model_ema.ema, str(run_root / "model_ema.pt"))

        # Load best checkpoint into self._model for apply phase
        if best_checkpoint_path is not None and best_checkpoint_path.exists():
            self._load_model(best_checkpoint_path, None)
        else:
            # No checkpoint saved (no val loader, no run_root) -- use current model
            model_ref = model
            if hasattr(model_ref, "_orig_mod"):
                model_ref = model_ref._orig_mod
            model_ref.eval()
            self._model = model_ref

        # --- Test set evaluation ---
        test_metrics = {}
        if (
            "test" in labels_json.get("splits", {})
            and labels_json["splits"]["test"]
        ):
            test_metrics = self._evaluate_partition(
                cfg, labels_json, num_classes, is_multilabel, class_names,
                "test", device,
            )

        # Store metrics
        self._metrics = {
            "class_names": {str(k): v for k, v in class_names.items()},
            "best_val_map": best_map,
            "training_metrics": training_metrics,
            "test_metrics": test_metrics,
        }

    def _evaluate_partition(
        self,
        cfg: dict,
        labels_json: dict,
        num_classes: int,
        is_multilabel: bool,
        class_names: dict[int, str],
        partition: str,
        device,
    ) -> dict:
        """Run evaluation on a given partition (e.g. 'test') and return metrics."""
        import torch
        from dataset import ClsDataset, collate_fn_val  # type: ignore[import-not-found]
        from utils import prep_for_answers  # type: ignore[import-not-found]
        from metrics import (  # type: ignore[import-not-found]
            calculate_multiclass_metrics,
            calc_frame_level_map,
            calculate_f1_metrics,
        )

        if self._model is None:
            return {}

        eval_dataset = ClsDataset(
            partition=partition,
            label_json=str(cfg["label_json"]),
            do_aa=False,
            predict_per_item=cfg["predict_per_item"],
            num_classes=num_classes,
            prefix=str(cfg["video_dir"]),
            resize_to=cfg["resize_to"],
            chunk_shift=cfg["chunk_shift"],
            chunk_length=cfg["chunk_length"],
            chunk_step=cfg["chunk_step"],
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            batch_size=cfg.get("val_bs", 4),
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn_val,
        )

        answers = []
        with torch.no_grad():
            for data, target, names in eval_loader:
                data = data.to(device)
                target = target.to(device).view(-1, num_classes)
                with torch.amp.autocast(
                    dtype=torch.bfloat16, device_type="cuda"
                ):
                    output = self._model(data)
                    output_prob = (
                        torch.sigmoid(output)
                        if is_multilabel
                        else torch.nn.functional.softmax(output, 1)
                    )
                    answers.extend(prep_for_answers(output_prob, target, names))

        if not answers:
            return {}

        metrics: dict = {}
        metrics.update(
            calculate_multiclass_metrics(answers, class_names, partition)
        )
        metrics[f"{partition}_frame_level_map"] = calc_frame_level_map(
            answers, labels_json, partition
        )
        metrics.update(
            calculate_f1_metrics(
                answers, labels_json, partition, is_multilabel, partition
            )
        )
        return metrics

    # --- Save state ---

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
                "model_name", self.params.model_name
            ),
            "predict_per_item": self._config.get(
                "predict_per_item", self.params.predict_per_item
            ),
            "fc_drop_rate": self._config.get("fc_drop_rate", 0.5),
            "freeze_encoder_layers": self._config.get(
                "freeze_encoder_layers", 14
            ),
            "version": self.version,
        }
        with open(run_root / "feral_config.json", "w") as f:
            json.dump(config_to_save, f, indent=2, default=str)

        # Save training metrics if available
        if self._metrics is not None:
            with open(run_root / "reports.json", "w") as f:
                json.dump(self._metrics, f, indent=2, default=str)
            if "training_metrics" in self._metrics:
                summary_df = pd.DataFrame(self._metrics["training_metrics"])
                summary_df.to_csv(run_root / "summary.csv", index=False)

    # --- Apply ---

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("No model loaded — call load_state() first")
        if df.empty:
            return pd.DataFrame()

        _import_feral(self.params.feral_code_dir)
        import torch
        import torchvision
        from dataset import get_frame_ids, read_range_video_decord, get_frame_count  # type: ignore[import-not-found]

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
