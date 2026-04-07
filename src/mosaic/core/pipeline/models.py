from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._utils import hash_params, json_ready
from .index_csv import IndexCSV, RunIndexRowBase
from .progress import NullProgressCallback

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from .progress import TrainingProgressCallback


# --- Helpers ---


def model_run_root(ds: Dataset, model_name: str, run_id: str) -> Path:
    return ds.get_root("models") / model_name / run_id


def model_index_path(ds: Dataset, model_name: str) -> Path:
    return ds.get_root("models") / model_name / "index.csv"


@dataclass(frozen=True, slots=True)
class ModelIndexRow(RunIndexRowBase):
    """Typed row for the model index CSV."""

    model: str
    version: str
    config_path: str
    config_hash: str
    metrics_path: str
    status: str
    notes: str


def model_index(path: Path) -> IndexCSV[ModelIndexRow]:
    return IndexCSV(path, ModelIndexRow)


def load_model_config(
    config: str | Path | dict[str, object] | None,
) -> dict[str, object]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if isinstance(config, (str, Path)):
        path = Path(config).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model config not found: {path}")
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc
    raise TypeError(f"Unsupported config type: {type(config)!r}")


def write_model_config(path: Path, config: dict[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(config), indent=2))


# --- Dataset method ---


def train_model(
    ds,
    model,
    config: str | Path | dict[str, object] | None = None,
    overwrite: bool = False,
    *,
    progress_callback: TrainingProgressCallback | None = None,
) -> str:
    """Train a registered model using a JSON (or dict) configuration.

    Args:
        ds: Dataset instance providing model storage roots.
        model: Model/trainer instance implementing ``name``, ``version``,
            and a ``train()`` method.  Optionally implements
            ``bind_dataset(ds)`` and ``configure(config, run_root)``.
        config: Path to a JSON config file or an in-memory dict of
            hyperparameters.
        overwrite: Reserved for future use (run_ids are hash-based).
        progress_callback: Callback for reporting training progress
            (epoch, phase, etc.).  If the model's ``train()`` method
            accepts a ``callback`` or ``progress_callback`` keyword
            argument, it is passed through automatically.

    Returns:
        The ``run_id`` for this training run.
    """
    if progress_callback is None:
        progress_callback = NullProgressCallback()
    storage_model_name = getattr(
        model, "storage_model_name", getattr(model, "name", None)
    )
    if not storage_model_name:
        raise ValueError("Model must define 'name' or 'storage_model_name'.")
    config_dict = load_model_config(config)
    config_hash = hash_params(config_dict or {})
    run_id = f"{model.version}-{config_hash}"
    run_root = model_run_root(ds, storage_model_name, run_id)
    run_root.mkdir(parents=True, exist_ok=True)

    idx_path = model_index_path(ds, storage_model_name)
    idx = model_index(idx_path)
    idx.ensure()

    config_path = run_root / "config.json"
    write_model_config(config_path, config_dict)

    if hasattr(model, "bind_dataset"):
        try:
            model.bind_dataset(ds)
        except Exception as exc:
            print(
                f"[model:{storage_model_name}] bind_dataset failed: {exc}",
                file=sys.stderr,
            )

    if hasattr(model, "configure"):
        model.configure(config_dict, run_root)
    else:
        setattr(model, "config", config_dict)
        setattr(model, "run_root", run_root)

    metrics = None
    metrics_path = run_root / "metrics.json"
    status = "success"
    notes = ""
    try:
        import inspect

        train_sig = inspect.signature(model.train)
        if "callback" in train_sig.parameters or "progress_callback" in train_sig.parameters:
            cb_name = "callback" if "callback" in train_sig.parameters else "progress_callback"
            metrics = model.train(**{cb_name: progress_callback})
        else:
            metrics = model.train()
    except Exception as exc:
        status = "failed"
        notes = str(exc)
        rows = [
            ModelIndexRow(
                run_id=run_id,
                abs_path=run_root,
                model=storage_model_name,
                version=model.version,
                config_path=str(config_path),
                config_hash=config_hash,
                metrics_path="",
                status=status,
                notes=notes[:500],
            )
        ]
        idx.append(rows)
        idx.mark_finished(run_id)
        raise

    if metrics:
        metrics_path.write_text(json.dumps(json_ready(metrics), indent=2))
    else:
        if metrics_path.exists():
            metrics_path.unlink()

    rows = [
        ModelIndexRow(
            run_id=run_id,
            abs_path=run_root,
            model=storage_model_name,
            version=model.version,
            config_path=str(config_path),
            config_hash=config_hash,
            metrics_path=str(metrics_path) if metrics and metrics_path.exists() else "",
            status=status,
            notes=notes[:500],
        )
    ]
    idx.append(rows)
    idx.mark_finished(run_id)
    print(f"[model:{storage_model_name}] completed run_id={run_id} -> {run_root}")
    return run_id
