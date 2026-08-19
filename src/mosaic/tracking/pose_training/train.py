"""Reading what a YOLO or POLO training run left behind.

The training itself is not here and cannot be: Ultralytics and the POLO fork are
AGPL-3.0, so they run in an environment the user builds, driven from
:mod:`mosaic.tracking.pose_training.ultralytics_train`. What stays is everything
that only ever needed a filesystem and pandas -- finding a checkpoint to resume
from, finding the best model of a run, and loading the curve a run recorded.

These are read *around* a training run rather than during one, which is why they
survive the move unchanged: none of them ever imported the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Annotation-only. ``load_training_curves`` imports pandas inside its body so
    # that importing this module costs nothing for the training paths that never
    # read a results.csv; the name still has to resolve for a type checker.
    import pandas as pd


def load_training_curves(run_dir: str | Path) -> "pd.DataFrame":
    """Load per-epoch training metrics from a YOLO training run.

    Parameters
    ----------
    run_dir : path
        Path to the training run directory (contains ``results.csv``).
        Also accepts a path to ``weights/best.pt`` — will resolve to
        the parent run directory automatically.

    Returns
    -------
    DataFrame
        One row per epoch with columns for train losses and val metrics.
    """
    import pandas as pd

    p = Path(run_dir)
    # Accept path to best.pt or weights/ dir
    if p.name == "best.pt" or p.name == "last.pt":
        p = p.parent.parent
    elif p.name == "weights":
        p = p.parent

    csv_path = p / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv in {p}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_best_model(project_dir: str | Path) -> Path | None:
    """Find the best.pt model from the most recent training run.

    Searches *project_dir* for subdirectories containing weights/best.pt,
    returns the most recently modified one.
    """
    project_path = Path(project_dir)
    candidates = sorted(
        project_path.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_last_checkpoint(project_dir: str | Path, name: str | None = None) -> Path:
    """Find the last.pt checkpoint for resuming training.

    Parameters
    ----------
    project_dir : path
        Project directory containing training runs.
    name : str, optional
        Specific run name.  If given, looks for
        ``project_dir/name/weights/last.pt``.  Otherwise searches all
        subdirectories and returns the most recently modified checkpoint.

    Returns
    -------
    Path
        Path to the ``last.pt`` checkpoint.

    Raises
    ------
    FileNotFoundError
        If no checkpoint is found.
    """
    project_path = Path(project_dir)
    if name is not None:
        checkpoint = project_path / name / "weights" / "last.pt"
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(
            f"No checkpoint at {checkpoint}. "
            f"Has a training run with name={name!r} completed at least one epoch?"
        )
    candidates = sorted(
        project_path.glob("*/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No last.pt checkpoint found under {project_path}. "
        f"Has a training run completed at least one epoch?"
    )
