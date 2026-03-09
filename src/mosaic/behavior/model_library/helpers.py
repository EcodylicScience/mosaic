"""
Helper functions for models in model_library.

Shared utilities used by multiple model implementations.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# XGBoost parameter presets for different training scenarios
XGB_PARAM_PRESETS: Dict[str, dict] = {
    "xgb_v0": dict(
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        random_state=123,
        eval_metric="logloss",
    ),
    "xgb_v1": dict(
        objective="binary:logistic",
        device="cuda",
        tree_method="hist",
        eval_metric=["logloss", "aucpr"],
        n_estimators=3000,
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=8,
        gamma=2.0,
        subsample=0.7,
        colsample_bytree=0.6,
        colsample_bylevel=0.8,
        reg_lambda=2.0,
        reg_alpha=0.5,
        max_bin=256,
        sampling_method="gradient_based",
        n_jobs=-1,
        random_state=123,
    ),
    "xgb_v2": dict(
        objective="binary:logistic",
        device="cuda",
        tree_method="hist",
        eval_metric=["logloss", "aucpr"],
        n_estimators=4000,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=3,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        max_bin=256,
        sampling_method="gradient_based",
        n_jobs=-1,
        random_state=123,
    ),
}


def to_jsonable(obj):
    """
    Convert various Python/NumPy types to JSON-serializable formats.

    Parameters
    ----------
    obj : any
        Object to convert (dict, list, ndarray, numeric types, etc.)

    Returns
    -------
    any
        JSON-serializable version of obj
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def undersample_then_smote(
    X,
    y,
    foreground_samples=None,
    undersample_ratio=1.0,
    use_smote=False,
    random_state=42,
):
    """
    Apply foreground cap + background undersampling + optional SMOTE.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels (n_samples,)
    foreground_samples : int | None
        Cap the minority (positive) class to this many samples.
        None keeps all foreground samples.
    undersample_ratio : float
        Target ratio of majority to minority samples after undersampling.
        1.0 means no background undersampling.
    use_smote : bool
        Whether to apply SMOTE oversampling after undersampling.
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Resampled (X, y) arrays
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 2:
        maj = classes[np.argmax(counts)]
        minc = classes[np.argmin(counts)]
        n_min = int(counts.min())

        if foreground_samples is not None and foreground_samples < n_min:
            n_min = foreground_samples

        n_maj_target = int(round(n_min * undersample_ratio))
        needs_resample = (
            foreground_samples is not None and foreground_samples < int(counts.min())
        ) or n_maj_target < int(counts.max())

        if needs_resample:
            sampling_strategy = {
                minc: n_min,
                maj: min(n_maj_target, int(counts.max())),
            }
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
            X, y = rus.fit_resample(X, y)

    if use_smote:
        sm = SMOTE(random_state=random_state)
        X, y = sm.fit_resample(X, y)
    return X, y
