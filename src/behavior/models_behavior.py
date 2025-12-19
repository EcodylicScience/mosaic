from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb

from .dataset import (
    _feature_index_path,
    _feature_run_root,
    _latest_feature_run_root,
    _resolve_inputs,
    _yield_feature_frames,
)
from .helpers import to_safe_name


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


def undersample_then_smote(X, y, use_undersample, undersample_ratio, use_smote, random_state=42):
    if use_undersample:
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) == 2:
            maj = classes[np.argmax(counts)]
            minc = classes[np.argmin(counts)]
            n_min = counts.min()
            n_maj_target = int(round(n_min * undersample_ratio))
            sampling_strategy = {
                minc: n_min,
                maj: min(n_maj_target, counts.max()),
            }
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
            X, y = rus.fit_resample(X, y)
    if use_smote:
        sm = SMOTE(random_state=random_state)
        X, y = sm.fit_resample(X, y)
    return X, y


class BehaviorXGBoostModel:
    """
    One-vs-rest XGBoost classifiers over per-frame behavioral features.

    Config keys (JSON or dict):
      - feature: str (single feature folder under dataset features/)
      - feature_run_id: optional str (defaults to latest finished run)
      - feature_loader: optional dict describing how to read per-sequence files
            {"kind": "parquet" (default) | "npz",
             "key": "Y" (for npz),
             "transpose": bool,
             "columns": [...],
             "drop_columns": [...],
             "numeric_only": true/false}
        (use this path when training from one feature)
      - OR specify an inputset/multi-input spec:
            "inputset": "social+ego@v1"   # saved via features.save_inputset
            # optional "inputs": [...] overrides or supplements the set, same schema as GlobalTSNE/WardAssign
        Each input entry supports {"feature","run_id","pattern","load":{...}}
      - label_kind: str (default "behavior")
      - train_sequences / test_sequences: optional lists (raw or safe names)
      - train_fraction: float (used when sequences not specified; default 0.8)
      - random_state: int (default 42) for shuffling/splits
      - classes: optional list[int] (defaults to unique labels in train split)
      - standardize: bool (default True) -> StandardScaler fit on train, applied to both splits
      - use_undersample: bool (default True)
      - undersample_ratio: float (default 3.0)
      - use_smote: bool (default False)
      - xgb_params: dict (overrides)
      - xgb_params_preset: str referencing XGB_PARAM_PRESETS (default "xgb_v0")
      - decision_threshold: float (default 0.5)
      - label_map: optional dict[int,str] for metadata only
    """

    name = "behavior-xgb"
    version = "0.1"

    def __init__(self, params: Optional[dict] = None):
        defaults = {
            "standardize": True,
            "use_undersample": True,
            "undersample_ratio": 3.0,
            "use_smote": False,
            "train_fraction": 0.8,
            "random_state": 42,
            "label_kind": "behavior",
            "decision_threshold": 0.5,
            "xgb_params_preset": "xgb_v0",
            "use_external_memory": False,
            "external_memory_chunk_rows": 500000,
        }
        self.params = {**defaults, **(params or {})}
        self._ds = None
        self._config: dict = {}
        self._run_root: Optional[Path] = None
        self._predict_models: Optional[Dict[int, Any]] = None
        self._predict_scaler = None
        self._predict_classes: list[int] = []
        self._predict_label_map: dict = {}
        self._predict_feature_columns: Optional[List[str]] = None
        self._predict_threshold: float = float(self.params.get("decision_threshold", 0.5))
        self._predict_run_root: Optional[Path] = None
        self._predict_config: dict = {}
        self._training_input_signature: Optional[dict] = None
        self._predict_input_signature: Optional[dict] = None

    def bind_dataset(self, ds):
        self._ds = ds

    def configure(self, config: dict, run_root: Path):
        cfg = dict(self.params)
        cfg.update(config or {})
        if not (cfg.get("feature") or cfg.get("inputset") or cfg.get("inputs")):
            raise ValueError("BehaviorXGBoostModel config requires 'feature' or 'inputset/inputs'.")
        self._config = cfg
        self._run_root = Path(run_root)

    def train(self) -> dict:
        if self._ds is None:
            raise RuntimeError("BehaviorXGBoostModel requires dataset binding.")
        if self._run_root is None:
            raise RuntimeError("BehaviorXGBoostModel.configure must be called before train().")

        ds = self._ds
        cfg = self._config
        run_root = self._run_root
        run_root.mkdir(parents=True, exist_ok=True)

        payloads = self._collect_sequence_payloads(ds, cfg)
        if not payloads:
            raise RuntimeError("No aligned feature/label sequences found for training.")

        split_info = self._split_sequences(payloads, cfg)
        train_payloads = [p for p in payloads if p["sequence_safe"] in split_info["train_sequences"]]
        test_payloads = [p for p in payloads if p["sequence_safe"] in split_info["test_sequences"]]
        if not train_payloads or not test_payloads:
            raise RuntimeError("Train/test splits are empty. Check sequence lists or train_fraction.")

        use_external = bool(cfg.get("use_external_memory", False))
        if use_external:
            X_train = y_train = X_test = y_test = groups_train = groups_test = None
        else:
            X_train, y_train, groups_train = self._stack_payloads(train_payloads)
            X_test, y_test, groups_test = self._stack_payloads(test_payloads)
            if X_train.size == 0 or X_test.size == 0:
                raise RuntimeError("Train/test matrices are empty after stacking.")

        scaler = None
        if cfg.get("standardize", True) and not use_external:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

        classes = cfg.get("classes")
        if not classes:
            classes = sorted(np.unique(y_train.astype(int)))
        classes = [int(c) for c in classes]

        models: Dict[int, dict] = {}
        reports: Dict[int, dict] = {}
        summary_rows: List[dict] = []

        if use_external:
            cache_dir = self._prepare_external_cache(run_root, cfg)
            (dtrain_path,
             dtest_path,
             groups_train,
             groups_test,
             n_train_samples,
             n_test_samples) = self._build_external_cache(cache_dir, train_payloads, test_payloads, cfg)
            dtrain_base = xgb.DMatrix(f"{dtrain_path}#dtrain.cache")
            dtest_base = xgb.DMatrix(f"{dtest_path}#dtest.cache")
            base_train_labels = dtrain_base.get_label().astype(int)
            base_test_labels = dtest_base.get_label().astype(int)
        else:
            cache_dir = None
            dtrain_path = dtest_path = None
            n_train_samples = int(X_train.shape[0])
            n_test_samples = int(X_test.shape[0])

        xgb_params = self._resolve_xgb_params(cfg)
        random_state = int(cfg.get("random_state", 42))
        threshold = float(cfg.get("decision_threshold", 0.5))

        for beh in classes:
            if use_external:
                y_train_base = base_train_labels
                y_test_base = base_test_labels
                y_train_bin = (y_train_base == beh).astype(int)
                y_test_bin = (y_test_base == beh).astype(int)
                pos = int(y_train_bin.sum())
                if pos == 0:
                    print(f"[behavior-xgb] WARN: no positives for class {beh}; skipping.")
                    continue
                neg = max(1, len(y_train_bin) - pos)
                spw = float(neg) / float(pos)
                params_core = dict(xgb_params)
                params_core.setdefault("scale_pos_weight", spw)
                num_round = int(params_core.pop("n_estimators", params_core.pop("num_boost_round", 100)))
                if params_core.get("device") == "cuda":
                    params_core.setdefault("tree_method", "gpu_hist")
                    params_core.setdefault("predictor", "gpu_predictor")
                    params_core.pop("device")
                dtrain_base.set_label(y_train_bin)
                dtest_base.set_label(y_test_bin)
                booster = xgb.train(
                    params_core,
                    dtrain_base,
                    num_boost_round=num_round,
                    evals=[(dtest_base, "test")],
                    verbose_eval=bool(cfg.get("xgb_verbose", False)),
                )
                clf = booster
                y_prob = booster.predict(dtest_base)
                y_pred = (y_prob >= threshold).astype(int)
            else:
                y_train_bin = (y_train == beh).astype(int)
                y_test_bin = (y_test == beh).astype(int)
                pos = int(y_train_bin.sum())
                if pos == 0:
                    print(f"[behavior-xgb] WARN: no positives for class {beh}; skipping.")
                    continue
                Xt, yt = undersample_then_smote(
                    X_train,
                    y_train_bin,
                    cfg.get("use_undersample", True),
                    cfg.get("undersample_ratio", 3.0),
                    cfg.get("use_smote", False),
                    random_state=random_state,
                )
                neg = max(1, int((y_train_bin == 0).sum()))
                spw = float(neg) / float(pos)
                if cfg.get("use_undersample") or cfg.get("use_smote"):
                    spw = 1.0
                params = dict(xgb_params)
                params["scale_pos_weight"] = spw

                clf = XGBClassifier(**params)
                clf.fit(
                    Xt,
                    yt,
                    eval_set=[(X_test, y_test_bin)],
                    verbose=bool(cfg.get("xgb_verbose", False)),
                )
                y_prob = clf.predict_proba(X_test)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)

            rep = classification_report(
                y_test_bin,
                y_pred,
                target_names=[f"not_{beh}", str(beh)],
                output_dict=True,
                zero_division=0,
            )
            cm = confusion_matrix(y_test_bin, y_pred)
            ap = average_precision_score(y_test_bin, y_prob)

            models[int(beh)] = {
                "model": clf,
                "scale_pos_weight": spw,
                "best_iteration": getattr(clf, "best_iteration", None),
            }
            reports[int(beh)] = {
                "report": rep,
                "confusion_matrix": cm,
                "average_precision": float(ap) if np.isfinite(ap) else None,
            }
            pos_key = str(beh)
            if pos_key in rep:
                summary_rows.append({
                    "behavior": int(beh),
                    "precision": rep[pos_key].get("precision", np.nan),
                    "recall": rep[pos_key].get("recall", np.nan),
                    "f1": rep[pos_key].get("f1-score", np.nan),
                    "support": rep[pos_key].get("support", np.nan),
                    "average_precision": float(ap) if np.isfinite(ap) else None,
                })

        if not models:
            raise RuntimeError("No classifiers were trained (all classes skipped).")

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = run_root / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        reports_json_path = run_root / "reports.json"
        reports_joblib_path = run_root / "reports.joblib"
        joblib.dump(reports, reports_joblib_path)
        reports_json_path.write_text(
            json.dumps({str(k): to_jsonable(v) for k, v in reports.items()}, indent=2)
        )

        models_joblib_path = run_root / "models.joblib"
        joblib.dump(
            {
                "xgb_params": xgb_params,
                "models": {int(k): v["model"] for k, v in models.items()},
                "model_meta": {
                    int(k): {
                        "scale_pos_weight": v.get("scale_pos_weight"),
                        "best_iteration": v.get("best_iteration"),
                    }
                    for k, v in models.items()
                },
                "scaler": scaler,
                "classes": [int(k) for k in models.keys()],
                "label_map": cfg.get("label_map"),
                "feature": cfg.get("feature"),
                "feature_run_id": cfg.get("feature_run_id"),
                "feature_columns": split_info["feature_columns"],
                "train_groups": groups_train,
                "test_groups": groups_test,
                "sequence_splits": split_info["sequence_meta"],
                "input_signature": self._training_input_signature,
            },
            models_joblib_path,
        )

        (run_root / "sequence_splits.json").write_text(
            json.dumps(to_jsonable(split_info["sequence_meta"]), indent=2)
        )

        metrics = {
            "classes_trained": [int(k) for k in models.keys()],
            "n_train_samples": int(n_train_samples),
            "n_test_samples": int(n_test_samples),
            "summary_csv": str(summary_csv),
            "reports_json": str(reports_json_path),
            "models_path": str(models_joblib_path),
            "train_sequences": list(split_info["train_sequences"]),
            "test_sequences": list(split_info["test_sequences"]),
        }
        return metrics

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_run_id(self, df: pd.DataFrame, run_id: Optional[str]) -> str:
        if run_id is not None:
            return str(run_id)
        if "finished_at" in df.columns:
            cand = df[df["finished_at"].fillna("").astype(str) != ""]
            base = cand if len(cand) else df
            order_col = "finished_at" if len(cand) else "started_at"
            if order_col in base.columns:
                base = base.sort_values(by=[order_col], ascending=False, kind="stable")
            run_id = str(base.iloc[0]["run_id"])
        else:
            base = df.sort_values(by=["started_at"], ascending=False, kind="stable")
            run_id = str(base.iloc[0]["run_id"])
        return run_id

    def _collect_sequence_payloads(self, ds, cfg: dict) -> List[dict]:
        label_lookup = self._build_label_lookup(ds, cfg.get("label_kind", "behavior"))
        if cfg.get("inputset") or cfg.get("inputs"):
            feature_payloads = self._collect_inputset_features(ds, cfg)
        else:
            feature_payloads = self._collect_single_feature(ds, cfg)

        payloads: List[dict] = []
        for safe_seq, feat_data in feature_payloads.items():
            label_info = label_lookup.get(safe_seq)
            if not label_info:
                continue
            labels = self._load_labels(label_info["path"])
            n = min(len(labels), feat_data["features"].shape[0])
            if n <= 0:
                continue
            features = feat_data["features"][:n]
            labels = labels[:n]
            mask = np.isfinite(features).all(axis=1)
            if not mask.any():
                continue
            features = features[mask].astype(np.float32, copy=False)
            labels = labels[mask].astype(np.int32, copy=False)
            if features.size == 0:
                continue
            payloads.append({
                "sequence_safe": safe_seq,
                "sequence": feat_data.get("sequence") or label_info["sequence"],
                "group": feat_data.get("group") or label_info["group"],
                "features": features,
                "labels": labels,
            })
        if not payloads:
            raise RuntimeError("No sequences had overlapping features and labels.")
        payloads.sort(key=lambda p: p["sequence_safe"])
        if self._feature_columns is None and payloads:
            self._feature_columns = [f"f{i}" for i in range(payloads[0]["features"].shape[1])]
        return payloads

    def _collect_single_feature(self, ds, cfg: dict) -> Dict[str, dict]:
        feature = cfg["feature"]
        loader = cfg.get("feature_loader", {})
        idx_path = ds.get_root("features") / feature / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Feature index missing: {idx_path}")
        df = pd.read_csv(idx_path)
        run_id = self._resolve_run_id(df, cfg.get("feature_run_id"))
        df = df[df["run_id"].astype(str) == str(run_id)].copy()
        if df.empty:
            raise RuntimeError(f"No rows found for feature {feature} run_id={run_id}")
        df["sequence"] = df["sequence"].fillna("").astype(str)
        df["group"] = df["group"].fillna("").astype(str)
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].apply(lambda v: to_safe_name(v) if v else "")

        feature_columns: Optional[List[str]] = None
        payloads: Dict[str, dict] = {}
        for _, row in df.iterrows():
            safe_seq = str(row.get("sequence_safe") or to_safe_name(row.get("sequence", "")))
            if not safe_seq:
                continue
            abs_raw = row.get("abs_path", "")
            if not abs_raw:
                continue
            abs_path = ds.remap_path(abs_raw)
            if not abs_path.exists():
                continue
            features, cols = self._load_feature_matrix(abs_path, loader)
            if features.size == 0:
                continue
            if feature_columns is None:
                if cols:
                    feature_columns = cols
                else:
                    feature_columns = self._capture_columns(None, features.shape[1], loader, prefix="")
            payloads[safe_seq] = {
                "features": features.astype(np.float32, copy=False),
                "sequence": row.get("sequence", ""),
                "group": row.get("group", ""),
            }
        if not payloads:
            raise RuntimeError("No sequences had usable feature outputs.")
        self._feature_columns = feature_columns
        self._training_input_signature = {
            "input_kind": "feature",
            "input_feature": feature,
            "input_run_id": run_id,
        }
        return payloads

    def _collect_inputset_features(self, ds, cfg: dict) -> Dict[str, dict]:
        explicit_inputs = cfg.get("inputs")
        inputset_name = cfg.get("inputset")
        explicit_override = bool(explicit_inputs)
        inputs, meta = _resolve_inputs(ds, explicit_inputs, inputset_name, explicit_override=explicit_override)
        n_inputs = len(inputs)
        per_seq: Dict[str, dict] = {}
        meta_cache: Dict[tuple[str, str], Dict[Path, dict]] = {}
        resolved_specs: list[dict] = []
        columns_by_input: List[Optional[List[str]]] = [None] * n_inputs

        for idx, spec in enumerate(inputs):
            feat_name = spec["feature"]
            run_id = spec.get("run_id")
            if run_id is None:
                run_id, run_root = _latest_feature_run_root(ds, feat_name)
            else:
                run_root = _feature_run_root(ds, feat_name, run_id)
            resolved_spec = dict(spec)
            resolved_spec["resolved_run_id"] = run_id
            resolved_spec["run_id"] = run_id
            resolved_specs.append(resolved_spec)
            pattern = spec.get("pattern", "*.parquet")
            files = sorted(run_root.glob(pattern))
            if not files:
                print(f"[behavior-xgb] WARN: no files for {feat_name} ({run_id}) pattern={pattern}")
                continue
            seq_map = self._feature_sequence_map(ds, feat_name, run_id)
            meta_map = self._feature_metadata_map(ds, feat_name, run_id, cache=meta_cache)
            load_spec = spec.get("load", {"kind": "parquet", "transpose": False})
            for pth in files:
                try:
                    resolved = pth.resolve()
                except Exception:
                    resolved = pth
                safe_seq = seq_map.get(resolved) or to_safe_name(pth.stem)
                arr, cols = self._load_feature_matrix(resolved, load_spec)
                if arr is None or arr.size == 0:
                    continue
                if columns_by_input[idx] is None:
                    prefix = spec.get("name") or spec.get("feature", f"input{idx}")
                    prefix = f"{prefix}::"
                    if cols:
                        columns_by_input[idx] = [f"{prefix}{c}" for c in cols]
                    else:
                        columns_by_input[idx] = self._derive_columns_from_loader(load_spec, arr.shape[1], prefix=prefix)
                entry = per_seq.setdefault(
                    safe_seq,
                    {
                        "mats": [None] * n_inputs,
                        "sequence": meta_map.get(resolved, {}).get("sequence"),
                        "group": meta_map.get(resolved, {}).get("group"),
                    },
                )
                entry["mats"][idx] = arr

        feature_payloads: Dict[str, dict] = {}
        first_columns: Optional[List[str]] = None
        for safe_seq, bundle in per_seq.items():
            mats = bundle["mats"]
            if any(m is None or m.size == 0 for m in mats):
                continue
            lengths = [m.shape[0] for m in mats]
            T_min = min(lengths)
            if T_min <= 0:
                continue
            mats_trim = [m[:T_min] for m in mats]
            features = np.hstack(mats_trim).astype(np.float32, copy=False)
            if first_columns is None:
                col_names: List[str] = []
                for idx, mat in enumerate(mats_trim):
                    cols = columns_by_input[idx]
                    if cols is None:
                        spec = inputs[idx]
                        prefix = spec.get("name") or spec.get("feature", f"input{idx}")
                        prefix = f"{prefix}::"
                        cols = self._derive_columns_from_loader(spec.get("load") or {}, mat.shape[1], prefix=prefix)
                    col_names.extend(cols)
                first_columns = col_names
            feature_payloads[safe_seq] = {
                "features": features,
                "sequence": bundle.get("sequence"),
                "group": bundle.get("group"),
            }

        if first_columns is not None:
            self._feature_columns = first_columns
        if not feature_payloads:
            raise RuntimeError("No sequences retained after aligning inputset features.")
        self._training_input_signature = {
            "input_kind": "inputset",
            "input_feature": inputset_name,
            "inputs": resolved_specs,
            "inputs_meta": meta,
        }
        return feature_payloads

    def _capture_columns(self, existing, width: int, loader: dict, prefix: str):
        if existing is not None:
            return existing
        cols = self._derive_columns_from_loader(loader, width, prefix=prefix)
        return cols

    def _derive_columns_from_loader(self, loader: dict, width: int, prefix: str) -> List[str]:
        cols = loader.get("columns") if isinstance(loader, dict) else None
        if cols:
            cols = [c for c in cols if c]
            if len(cols) == width:
                return [f"{prefix}{c}" for c in cols]
        return [f"{prefix}f{i}" for i in range(width)]

    def _load_feature_matrix(self, path: Path, loader: dict) -> tuple[np.ndarray, Optional[List[str]]]:
        kind = str(loader.get("kind", "parquet")).lower()
        if kind == "parquet":
            df = pd.read_parquet(path)
            drop_cols = loader.get("drop_columns")
            if drop_cols:
                df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
            columns = loader.get("columns")
            if columns:
                df = df[[c for c in columns if c in df.columns]]
            elif loader.get("numeric_only", True):
                df = df.select_dtypes(include=[np.number])
            else:
                df = df.apply(pd.to_numeric, errors="coerce")
            return df.to_numpy(dtype=np.float32, copy=False), list(df.columns)
        if kind == "npz":
            key = loader.get("key")
            if not key:
                raise ValueError("feature_loader.kind='npz' requires 'key'.")
            with np.load(path, allow_pickle=True) as npz:
                if key not in npz.files:
                    raise KeyError(f"Key '{key}' missing in {path.name}")
                arr = np.asarray(npz[key])
            if loader.get("transpose"):
                arr = arr.T
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr.astype(np.float32, copy=False), None
        raise ValueError(f"Unsupported feature_loader.kind='{kind}'")

    def _load_labels(self, path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=True) as npz:
            if "labels" not in npz.files:
                raise KeyError(f"'labels' key missing in {path.name}")
            return np.asarray(npz["labels"], dtype=np.int32)

    def _build_label_lookup(self, ds, label_kind: str) -> Dict[str, dict]:
        label_root = ds.get_root("labels") / label_kind
        idx_path = label_root / "index.csv"
        if not idx_path.exists():
            raise FileNotFoundError(f"Labels index missing: {idx_path}")
        df = pd.read_csv(idx_path)
        df["sequence"] = df["sequence"].fillna("").astype(str)
        df["group"] = df["group"].fillna("").astype(str)
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].apply(lambda v: to_safe_name(v) if v else "")
        lookup = {}
        for _, row in df.iterrows():
            safe_seq = str(row.get("sequence_safe") or to_safe_name(row.get("sequence", "")))
            path = str(row.get("abs_path", "")).strip()
            if not safe_seq or not path:
                continue
            lookup[safe_seq] = {
                "path": ds.remap_path(path),
                "sequence": row.get("sequence", ""),
                "group": row.get("group", ""),
            }
        return lookup

    def _feature_sequence_map(self, ds, feature_name: str, run_id: str) -> Dict[Path, str]:
        idx_path = _feature_index_path(ds, feature_name)
        if not idx_path.exists():
            return {}
        df = pd.read_csv(idx_path)
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            return {}
        if "sequence_safe" not in df.columns:
            df["sequence_safe"] = df["sequence"].fillna("").apply(lambda v: to_safe_name(v) if v else "")
        mapping = {}
        for _, row in df.iterrows():
            abs_path = str(row.get("abs_path", "")).strip()
            if not abs_path:
                continue
            resolved = ds.remap_path(abs_path)
            try:
                resolved = resolved.resolve()
            except Exception:
                pass
            mapping[resolved] = row.get("sequence_safe") or to_safe_name(row.get("sequence", ""))
        return mapping

    def _feature_metadata_map(
        self,
        ds,
        feature_name: str,
        run_id: str,
        cache: Optional[dict] = None,
    ) -> Dict[Path, dict]:
        if cache is not None and (feature_name, run_id) in cache:
            return cache[(feature_name, run_id)]
        idx_path = _feature_index_path(ds, feature_name)
        if not idx_path.exists():
            return {}
        df = pd.read_csv(idx_path)
        df = df[df["run_id"].astype(str) == str(run_id)]
        if df.empty:
            return {}
        mapping = {}
        for _, row in df.iterrows():
            abs_path = str(row.get("abs_path", "")).strip()
            if not abs_path:
                continue
            resolved = ds.remap_path(abs_path)
            try:
                resolved = resolved.resolve()
            except Exception:
                pass
            mapping[resolved] = {
                "sequence": row.get("sequence", ""),
                "group": row.get("group", ""),
                "sequence_safe": row.get("sequence_safe", ""),
            }
        if cache is not None:
            cache[(feature_name, run_id)] = mapping
        return mapping

    def _split_sequences(self, payloads: List[dict], cfg: dict) -> dict:
        provided_train = self._normalize_sequence_list(cfg.get("train_sequences"))
        provided_test = self._normalize_sequence_list(cfg.get("test_sequences"))
        all_safe = [p["sequence_safe"] for p in payloads]
        payload_map = {p["sequence_safe"]: p for p in payloads}
        if provided_train or provided_test:
            train_safe = [s for s in all_safe if s in provided_train]
            test_safe = [s for s in all_safe if s in provided_test]
        else:
            rng = np.random.default_rng(cfg.get("random_state", 42))
            shuffled = all_safe[:]
            rng.shuffle(shuffled)
            split_idx = max(1, int(len(shuffled) * float(cfg.get("train_fraction", 0.8))))
            split_idx = min(split_idx, len(shuffled) - 1)
            train_safe = shuffled[:split_idx]
            test_safe = shuffled[split_idx:]
        if not train_safe or not test_safe:
            raise RuntimeError("Train/test split produced empty sets; adjust train_fraction or provide sequences.")
        def _meta(seq_list):
            return [
                {
                    "sequence_safe": s,
                    "sequence": payload_map[s]["sequence"],
                    "group": payload_map[s]["group"],
                    "n_frames": int(payload_map[s]["labels"].shape[0]),
                }
                for s in seq_list
            ]

        sequence_meta = {"train": _meta(train_safe), "test": _meta(test_safe)}
        return {
            "train_sequences": set(train_safe),
            "test_sequences": set(test_safe),
            "sequence_meta": sequence_meta,
            "feature_columns": getattr(self, "_feature_columns", None),
        }

    def _normalize_sequence_list(self, seqs: Optional[Iterable[str]]) -> set[str]:
        if not seqs:
            return set()
        safe = set()
        for item in seqs:
            if item is None:
                continue
            item = str(item)
            safe.add(item)
            safe.add(to_safe_name(item))
        return safe

    def _stack_payloads(self, payloads: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = [p["features"] for p in payloads]
        labels = [p["labels"] for p in payloads]
        groups = [np.full(len(p["labels"]), p["sequence_safe"], dtype=object) for p in payloads]
        X = np.vstack(features)
        y = np.concatenate(labels)
        g = np.concatenate(groups)
        return X, y, g

    def _prepare_external_cache(self, run_root: Path, cfg: dict) -> Path:
        cache_dir = run_root / "xgb_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _build_external_cache(self,
                              cache_dir: Path,
                              train_payloads: List[dict],
                              test_payloads: List[dict],
                              cfg: dict) -> tuple[str, str, list[str], list[str], int, int]:
        train_path = cache_dir / "train.libsvm"
        test_path = cache_dir / "test.libsvm"
        n_train = self._write_libsvm(train_payloads, train_path)
        n_test = self._write_libsvm(test_payloads, test_path)
        groups_train = [p["sequence_safe"] for p in train_payloads]
        groups_test = [p["sequence_safe"] for p in test_payloads]
        return str(train_path), str(test_path), groups_train, groups_test, n_train, n_test

    def _write_libsvm(self, payloads: List[dict], path: Path) -> int:
        total = 0
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for payload in payloads:
                feats = payload.get("features")
                labels = payload.get("labels")
                if feats is None or labels is None:
                    continue
                for row, label in zip(feats, labels):
                    fh.write(self._row_to_libsvm(label, row))
                    fh.write("\n")
                    total += 1
                payload["features"] = None
                payload["labels"] = None
        return total

    @staticmethod
    def _row_to_libsvm(label: np.ndarray | float | int, row: np.ndarray) -> str:
        label_val = float(label)
        if row.ndim != 1:
            row = row.ravel()
        nz = np.nonzero(row)[0]
        if nz.size == 0:
            return f"{label_val:g}"
        parts = [f"{label_val:g}"]
        for idx in nz:
            parts.append(f"{idx + 1}:{row[idx]:g}")
        return " ".join(parts)

    def _resolve_xgb_params(self, cfg: dict) -> dict:
        params = {}
        preset = cfg.get("xgb_params_preset")
        if preset:
            params.update(XGB_PARAM_PRESETS.get(preset, {}))
        params.update(cfg.get("xgb_params", {}))
        if not params:
            params.update(XGB_PARAM_PRESETS["xgb_v0"])
        return params

    # ------------------------------------------------------------------ #
    # Prediction helpers
    # ------------------------------------------------------------------ #
    def load_trained_model(self, run_root: Path) -> None:
        run_root = Path(run_root)
        models_path = run_root / "models.joblib"
        if not models_path.exists():
            raise FileNotFoundError(f"Trained model bundle not found: {models_path}")
        bundle = joblib.load(models_path)
        models_raw = bundle.get("models") or {}
        normalized: Dict[int, Any] = {}
        for key, val in models_raw.items():
            try:
                normalized[int(key)] = val
            except Exception:
                continue
        if not normalized:
            raise RuntimeError("No trained classifiers found in models.joblib.")
        self._predict_models = normalized
        self._predict_scaler = bundle.get("scaler")
        classes = bundle.get("classes")
        if classes:
            self._predict_classes = [int(c) for c in classes]
        else:
            self._predict_classes = sorted(normalized.keys())
        self._predict_label_map = bundle.get("label_map") or {}
        self._predict_feature_columns = bundle.get("feature_columns")
        self._predict_run_root = run_root
        cfg_path = run_root / "config.json"
        if cfg_path.exists():
            try:
                self._predict_config = json.loads(cfg_path.read_text())
            except Exception:
                self._predict_config = {}
        else:
            self._predict_config = {}
        thresh = self._predict_config.get("decision_threshold")
        if thresh is None:
            thresh = self.params.get("decision_threshold", 0.5)
        self._predict_threshold = float(thresh)
        self._predict_input_signature = bundle.get("input_signature")

    def predict_sequence(self, df_feat: pd.DataFrame, meta: dict) -> pd.DataFrame:
        if self._predict_models is None:
            raise RuntimeError("BehaviorXGBoostModel is not loaded for prediction.")
        features, frames = self._prepare_features_from_df(df_feat)
        if features.size == 0:
            return pd.DataFrame({
                "frame": frames,
                "sequence": meta.get("sequence", ""),
                "group": meta.get("group", ""),
            })
        if self._predict_scaler is not None:
            features = self._predict_scaler.transform(features)
        prob_arrays: Dict[int, np.ndarray] = {}
        for cls in self._predict_classes:
            model = self._predict_models.get(cls) or self._predict_models.get(int(cls))
            if model is None:
                continue
            scores = self._predict_scores(model, features)
            prob_arrays[int(cls)] = scores
        if not prob_arrays:
            return pd.DataFrame({
                "frame": frames,
                "sequence": meta.get("sequence", ""),
                "group": meta.get("group", ""),
            })
        ordered_classes = [cls for cls in self._predict_classes if cls in prob_arrays]
        probs_stack = np.vstack([prob_arrays[cls] for cls in ordered_classes]).T
        best_idx = np.argmax(probs_stack, axis=1)
        best_scores = probs_stack[np.arange(len(best_idx)), best_idx]
        best_labels = [int(ordered_classes[i]) for i in best_idx]
        label_names = [self._predict_label_map.get(lbl, str(lbl)) for lbl in best_labels]
        data = {
            "frame": frames,
            "sequence": meta.get("sequence", ""),
            "group": meta.get("group", ""),
            "label_id": best_labels,
            "label_name": label_names,
            "score": best_scores,
            "model_run_id": meta.get("model_run_id", ""),
        }
        for cls in ordered_classes:
            data[f"prob_{cls}"] = prob_arrays[cls]
        return pd.DataFrame(data)

    def _prepare_features_from_df(self, df_feat: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if "frame" in df_feat.columns:
            frames = df_feat["frame"].to_numpy(dtype=np.int64, copy=False)
        else:
            frames = np.arange(len(df_feat), dtype=np.int64)
        cols = self._predict_feature_columns
        matrix: Optional[np.ndarray] = None
        if cols:
            overlap = [c for c in cols if c in df_feat.columns]
            if overlap:
                # Align to expected columns; any missing columns are filled with zeros to preserve width.
                df_use = df_feat.reindex(columns=cols, fill_value=0.0)
                matrix = df_use.to_numpy(dtype=np.float32, copy=False)
            else:
                # No overlapping names (e.g., training used positional f0.. while inputs carry full names).
                numeric = df_feat.select_dtypes(include=[np.number]).copy()
                numeric = numeric.drop(columns=[c for c in ("frame",) if c in numeric.columns], errors="ignore")
                arr = numeric.to_numpy(dtype=np.float32, copy=False)
                # Match expected width by padding/truncating.
                target_w = len(cols)
                if arr.shape[1] > target_w:
                    arr = arr[:, :target_w]
                elif arr.shape[1] < target_w:
                    pad = np.zeros((arr.shape[0], target_w - arr.shape[1]), dtype=np.float32)
                    arr = np.hstack([arr, pad])
                matrix = arr
        else:
            numeric = df_feat.select_dtypes(include=[np.number]).copy()
            drop_cols = [c for c in ("frame",) if c in numeric.columns]
            if drop_cols:
                numeric = numeric.drop(columns=drop_cols, errors="ignore")
            matrix = numeric.to_numpy(dtype=np.float32, copy=False)
        if matrix.ndim != 2:
            matrix = np.atleast_2d(matrix)
        matrix = np.nan_to_num(matrix, copy=False)
        if matrix.shape[0] != len(frames):
            min_len = min(matrix.shape[0], len(frames))
            matrix = matrix[:min_len]
            frames = frames[:min_len]
        return matrix, frames

    @staticmethod
    def _predict_scores(model, X: np.ndarray) -> np.ndarray:
        """
        Shared scoring helper that matches the training-time branch.
        """
        if hasattr(model, "predict_proba"):
            out = model.predict_proba(X)
            out = np.asarray(out)
            if out.ndim == 2 and out.shape[1] > 1:
                return out[:, 1]
            return out.ravel()
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X)
            return model.predict(dmat)
        if hasattr(model, "predict"):
            out = model.predict(X)
            out = np.asarray(out)
            if out.ndim > 1:
                out = out[:, 0]
            return out.ravel()
        raise RuntimeError("Unsupported model type for prediction.")

def _predict_scores(self, model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            out = model.predict_proba(X)
            out = np.asarray(out)
            if out.ndim == 2 and out.shape[1] > 1:
                return out[:, 1]
            return out.ravel()
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X)
            return model.predict(dmat)
        if hasattr(model, "predict"):
            out = model.predict(X)
            out = np.asarray(out)
            if out.ndim > 1:
                out = out[:, 0]
            return out.ravel()
        raise RuntimeError("Unsupported model type for prediction.")


class OrientationPCAModel:
    """PCA over orientation-rel features for reusable projection."""

    name = "orientation-pca"
    version = "0.1"

    def __init__(self, params: Optional[dict] = None):
        defaults = {
            "feature": "orientation-rel",
            "feature_run_id": None,
            "n_components": 6,
        }
        self.params = {**defaults, **(params or {})}
        self._ds = None
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._feature_cols: Optional[List[str]] = None

    def bind_dataset(self, ds):
        self._ds = ds

    def needs_fit(self) -> bool: return True
    def supports_partial_fit(self) -> bool: return False

    def train(self) -> dict:
        if self._ds is None:
            raise RuntimeError("OrientationPCAModel requires dataset binding.")
        feat_name = self.params.get("feature")
        run_id = self.params.get("feature_run_id")
        if run_id is None:
            run_id, _ = _latest_feature_run_root(self._ds, feat_name)
        X_blocks: List[np.ndarray] = []
        cols_ref: Optional[List[str]] = None
        for _, _, df_feat in _yield_feature_frames(self._ds, feat_name, run_id=run_id):
            if df_feat is None or df_feat.empty:
                continue
            df_use = df_feat.select_dtypes(include=[np.number])
            drop_cols = [c for c in ("frame", "sequence", "group", "id_a", "id_b") if c in df_use.columns]
            df_use = df_use.drop(columns=drop_cols, errors="ignore")
            if df_use.empty:
                continue
            if cols_ref is None:
                cols_ref = list(df_use.columns)
            else:
                df_use = df_use.reindex(columns=cols_ref, fill_value=np.nan)
            X_blocks.append(df_use.to_numpy(dtype=np.float32, copy=False))
        if not X_blocks:
            raise RuntimeError("No data found to fit OrientationPCAModel.")
        X = np.vstack(X_blocks)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(col_means, inds[1])
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._pca = PCA(n_components=int(self.params.get("n_components", 6)))
        self._pca.fit(Xs)
        self._feature_cols = cols_ref
        return {"explained_variance": self._pca.explained_variance_ratio_.tolist()}

    def save_model(self, path: Path) -> None:
        if self._pca is None or self._scaler is None:
            raise RuntimeError("Model not trained.")
        bundle = {
            "scaler": self._scaler,
            "pca": self._pca,
            "feature_cols": self._feature_cols,
            "params": self.params,
        }
        joblib.dump(bundle, path)

    def load_trained_model(self, run_root: Path) -> None:
        model_path = Path(run_root) / "model.joblib"
        bundle = joblib.load(model_path)
        self._scaler = bundle.get("scaler")
        self._pca = bundle.get("pca")
        self._feature_cols = bundle.get("feature_cols")
        self.params = {**self.params, **bundle.get("params", {})}

    def predict_sequence(self, df_feat: pd.DataFrame, meta: dict) -> pd.DataFrame:
        if self._scaler is None or self._pca is None:
            raise RuntimeError("OrientationPCAModel not loaded.")
        if df_feat is None or df_feat.empty:
            return pd.DataFrame()
        df_use = df_feat.select_dtypes(include=[np.number])
        drop_cols = [c for c in ("frame", "sequence", "group", "id_a", "id_b") if c in df_use.columns]
        df_use = df_use.drop(columns=drop_cols, errors="ignore")
        if self._feature_cols is not None:
            df_use = df_use.reindex(columns=self._feature_cols, fill_value=np.nan)
        X = df_use.to_numpy(dtype=np.float32, copy=False)
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        if inds[0].size:
            X[inds] = np.take(col_means, inds[1])
        Xs = self._scaler.transform(X)
        Z = self._pca.transform(Xs)
        out = pd.DataFrame(Z, columns=[f"PC{i}" for i in range(Z.shape[1])])
        if "frame" in df_feat.columns:
            out["frame"] = df_feat["frame"].to_numpy()
        for col in ("sequence", "group", "id_a", "id_b"):
            if col in df_feat.columns:
                out[col] = df_feat[col]
        return out

    def get_prediction_input_signature(self) -> Optional[dict]:
        if self._predict_input_signature is None:
            return None
        return json.loads(json.dumps(self._predict_input_signature))
