from pathlib import Path
from typing import Optional, Dict, Any, Iterable
import numpy as np
import pandas as pd

from behavior.dataset import register_feature


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


@register_feature
class IdTagColumns:
    """
    Attach per-id label fields (from labels/<label_kind>) to each frame, so they can
    be merged via inputsets and used as categories (e.g., focal/nonfocal).

    Outputs per row (same granularity as input tracks/feature):
      frame/time/id/group/sequence + one column per requested label field.
    """

    name = "id-tag-columns"
    version = "0.1"
    parallelizable = True

    _defaults = dict(
        label_kind="id_tags",
        fields=None,                # list of fields to include; None -> all fields found
        field_renames=None,         # optional mapping {field: new_column_name}
        id_col="id",
        frame_col="frame",
        time_col="time",
        group_col="group",
        sequence_col="sequence",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = _merge_params(params, self._defaults)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False
        self._ds = None
        self._labels: dict[tuple[str, str], dict] = {}

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds
        try:
            loaded = ds.load_id_labels(kind=self.params["label_kind"])
        except Exception:
            loaded = {}
        # Normalize: {(group, sequence): labels dict}
        for key, payload in loaded.items():
            labels = payload.get("labels") or {}
            self._labels[key] = labels

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        return

    def partial_fit(self, df: pd.DataFrame) -> None:
        return

    def finalize_fit(self) -> None:
        return

    def save_model(self, path: Path) -> None:
        return

    def load_model(self, path: Path) -> None:
        return

    # ----------------------- Core logic --------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        p = self.params
        id_col = p["id_col"]
        frame_col = p["frame_col"]
        time_col = p["time_col"]
        group_col = p["group_col"]
        sequence_col = p["sequence_col"]

        group_val = str(df[group_col].iloc[0]) if group_col in df.columns and not df.empty else ""
        sequence_val = str(df[sequence_col].iloc[0]) if sequence_col in df.columns and not df.empty else ""
        labels = self._labels.get((group_val, sequence_val))
        if not labels:
            return pd.DataFrame()  # nothing to attach

        # Determine fields
        fields = p["fields"]
        if fields is None:
            # union of all fields in labels
            field_set = set()
            for tags in labels.values():
                if tags:
                    field_set.update(tags.keys())
            fields = sorted(field_set)
        rename_map = p.get("field_renames") or {}

        # Build output columns
        out = pd.DataFrame()
        if frame_col in df.columns:
            out[frame_col] = df[frame_col].values
        if time_col in df.columns:
            out[time_col] = df[time_col].values
        if group_col in df.columns:
            out[group_col] = df[group_col].values
        else:
            out[group_col] = group_val
        if sequence_col in df.columns:
            out[sequence_col] = df[sequence_col].values
        else:
            out[sequence_col] = sequence_val
        out[id_col] = df[id_col].values if id_col in df.columns else np.nan

        # Vectorized map per field
        ids_series = out[id_col]
        for field in fields:
            col_name = rename_map.get(field, field)
            out[col_name] = ids_series.map(lambda i: labels.get(i, {}).get(field, np.nan))

        return out

