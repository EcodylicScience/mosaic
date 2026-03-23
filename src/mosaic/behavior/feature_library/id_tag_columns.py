from __future__ import annotations

from pathlib import Path
from typing import Iterable, final

import numpy as np
import pandas as pd

from .spec import register_feature

from .spec import COLUMNS, Inputs, OutputType, Params, TrackInput


@final
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
    output_type: OutputType = "per_frame"

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        label_kind: str = "id_tags"
        fields: list[str] | None = None
        field_renames: dict[str, str] | None = None

    def __init__(
        self,
        inputs: IdTagColumns.Inputs = Inputs(("tracks",)),
        params: dict[str, object] | None = None,
    ):
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self.storage_feature_name = self.name
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False
        self._ds = None
        self._labels: dict[tuple[str, str], dict] = {}
        self._scope_filter: dict[str, object] = {}

    # ----------------------- Dataset hooks -----------------------
    def bind_dataset(self, ds):
        self._ds = ds
        try:
            loaded = ds.load_id_labels(kind=self.params.label_kind)
        except Exception:
            loaded = {}
        # Normalize: {(group, sequence): labels dict}
        for key, payload in loaded.items():
            labels = payload.get("labels") or {}
            self._labels[key] = labels

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}

    # ----------------------- Fit protocol ------------------------
    def needs_fit(self) -> bool:
        return False

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
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
        id_col = COLUMNS.id_col
        frame_col = COLUMNS.frame_col
        time_col = COLUMNS.time_col
        group_col = COLUMNS.group_col
        sequence_col = COLUMNS.seq_col

        group_val = (
            str(df[group_col].iloc[0])
            if group_col in df.columns and not df.empty
            else ""
        )
        sequence_val = (
            str(df[sequence_col].iloc[0])
            if sequence_col in df.columns and not df.empty
            else ""
        )
        labels = self._labels.get((group_val, sequence_val))
        if not labels:
            return pd.DataFrame()  # nothing to attach

        # Determine fields
        fields = p.fields
        if fields is None:
            # union of all fields in labels
            field_set = set()
            for tags in labels.values():
                if tags:
                    field_set.update(tags.keys())
            fields = sorted(field_set)
        rename_map = p.field_renames or {}

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
            out[col_name] = ids_series.map(
                lambda i: labels.get(i, {}).get(field, np.nan)
            )

        return out
