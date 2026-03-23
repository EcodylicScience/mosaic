from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field

from .spec import COLUMNS as C
from .spec import Inputs, LabelsSource, Params, TrackInput, register_feature


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
    scope_dependent = False

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        labels: LabelsSource = Field(default_factory=lambda: LabelsSource(kind="id_tags"))
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
        self._labels: dict[tuple[str, str], dict] = {}

    def load_state(self, run_root: Path, artifact_paths: dict[str, Path]) -> bool:
        self._labels = {}
        labels_root = artifact_paths.get("labels")
        if labels_root is None:
            return True
        import json

        for path in sorted(labels_root.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                key = tuple(path.stem.split("__", 1)) if "__" in path.stem else ("", path.stem)
                labels = data.get("labels") or {}
                self._labels[key] = labels
            except Exception:
                continue
        return True

    def fit(self, inputs: Iterator[tuple[str, pd.DataFrame]]) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        group_val = (
            str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
        )
        sequence_val = (
            str(df[C.seq_col].iloc[0]) if C.seq_col in df.columns else ""
        )
        labels = self._labels.get((group_val, sequence_val))
        if not labels:
            return pd.DataFrame()

        fields = self.params.fields
        if fields is None:
            field_set: set[str] = set()
            for tags in labels.values():
                if tags:
                    field_set.update(tags.keys())
            fields = sorted(field_set)
        rename_map = self.params.field_renames or {}

        out = pd.DataFrame(index=df.index)
        for col in (C.frame_col, C.time_col, C.group_col, C.seq_col, C.id_col):
            if col in df.columns:
                out[col] = df[col]
        if C.group_col not in df.columns:
            out[C.group_col] = group_val
        if C.seq_col not in df.columns:
            out[C.seq_col] = sequence_val
        if C.id_col not in df.columns:
            out[C.id_col] = np.nan

        ids_series = out[C.id_col]
        for field in fields:
            col_name = rename_map.get(field, field)
            out[col_name] = ids_series.map(
                lambda i, f=field: labels.get(i, {}).get(f, np.nan)
            )

        return out
