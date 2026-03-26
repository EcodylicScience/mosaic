from __future__ import annotations

from pathlib import Path
from typing import final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    Inputs,
    InputStream,
    LabelsSource,
    Params,
    TrackInput,
)

from .registry import register_feature


@final
@register_feature
class IdTagColumns:
    """
    Attach per-id label fields (from labels/<label_kind>) to each frame, so they can
    be merged via Inputs() and used as categories (e.g., focal/nonfocal).

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
        labels: LabelsSource = Field(
            default_factory=lambda: LabelsSource(kind="id_tags")
        )
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
        # Sync labels.kind with label_kind so _resolve_dependencies finds the right dir
        if self.params.label_kind != self.params.labels.kind:
            self.params.labels = LabelsSource(kind=self.params.label_kind)
        self._labels: dict[tuple[str, str], dict] = {}

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._labels = {}
        labels_lookup = dependency_lookups.get("labels", {})
        for (group, sequence), path in labels_lookup.items():
            if not path.exists():
                continue
            try:
                if path.suffix == ".npz":
                    data = np.load(path, allow_pickle=True)
                    ids = data.get("ids", data.get("id", np.array([])))
                    fields = [k for k in data.keys() if k not in ("ids", "id")]
                    labels: dict = {}
                    for idx, fid in enumerate(ids):
                        fid_key = int(fid) if hasattr(fid, 'item') else fid
                        labels[fid_key] = {
                            f: int(data[f][idx]) if hasattr(data[f][idx], 'item')
                            else data[f][idx]
                            for f in fields
                        }
                    self._labels[(group, sequence)] = labels
                elif path.suffix == ".json":
                    import json
                    raw = json.loads(path.read_text())
                    self._labels[(group, sequence)] = raw.get("labels", {})
            except Exception:
                continue
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        group_val = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
        sequence_val = str(df[C.seq_col].iloc[0]) if C.seq_col in df.columns else ""
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
