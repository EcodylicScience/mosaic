"""
WardAssignClustering feature.

Extracted from features.py as part of feature_library modularization.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import final

import joblib
import numpy as np
import pandas as pd
from pydantic import Field
from scipy.cluster.hierarchy import fcluster
from sklearn.neighbors import NearestNeighbors

from mosaic.core.dataset import _dataset_base_dir, register_feature
from mosaic.core.helpers import to_safe_name

from .helpers import (
    StreamingFeatureHelper,
    _build_index_row,
    _build_sequence_lookup,
    _get_feature_run_root,
    _load_artifact_matrix,
    _load_joblib_artifact,
    _parse_scope_filter,
    _resolve_sequence_identity,
)
from .global_tsne import GlobalTSNE
from .global_ward import GlobalWardClustering
from .params import (
    ArtifactSpec,
    Inputs,
    JoblibLoadSpec,
    LoadSpec,
    NNResult,
    NpzLoadSpec,
    OutputType,
    Params,
    Result,
)


@final
@register_feature
class WardAssignClustering:
    """
    Assign Ward clusters (cut at n_clusters) to full-frame feature streams by stacking
    multiple prior features (e.g., pair-wavelet social + ego) and reusing the scaler
    from GlobalTSNE.

    Data inputs are specified via constructor ``inputs`` (Result-based).
    Artifact references (ward model, templates, scaler) are in Params.
    """

    name = "ward-assign"
    version = "0.2"
    parallelizable = True
    output_type: OutputType = "global"
    skip_transform_phase = True

    class ModelArtifact(ArtifactSpec):
        """Params dump (model.joblib)."""

        feature: str = "ward-assign"
        pattern: str = "model.joblib"
        load: LoadSpec = Field(default_factory=JoblibLoadSpec)

    class SeqLabelsArtifact(ArtifactSpec):
        """Per-sequence cluster labels (global_ward_labels_seq=*.npz)."""

        feature: str = "ward-assign"
        pattern: str = "global_ward_labels_seq=*.npz"
        load: LoadSpec = Field(default_factory=lambda: NpzLoadSpec(key="labels"))

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        """Ward-assign clustering parameters.

        Attributes:
            ward: Ward model to load for linkage matrix.
            templates: Templates artifact for centroid computation.
            scaler: Optional scaler specification.
            n_clusters: Number of clusters to cut. Default 20.
            recalc: Force recomputation. Default False.
            pair_filter: Nearest-neighbor pair filter. Default None.
        """

        ward: GlobalWardClustering.ModelArtifact = Field(
            default_factory=GlobalWardClustering.ModelArtifact
        )
        templates: GlobalTSNE.TemplatesArtifact | None = Field(
            default_factory=GlobalTSNE.TemplatesArtifact
        )
        scaler: GlobalTSNE.ScalerArtifact | None = None
        n_clusters: int = Field(default=20, ge=1)
        recalc: bool = False
        pair_filter: NNResult | None = None

    def __init__(
        self,
        inputs: WardAssignClustering.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self.storage_feature_name = (
            f"ward-assign__from__{self.params.ward.feature}"
        )
        self.storage_use_input_suffix = True
        self.skip_existing_outputs = False

        self._ds: object = None
        self._Z: np.ndarray | None = None
        self._templates: np.ndarray | None = None
        self._cluster_ids: np.ndarray | None = None
        self._assign_nn: NearestNeighbors | None = None
        self._scaler: object = None
        self._processed_sequences: set[str] = set()
        self._run_root: Path | None = None
        self._allowed_safe_sequences: set[str] | None = None
        self._pair_map: dict[str, tuple[str, str]] = {}
        self._scope_filter: dict[str, object] | None = None
        self._sequence_lookup_cache: dict[str, tuple[str, str]] | None = None
        self._additional_index_rows: list[dict[str, object]] = []

    def bind_dataset(self, ds: object) -> None:
        self._ds = ds

    def set_run_root(self, path: Path) -> None:
        """Set the output directory for immediate file writes during fit()."""
        self._run_root = path

    def set_scope_filter(self, scope: dict[str, object] | None) -> None:
        self._scope_filter = scope or {}
        self._allowed_safe_sequences, self._pair_map = _parse_scope_filter(scope)
        self._sequence_lookup_cache = None

    def needs_fit(self) -> bool:
        return True

    def supports_partial_fit(self) -> bool:
        return False

    def loads_own_data(self) -> bool:
        # NOTE: time/frame scope filters (filter_start_frame, etc.) are not
        # applied when loading own data. run_feature() raises RuntimeError
        # if these filters are set. Future work: apply them during loading.
        return True

    def partial_fit(self, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def finalize_fit(self) -> None:
        pass

    def load_model(self, path: Path) -> None:
        raise NotImplementedError

    # --- helpers ---

    def _get_sequence_lookup(self) -> dict[str, tuple[str, str]]:
        if self._sequence_lookup_cache is not None:
            return self._sequence_lookup_cache
        self._sequence_lookup_cache = _build_sequence_lookup(
            self._ds, self._allowed_safe_sequences
        )
        return self._sequence_lookup_cache

    def _write_sequence_outputs(
        self,
        safe_seq: str,
        labels: np.ndarray,
        frames: np.ndarray | None,
        id1_vals: np.ndarray | None = None,
        id2_vals: np.ndarray | None = None,
        entity_level: str = "global",
    ) -> None:
        if self._run_root is None:
            print(
                f"[ward-assign] WARN: _run_root not set, labels for {safe_seq} not written",
                file=sys.stderr,
            )
            return
        labels_arr = np.asarray(labels, dtype=np.int32).ravel()
        frame_arr = (
            None if frames is None else np.asarray(frames, dtype=np.int64).ravel()
        )
        if frame_arr is None or len(frame_arr) != len(labels_arr):
            frame_arr = np.arange(len(labels_arr), dtype=np.int64)
        if id1_vals is None or len(id1_vals) != len(labels_arr):
            id1_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)
        if id2_vals is None or len(id2_vals) != len(labels_arr):
            id2_vals = np.full(len(labels_arr), np.nan, dtype=np.float64)

        # Legacy npz artifact (for backward compatibility with viz/analysis code).
        npz_path = self._run_root / f"global_ward_labels_seq={safe_seq}.npz"
        np.savez_compressed(
            npz_path,
            labels=labels_arr,
            frames=frame_arr,
            id1=id1_vals,
            id2=id2_vals,
            entity_level=np.array([str(entity_level or "global")], dtype=object),
        )

        # Standard per-sequence parquet artifact for index-based loading.
        group, sequence = _resolve_sequence_identity(
            safe_seq, self._pair_map, self._get_sequence_lookup()
        )
        safe_group = to_safe_name(group) if group else ""
        out_name = f"{safe_group + '__' if safe_group else ''}{safe_seq}.parquet"
        out_path = self._run_root / out_name
        df_out = pd.DataFrame(
            {
                "frame": frame_arr.astype(np.int64, copy=False),
                "cluster": labels_arr,
                "id1": pd.array(id1_vals, dtype="Int64"),
                "id2": pd.array(id2_vals, dtype="Int64"),
                "entity_level": np.full(
                    len(labels_arr), str(entity_level or "global"), dtype=object
                ),
                "sequence": sequence,
                "group": group,
            }
        )
        df_out.to_parquet(out_path, index=False)
        self._additional_index_rows.append(
            _build_index_row(
                safe_seq,
                group,
                sequence,
                out_path,
                int(len(df_out)),
                dataset_root=_dataset_base_dir(self._ds) if self._ds else None,
            )
        )
        self._processed_sequences.add(safe_seq)

    def fit(self, X_iter: Iterable[pd.DataFrame]) -> None:
        import gc

        import pyarrow as pa

        if self._ds is None:
            raise RuntimeError(
                "[ward-assign] Dataset not bound. Use dataset.run_feature(...)."
            )
        self._processed_sequences = set()
        self._additional_index_rows = []

        # Load Ward linkage
        ward_spec = self.params.ward
        _, ward_root = _get_feature_run_root(
            self._ds, ward_spec.feature, ward_spec.run_id
        )
        pattern = ward_spec.pattern
        files = sorted(ward_root.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"[ward-assign] No Ward model '{pattern}' in {ward_root}"
            )
        bundle = joblib.load(files[0])
        self._Z = bundle.get("linkage_matrix")
        if self._Z is None:
            raise ValueError("[ward-assign] Ward model missing linkage_matrix.")

        # Reload artifact matrix (templates) to derive centroids
        self._templates = _load_artifact_matrix(self._ds, self.params.templates)
        n_clusters = self.params.n_clusters
        labels_templates = fcluster(self._Z, n_clusters, criterion="maxclust")
        uniq = np.unique(labels_templates)
        centroids = []
        for cid in uniq:
            mask = labels_templates == cid
            if not mask.any():
                continue
            centroids.append(self._templates[mask].mean(axis=0))
        centroids = np.vstack(centroids)
        self._cluster_ids = uniq.astype(int)
        self._assign_nn = NearestNeighbors(n_neighbors=1).fit(centroids)

        # Free templates and linkage matrix - no longer needed after centroid computation
        del self._templates, self._Z, labels_templates, centroids
        self._templates = None
        self._Z = None
        gc.collect()

        # optional scaler
        scaler_spec = self.params.scaler
        self._scaler = (
            _load_joblib_artifact(self._ds, scaler_spec) if scaler_spec else None
        )

        # Build manifest from Result-based inputs
        helper = StreamingFeatureHelper(self._ds, "ward-assign")
        if self.params.pair_filter:
            helper.set_pair_filter(self.params.pair_filter)
        scope_filter = (
            {"safe_sequences": self._allowed_safe_sequences}
            if self._allowed_safe_sequences
            else None
        )
        manifest = helper.build_manifest_from_results(
            self.inputs.feature_inputs, scope_filter=scope_filter
        )
        if not manifest:
            raise RuntimeError("[ward-assign] No usable inputs found for assignment.")

        # Process sequences one at a time using direct loading
        keys = list(manifest.keys())
        n_keys = len(keys)
        for i, safe_seq in enumerate(keys):
            X_full, frames, id1_vals, id2_vals, entity_level = (
                helper.load_key_data_with_identity(
                    manifest[safe_seq],
                    extract_frames=True,
                    key=safe_seq,
                )
            )
            if X_full is None:
                continue

            if self._scaler is not None:
                if not hasattr(self._scaler, "transform"):
                    raise ValueError("[ward-assign] scaler object missing transform().")
                X_use = self._scaler.transform(X_full)
                del X_full
            else:
                X_use = X_full

            idxs = self._assign_nn.kneighbors(X_use, return_distance=False)
            labels = self._cluster_ids[idxs.ravel()]
            self._write_sequence_outputs(
                safe_seq,
                labels,
                frames,
                id1_vals=id1_vals,
                id2_vals=id2_vals,
                entity_level=entity_level,
            )

            # Free memory after each sequence
            del X_use, idxs, labels
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % 10 == 0 or i == n_keys - 1:
                print(
                    f"[ward-assign] Processed {i + 1}/{n_keys} sequences",
                    file=sys.stderr,
                )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Outputs are generated during fit(); transform is intentionally skipped.
        return pd.DataFrame(index=[])

    def save_model(self, path: Path) -> None:
        run_root = path.parent
        run_root.mkdir(parents=True, exist_ok=True)
        marker_seq = "__global__"
        safe_marker_seq = to_safe_name(marker_seq)
        marker_path = run_root / f"{safe_marker_seq}.parquet"
        marker_df = pd.DataFrame({"run_marker": [True]})
        marker_df.to_parquet(marker_path, index=False)
        self._additional_index_rows.append(
            _build_index_row(
                safe_marker_seq,
                "",
                marker_seq,
                marker_path,
                int(len(marker_df)),
                dataset_root=_dataset_base_dir(self._ds) if self._ds else None,
            )
        )

        # Labels are already written to disk during fit(); just save model params
        joblib.dump(
            {
                "params": self.params.model_dump(),
                "processed_sequences": list(self._processed_sequences),
            },
            path,
        )

    def get_additional_index_rows(self) -> list[dict[str, object]]:
        return list(self._additional_index_rows)


#### VISUALIZATION
