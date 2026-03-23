"""
Shared helper functions for feature implementations.

This module contains utility functions used across multiple features in the
feature_library to avoid code duplication.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import pandas as pd

from .spec import ArtifactSpec, ParquetLoadSpec

if TYPE_CHECKING:
    from .spec import DictModel, LoadSpec, Result

from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.loading import (
    EntryData,
    _build_nn_lookup,
    _get_feature_run_root,
    _load_array_from_spec,
    _load_artifact_matrix,
    _load_joblib_artifact,
    _load_parquet_dataframe,
    _nn_pair_mask,
    _normalize_identity_columns,
    _pose_column_pairs,
    _resolve_sequence_identity,
)
from mosaic.core.pipeline.loading import (
    load_entry_data as _load_entry_data,
)

__all__ = [
    "EntryData",
    "PartialIndexRow",
    "StreamingFeatureHelper",
    "_build_index_row",
    "_build_nn_lookup",
    "_get_feature_run_root",
    "_load_artifact_matrix",
    "_load_array_from_spec",
    "_load_joblib_artifact",
    "_load_parquet_dataframe",
    "_nn_pair_mask",
    "_normalize_identity_columns",
    "_pose_column_pairs",
    "_resolve_sequence_identity",
]

# --- StreamingFeatureHelper - Unified streaming processor for global features ---


class StreamingFeatureHelper:
    """
    Unified streaming processor for global features.

    Provides common functionality for:
    - Building file manifests without loading data
    - Streaming data loading with automatic memory cleanup
    - Progress logging
    - Scope filtering

    Usage:
        helper = StreamingFeatureHelper(ds, "my-feature")
        manifest = helper.build_manifest(inputs, scope)
        for entry_key, kd in helper.iter_sequences(manifest):
            # process kd.features, kd.frames
            pass
    """

    def __init__(self, ds, feature_name: str):
        """
        Initialize the streaming helper.

        Parameters
        ----------
        ds : Dataset
            Dataset instance
        feature_name : str
            Name of the feature (for logging)
        """
        self.ds = ds
        self.feature_name = feature_name
        self._pair_filter_spec: dict | None = None
        self._nn_lookup_cache: dict[str, dict] = {}

    def set_pair_filter(self, pair_filter_spec: DictModel | dict | None) -> None:
        """Configure pair filtering (e.g. nearest-neighbor) for subsequent loads."""
        self._pair_filter_spec = pair_filter_spec

    def _get_nn_lookup(self, entry_key: str) -> dict:
        """Build and cache the NN lookup for an entry key."""
        if entry_key in self._nn_lookup_cache:
            return self._nn_lookup_cache[entry_key]
        lookup = (
            _build_nn_lookup(self.ds, entry_key, self._pair_filter_spec)
            if self._pair_filter_spec
            else {}
        )
        self._nn_lookup_cache[entry_key] = lookup
        return lookup

    def _make_df_filter(
        self, entry_key: str
    ) -> Callable[[pd.DataFrame], pd.DataFrame] | None:
        """Return a DataFrame filter callable for the given entry, or None."""
        if self._pair_filter_spec is None:
            return None
        nn_lookup = self._get_nn_lookup(entry_key)
        if not nn_lookup:
            return None

        def _filter(df: pd.DataFrame) -> pd.DataFrame:
            mask = _nn_pair_mask(df, nn_lookup)
            return df.loc[mask].reset_index(drop=True)

        return _filter

    def build_manifest_from_results(
        self,
        results: tuple[Result, ...],
        scope: Scope | None = None,
    ) -> dict[str, list[tuple[Path, LoadSpec]]]:
        """Build manifest from Result objects (per-frame parquet outputs)."""
        specs = [
            ArtifactSpec(feature=r.feature, run_id=r.run_id, load=ParquetLoadSpec())
            for r in results
        ]
        manifest = self.build_manifest(specs, scope=scope)
        if not manifest:
            labels = ", ".join(f"{r.feature} ({r.run_id})" for r in results)
            raise RuntimeError(
                f"[{self.feature_name}] build_manifest_from_results: "
                f"no per-sequence parquet files found for inputs: {labels}"
            )
        return manifest

    def build_manifest(
        self,
        inputs: list[ArtifactSpec],
        scope: Scope | None = None,
    ) -> dict[str, list[tuple[Path, LoadSpec]]]:
        """
        Build manifest of file paths per sequence WITHOUT loading data.

        Parameters
        ----------
        inputs : list[ArtifactSpec]
            List of typed artifact specifications.
        scope : Scope, optional
            Resolved scope with entries to filter by.

        Returns
        -------
        dict[str, list[tuple[Path, LoadSpec]]]
            Mapping from sequence key to list of (path, load_spec) tuples
        """
        from mosaic.core.pipeline.index import feature_run_root, latest_feature_run_root

        manifest: dict[str, list[tuple[Path, LoadSpec]]] = {}

        allowed_keys = scope.entry_keys if scope and scope.entries else None

        for spec in inputs:
            feat_name = spec.feature
            run_id = spec.run_id
            if run_id is None:
                run_id, run_root = latest_feature_run_root(self.ds, feat_name)
            else:
                run_root = feature_run_root(self.ds, feat_name, run_id)

            pattern = spec.pattern
            load_spec = spec.load
            files = sorted(run_root.glob(pattern))

            if not files:
                print(
                    f"[{self.feature_name}] WARN: no files for {feat_name} ({run_id}) pattern={pattern}",
                    file=sys.stderr,
                )
                continue

            for pth in files:
                entry_key = self._extract_key(pth)
                if allowed_keys is not None and entry_key not in allowed_keys:
                    continue
                if entry_key not in manifest:
                    manifest[entry_key] = []
                manifest[entry_key].append((pth, load_spec))

        return manifest

    def iter_sequences(
        self,
        manifest: dict[str, list[tuple[Path, LoadSpec]]],
        progress_interval: int = 10,
    ) -> Iterator[tuple[str, EntryData]]:
        """Iterate through manifest, yielding (entry_key, EntryData) one at a time."""
        import pyarrow as pa

        n_entries = len(manifest)
        for i, (entry_key, file_specs) in enumerate(manifest.items()):
            kd = self.load_entry_data(file_specs, entry_key=entry_key)
            if kd is None:
                continue

            yield entry_key, kd

            del kd
            gc.collect()
            pa.default_memory_pool().release_unused()

            if (i + 1) % progress_interval == 0 or i == n_entries - 1:
                print(
                    f"[{self.feature_name}] Processed {i + 1}/{n_entries} sequences",
                    file=sys.stderr,
                )

    def load_entry_data(
        self,
        file_specs: list[tuple[Path, LoadSpec]],
        entry_key: str | None = None,
    ) -> EntryData | None:
        """Load and merge data for a single manifest entry."""
        df_filter = self._make_df_filter(entry_key) if entry_key else None
        return _load_entry_data(file_specs, df_filter=df_filter)

    def _extract_key(self, path: Path) -> str:
        """Extract entry key from file path (the bare stem)."""
        return path.stem


@dataclass(frozen=True, slots=True)
class PartialIndexRow:
    """Partial index row from global features (group/sequence/path/n_rows).

    Produced by _build_index_row(), consumed by _append_external_row()
    in run.py which fills in the remaining FeatureIndexRow fields.
    """

    group: str
    sequence: str
    abs_path: str
    n_rows: int = 0


def _build_index_row(
    group: str,
    sequence: str,
    output_path: Path,
    n_rows: int = 0,
) -> PartialIndexRow:
    """Build a partial index row for get_additional_index_rows()."""
    return PartialIndexRow(
        group=group,
        sequence=sequence,
        abs_path=str(output_path.resolve()),
        n_rows=n_rows,
    )
