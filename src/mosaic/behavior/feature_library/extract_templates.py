from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar, Literal, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    InputRequire,
    Inputs,
    Params,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
)

from .helpers import ensure_columns, feature_columns
from .registry import register_feature
from .types import PoolConfig


class TemplatesArtifact(ParquetArtifact):
    """Template feature vectors (templates.parquet)."""

    feature: str = "extract-templates"
    pattern: str = "templates.parquet"
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


class ProvenanceArtifact(ParquetArtifact):
    """Per-entry template provenance (template_provenance.parquet)."""

    feature: str = "extract-templates"
    pattern: str = "template_provenance.parquet"
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


@final
@register_feature
class ExtractTemplates:
    """Subsample per-sequence data into a representative template matrix.

    Entry point for the global feature pipeline. Streams per-sequence
    inputs, builds a candidate pool with proportional per-entry
    contribution, and selects templates using the configured strategy.
    """

    name = "extract-templates"
    version = "0.1"
    parallelizable = False
    scope_dependent = True

    TemplatesArtifact = TemplatesArtifact
    ProvenanceArtifact = ProvenanceArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        """ExtractTemplates parameters.

        Attributes:
            strategy: Selection strategy. Default "random".
            n_templates: Number of templates to select. Required.
            pool: Pool configuration. Default PoolConfig().
            random_state: Random seed. Default 42.
        """

        strategy: Literal["random", "farthest_first"] = "random"
        n_templates: int = Field(ge=1)
        pool: PoolConfig = Field(default_factory=PoolConfig)
        random_state: int = 42

    def __init__(
        self,
        inputs: ExtractTemplates.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._templates: np.ndarray | None = None
        self._template_entries: np.ndarray | None = None
        self._entry_map: dict[int, tuple[str, str]] = {}
        self._provenance: pd.DataFrame | None = None
        self._rng = np.random.default_rng(self.params.random_state)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_indices: dict[str, pd.DataFrame],
    ) -> bool:
        self._feature_columns = None
        self._templates = None
        self._template_entries = None
        self._entry_map = {}
        self._provenance = None

        path = run_root / "templates.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self._feature_columns = list(df.columns)
            self._templates = df.to_numpy(dtype=np.float64)

            provenance_path = run_root / "template_provenance.parquet"
            if provenance_path.exists():
                self._provenance = pd.read_parquet(provenance_path)

            return True
        return False

    def fit(
        self,
        inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]],
    ) -> None:
        n_templates = self.params.n_templates
        pool_cfg = self.params.pool
        pool_size = pool_cfg.size if pool_cfg.size is not None else n_templates

        if pool_cfg.allocation == "exact":
            self._fit_exact(inputs, pool_size)
        else:
            self._fit_reservoir(inputs, pool_size)

        # Both _fit_reservoir and _fit_exact raise on empty inputs,
        # so _templates is guaranteed set here. Narrow for type checker.
        if self._templates is None:
            msg = "[extract-templates] No data after pool construction."
            raise RuntimeError(msg)

        # Farthest-first selection from pool down to n_templates
        if (
            self.params.strategy == "farthest_first"
            and self._templates.shape[0] > n_templates
        ):
            self._templates, sel_idx = self._farthest_first(
                self._templates, n_templates
            )
            if self._template_entries is not None:
                self._template_entries = self._template_entries[sel_idx]

        # Compute per-entry provenance
        self._provenance = self._build_provenance()

    def _fit_reservoir(
        self,
        inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]],
        pool_size: int,
    ) -> None:

        reservoir: np.ndarray | None = None
        entry_indices = np.empty(pool_size, dtype=np.int32)
        filled = 0
        n_seen = 0  # global count of all rows processed
        n_entries = 0
        max_frac = self.params.pool.max_entry_fraction

        for _entry_key, df in inputs():
            if self._feature_columns is None:
                self._feature_columns = feature_columns(df)
                ensure_columns(df, [C.seq_col])
                reservoir = np.empty(
                    (pool_size, len(self._feature_columns)), dtype=np.float64
                )

            # Track entry metadata
            if n_entries not in self._entry_map:
                group = (
                    str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
                )
                sequence = str(df[C.seq_col].iloc[0])
                self._entry_map[n_entries] = (group, sequence)
            entry_idx = n_entries

            cols = self._feature_columns
            matrix = df[cols].to_numpy(dtype=np.float64)
            n_rows = matrix.shape[0]

            # Apply per-entry cap
            n_entries += 1
            if max_frac is not None:
                effective_cap = max(int(max(max_frac, 1.0 / n_entries) * pool_size), 1)
                if n_rows > effective_cap:
                    idx = self._rng.choice(n_rows, size=effective_cap, replace=False)
                    matrix = matrix[idx]
                    n_rows = effective_cap

            assert reservoir is not None

            if filled < pool_size:
                space = pool_size - filled
                if n_rows <= space:
                    reservoir[filled : filled + n_rows] = matrix
                    entry_indices[filled : filled + n_rows] = entry_idx
                    filled += n_rows
                    n_seen += n_rows
                    continue
                else:
                    # Fill remaining space, then do replacement for the rest
                    reservoir[filled:pool_size] = matrix[:space]
                    entry_indices[filled:pool_size] = entry_idx
                    filled = pool_size
                    n_seen += space
                    matrix = matrix[space:]
                    n_rows = matrix.shape[0]
                    # Fall through to replacement below

            if n_rows > 0:
                # Vectorized reservoir replacement (Algorithm R)
                positions = np.arange(n_seen + 1, n_seen + n_rows + 1, dtype=np.float64)
                probabilities = pool_size / positions
                randoms = self._rng.random(n_rows)
                include_mask = randoms < probabilities
                included = matrix[include_mask]
                if included.shape[0] > 0:
                    slots = self._rng.integers(0, pool_size, size=included.shape[0])
                    reservoir[slots] = included
                    entry_indices[slots] = entry_idx
                n_seen += n_rows

        if reservoir is None or filled == 0:
            msg = "[extract-templates] No data in inputs."
            raise RuntimeError(msg)

        self._templates = reservoir[:filled]
        self._template_entries = entry_indices[:filled]

    def _fit_exact(
        self,
        inputs: Callable[[], Iterator[tuple[str, pd.DataFrame]]],
        pool_size: int,
    ) -> None:
        from mosaic.core.pipeline.types import COLUMNS as C

        # Pass 1: count rows per entry, capture feature_columns and metadata
        entry_counts: dict[str, int] = {}
        entry_idx_map: dict[str, int] = {}
        for entry_key, df in inputs():
            if self._feature_columns is None:
                self._feature_columns = feature_columns(df)
                ensure_columns(df, [C.seq_col])
            if entry_key not in entry_idx_map:
                idx = len(entry_idx_map)
                entry_idx_map[entry_key] = idx
                group = (
                    str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
                )
                sequence = str(df[C.seq_col].iloc[0])
                self._entry_map[idx] = (group, sequence)
            entry_counts[entry_key] = len(df)

        if not entry_counts:
            msg = "[extract-templates] No data in inputs."
            raise RuntimeError(msg)

        # Compute quotas
        quotas = self._compute_quotas(entry_counts, pool_size)

        # Pass 2: sample exactly quota rows per entry
        blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        cols = self._feature_columns
        assert cols is not None

        for entry_key, df in inputs():
            quota = quotas.get(entry_key, 0)
            if quota <= 0:
                continue
            matrix = df[cols].to_numpy(dtype=np.float64)
            entry_idx = entry_idx_map[entry_key]
            if matrix.shape[0] <= quota:
                blocks.append(matrix)
                label_blocks.append(np.full(matrix.shape[0], entry_idx, dtype=np.int32))
            else:
                idx = self._rng.choice(matrix.shape[0], size=quota, replace=False)
                blocks.append(matrix[idx])
                label_blocks.append(np.full(quota, entry_idx, dtype=np.int32))

        self._templates = np.vstack(blocks)
        self._template_entries = np.concatenate(label_blocks)

    def _compute_quotas(
        self,
        entry_counts: dict[str, int],
        pool_size: int,
    ) -> dict[str, int]:
        """Compute exact per-entry quotas proportional to row count, with cap."""
        total_rows = sum(entry_counts.values())
        n_entries = len(entry_counts)
        max_frac = self.params.pool.max_entry_fraction

        effective_cap = pool_size  # no cap
        if max_frac is not None:
            effective_cap = max(int(max(max_frac, 1.0 / n_entries) * pool_size), 1)

        # First pass: proportional allocation with cap
        quotas: dict[str, int] = {}
        uncapped_rows = 0

        for entry_key, count in entry_counts.items():
            raw_quota = int(count / total_rows * pool_size)
            if raw_quota > effective_cap:
                quotas[entry_key] = effective_cap
            else:
                quotas[entry_key] = raw_quota
                uncapped_rows += count

        # Redistribute excess from capped entries
        remainder = pool_size - sum(quotas.values())
        if remainder > 0 and uncapped_rows > 0:
            for entry_key, count in entry_counts.items():
                if quotas[entry_key] < effective_cap:
                    extra = int(count / uncapped_rows * remainder)
                    quotas[entry_key] = min(quotas[entry_key] + extra, effective_cap)

        return quotas

    def _farthest_first(
        self, pool: np.ndarray, n_select: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select n_select points by iterative farthest-point sampling.

        Returns (selected_points, selected_indices) so callers can subset
        parallel arrays (e.g. entry labels).
        """
        first = int(self._rng.integers(0, int(pool.shape[0])))
        selected: list[int] = [first]
        distances = np.sum((pool - pool[selected[0]]) ** 2, axis=1)

        while len(selected) < min(n_select, pool.shape[0]):
            farthest = int(np.argmax(distances))
            selected.append(farthest)
            new_distances = np.sum((pool - pool[farthest]) ** 2, axis=1)
            distances = np.minimum(distances, new_distances)

        sel = np.array(selected)
        return pool[sel], sel

    def _build_provenance(self) -> pd.DataFrame:
        """Build per-entry template provenance table.

        Every entry from _entry_map appears, even those with 0 templates.
        """
        if self._template_entries is None:
            total = 0
            count_map: dict[int, int] = {}
        else:
            unique_indices, counts = np.unique(
                self._template_entries, return_counts=True
            )
            total = int(counts.sum())
            count_map = dict(zip(unique_indices.tolist(), counts.tolist()))

        rows: list[dict[str, str | int | float]] = []
        for entry_idx in sorted(self._entry_map):
            group, sequence = self._entry_map[entry_idx]
            count = count_map.get(entry_idx, 0)
            rows.append(
                {
                    "group": group,
                    "sequence": sequence,
                    "count": count,
                    "proportion": float(count) / total if total > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def save_state(self, run_root: Path) -> None:
        if self._templates is None or self._feature_columns is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._templates, columns=self._feature_columns)
        df.to_parquet(run_root / "templates.parquet", index=False)

        if self._provenance is not None:
            self._provenance.to_parquet(
                run_root / "template_provenance.parquet", index=False
            )
