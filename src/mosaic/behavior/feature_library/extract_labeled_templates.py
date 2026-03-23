from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, Self, final

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from mosaic.core.helpers import load_labels_for_feature_frames
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    DependencyLookup,
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    InputStream,
    Params,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
)

from .helpers import feature_columns
from .registry import register_feature
from .types import PoolConfig


@dataclass
class _Reservoir:
    """Per-(split, class) reservoir buffer."""

    buffer: np.ndarray
    entry_indices: np.ndarray
    filled: int = 0
    n_seen: int = 0


class LabeledTemplatesArtifact(ParquetArtifact):
    """Labeled template feature vectors (templates.parquet).

    Uses numeric_only=False because the parquet contains the str 'split'
    column alongside numeric feature columns and int 'label'.
    """

    feature: str = "extract-labeled-templates"
    pattern: str = "templates.parquet"
    load: ParquetLoadSpec = Field(
        default_factory=lambda: ParquetLoadSpec(numeric_only=False)
    )


class LabeledProvenanceArtifact(ParquetArtifact):
    """Per-entry template provenance (template_provenance.parquet)."""

    feature: str = "extract-labeled-templates"
    pattern: str = "template_provenance.parquet"
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


@final
@register_feature
class ExtractLabeledTemplates:
    """Extract labeled, split-annotated templates from upstream features.

    Streams upstream feature data, aligns ground truth labels from NPZ
    files, assigns train/test splits by sequence, and subsamples per
    class. Produces a templates parquet with feature columns + label
    (int) + split (str).
    """

    name = "extract-labeled-templates"
    version = "0.1"
    parallelizable = False
    scope_dependent = True

    LabeledTemplatesArtifact = LabeledTemplatesArtifact
    LabeledProvenanceArtifact = LabeledProvenanceArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        labels: GroundTruthLabelsSource
        strategy: Literal["random", "farthest_first"] = "random"
        n_per_class: int | Mapping[int, int] | None = None
        n_total: int | None = None
        pool: PoolConfig = Field(default_factory=PoolConfig)
        test_fraction: float = Field(default=0.2, ge=0.0, le=1.0)
        random_state: int = 42

        @model_validator(mode="after")
        def _require_sampling_spec(self) -> Self:
            has_per_class = self.n_per_class is not None
            has_total = self.n_total is not None
            if has_per_class == has_total:
                msg = "Exactly one of 'n_per_class' or 'n_total' must be set"
                raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: ExtractLabeledTemplates.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)

        self._feature_columns: list[str] | None = None
        self._templates: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._splits: np.ndarray | None = None
        self._template_entries: np.ndarray | None = None
        self._entry_map: dict[int, tuple[str, str]] = {}
        self._provenance: pd.DataFrame | None = None
        self._sequence_splits: dict[tuple[str, str], str] = {}
        self._labels_lookup: dict[tuple[str, str], Path] = {}
        self._rng = np.random.default_rng(self.params.random_state)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._feature_columns = None
        self._templates = None
        self._labels = None
        self._splits = None
        self._template_entries = None
        self._entry_map = {}
        self._provenance = None
        self._sequence_splits = {}
        self._labels_lookup = dependency_lookups.get("labels", {})

        path = run_root / "templates.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            label_col = df["label"].to_numpy(dtype=np.int64)
            split_col = df["split"].to_numpy()
            feature_cols = feature_columns(df)
            self._feature_columns = feature_cols
            self._templates = df[feature_cols].to_numpy(dtype=np.float64)
            self._labels = label_col
            self._splits = split_col

            provenance_path = run_root / "template_provenance.parquet"
            if provenance_path.exists():
                self._provenance = pd.read_parquet(provenance_path)
                for _, row in self._provenance.iterrows():
                    self._sequence_splits[(row["group"], row["sequence"])] = row["split"]

            return True
        return False

    def fit(self, inputs: InputStream) -> None:
        # NOTE: bout-aware sampling strategy
        #
        # The current implementation uses proportional random sampling, which
        # over-represents long stationary periods and under-represents transitions
        # and short behavioral events. A bout-aware strategy would identify
        # consecutive frames with the same label as "bouts" and distribute
        # sampling across bouts, ensuring each distinct behavioral episode
        # contributes proportionally. This captures the diversity of behavioral
        # expression rather than just the most common states. Bout-aware
        # sampling is a planned future enhancement.

        # Pre-compute deterministic split assignments from entry count.
        # Guarantees exactly n_test entries in test (minimum 1 when
        # test_fraction > 0, maximum n-1 when test_fraction < 1).
        split_assignments = self._compute_split_assignments(inputs.n_entries)

        if self.params.pool.allocation == "exact":
            self._fit_exact(inputs, split_assignments)
        else:
            self._fit_reservoir(inputs, split_assignments)

        if self._templates is None:
            msg = "[extract-labeled-templates] No data after pool construction."
            raise RuntimeError(msg)

        if self.params.strategy == "farthest_first":
            self._apply_farthest_first_per_class()

        self._provenance = self._build_provenance()

    def _final_target_for_class(
        self,
        class_id: int,
        n_classes_seen: int,
    ) -> int:
        """Compute the final target count for a given class."""
        n_per_class = self.params.n_per_class
        if isinstance(n_per_class, int):
            return n_per_class
        if isinstance(n_per_class, Mapping):
            return n_per_class.get(class_id, 0)
        # n_total mode
        if self.params.n_total is not None and n_classes_seen > 0:
            return self.params.n_total // n_classes_seen
        return 0

    def _collection_target_for_class(
        self,
        class_id: int,
        n_classes_seen: int,
    ) -> int:
        """Compute the reservoir/collection target for a given class.

        For farthest_first strategy with a pool.size, the collection target
        is the pool size (per class) so the oversized pool can be built
        before farthest-first selection reduces it to the final target.
        """
        final = self._final_target_for_class(class_id, n_classes_seen)
        if (
            self.params.strategy == "farthest_first"
            and self.params.pool.size is not None
        ):
            return self.params.pool.size
        return final

    def _compute_split_assignments(self, n_entries: int) -> list[str]:
        """Compute exact train/test split assignments for n_entries sequences.

        Produces a list of length n_entries where each element is "train"
        or "test". The number of test entries is ``max(1, round(f * n))``
        clamped to ``[1, n-1]`` when ``test_fraction`` is in ``(0, 1)``.
        Which entries are test is randomized via ``self._rng``.
        """
        f = self.params.test_fraction
        if f <= 0.0 or n_entries <= 1:
            return ["train"] * n_entries
        if f >= 1.0:
            return ["test"] * n_entries
        n_test = max(1, min(n_entries - 1, round(f * n_entries)))
        indices = list(range(n_entries))
        self._rng.shuffle(indices)
        test_set = set(indices[:n_test])
        return ["test" if i in test_set else "train" for i in range(n_entries)]

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

    def _apply_farthest_first_per_class(self) -> None:
        """Apply farthest-first selection per (split, label) group.

        Reduces the oversized collection pool down to the final target
        count for each class.
        """
        assert self._templates is not None
        assert self._labels is not None
        assert self._splits is not None

        n_classes = len(np.unique(self._labels))

        template_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        split_blocks: list[np.ndarray] = []
        entry_blocks: list[np.ndarray] = []

        # Iterate unique (split, label) pairs
        unique_splits = np.unique(self._splits)
        unique_labels = np.unique(self._labels)

        for split in unique_splits:
            for label_int in unique_labels:
                mask = (self._splits == split) & (self._labels == label_int)
                subset = self._templates[mask]
                if subset.shape[0] == 0:
                    continue

                target = self._final_target_for_class(int(label_int), n_classes)
                if target <= 0:
                    continue

                if subset.shape[0] > target:
                    selected, sel_idx = self._farthest_first(subset, target)
                    template_blocks.append(selected)
                    if self._template_entries is not None:
                        entry_blocks.append(self._template_entries[mask][sel_idx])
                else:
                    template_blocks.append(subset)
                    if self._template_entries is not None:
                        entry_blocks.append(self._template_entries[mask])

                n_out = int(template_blocks[-1].shape[0])
                label_arr: np.ndarray = np.full(n_out, int(label_int), dtype=np.int64)
                label_blocks.append(label_arr)
                split_arr: np.ndarray = np.array([split] * n_out, dtype=object)
                split_blocks.append(split_arr)

        if not template_blocks:
            return

        self._templates = np.vstack(template_blocks)
        self._labels = np.concatenate(label_blocks)
        self._splits = np.concatenate(split_blocks)
        if entry_blocks:
            self._template_entries = np.concatenate(entry_blocks)

    def _fit_reservoir(
        self,
        inputs: InputStream,
        split_assignments: list[str],
    ) -> None:
        reservoirs: dict[tuple[str, int], _Reservoir] = {}
        n_entries = 0
        classes_seen: set[int] = set()

        for _entry_key, df in inputs():
            if self._feature_columns is None:
                self._feature_columns = feature_columns(df)

            group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
            sequence = str(df[C.seq_col].iloc[0])
            entry_idx = n_entries
            self._entry_map[entry_idx] = (group, sequence)
            split = split_assignments[n_entries]
            self._sequence_splits[(group, sequence)] = split
            n_entries += 1

            # Look up label file
            npz_path = self._labels_lookup.get((group, sequence))
            if npz_path is None:
                continue

            labels = load_labels_for_feature_frames(
                npz_path, df[C.frame_col].to_numpy()
            )
            matrix = df[self._feature_columns].to_numpy(dtype=np.float64)

            # Check for newly discovered classes and rebalance if using n_total
            unique_labels = set(np.unique(labels).tolist())
            new_classes = unique_labels - classes_seen
            if new_classes and self.params.n_total is not None:
                classes_seen |= new_classes
                # Use collection target (which accounts for farthest_first pool)
                new_target = self._collection_target_for_class(0, len(classes_seen))
                # Truncate existing reservoirs to new target
                for key, reservoir in reservoirs.items():
                    if reservoir.filled > new_target:
                        reservoir.buffer = reservoir.buffer[:new_target].copy()
                        reservoir.entry_indices = reservoir.entry_indices[
                            :new_target
                        ].copy()
                        reservoir.filled = new_target
            else:
                classes_seen |= new_classes

            # Process rows grouped by label
            for label in sorted(unique_labels):
                label_int = int(label)
                mask = labels == label_int
                label_matrix = matrix[mask]
                n_rows = int(label_matrix.shape[0])
                if n_rows == 0:
                    continue

                key = (split, label_int)
                target = self._collection_target_for_class(label_int, len(classes_seen))
                if target <= 0:
                    continue

                if key not in reservoirs:
                    n_features = label_matrix.shape[1]
                    reservoirs[key] = _Reservoir(
                        buffer=np.empty((target, n_features), dtype=np.float64),
                        entry_indices=np.empty(target, dtype=np.int32),
                    )

                reservoir = reservoirs[key]

                # Resize if target changed (n_total rebalancing)
                if reservoir.buffer.shape[0] != target:
                    new_buffer = np.empty(
                        (target, reservoir.buffer.shape[1]), dtype=np.float64
                    )
                    new_entries = np.empty(target, dtype=np.int32)
                    keep = min(reservoir.filled, target)
                    new_buffer[:keep] = reservoir.buffer[:keep]
                    new_entries[:keep] = reservoir.entry_indices[:keep]
                    reservoir.buffer = new_buffer
                    reservoir.entry_indices = new_entries
                    reservoir.filled = keep

                # Feed rows into the reservoir (Algorithm R, vectorized)
                filled = reservoir.filled
                n_seen = reservoir.n_seen

                if filled < target:
                    space = target - filled
                    if n_rows <= space:
                        reservoir.buffer[filled : filled + n_rows] = label_matrix
                        reservoir.entry_indices[filled : filled + n_rows] = entry_idx
                        reservoir.filled += n_rows
                        reservoir.n_seen += n_rows
                        continue
                    else:
                        reservoir.buffer[filled:target] = label_matrix[:space]
                        reservoir.entry_indices[filled:target] = entry_idx
                        reservoir.filled = target
                        reservoir.n_seen += space
                        label_matrix = label_matrix[space:]
                        n_rows = int(label_matrix.shape[0])
                        n_seen = reservoir.n_seen

                if n_rows > 0:
                    positions = np.arange(
                        n_seen + 1, n_seen + n_rows + 1, dtype=np.float64
                    )
                    probabilities = target / positions
                    randoms: np.ndarray = self._rng.random(n_rows)
                    include_mask: np.ndarray = randoms < probabilities
                    included = label_matrix[include_mask]
                    n_included = int(included.shape[0])
                    if n_included > 0:
                        slots: np.ndarray = self._rng.integers(
                            0, target, size=n_included
                        )
                        reservoir.buffer[slots] = included
                        reservoir.entry_indices[slots] = entry_idx
                    reservoir.n_seen += n_rows

        if not reservoirs:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        # Concatenate all reservoirs
        template_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        split_blocks: list[np.ndarray] = []
        entry_blocks: list[np.ndarray] = []

        for (split, label), reservoir in sorted(reservoirs.items()):
            if reservoir.filled == 0:
                continue
            template_blocks.append(reservoir.buffer[: reservoir.filled])
            label_blocks.append(np.full(reservoir.filled, label, dtype=np.int64))
            split_blocks.append(np.array([split] * reservoir.filled, dtype=object))
            entry_blocks.append(reservoir.entry_indices[: reservoir.filled])

        if not template_blocks:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        self._templates = np.vstack(template_blocks)
        self._labels = np.concatenate(label_blocks)
        self._splits = np.concatenate(split_blocks)
        self._template_entries = np.concatenate(entry_blocks)

    def _fit_exact(
        self,
        inputs: InputStream,
        split_assignments: list[str],
    ) -> None:
        """Two-pass exact allocation: count, then sample per (split, class)."""

        # Pass 1: count rows per (split, class), capture metadata
        class_counts: dict[tuple[str, int], int] = {}
        entry_splits: dict[str, str] = {}
        entry_idx_map: dict[str, int] = {}
        n_seen = 0

        for entry_key, df in inputs():
            if self._feature_columns is None:
                self._feature_columns = feature_columns(df)

            group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
            sequence = str(df[C.seq_col].iloc[0])

            if entry_key not in entry_idx_map:
                idx = len(entry_idx_map)
                entry_idx_map[entry_key] = idx
                self._entry_map[idx] = (group, sequence)

            split = split_assignments[n_seen]
            n_seen += 1
            entry_splits[entry_key] = split
            self._sequence_splits[(group, sequence)] = split

            npz_path = self._labels_lookup.get((group, sequence))
            if npz_path is None:
                continue

            labels = load_labels_for_feature_frames(
                npz_path, df[C.frame_col].to_numpy()
            )
            for label_int in np.unique(labels).tolist():
                key = (split, int(label_int))
                class_counts[key] = class_counts.get(key, 0) + int(
                    (labels == label_int).sum()
                )

        if not class_counts:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        # Compute targets per (split, class)
        all_classes = sorted({label for _, label in class_counts})
        n_classes = len(all_classes)

        targets: dict[tuple[str, int], int] = {}
        for key in class_counts:
            split, label_int = key
            targets[key] = self._collection_target_for_class(label_int, n_classes)

        # Compute per-(split, class) quotas from counts
        quotas: dict[tuple[str, int], int] = {}
        for key, count in class_counts.items():
            target = targets.get(key, 0)
            quotas[key] = min(count, target)

        # Pass 2: sample exactly quota rows per (split, class)
        # Use a derived seed for sampling RNG (independent of split-assignment draws)
        sampling_rng = np.random.default_rng(self.params.random_state + 1)
        # Track how many we've sampled per key so far
        sampled: dict[tuple[str, int], int] = {k: 0 for k in quotas}
        remaining: dict[tuple[str, int], int] = dict(class_counts)

        template_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        split_blocks: list[np.ndarray] = []
        entry_blocks: list[np.ndarray] = []

        cols = self._feature_columns
        assert cols is not None

        for entry_key, df in inputs():
            group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
            sequence = str(df[C.seq_col].iloc[0])
            split = entry_splits[entry_key]
            entry_idx = entry_idx_map[entry_key]

            npz_path = self._labels_lookup.get((group, sequence))
            if npz_path is None:
                continue

            labels = load_labels_for_feature_frames(
                npz_path, df[C.frame_col].to_numpy()
            )
            matrix = df[cols].to_numpy(dtype=np.float64)

            for label_int in np.unique(labels).tolist():
                label_int = int(label_int)
                key = (split, label_int)
                quota = quotas.get(key, 0)
                already = sampled.get(key, 0)
                need = quota - already
                if need <= 0:
                    continue

                mask = labels == label_int
                label_matrix = matrix[mask]
                n_rows = int(label_matrix.shape[0])
                rem = remaining.get(key, n_rows)

                # Sample proportionally: this entry contributes n_rows / rem
                # of the remaining quota
                take = min(n_rows, need)
                if n_rows > take:
                    sample_idx: np.ndarray = sampling_rng.choice(
                        n_rows, size=take, replace=False
                    )
                    label_matrix = label_matrix[sample_idx]

                sampled_block: np.ndarray = label_matrix[:take]
                template_blocks.append(sampled_block)
                label_blocks.append(np.full(take, label_int, dtype=np.int64))
                split_blocks.append(np.array([split] * take, dtype=object))
                entry_blocks.append(np.full(take, entry_idx, dtype=np.int32))
                sampled[key] = already + take
                remaining[key] = rem - n_rows

        if not template_blocks:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        self._templates = np.vstack(template_blocks)
        self._labels = np.concatenate(label_blocks)
        self._splits = np.concatenate(split_blocks)
        self._template_entries = np.concatenate(entry_blocks)

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
            split = self._sequence_splits.get((group, sequence), "train")
            rows.append(
                {
                    "group": group,
                    "sequence": sequence,
                    "split": split,
                    "count": count,
                    "proportion": float(count) / total if total > 0 else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        group = str(df[C.group_col].iloc[0]) if C.group_col in df.columns else ""
        sequence = str(df[C.seq_col].iloc[0])

        # Add split assignment
        split = self._sequence_splits.get((group, sequence), "train")
        df = df.copy()
        df["split"] = split

        # Add aligned labels
        npz_path = self._labels_lookup.get((group, sequence))
        if npz_path is not None and C.frame_col in df.columns:
            labels = load_labels_for_feature_frames(
                npz_path, df[C.frame_col].to_numpy()
            )
            df["label"] = labels
        else:
            df["label"] = 0

        return df

    def save_state(self, run_root: Path) -> None:
        if (
            self._templates is None
            or self._feature_columns is None
            or self._labels is None
            or self._splits is None
            or self._provenance is None
        ):
            return
        run_root.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(self._templates, columns=self._feature_columns)
        df["label"] = self._labels
        df["split"] = self._splits
        df.to_parquet(run_root / "templates.parquet", index=False)

        self._provenance.to_parquet(
            run_root / "template_provenance.parquet", index=False
        )
