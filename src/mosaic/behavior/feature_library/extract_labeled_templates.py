from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self, cast, final

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from mosaic.core.helpers import UNLABELED
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.column_contract import (
    Column,
    Payload,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import (
    COLUMNS as C,
    EmitsLevel,
)
from mosaic.core.pipeline.types import (
    TEMPLATES_ARTIFACT_NAME,
    DependencyLookup,
    GroundTruthLabelsSource,
    InputRequire,
    Inputs,
    InputStream,
    LabeledTemplatesRef,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
)
from mosaic.core.params import Declared, Params

from .helpers import (
    check_memory_headroom,
    chosen_payload_columns,
    ensure_columns,
    entry_identity,
    farthest_first,
    feature_columns,
)
from .registry import register_feature
from .types import PAYLOAD_COLUMNS_DESCRIPTION, PayloadColumnNames, PoolConfig
from mosaic.core.pipeline.writers import write_parquet_atomic

_LABELS_DESCRIPTION = (
    "Reference to the per-frame ground-truth labels used to assign a "
    "class to every template, resolved from labels/<kind>/index.csv."
)

_STRATEGY_DESCRIPTION = (
    "How templates are selected from the candidate pool. random keeps "
    "the pool as collected. farthest_first reduces an oversized pool to "
    "the target count per class by iterative farthest-point sampling."
)

_N_PER_CLASS_DESCRIPTION = (
    "The number of templates per class. An int applies the same count to "
    "every class, and a mapping sets each class's count individually. A "
    "mapping value of null (or the string all) takes every labeled frame of "
    "that class, and a class the mapping does not list contributes none. "
    "Exactly one of n_per_class and n_total must be set."
)

_N_TOTAL_DESCRIPTION = (
    "The total number of templates, split equally across the classes "
    "seen. Each class gets n_total // (number of classes), so the "
    "realized total can be lower. Exactly one of n_per_class and n_total "
    "must be set."
)

_POOL_DESCRIPTION = (
    "Candidate pool size and per-entry allocation strategy consulted "
    "before final template selection."
)

_TEST_FRACTION_DESCRIPTION = "Fraction of sequences held out for the test split."

_RANDOM_STATE_DESCRIPTION = (
    "The random seed for split assignment and template sampling."
)


@dataclass
class _Reservoir:
    """Per-(split, class) reservoir buffer.

    ``buffer`` holds ``filled`` rows and room for more. It grows with the rows
    it receives, up to ``target`` rows, or without a limit when ``target`` is
    ``None``, which takes every row.
    """

    buffer: np.ndarray
    entry_indices: np.ndarray
    frames: np.ndarray
    target: int | None
    filled: int = 0
    n_seen: int = 0

    @property
    def capacity(self) -> int:
        return int(self.buffer.shape[0])

    def grown_capacity(self, rows: int) -> int:
        """Return the capacity that :meth:`reserve` gives for *rows* rows in all."""
        if rows <= self.capacity:
            return self.capacity
        grown = max(rows, 2 * self.capacity)
        return grown if self.target is None else min(grown, self.target)

    def reserve(self, rows: int) -> None:
        """Make room for *rows* rows in all, doubling the buffer as it grows."""
        grown = self.grown_capacity(rows)
        if grown == self.capacity:
            return
        buffer = np.empty((grown, self.buffer.shape[1]), dtype=np.float64)
        entries = np.empty(grown, dtype=np.int32)
        frames = np.empty(grown, dtype=np.int64)
        buffer[: self.filled] = self.buffer[: self.filled]
        entries[: self.filled] = self.entry_indices[: self.filled]
        frames[: self.filled] = self.frames[: self.filled]
        self.buffer, self.entry_indices, self.frames = buffer, entries, frames

    def truncate(self, rows: int) -> None:
        """Keep the first *rows* rows, and cap the reservoir at that many."""
        self.target = rows
        if self.filled > rows:
            self.buffer = self.buffer[:rows].copy()
            self.entry_indices = self.entry_indices[:rows].copy()
            self.frames = self.frames[:rows].copy()
            self.filled = rows


def _check_draw_memory(rows: int, n_features: int) -> None:
    """Refuse a sample of *rows* templates that would not fit in memory.

    The estimate counts the sample's buffers and the copy that stacks them into
    the table that is written.
    """
    check_memory_headroom(
        2 * rows * n_features * 8,
        "extract-labeled-templates",
        f"drawing {rows:,} templates of {n_features} columns",
        "Lower n_per_class or n_total, give a large class a count instead of "
        "null, or name fewer columns with `columns`.",
    )


def _proportional_shares(counts: Mapping[int, int], quota: int) -> dict[int, int]:
    """Split *quota* across entries in proportion to the rows each holds.

    The largest-remainder method on integers: each entry takes the floor of its
    exact share, and the rows still owed go one each to the entries with the
    largest remainders, the earlier entry first on a tie. The shares add to
    *quota* exactly and no entry's share exceeds its rows, because *quota* is at
    most the total.

    Args:
        counts: The rows each entry holds, keyed by entry index.
        quota: The rows to draw, at most ``sum(counts.values())``.

    Returns:
        The rows each entry draws, keyed as *counts* is.
    """
    total = sum(counts.values())
    if quota >= total:
        return dict(counts)
    shares = {entry: quota * rows // total for entry, rows in counts.items()}
    owed = quota - sum(shares.values())
    by_remainder = sorted(
        counts, key=lambda entry: (-(quota * counts[entry] % total), entry)
    )
    for entry in by_remainder[:owed]:
        shares[entry] += 1
    return shares


class LabeledTemplatesArtifact(LabeledTemplatesRef):
    """Labeled template feature vectors, named for the run that produced them.

    The filename and the ``numeric_only=False`` load spec come from
    ``LabeledTemplatesRef``; all this adds is which feature wrote them, which is
    what ``from_result`` validates against.
    """

    feature: str = "extract-labeled-templates"


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
    (int) + split (str) + group and sequence (str), the entry each row was
    drawn from, and frame (int), the frame it was drawn at.

    The run writes ``templates.parquet`` and ``template_provenance.parquet``.
    The provenance records each entry's split and its labeled frames: the
    ``labeled_start`` and ``labeled_stop`` of an entry run from the first frame
    of any series that a label covers to one past the last, empty for an entry
    with no label. A classifier that cross-validates by block cuts those frames
    into blocks. Each entry's table is empty. A
    step that consumes the templates reads this step's inputs and names the
    templates through a params field.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.extract_labeled_templates.ExtractLabeledTemplates.Params`.
    """

    category = "global"
    costly_fit = False  # cheap to refit, so a cleanup may remove a superseded run
    fits = True  # learns from its inputs, so a held-out group must stay out of its fit
    name = "extract-labeled-templates"
    # 0.2: farthest_first compares rows over their finite columns, draws its
    # first row from those with a finite value, and selects each row once. A
    # row containing NaN made every distance NaN, a first row with no finite
    # value made every distance 0, and either way the selection repeated row 0.
    # 0.3: ``nn_id`` and ``attention_target_id``, track ids, are no longer
    # template columns (``BOOKKEEPING_COLS``).
    # 0.4: a row no label event matches is unlabeled (-1) and never sampled,
    # where it was class 0, and a pair row takes the label of its own direction
    # (``labels_for_rows``), so a CalMS21 intruder-focal row is unlabeled.
    # 0.5: each entry's table is empty, where it copied the entry's input with
    # its split and label. A consumer reads this step's inputs, and the templates
    # through a params field. ``template_provenance.parquet`` records each
    # entry's split.
    # 0.6: ``group`` and ``sequence`` (text) are written beside ``label`` and
    # ``split``, naming the entry each row was drawn from, so that a classifier
    # can hold a sequence out of its own fit. They are identity columns and no
    # feature (``feature_columns``). ``pool.allocation="exact"`` gives every
    # entry its proportional share of a class's quota, where the first entries
    # took the whole quota and the later ones drew nothing. A mapping count of
    # ``None`` takes every labeled frame of a class, a reservoir grows with the
    # rows it receives instead of being allocated at its target, and a draw
    # that would not fit in memory is refused before it starts.
    # 0.7: ``frame`` is written beside ``group`` and ``sequence``, the frame each
    # row was drawn at, and the provenance records each entry's labeled frames
    # (``labeled_start``, ``labeled_stop``). ``frame`` is an identity column and
    # no feature. A classifier that cross-validates by block (``cv_unit="block"``)
    # cuts the labeled frames of each train entry into blocks and tells a template
    # row's block by its frame. The sample itself is as 0.6 drew it.
    version = "0.7"
    parallelizable = False
    scope_dependent = True
    accepts_overlap = False  # computes within a frame, so gains nothing
    temporal_reach = 0  # draws rows of its inputs as they are
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    # Fit aligns each entry's labels by frame and samples its payload columns,
    # every one or those that ``columns`` names. Apply returns an empty table,
    # because the sample in ``save_state`` is the output. ``columns`` is not
    # declared, because a requirement on a list param is met by any one of its
    # names. The fit refuses each named column that the input lacks.
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.seq_col),
        Column(C.frame_col),
        Payload(),
    )
    provides: ClassVar[Provides] = Provides()
    LabeledTemplatesArtifact = LabeledTemplatesArtifact
    LabeledProvenanceArtifact = LabeledProvenanceArtifact

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

    class Params(Params):
        labels: Annotated[GroundTruthLabelsSource, Declared(_LABELS_DESCRIPTION)]
        strategy: Annotated[
            Literal["random", "farthest_first"], Declared(_STRATEGY_DESCRIPTION)
        ] = "random"
        n_per_class: Annotated[
            int | Mapping[int, int | None] | None, Declared(_N_PER_CLASS_DESCRIPTION)
        ] = None
        n_total: Annotated[int | None, Declared(_N_TOTAL_DESCRIPTION)] = None
        pool: Annotated[PoolConfig, Declared(_POOL_DESCRIPTION)] = Field(
            default_factory=PoolConfig
        )
        test_fraction: Annotated[float, Declared(_TEST_FRACTION_DESCRIPTION)] = Field(
            default=0.2, ge=0.0, le=1.0
        )
        random_state: Annotated[int, Declared(_RANDOM_STATE_DESCRIPTION)] = 42
        columns: Annotated[
            PayloadColumnNames | None, Declared(PAYLOAD_COLUMNS_DESCRIPTION)
        ] = None

        @field_validator("n_per_class", mode="before")
        @classmethod
        def _all_is_null(cls, value: object) -> object:
            """Spell "every frame" one way, so that one sample has one run id."""
            if isinstance(value, Mapping):
                counts = cast("Mapping[object, object]", value)
                return {
                    label: None if count == "all" else count
                    for label, count in counts.items()
                }
            return value

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
        # The group and the sequence of each template row, as text.
        self._row_identity: tuple[np.ndarray, np.ndarray] | None = None
        # The frame of each template row, in the order of the rows.
        self._template_frames: np.ndarray | None = None
        self._entry_map: dict[int, tuple[str, str]] = {}
        self._provenance: pd.DataFrame | None = None
        self._sequence_splits: dict[tuple[str, str], str] = {}
        # The first labeled frame of each entry and one past its last.
        self._labeled_frames: dict[tuple[str, str], tuple[int, int]] = {}
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
        self._row_identity = None
        self._template_frames = None
        self._entry_map = {}
        self._provenance = None
        self._sequence_splits = {}
        self._labeled_frames = {}
        self._labels_lookup = dependency_lookups.get("labels", {})

        path = run_root / TEMPLATES_ARTIFACT_NAME
        if path.exists():
            df = pd.read_parquet(path)
            label_col = df["label"].to_numpy(dtype=np.int64)
            split_col = df["split"].to_numpy()
            feature_cols = feature_columns(df)
            self._feature_columns = feature_cols
            self._templates = df[feature_cols].to_numpy(dtype=np.float64)
            self._labels = label_col
            self._splits = split_col
            if {C.group_col, C.seq_col} <= set(df.columns):
                self._row_identity = (
                    df[C.group_col].to_numpy(dtype=object),
                    df[C.seq_col].to_numpy(dtype=object),
                )
            if C.frame_col in df.columns:
                self._template_frames = df[C.frame_col].to_numpy(dtype=np.int64)

            provenance_path = run_root / "template_provenance.parquet"
            if provenance_path.exists():
                self._provenance = pd.read_parquet(provenance_path)

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

        self._row_identity = self._identity_of_rows()
        self._provenance = self._build_provenance()

    def _identity_of_rows(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The group and the sequence of the entry each template row was drawn from."""
        if self._template_entries is None:
            return None
        order = sorted(self._entry_map)
        groups = np.array([self._entry_map[i][0] for i in order], dtype=object)
        sequences = np.array([self._entry_map[i][1] for i in order], dtype=object)
        return groups[self._template_entries], sequences[self._template_entries]

    def _final_target_for_class(
        self,
        class_id: int,
        n_classes_seen: int,
    ) -> int | None:
        """Compute the final target count for a given class, ``None`` for every frame."""
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
    ) -> int | None:
        """Compute the reservoir/collection target for a given class.

        For farthest_first strategy with a pool.size, the collection target
        is the pool size (per class) so the oversized pool can be built
        before farthest-first selection reduces it to the final target.
        ``None`` collects every frame of the class.
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
        frame_blocks: list[np.ndarray] = []

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
                if target is not None and target <= 0:
                    continue

                if target is not None and subset.shape[0] > target:
                    selected, sel_idx = farthest_first(subset, target, self._rng)
                    template_blocks.append(selected)
                    if self._template_entries is not None:
                        entry_blocks.append(self._template_entries[mask][sel_idx])
                    if self._template_frames is not None:
                        frame_blocks.append(self._template_frames[mask][sel_idx])
                else:
                    template_blocks.append(subset)
                    if self._template_entries is not None:
                        entry_blocks.append(self._template_entries[mask])
                    if self._template_frames is not None:
                        frame_blocks.append(self._template_frames[mask])

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
        if frame_blocks:
            self._template_frames = np.concatenate(frame_blocks)

    def _note_labeled_frames(
        self, entry: tuple[str, str], df: pd.DataFrame, labels: np.ndarray
    ) -> None:
        """Widen the labeled frames of *entry* by those of the labeled rows of *df*.

        Args:
            entry: The ``(group, sequence)`` *df* belongs to.
            df: The entry's input, with a ``frame`` column.
            labels: The label of each row of *df*, ``UNLABELED`` for none.
        """
        labeled = df[C.frame_col].to_numpy()[labels != UNLABELED]
        if len(labeled) == 0:
            return
        first, stop = self._labeled_frames.get(entry, (int(labeled.min()), 0))
        self._labeled_frames[entry] = (
            min(first, int(labeled.min())),
            max(stop, int(labeled.max()) + 1),
        )

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
                self._feature_columns = chosen_payload_columns(
                    df, self.params.columns, self.name
                )
                ensure_columns(df, [C.seq_col])

            group, sequence = entry_identity(df)
            entry_idx = n_entries
            self._entry_map[entry_idx] = (group, sequence)
            split = split_assignments[n_entries]
            self._sequence_splits[(group, sequence)] = split
            n_entries += 1

            # Look up label file
            npz_path = self._labels_lookup.get((group, sequence))
            if npz_path is None:
                continue

            labels = labels_for_rows(npz_path, df)
            keep = labels != UNLABELED
            self._note_labeled_frames((group, sequence), df, labels)
            labels = labels[keep]
            matrix = df[self._feature_columns].to_numpy(dtype=np.float64)[keep]
            row_frames = df[C.frame_col].to_numpy(dtype=np.int64)[keep]

            # Check for newly discovered classes and rebalance if using n_total
            unique_labels = set(np.unique(labels).tolist())
            new_classes = unique_labels - classes_seen
            if new_classes and self.params.n_total is not None:
                classes_seen |= new_classes
                # Use collection target (which accounts for farthest_first pool)
                new_target = self._collection_target_for_class(0, len(classes_seen))
                assert new_target is not None
                # Truncate existing reservoirs to new target
                for reservoir in reservoirs.values():
                    reservoir.truncate(new_target)
            else:
                classes_seen |= new_classes

            # Process rows grouped by label
            for label in sorted(unique_labels):
                label_int = int(label)
                mask = labels == label_int
                label_matrix = matrix[mask]
                label_frames = row_frames[mask]
                n_rows = int(label_matrix.shape[0])
                if n_rows == 0:
                    continue

                key = (split, label_int)
                target = self._collection_target_for_class(label_int, len(classes_seen))
                if target is not None and target <= 0:
                    continue

                if key not in reservoirs:
                    n_features = label_matrix.shape[1]
                    reservoirs[key] = _Reservoir(
                        buffer=np.empty((0, n_features), dtype=np.float64),
                        entry_indices=np.empty(0, dtype=np.int32),
                        frames=np.empty(0, dtype=np.int64),
                        target=target,
                    )

                reservoir = reservoirs[key]

                # The target changed (n_total rebalancing)
                if reservoir.target != target:
                    assert target is not None
                    reservoir.truncate(target)

                # Feed rows into the reservoir (Algorithm R, vectorized)
                filled = reservoir.filled
                n_seen = reservoir.n_seen

                if target is None or filled < target:
                    space = n_rows if target is None else min(n_rows, target - filled)
                    self._reserve(reservoirs, reservoir, filled + space)
                    reservoir.buffer[filled : filled + space] = label_matrix[:space]
                    reservoir.entry_indices[filled : filled + space] = entry_idx
                    reservoir.frames[filled : filled + space] = label_frames[:space]
                    reservoir.filled += space
                    reservoir.n_seen += space
                    label_matrix = label_matrix[space:]
                    label_frames = label_frames[space:]
                    n_rows = int(label_matrix.shape[0])
                    n_seen = reservoir.n_seen

                if n_rows > 0:
                    assert target is not None
                    positions = np.arange(
                        n_seen + 1, n_seen + n_rows + 1, dtype=np.float64
                    )
                    probabilities = target / positions
                    randoms: np.ndarray = self._rng.random(n_rows)
                    include_mask: np.ndarray = randoms < probabilities
                    included = label_matrix[include_mask]
                    included_frames = label_frames[include_mask]
                    n_included = int(included.shape[0])
                    if n_included > 0:
                        slots: np.ndarray = self._rng.integers(
                            0, target, size=n_included
                        )
                        reservoir.buffer[slots] = included
                        reservoir.entry_indices[slots] = entry_idx
                        reservoir.frames[slots] = included_frames
                    reservoir.n_seen += n_rows

        if not reservoirs:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        # Concatenate all reservoirs
        template_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        split_blocks: list[np.ndarray] = []
        entry_blocks: list[np.ndarray] = []
        frame_blocks: list[np.ndarray] = []

        for (split, label), reservoir in sorted(reservoirs.items()):
            if reservoir.filled == 0:
                continue
            template_blocks.append(reservoir.buffer[: reservoir.filled])
            label_blocks.append(np.full(reservoir.filled, label, dtype=np.int64))
            split_blocks.append(np.array([split] * reservoir.filled, dtype=object))
            entry_blocks.append(reservoir.entry_indices[: reservoir.filled])
            frame_blocks.append(reservoir.frames[: reservoir.filled])

        if not template_blocks:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        self._templates = np.vstack(template_blocks)
        self._labels = np.concatenate(label_blocks)
        self._splits = np.concatenate(split_blocks)
        self._template_entries = np.concatenate(entry_blocks)
        self._template_frames = np.concatenate(frame_blocks)

    def _reserve(
        self,
        reservoirs: dict[tuple[str, int], _Reservoir],
        reservoir: _Reservoir,
        rows: int,
    ) -> None:
        """Grow *reservoir* to hold *rows* rows, once the sample is known to fit.

        The check counts every reservoir at the size it would have, and the copy
        that stacks them into the sample.
        """
        if rows <= reservoir.capacity:
            return
        held = sum(r.capacity for r in reservoirs.values() if r is not reservoir)
        _check_draw_memory(
            held + reservoir.grown_capacity(rows), int(reservoir.buffer.shape[1])
        )
        reservoir.reserve(rows)

    def _fit_exact(
        self,
        inputs: InputStream,
        split_assignments: list[str],
    ) -> None:
        """Two-pass exact allocation: count, then sample each entry's share of a quota."""

        # Pass 1: count rows per (split, class) and entry, capture metadata
        entry_rows: dict[tuple[str, int], dict[int, int]] = {}
        entry_splits: dict[str, str] = {}
        entry_idx_map: dict[str, int] = {}
        n_seen = 0

        for entry_key, df in inputs():
            if self._feature_columns is None:
                self._feature_columns = chosen_payload_columns(
                    df, self.params.columns, self.name
                )
                ensure_columns(df, [C.seq_col])

            group, sequence = entry_identity(df)

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

            labels = labels_for_rows(npz_path, df)
            self._note_labeled_frames((group, sequence), df, labels)
            labels = labels[labels != UNLABELED]
            for label_int in np.unique(labels).tolist():
                key = (split, int(label_int))
                entry_rows.setdefault(key, {})[entry_idx_map[entry_key]] = int(
                    (labels == label_int).sum()
                )

        if not entry_rows:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        # Each (split, class) draws its target, or every row when the class has
        # fewer, and splits that quota across the entries in proportion to the
        # rows each holds.
        n_classes = len({label for _, label in entry_rows})
        shares: dict[tuple[str, int], dict[int, int]] = {}
        for key, counts in entry_rows.items():
            target = self._collection_target_for_class(key[1], n_classes)
            total = sum(counts.values())
            shares[key] = _proportional_shares(
                counts, total if target is None else min(total, target)
            )
        assert self._feature_columns is not None
        _check_draw_memory(
            sum(sum(per_entry.values()) for per_entry in shares.values()),
            len(self._feature_columns),
        )

        # Pass 2: sample exactly each entry's share per (split, class)
        # Use a derived seed for sampling RNG (independent of split-assignment draws)
        sampling_rng = np.random.default_rng(self.params.random_state + 1)

        template_blocks: list[np.ndarray] = []
        label_blocks: list[np.ndarray] = []
        split_blocks: list[np.ndarray] = []
        entry_blocks: list[np.ndarray] = []
        frame_blocks: list[np.ndarray] = []

        cols = self._feature_columns
        assert cols is not None

        for entry_key, df in inputs():
            group, sequence = entry_identity(df)
            split = entry_splits[entry_key]
            entry_idx = entry_idx_map[entry_key]

            npz_path = self._labels_lookup.get((group, sequence))
            if npz_path is None:
                continue

            labels = labels_for_rows(npz_path, df)
            keep = labels != UNLABELED
            labels = labels[keep]
            matrix = df[cols].to_numpy(dtype=np.float64)[keep]
            row_frames = df[C.frame_col].to_numpy(dtype=np.int64)[keep]

            for label_int in np.unique(labels).tolist():
                label_int = int(label_int)
                take = shares.get((split, label_int), {}).get(entry_idx, 0)
                if take <= 0:
                    continue

                label_matrix = matrix[labels == label_int]
                label_frames = row_frames[labels == label_int]
                n_rows = int(label_matrix.shape[0])
                if n_rows > take:
                    sample_idx: np.ndarray = sampling_rng.choice(
                        n_rows, size=take, replace=False
                    )
                    label_matrix = label_matrix[sample_idx]
                    label_frames = label_frames[sample_idx]

                template_blocks.append(label_matrix)
                frame_blocks.append(label_frames)
                label_blocks.append(np.full(take, label_int, dtype=np.int64))
                split_blocks.append(np.array([split] * take, dtype=object))
                entry_blocks.append(np.full(take, entry_idx, dtype=np.int32))

        if not template_blocks:
            msg = "No data received during fit"
            raise RuntimeError(msg)

        self._templates = np.vstack(template_blocks)
        self._labels = np.concatenate(label_blocks)
        self._splits = np.concatenate(split_blocks)
        self._template_entries = np.concatenate(entry_blocks)
        self._template_frames = np.concatenate(frame_blocks)

    def _build_provenance(self) -> pd.DataFrame:
        """Build per-entry template provenance table.

        Every entry from _entry_map appears, even those with 0 templates. An
        entry with no labeled frame has no ``labeled_start`` or ``labeled_stop``.
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

        rows: list[dict[str, str | int | float | None]] = []
        for entry_idx in sorted(self._entry_map):
            group, sequence = self._entry_map[entry_idx]
            count = count_map.get(entry_idx, 0)
            split = self._sequence_splits.get((group, sequence), "train")
            labeled = self._labeled_frames.get((group, sequence))
            rows.append(
                {
                    "group": group,
                    "sequence": sequence,
                    "split": split,
                    "count": count,
                    "proportion": float(count) / total if total > 0 else 0.0,
                    "labeled_start": None if labeled is None else labeled[0],
                    "labeled_stop": None if labeled is None else labeled[1],
                }
            )
        return pd.DataFrame(rows).astype(
            {"labeled_start": "Int64", "labeled_stop": "Int64"}
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

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
        if self._row_identity is not None:
            df[C.group_col], df[C.seq_col] = self._row_identity
        if self._template_frames is not None:
            df[C.frame_col] = self._template_frames
        _ = write_parquet_atomic(df, run_root / TEMPLATES_ARTIFACT_NAME)

        _ = write_parquet_atomic(
            self._provenance, run_root / "template_provenance.parquet"
        )
