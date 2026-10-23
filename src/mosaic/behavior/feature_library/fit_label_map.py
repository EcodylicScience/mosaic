"""``fit-label-map``: which behavior class each cluster stands for.

A clustering finer than the behavior vocabulary is scored by mapping each
cluster to a class. Mapping on the rows being scored measures how pure the
clusters are; mapping on held-out rows measures how well the clustering
predicts behavior. This feature fits the map on its own entries -- the training
groups, under a step scope -- and ``evaluate-labels`` applies it to others
through its ``label_map`` reference.

The map is :func:`~mosaic.core.evaluation.majority_mapping`: each cluster's most
frequent class among the labeled rows, a tie going to the lower class id. A
cluster no labeled row falls in is left out, and ``evaluate-labels`` counts a
row of it as wrong.

**Which weighting.** ``count`` gives a cluster to the class with the most rows in
it, which follows prevalence: a behavior that is 6% of the training frames and
12% of the test frames can end up with no cluster, and then scores an F1 of 0
whatever the clustering holds. ``class_balanced`` gives a cluster to the class
whose frames it holds the largest share of, which weights every class alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal, final

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.evaluation import Weighting, majority_mapping, select_scored_rows
from mosaic.core.helpers import UNLABELED
from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    ParamColumn,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.loading import labels_for_rows
from mosaic.core.pipeline.types import (
    COLUMNS as C,
)
from mosaic.core.pipeline.types import (
    META_COLS,
    DependencyLookup,
    EmitsLevel,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    ParquetArtifact,
    ParquetLoadSpec,
    Result,
)
from mosaic.core.pipeline.writers import write_parquet_atomic

from .helpers import ensure_columns, entry_identity
from .registry import register_feature

LABEL_MAP_FILE = "label_map.parquet"
"""The file a label map is saved in, one row per mapped cluster."""

MAPPED_COLUMN = "mapped_label"
"""The column ``fit-label-map`` writes: each row's cluster, mapped to its class."""

_LABELS_DESCRIPTION = (
    "The ground-truth labels the map is fitted on, resolved from "
    "labels/<kind>/index.csv. A {step: <id>} reference names a convert-labels "
    "step and pins the variant it writes."
)
_PREDICTION_COLUMN_DESCRIPTION = "The column holding each row's cluster."
_PERSPECTIVE_DESCRIPTION = (
    "Which rows of a pair output to fit on: the rows of one perspective, or "
    "'pooled' for both. Unset fits on the rows as they are, and refuses rows "
    "of more than one perspective."
)
_WEIGHTING_DESCRIPTION = (
    "How a cluster picks its class. count takes the class with the most rows "
    "in the cluster, which follows prevalence and can leave a rare class with "
    "no cluster. class_balanced takes the class whose rows the cluster holds "
    "the largest share of."
)


class LabelMapRef(ParquetArtifact):
    """A cluster to class map, as ``fit-label-map`` saves it.

    ``feature`` defaults to empty, which means no reference.
    """

    feature: str = ""
    pattern: str = LABEL_MAP_FILE
    load: ParquetLoadSpec = Field(default_factory=ParquetLoadSpec)


class LabelMapArtifact(LabelMapRef):
    """The label map of a ``fit-label-map`` run."""

    feature: str = "fit-label-map"


def read_label_map(path: Path) -> dict[int, int]:
    """The ``{cluster: class}`` map saved at *path*."""
    frame = pd.read_parquet(path)
    return {
        int(cluster): int(label)
        for cluster, label in zip(frame["cluster"], frame["label"], strict=True)
    }


@final
@register_feature
class FitLabelMap:
    """Map each cluster to the behavior class its labeled rows favor.

    Fit on one set of entries, such as the training groups, the map scores a
    clustering on others through ``evaluate-labels``. Apply writes each row's
    mapped class beside its cluster.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.fit_label_map.FitLabelMap.Params`.
    """

    category = "global"
    costly_fit = False  # cheap to refit, so a cleanup may remove a superseded run
    fits = True  # learns from its inputs, so a held-out group must stay out of its fit
    name = "fit-label-map"
    version = "0.2"
    parallelizable = False
    scope_dependent = True
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    requires: ClassVar[tuple[Requirement, ...]] = (
        Column(C.frame_col),
        ParamColumn("prediction_column"),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(ParamColumn("prediction_column"), Column(MAPPED_COLUMN)),
        conditional=META_COLUMNS_KEPT,
    )
    LabelMapArtifact = LabelMapArtifact

    class Inputs(Inputs[Result]):
        pass

    class Params(Params):
        labels: Annotated[GroundTruthLabelsSource, Declared(_LABELS_DESCRIPTION)] = (
            Field(default_factory=GroundTruthLabelsSource)
        )
        prediction_column: Annotated[str, Declared(_PREDICTION_COLUMN_DESCRIPTION)] = (
            "cluster"
        )
        perspective: Annotated[
            int | Literal["pooled"] | None, Declared(_PERSPECTIVE_DESCRIPTION)
        ] = None
        weighting: Annotated[Weighting, Declared(_WEIGHTING_DESCRIPTION)] = "count"

    def __init__(
        self, inputs: FitLabelMap.Inputs, params: dict[str, object] | None = None
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        self._labels_lookup: DependencyLookup = {}
        self._map: dict[int, int] | None = None
        self._counts: dict[int, int] = {}

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._labels_lookup = dependency_lookups.get("labels", {})
        self._map = None
        path = run_root / LABEL_MAP_FILE
        if path.exists():
            self._map = read_label_map(path)
            return True
        return False

    def fit(self, inputs: InputStream) -> None:
        true: list[np.ndarray] = []
        pred: list[np.ndarray] = []
        for _key, df in inputs():
            labels = self._labels_of(df)
            if labels is None:
                continue
            rows = df.assign(_label=labels)
            scored = select_scored_rows(
                rows,
                perspective=self.params.perspective,
                prediction_column=self.params.prediction_column,
                label_column="_label",
            ).frame
            # A row with no cluster is dropped before the cast, which would
            # turn its NaN into an arbitrary integer.
            scored = scored.loc[scored[self.params.prediction_column].notna()]
            true.append(scored["_label"].to_numpy(dtype=np.int64))
            pred.append(scored[self.params.prediction_column].to_numpy(dtype=np.int64))
        y_true = np.concatenate(true) if true else np.empty(0, dtype=np.int64)
        y_pred = np.concatenate(pred) if pred else np.empty(0, dtype=np.int64)
        self._map = majority_mapping(y_true, y_pred, weighting=self.params.weighting)
        if not self._map:
            msg = (
                "[fit-label-map] no labeled row to fit a map on. Check that the "
                "entries in scope have labels."
            )
            raise RuntimeError(msg)
        labeled = y_true != UNLABELED
        clusters, counts = np.unique(y_pred[labeled], return_counts=True)
        self._counts = {int(c): int(n) for c, n in zip(clusters, counts, strict=True)}

    def _labels_of(self, df: pd.DataFrame) -> np.ndarray | None:
        """Each row's label, or ``None`` for an entry with no label file."""
        path = self._labels_lookup.get(entry_identity(df))
        if path is None:
            return None
        return labels_for_rows(path, df)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._map is None:
            msg = "fit-label-map has no map yet."
            raise RuntimeError(msg)
        column = self.params.prediction_column
        ensure_columns(df, [C.frame_col, column])
        kept = [name for name in df.columns if name in META_COLS and name != column]
        out = df[[*kept, column]].copy()
        clusters = df[column].to_numpy()
        out[MAPPED_COLUMN] = np.array(
            [
                self._map.get(int(value), UNLABELED) if pd.notna(value) else UNLABELED
                for value in clusters
            ],
            dtype=np.int64,
        )
        return out

    def save_state(self, run_root: Path) -> None:
        if self._map is None:
            return
        frame = pd.DataFrame(
            {
                "cluster": list(self._map),
                "label": list(self._map.values()),
                "n_rows": [self._counts.get(cluster, 0) for cluster in self._map],
            }
        ).sort_values("cluster", ignore_index=True)
        _ = write_parquet_atomic(frame, run_root / LABEL_MAP_FILE)
