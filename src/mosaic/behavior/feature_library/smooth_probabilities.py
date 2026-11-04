"""``smooth-probabilities``: smooth a classifier's class probabilities over consecutive frames.

The input is one classifier run, or one ensemble of them. The step smooths each
``prob_<class>`` column along each series' frames, divides each row by its sum,
and chooses ``predicted_label`` again from the smoothed probabilities, with the
columns and identity of any classifier. The input's ``predicted_label`` and the
threshold it was chosen by are not read.

**Segments.** Smoothing reads within a segment only: a run of consecutive frames
of one series (an individual, or one perspective of a pair) whose probabilities
are all finite and whose rows share one scoring fold of the input's
``cv_folds.parquet``. A frame gap, a row with a NaN probability and the edge of a
fold's block each cut a segment, and the smoothing is truncated there, as at the
ends of a series. A row with a NaN probability is written as it came, with
``default_class``. Without a fold table every row is scored by the full model, and
only gaps and NaN rows cut.

**Out of fold.** A fold's model scored every row of a segment and never trained on
it, so a smoothed row reads only probabilities of models that did not see it,
whatever the gap between blocks. A fold table by sequence gives a whole entry one
fold, and only frame gaps cut within it. The step copies the input's fold table to
its own run, so ``tune-threshold`` and ``probability-ensemble`` downstream find it
where they find a classifier's.

**Bouts.** With ``min_bout_frames`` above 1, a run of one predicted class shorter
than it, within a segment, takes the class of a neighbor bout, the shortest bout
first, until none is shorter or the segment is one bout. The probabilities are not
changed, so a merged row's ``predicted_label`` may differ from its thresholded
argmax.

**Tuned thresholds.** ``tune-threshold`` scores thresholds on the probabilities of
its input, and smoothing changes them. A ``decision_threshold`` naming a
``tune-threshold`` run therefore applies only with ``method: "none"``, and only to
the run it was tuned on. The chain is a classifier, a smoothing step,
``tune-threshold`` over the smoothing step, then a ``method: "none"`` step over the
smoothing step that applies the tuned thresholds and merges bouts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self, final

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter1d, median_filter

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.column_contract import (
    META_COLUMNS_KEPT,
    Column,
    Prefix,
    Provides,
    Requirement,
)
from mosaic.core.pipeline.types import (
    COLUMNS,
    DependencyLookup,
    EmitsLevel,
    InputRequire,
    Inputs,
    InputStream,
    Result,
)
from mosaic.core.pipeline.types.feature import RelatesLevel
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.track_quality import consecutive_runs

from .classifier_common import (
    FOLD_TABLE_FILE,
    DEFAULT_CLASS_DESCRIPTION,
    FoldLookup,
    ThresholdsArtifact,
    TunedDecisionThreshold,
    assemble_prediction,
    input_fold_table_path,
    predicted_labels,
    probability_classes,
    tuned_threshold,
)
from .helpers import (
    ensure_columns,
    gaussian_radius,
    iter_series_positions,
    refuse_repeated_series_rows,
)
from .registry import register_feature

__all__ = ["SmoothProbabilitiesFeature"]

_TRUNCATE: Final = 4.0
"""The standard deviations at which the Gaussian kernel is cut, scipy's default."""

_NO_PROBABILITIES: Final = (
    "[smooth-probabilities] the input has no prob_<class> columns. Its input is "
    "a classifier run or an ensemble of them"
)

_METHOD_DESCRIPTION = (
    "How each class probability is smoothed along consecutive frames: gaussian "
    "convolves it with a Gaussian of sigma frames, median takes the median of "
    "window frames, and none leaves it as it is."
)

_SIGMA_DESCRIPTION = (
    "The standard deviation, in frames, of the Gaussian that method gaussian "
    "smooths with."
)

_WINDOW_DESCRIPTION = (
    "The odd number of frames whose median method median takes, at least 3."
)

_MIN_BOUT_FRAMES_DESCRIPTION = (
    "The fewest frames a run of one predicted class keeps, a shorter run taking "
    "the class of the neighbor run whose class has the higher mean probability "
    "over it; 1 merges nothing."
)


def _merged_bouts(
    labels: npt.NDArray[np.intp],
    probs: npt.NDArray[np.float64],
    column_of: Mapping[int, int],
    min_frames: int,
) -> npt.NDArray[np.intp]:
    """Return the labels of one segment with every bout shorter than *min_frames* merged.

    While the segment holds more than one bout and one is shorter than
    *min_frames*, the shortest (the first in frame order on a tie) takes the class
    of the neighbor whose class has the higher mean probability over its rows, the
    earlier neighbor on a tie and the one neighbor at a segment edge. A class with
    no probability column has mean 0.

    Args:
        labels: The predicted class of each row, in frame order.
        probs: The probabilities of the rows, one column per class.
        column_of: The column of *probs* of each class.
        min_frames: The fewest frames a bout keeps.
    """
    if min_frames <= 1 or labels.size == 0:
        return labels
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges]).astype(np.intp)
    stops = np.concatenate([edges, [labels.size]]).astype(np.intp)
    kinds = labels[starts].copy()

    def mean_of(kind: int, start: int, stop: int) -> float:
        column = column_of.get(kind)
        return 0.0 if column is None else float(probs[start:stop, column].mean())

    while starts.size > 1:
        lengths = stops - starts
        short = int(np.argmin(lengths))
        if lengths[short] >= min_frames:
            break
        start, stop = int(starts[short]), int(stops[short])
        before, after = short - 1, short + 1
        if before < 0:
            target = after
        elif after >= starts.size:
            target = before
        else:
            earlier = mean_of(int(kinds[before]), start, stop)
            later = mean_of(int(kinds[after]), start, stop)
            target = before if earlier >= later else after
        if target == before:
            stops[before] = stop
        else:
            starts[after] = start
        starts = np.delete(starts, short)
        stops = np.delete(stops, short)
        kinds = np.delete(kinds, short)
        # The bouts on either side of the merged one now touch, and join when
        # they share a class.
        left = short - 1
        if 0 <= left < starts.size - 1 and kinds[left] == kinds[left + 1]:
            stops[left] = stops[left + 1]
            starts = np.delete(starts, left + 1)
            stops = np.delete(stops, left + 1)
            kinds = np.delete(kinds, left + 1)
    return np.repeat(kinds, stops - starts)


@final
@register_feature
class SmoothProbabilitiesFeature:
    """Smooth a classifier's class probabilities along consecutive frames.

    Reads one run that writes ``prob_<class>`` columns and writes the smoothed
    probabilities and a ``predicted_label`` chosen from them, as a classifier
    would. Smoothing never crosses a frame gap, a row without probabilities or the
    edge of a block of the input's cross-validation folds, so out-of-fold
    probabilities stay out of fold. Short bouts of one predicted class can be
    merged into their neighbors.

    Field documentation is on
    :class:`~mosaic.behavior.feature_library.smooth_probabilities.SmoothProbabilitiesFeature.Params`.
    """

    category = "per-frame"
    name = "smooth-probabilities"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    fits = False  # learns nothing, and copies the input's fold table
    relates: RelatesLevel = "none"
    accepts_overlap = False
    consumed_roots: tuple[str, ...] = ()
    emits: EmitsLevel = "as-input"
    follows_ids: bool = True  # smooths each id's series along its frames
    requires: ClassVar[tuple[Requirement, ...]] = (
        Prefix("prob_"),
        Column(COLUMNS.frame_col),
    )
    provides: ClassVar[Provides] = Provides(
        adds=(Prefix("prob_"), Column("predicted_label")),
        conditional=META_COLUMNS_KEPT,
    )

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "nonempty"

        @model_validator(mode="after")
        def _one_classifier(self) -> Self:
            """Smooth one run, whose fold table says where the folds cut."""
            if len(self.root) != 1:
                msg = (
                    "smooth-probabilities reads one classifier run, or one "
                    f"ensemble of them, and got {len(self.root)} inputs"
                )
                raise ValueError(msg)
            return self

    class Params(Params):
        method: Annotated[
            Literal["gaussian", "median", "none"], Declared(_METHOD_DESCRIPTION)
        ] = "gaussian"
        sigma: Annotated[float, Field(gt=0.0), Declared(_SIGMA_DESCRIPTION)] = 2.0
        window: Annotated[int, Field(ge=3), Declared(_WINDOW_DESCRIPTION)] = 5
        min_bout_frames: Annotated[
            int, Field(ge=1), Declared(_MIN_BOUT_FRAMES_DESCRIPTION)
        ] = 1
        decision_threshold: TunedDecisionThreshold = None
        default_class: Annotated[int, Declared(DEFAULT_CLASS_DESCRIPTION)]

        @model_validator(mode="after")
        def _settings_of_the_method(self) -> Self:
            """Refuse a setting that the chosen method does not read."""
            if self.window % 2 == 0:
                msg = f"window must be odd, and is {self.window}"
                raise ValueError(msg)
            if self.method != "gaussian" and self.sigma != 2.0:
                msg = "sigma only applies with method gaussian"
                raise ValueError(msg)
            if self.method != "median" and self.window != 5:
                msg = "window only applies with method median"
                raise ValueError(msg)
            if (
                isinstance(self.decision_threshold, ThresholdsArtifact)
                and self.method != "none"
            ):
                msg = (
                    "a decision_threshold that names a tune-threshold run only "
                    "applies with method none: the thresholds were scored on the "
                    "probabilities that step read, and smoothing changes them. "
                    "Tune over this step and apply the thresholds in a method "
                    "none step after it"
                )
                raise ValueError(msg)
            return self

    def __init__(
        self,
        inputs: SmoothProbabilitiesFeature.Inputs,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs
        self.params = self.Params.from_overrides(params)
        # A threshold that names a ``tune-threshold`` run is read in load_state.
        self._threshold: float | Mapping[int, float] | None = (
            None
            if isinstance(self.params.decision_threshold, ThresholdsArtifact)
            else self.params.decision_threshold
        )
        self._fold_table: pd.DataFrame | None = None
        self._folds: FoldLookup | None = None

    @staticmethod
    def temporal_reach(params: SmoothProbabilitiesFeature.Params) -> int | None:
        """Frames on each side that one row reads.

        The Gaussian reads ``int(4 * sigma + 0.5)`` frames and the median half its
        window. Merging bouts has no bound, since a chain of short bouts carries a
        merge any distance.
        """
        if params.min_bout_frames > 1:
            return None
        if params.method == "gaussian":
            return gaussian_radius(params.sigma, _TRUNCATE)
        if params.method == "median":
            return params.window // 2
        return 0

    # --- Feature protocol ---

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        self._threshold = tuned_threshold(
            self.name,
            self.params.decision_threshold,
            artifact_paths,
            self.params.default_class,
            self.inputs,
        )
        path = input_fold_table_path(dependency_lookups.get("_input_0", {}))
        if path is None:
            self._fold_table = None
            self._folds = None
            return True
        self._fold_table = pd.read_parquet(path)
        self._folds = FoldLookup.from_table(self._fold_table)
        return (run_root / FOLD_TABLE_FILE).exists()

    def fit(self, inputs: InputStream) -> None:
        """Read nothing: the step learns nothing, and only copies the fold table."""

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if (
            isinstance(self.params.decision_threshold, ThresholdsArtifact)
            and self._threshold is None
        ):
            msg = f"[{self.name}] no thresholds -- call load_state() first"
            raise RuntimeError(msg)
        frame_col = COLUMNS.frame_col
        ensure_columns(df, [frame_col])
        classes = probability_classes(df.columns)
        if not classes:
            raise ValueError(_NO_PROBABILITIES)
        frames = self._whole_frames(df)
        refuse_repeated_series_rows(df, f"[{self.name}]")
        probs = df[[f"prob_{cls}" for cls in classes]].to_numpy(dtype=np.float64)
        folds = (
            np.full(len(df), -1, dtype=np.intp)
            if self._folds is None or df.empty
            else self._folds.scoring_folds(df, self.name)
        )
        finite = np.isfinite(probs).all(axis=1)

        smoothed = probs.copy()
        segments: list[npt.NDArray[np.intp]] = []
        for positions in iter_series_positions(df):
            series_frames = frames[positions]
            if series_frames.size > 1:
                step = int(np.diff(series_frames).min())
                if step != 1:
                    msg = (
                        f"[{self.name}] the frames of a series are a step of {step} "
                        "apart at the least, and smoothing reads runs of "
                        "consecutive frames, so it would smooth nowhere. Smooth "
                        "the probabilities before the table is subsampled"
                    )
                    raise ValueError(msg)
            series_folds = folds[positions]
            for value in np.unique(series_folds):
                keep = finite[positions] & (series_folds == value)
                for start, stop in consecutive_runs(series_frames, keep):
                    rows = positions[start:stop]
                    smoothed[rows] = self._smoothed(probs[rows])
                    segments.append(rows)

        result = assemble_prediction(
            df, smoothed, classes, self._threshold, self.params.default_class
        )
        if self.params.min_bout_frames > 1:
            labels = predicted_labels(
                smoothed, classes, self._threshold, self.params.default_class
            )
            column_of = {cls: column for column, cls in enumerate(classes)}
            for rows in segments:
                labels[rows] = _merged_bouts(
                    labels[rows],
                    smoothed[rows],
                    column_of,
                    self.params.min_bout_frames,
                )
            result["predicted_label"] = labels
        return result

    def save_state(self, run_root: Path) -> None:
        if self._fold_table is None:
            return
        run_root.mkdir(parents=True, exist_ok=True)
        _ = write_parquet_atomic(self._fold_table, run_root / FOLD_TABLE_FILE)

    # --- Smoothing ---

    def _whole_frames(self, df: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Return the frames of *df* as integers, refusing a value that is not whole."""
        values: npt.NDArray[np.float64] = df[COLUMNS.frame_col].to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        whole = np.isfinite(values) & (values == np.floor(values))
        if not whole.all():
            first = values[int(np.flatnonzero(~whole)[0])]
            msg = (
                f"[{self.name}] frame {first} is not a whole number. Smoothing "
                "reads runs of consecutive frames, which only whole frame numbers "
                "make"
            )
            raise ValueError(msg)
        return values.astype(np.int64)

    def _smoothed(self, block: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the probabilities of one segment smoothed and divided by each row's sum."""
        method = self.params.method
        if method == "none":
            return block
        if method == "gaussian":
            filtered = gaussian_filter1d(
                block,
                self.params.sigma,
                axis=0,
                mode="nearest",
                truncate=_TRUNCATE,
            )
        else:
            filtered = median_filter(
                block, size=(self.params.window, 1), mode="nearest"
            )
        total = filtered.sum(axis=1, keepdims=True)
        return np.divide(filtered, total, out=filtered.copy(), where=total > 0.0)
