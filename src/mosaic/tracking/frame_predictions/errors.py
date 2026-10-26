"""Declares the errors that predict-frames and score-predictions runs raise.

Each op raises one error class per kind of failure. Every class is a
:class:`~mosaic.core.pipeline.error_kind.ClassifiedError` and is not a
:class:`~mosaic.core.pipeline.refusal.Refusal`. The ``refused`` kind names a
member of the closed ``RefusalReason`` set, whose members describe other
failures. A reference that is not a run id keeps
:class:`~mosaic.tracking.model_refs.ModelReferenceRefusedError`.

Imports only ``core``.
"""

from __future__ import annotations

from typing import ClassVar

from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind

__all__ = [
    "PredictFramesConflict",
    "PredictFramesInvalid",
    "PredictFramesNotFound",
    "PredictFramesUnavailable",
    "PredictionRunNotFound",
    "ScorePredictionsConflict",
    "ScorePredictionsInvalid",
]


class PredictFramesInvalid(ClassifiedError, ValueError):
    """Refuses a request that does not fit the models it names.

    Raised for references of mixed training kinds, a field set for another
    framework's model, and a model or device that single-frame prediction
    cannot run.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"


class PredictFramesConflict(ClassifiedError, ValueError):
    """Refuses a request that disagrees with the dataset.

    Raised when every scoped entry lacks frames in the extraction run, and
    when the tool's keypoints differ from the pose that the model's provenance
    records.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"


class PredictFramesNotFound(ClassifiedError, LookupError):
    """Reports an extraction run that no frames-index row names."""

    error_kind: ClassVar[ErrorKind] = "not_found"


class PredictFramesUnavailable(ClassifiedError, RuntimeError):
    """Reports an installed tool too old to predict on single frames."""

    error_kind: ClassVar[ErrorKind] = "unavailable"


class PredictionRunNotFound(ClassifiedError, LookupError):
    """Reports a prediction run that no finished predict-frames row names."""

    error_kind: ClassVar[ErrorKind] = "not_found"


class ScorePredictionsInvalid(ClassifiedError, ValueError):
    """Refuses prediction runs or labels that cannot be scored together.

    Raised for prediction runs of different models or of different predict
    params, prediction runs that name the model's keypoints in more than one
    way, revisions that do not narrow to one pose, and a keypoint map that does
    not resolve, such as one that resolves empty.
    """

    error_kind: ClassVar[ErrorKind] = "invalid"


class ScorePredictionsConflict(ClassifiedError, ValueError):
    """Refuses a score whose prediction runs and labels disagree about images.

    Raised for an image that two prediction runs both predict, an image that
    two revisions both contain as a usable frame of the pose, and labels of
    which no frame is covered by a prediction frame.
    """

    error_kind: ClassVar[ErrorKind] = "conflict"
