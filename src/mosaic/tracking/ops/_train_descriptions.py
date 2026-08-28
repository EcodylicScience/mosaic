"""Field prose shared by the training ops.

``PoseTrainParams`` and ``LocalizerTrainParams`` (:mod:`mosaic.tracking.ops.train`),
``TrainLitposeParams`` (:mod:`mosaic.tracking.ops.train_litpose`) and
``TrainSleapParams`` (:mod:`mosaic.tracking.ops.train_sleap`) each train a
model from a labeled dataset. ``PointTrainParams`` inherits
``PoseTrainParams``'s fields as a fourth consumer.

BASE_MODEL_DESCRIPTION and EPOCHS_DESCRIPTION cover what a run fine-tunes from
and how long it trains. All four classes declare them.

IDLE_TIMEOUT_DESCRIPTION and MAX_RUNTIME_DESCRIPTION belong only to
``TrainLitposeParams`` and ``TrainSleapParams``, the two ops that watch the
trainer as a subprocess and can kill it. train.py's trainers bound their
runtime with the module constant ``_TRAIN_IDLE_SECONDS`` instead.
"""

from __future__ import annotations

BASE_MODEL_DESCRIPTION = (
    "Weights to fine-tune from, as a path or as the run id of the training "
    "op that produced them. Identity records the training run id when the "
    "reference is one, and the weights' content digest when it is a bare "
    "path."
)

EPOCHS_DESCRIPTION = "How long the model trains at most."

IDLE_TIMEOUT_DESCRIPTION = (
    "How long the training subprocess may go without output before it is "
    "killed. A generous default, because an epoch on a large set is slow "
    "and a watchdog must not mistake slow for dead."
)

MAX_RUNTIME_DESCRIPTION = (
    "Absolute wall-clock ceiling for the training run. Unset leaves the "
    "ceiling to whatever queue submitted the run, and idle_timeout still "
    "applies."
)
