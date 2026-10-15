"""The order of the graces a cancelled training run passes through."""

from mosaic.core.pipeline.graph import STEP_CANCEL_GRACE_SECONDS
from mosaic.tracking.ops.train import TRAIN_CANCEL_GRACE_SECONDS


def test_a_request_driver_outwaits_the_training_grace() -> None:
    # A driver that killed the step first would leave the trainer, which runs in
    # a session of its own, writing into a run root recorded as cancelled.
    assert STEP_CANCEL_GRACE_SECONDS > TRAIN_CANCEL_GRACE_SECONDS
