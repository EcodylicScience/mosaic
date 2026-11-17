"""Record a step that needs no work, after measuring it.

:func:`settle_step` is for a scheduler that drives steps by itself and wants a step
that has nothing to compute to read as finished without a job. Its contract:

* It measures the step as :func:`~.input_size.step_input_size` does, so the report
  it returns holds every field of that one.
* It records under the step's own execution id, which is where the readers of a step
  look: the children's pinned parents, the request's rollup and its status.
* It never computes. A step with entries to compute, or one that only quarantine
  narrowing would leave complete, is declined.
* It writes nothing unless it settles. A decline, a refusal, a cancel and any other
  error leave the dataset as it was, and a step whose run-log exists is left to
  whoever wrote it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mosaic.runlog import run_log_path

from ..refusal import RefusalReason
from .input_size import StepInputSizeWire, step_input_size
from .model import GraphModel
from .step import SettleDeclined, SettleDeclineReason, execute_step

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

    from ..job import CancelToken
    from .model import Request

__all__ = [
    "SettleOutcome",
    "SettleReason",
    "SettledState",
    "StepSettleOutcomeWire",
    "StepSettleWire",
    "settle_step",
]

type SettleOutcome = Literal["settled", "declined", "not_attempted"]
"""What a settle did: recorded the step, left it to an attempt, or did not try."""

type SettleReason = (
    Literal["", "attempt_recorded", "entries_to_compute", "inputs_not_ready", "op_step"]
    | SettleDeclineReason
)
"""Why a settle did not record the step, ``""`` when it did."""

type SettledState = Literal["cached", "bound"]
"""The answer a settle records: the step found done, or bound to a run."""


class StepSettleOutcomeWire(GraphModel):
    """What a settle did with one step.

    Attributes:
        outcome: ``settled`` when the step's attempt was recorded, ``declined``
            when the step could have settled and did not, ``not_attempted`` when
            it has work to do, so no settle was tried.
        reason: For ``declined``, ``would_compute`` (the attempt would compute),
            ``refused`` (an ordinary attempt refuses the step) or
            ``attempt_recorded`` (the step's run-log exists). For
            ``not_attempted``, ``entries_to_compute``, ``inputs_not_ready`` or
            ``op_step``. ``""`` for ``settled``.
        refusal: The reason of the refusal behind a ``refused`` decline, else
            ``""``.
        execution_id: The step's own execution id, whatever the outcome.
        state: The answer recorded, ``cached`` or ``bound``, for ``settled``,
            else ``None``.
        run_id: The run the record names, for ``settled``, else ``None``.
    """

    outcome: SettleOutcome
    reason: SettleReason
    refusal: RefusalReason | Literal[""]
    execution_id: str
    state: SettledState | None
    run_id: str | None


class StepSettleWire(StepInputSizeWire):
    """The input size of one step, and what a settle did with it.

    Attributes:
        settle: The settle's outcome. Every other field reads as in
            :class:`~.input_size.StepInputSizeWire`.
    """

    settle: StepSettleOutcomeWire


def settle_step(
    ds: Dataset,
    request: Request,
    step_id: str,
    *,
    owner: str,
    cancel_token: CancelToken | None = None,
) -> StepSettleWire:
    """Measure step *step_id* of *request* and record it when it needs no work.

    Args:
        ds: The dataset, opened for writing.
        request: The submission the step belongs to.
        step_id: The step.
        owner: Who records the attempt.
        cancel_token: Checked after the size is read and again right before the
            record is written.

    Returns:
        The step's input size and the settle's outcome.

    Raises:
        KeyError: The recipe has no step *step_id*.
        StepRefused: The size read refuses the step, as a parent step whose
            run-log records no run does. Nothing is written.
        Cancelled: *cancel_token* is cancelled before the settle returns or
            records. Nothing is written.
    """
    sized = step_input_size(ds, request, step_id)
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    execution_id = request.execution_of(step_id)

    def report(
        outcome: SettleOutcome,
        reason: SettleReason = "",
        *,
        refusal: RefusalReason | Literal[""] = "",
        state: SettledState | None = None,
        run_id: str | None = None,
    ) -> StepSettleWire:
        return StepSettleWire(
            **dict(sized),
            settle=StepSettleOutcomeWire(
                outcome=outcome,
                reason=reason,
                refusal=refusal,
                execution_id=execution_id,
                state=state,
                run_id=run_id,
            ),
        )

    if sized.kind == "op":
        return report("not_attempted", "op_step")
    if not sized.inputs_ready:
        return report("not_attempted", "inputs_not_ready")
    if sized.entries_to_compute > 0:
        return report("not_attempted", "entries_to_compute")
    if run_log_path(ds.base_dir, execution_id).exists():
        return report("declined", "attempt_recorded")
    if cancel_token is not None:
        cancel_token.raise_if_cancelled()
    try:
        outcome = execute_step(
            ds,
            request,
            step_id,
            owner=owner,
            cancel_token=cancel_token,
            settle_only=True,
        )
    except SettleDeclined as declined:
        return report("declined", declined.reason, refusal=declined.refusal)
    state = outcome.state
    if state != "cached" and state != "bound":
        raise AssertionError(f"a settle-only attempt ended {state!r}")
    return report("settled", state=state, run_id=outcome.run_id)
