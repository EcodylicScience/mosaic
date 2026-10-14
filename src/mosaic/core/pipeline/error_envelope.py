"""One typed failure shape for every caller that reports an error as data.

:func:`error_envelope` renders a
:class:`~mosaic.core.pipeline.error_kind.ClassifiedError` as an
:class:`ErrorEnvelope`. The CLI's ``--json`` failure output is that model dumped.
It takes classified errors only. A caller catches :class:`ClassifiedError` and
lets every other exception propagate.
"""

from __future__ import annotations

from mosaic.core.json_value import JsonValue
from mosaic.core.strict_model import StrictModel

from .error_kind import ClassifiedError, ErrorKind

__all__ = ["ErrorEnvelope", "error_envelope"]


class ErrorEnvelope(StrictModel):
    """A failure, as the kind a caller branches on and the facts that explain it.

    Attributes:
        kind: Which kind of failure this is.
        reason: Which failure of its kind, when the error names one. Under
            ``refused`` it is always a ``RefusalReason``.
        step: The pipeline step that failed, when it is known.
        message: The sentence a person reads.
        detail: The names and numbers that make the failure actionable.
    """

    kind: ErrorKind
    reason: str | None = None
    step: str | None = None
    message: str
    detail: dict[str, JsonValue] | None = None


def error_envelope(exc: ClassifiedError) -> ErrorEnvelope:
    """Return *exc* as an :class:`ErrorEnvelope`.

    Args:
        exc: The error, which declares its kind.

    Returns:
        The envelope, with the message that ``str(exc)`` gives.
    """
    return ErrorEnvelope(
        kind=exc.error_kind,
        reason=exc.error_reason(),
        step=exc.error_step(),
        message=str(exc),
        detail=exc.error_detail(),
    )
