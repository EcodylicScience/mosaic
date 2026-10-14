"""The kinds of failure that a caller answers differently, and the base declaring one.

An error declares its kind by subclassing :class:`ClassifiedError` and setting
:attr:`ClassifiedError.error_kind`, from any module. A subclass keeps whatever
other base its callers catch it by. :func:`~mosaic.core.pipeline.error_envelope.error_envelope`
renders a classified error as the one failure shape that the CLI's ``--json`` and
other programmatic callers return.

The module is a leaf. It imports the standard library only, and the bottom-level
:mod:`mosaic.optional_dependency` subclasses its base without loading anything
else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

__all__ = ["ClassifiedError", "ErrorKind"]

type ErrorKind = Literal["refused", "invalid", "unavailable", "not_found", "conflict"]
"""Which kind of failure an error is.

``refused``
    A :class:`~mosaic.core.pipeline.refusal.Refusal`: the work declined to run for
    one of the closed set of reasons in ``RefusalReason``. Only a refusal has this
    kind. A ``refused`` failure therefore always names a ``RefusalReason``, and a
    class declaring the kind without overriding ``error_reason`` raises
    ``TypeError`` at definition.
``invalid``
    The request is malformed or does not fit a declaration: a recipe, a step's
    params, a scope or a model reference. Changing the request answers it.
``unavailable``
    This installation cannot do it: an optional dependency is missing, or a
    manifest is newer than this version reads. Changing the installation answers it.
``not_found``
    A named thing does not exist.
``conflict``
    The request disagrees with the current state of the dataset.
"""


class ClassifiedError(Exception):
    """An error with a declared kind of failure.

    A subclass sets :attr:`error_kind`, and overrides the three ``error_*``
    methods where it knows more than its message.
    """

    error_kind: ClassVar[ErrorKind]
    """The kind of every instance of this class."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, "error_kind"):
            raise TypeError(
                f"{cls.__name__} subclasses ClassifiedError with no error_kind"
            )
        if (
            cls.error_kind == "refused"
            and cls.error_reason is ClassifiedError.error_reason
        ):
            raise TypeError(
                f"{cls.__name__} declares error_kind 'refused' and names no "
                "reason. Subclass Refusal instead."
            )

    def error_reason(self) -> str | None:
        """Return which failure of its kind this is, or ``None`` when one is not named."""
        return None

    def error_step(self) -> str | None:
        """Return the pipeline step that failed, or ``None`` when it is not known."""
        return None

    def error_detail(self) -> dict[str, JsonValue] | None:
        """Return the names and numbers that make the failure actionable, as JSON."""
        return None
