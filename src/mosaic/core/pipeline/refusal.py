"""A refusal: a failure that names which of a closed set of reasons it is.

**A refusal is not a new terminal status.** It is an ordinary failure carrying a
reason: the exit code is :data:`REFUSED_EXIT_CODE`, the run-log status stays
``failed``, and the reason is recorded in ``error_json``. Adding a status would
mean adding a member to ``runlog.TERMINAL_STATUSES``, which three repositories
read and mosaic-api's sweeper reaps, the same reason ``partial`` was kept out of
it.

An error declares itself a refusal by subclassing :class:`Refusal`, from any
module. :func:`~mosaic.core.pipeline.job.job_context` then records the refusal's
``error_json`` for its attempt instead of a traceback, and ``mosaic run``,
``mosaic track`` and ``mosaic pipeline run`` exit with :data:`REFUSED_EXIT_CODE`.

The module is a leaf. The run-log recorder and the pipeline graph both read it,
and it imports neither.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

__all__ = [
    "REFUSED_EXIT_CODE",
    "Refusal",
    "RefusalReason",
]

REFUSED_EXIT_CODE: Final = 65
"""The exit code of an attempt that a refusal ended.

A refusal usually comes before any work, but not always. A tracker refuses a
file's codec when it reaches that entry, so earlier entries of the same run may
already be published. An inference op refuses one before any entry runs.

Reserved so a driver can tell a refusal from a crash without parsing anything,
and chosen from the codes that ``terminal_status_for_exit`` maps to ``failed``:
not zero, not the cooperative-cancel code, not negative. The ledger row therefore
reads ``failed``, with the reason beside it in ``error_json``.
"""

type RefusalReason = Literal[
    "coverage_shortfall",
    "upstream_empty",
    "schema_family_mismatch",
    "variant_mismatch",
    "version_moved",
    "parent_unrecorded",
    "recipe_missing",
    "digest_mismatch",
    "undecodable_codec",
]
"""The reasons a :class:`Refusal` can name. A closed set on purpose.

It crosses two repository boundaries as the ``reason`` field of ``error_json``,
so a new member is a wire addition rather than a local choice, and a free-text
reason would be one nobody downstream can branch on.

Most are a pipeline step declining before it runs. ``undecodable_codec`` is a
tracker's or an inference op's refusal to hand its tool a file whose codec the
tool's environment does not decode, inside a step or in a run outside any
pipeline. A tracker raises it when it reaches that entry, and an inference op
before any entry runs.

Not every error that declines work is a :class:`Refusal`. The rest exit 1 and
name no reason: among them a tracker's refusal of an entry's media before its
tool runs, such as a missing join or a clip without an identity, and a model
reference that does not resolve (``ModelReferenceRefusedError``).
"""


class Refusal(Exception):
    """An error that is a refusal, and names which one.

    Subclassed by every error a caller should read as a refusal rather than a
    crash. A subclass keeps whatever other base its callers catch it by.

    Attributes:
        reason: Which refusal this is.
        step_id: The step that refused, or ``""`` when the refusal does not know
            it: raised outside a pipeline, or below the step that ran it.
        detail: The numbers or names that make the reason actionable. JSON, so
            the run-log records it unchanged.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: RefusalReason,
        step_id: str = "",
        detail: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason: RefusalReason = reason
        self.step_id: str = step_id
        self.detail: dict[str, JsonValue] = dict(detail or {})

    def error_json(self, step_id: str = "") -> str:
        """Return the refusal as the ``error_json`` blob that a ledger row records.

        Args:
            step_id: The step to name when the refusal does not name its own.

        Returns:
            ``reason``, ``step`` and ``message``, then the detail, as JSON text.
        """
        payload: dict[str, JsonValue] = {
            "reason": self.reason,
            "step": self.step_id or step_id,
            "message": str(self),
            **self.detail,
        }
        return json.dumps(payload)
