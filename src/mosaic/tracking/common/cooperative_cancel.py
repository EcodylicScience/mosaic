"""Asking an external tool to stop, before killing it.

:func:`~mosaic.core.pipeline.subprocess_util.run_supervised` answers a fired
cancel predicate with a process-group kill, and that is right for almost
everything mosaic drives: a tracker's unit of loss is one video it will redo.
Training is the exception. Ultralytics cannot be interrupted inside an epoch, so
a kill loses whichever epoch was in flight, where a flag it reads between them
ends the run with ``last.pt`` and ``results.csv`` complete.

Nothing in the supervision helper changes. What changes is only the predicate it
is handed: this one writes a file the tool is watching for, keeps answering
``False`` while the tool finishes its epoch, and answers ``True`` once a grace
period has passed -- at which point the kill that was always there happens
anyway. A tool that honors the file is never killed; a tool that ignores it is
not immortal.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["stop_then_kill"]


def stop_then_kill(
    is_cancelled: Callable[[], bool],
    sentinel: Path,
    grace: float,
) -> Callable[[], bool]:
    """A cancel predicate that asks first and kills second.

    Args:
        is_cancelled: What the job says about its own cancellation, polled by the
            supervisor. Typically ``ctx.cancel_token.is_cancelled``.
        sentinel: The path the tool stats. Created once, on the first poll after
            the token fires, with an exclusive create so a racing writer cannot
            produce a half-written file the tool reads.
        grace: How long the tool is given, after being asked, before the kill.
            **Must exceed one epoch**, or the kill always wins and the file is
            decorative. It must also stay *under* whatever grace the substrate
            running mosaic imposes -- a container runtime that SIGKILLs the whole
            process tree on its own timer makes this one moot, and the ordering
            has to be epoch, then this, then that.

    Returns:
        A predicate for
        :func:`~mosaic.core.pipeline.subprocess_util.run_supervised`. It answers
        ``False`` while the tool is being asked politely, so the supervisor keeps
        waiting; ``True`` once *grace* has elapsed, which is the ordinary kill.

    The clock starts when the token first fires rather than when the sentinel is
    created, and the two are the same moment by construction: the file is written
    on that same poll.
    """
    asked_at: list[float | None] = [None]

    def check() -> bool:
        if not is_cancelled():
            return False
        if asked_at[0] is None:
            asked_at[0] = time.monotonic()
            _ask(sentinel)
            return False
        return (time.monotonic() - asked_at[0]) >= grace

    return check


def _ask(sentinel: Path) -> None:
    """Write the file the tool is watching for, and say so if that is impossible.

    An exclusive create, which is also how mosaic takes every other claim on
    disk. A file that is already there is the answer this wanted, so that case is
    silent.

    A filesystem that refuses the write is **not** made fatal. The polite path
    being unavailable is a reason to fall back to the kill, which is what the
    caller does when the grace expires -- raising here would turn a cancel into a
    crash. It is logged, because an operator who sees a training run killed rather
    than stopped is owed the reason.
    """
    try:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        handle = os.open(sentinel, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return
    except OSError as failure:
        logger.warning(
            "could not write the cancel sentinel at %s (%s), so this run will be "
            "stopped by killing it rather than by asking it to finish its epoch",
            sentinel,
            failure,
        )
        return
    os.close(handle)
