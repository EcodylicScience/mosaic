"""Handle the terminating signals of a program, except the ones that the caller
ignores."""

import os
import signal
from collections.abc import Callable
from types import FrameType

TERMINATING_SIGNALS = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)

Handler = Callable[[int, FrameType | None], None]


class Terminated(Exception):
    """Signal a terminating signal, which unwinds the stack.

    `raise_terminated` raises it.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum: int = signum


def raise_terminated(signum: int, _frame: FrameType | None) -> None:
    raise Terminated(signum)


def handle_unless_ignored(signum: int, handler: Handler) -> None:
    """Install `handler` for `signum`, unless the signal is ignored.

    Ignored signals stay ignored for every command that this program starts, as
    under a shell. Skipping the handler keeps them from resetting to the
    default in those commands.
    """
    if signal.getsignal(signum) is not signal.SIG_IGN:
        _ = signal.signal(signum, handler)


def handle_terminating_signals(handler: Handler) -> None:
    for signum in TERMINATING_SIGNALS:
        handle_unless_ignored(signum, handler)


def die_of_signal(signum: int) -> None:
    """End this process by `signum` itself. The caller then sees 128+signum."""
    _ = signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def run_with_signal_handling(main: Callable[[], int]) -> int:
    """Run `main`. Terminating signals unwind its stack, run its cleanup, and
    then end this process by that signal."""
    handle_terminating_signals(raise_terminated)
    try:
        return main()
    except Terminated as terminated:
        die_of_signal(terminated.signum)
        raise
