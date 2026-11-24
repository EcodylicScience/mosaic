"""Run the ``mosaic`` CLI in this process, for the bench scripts that observe it.

``pipes.py`` and ``stacksample.py`` import this module. A process worker under
spawn imports the script that started its parent, and through it this module,
so ``mosaic.cli`` is imported inside ``run_mosaic`` and not at module level.
"""

import time
from collections.abc import Sequence


def run_mosaic(args: Sequence[str]) -> tuple[int, float]:
    """Run ``mosaic ARGS`` through the CLI's entry function.

    Args:
        args: The command line after ``mosaic``.

    Returns:
        The exit status and the wall seconds of the command. A ``SystemExit``
        with no code is status 0, and one whose code is not an integer is
        status 1. Any other exception propagates.
    """
    started = time.perf_counter()
    status = 0
    try:
        from mosaic.cli import app

        app(args=list(args), prog_name="mosaic")
    except SystemExit as exit_:
        code = exit_.code
        status = 0 if code is None else code if isinstance(code, int) else 1
    return status, time.perf_counter() - started
