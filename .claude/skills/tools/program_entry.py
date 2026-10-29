"""Run a program's main function as the program."""

import sys
from collections.abc import Callable
from typing import NoReturn

from lossless_text import pass_undecodable_bytes
from terminating_signals import run_with_signal_handling


def run_main(main: Callable[[], int]) -> NoReturn:
    """Run `main` with undecodable bytes written as they are and terminating
    signals handled, then exit with its status."""
    pass_undecodable_bytes()
    sys.exit(run_with_signal_handling(main))
