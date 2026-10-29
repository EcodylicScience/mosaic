# remote-tool-minimum-version: 3.12
"""Parse the command line of every program here the same way."""

import argparse
import sys
from typing import NoReturn, override

from lossless_text import encoded, write_error

EXIT_USAGE = 2


class CommandLineParser(argparse.ArgumentParser):
    """Answer a usage error with the full help on standard error, then the
    error line, then `closing_line` when it is set."""

    closing_line: str = ""

    @override
    def error(self, message: str) -> NoReturn:
        closing = f"{self.closing_line}\n" if self.closing_line else ""
        write_error(encoded(f"{self.format_help()}{self.prog}: error: {message}\n{closing}"))
        sys.exit(EXIT_USAGE)


def program_parser(
    description: str, epilog: str | None = None, closing_line: str = "", usage: str | None = None
) -> CommandLineParser:
    """Return a parser that refuses an abbreviated option and prints the
    description and epilog as written."""
    parser = CommandLineParser(
        description=description,
        epilog=epilog,
        usage=usage,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.closing_line = closing_line
    return parser


def nonempty_value(text: str) -> str:
    """Return an option's value, refusing an empty one."""
    if not text:
        raise argparse.ArgumentTypeError("needs a nonempty value")
    return text
