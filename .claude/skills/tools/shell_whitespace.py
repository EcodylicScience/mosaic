"""Define the characters that a shell's `[[:space:]]` matches in the C locale,
and split words on blanks."""

import re

SHELL_WHITESPACE = " \t\n\v\f\r"
# A regular expression that matches one of those characters.
SHELL_WHITESPACE_CLASS = f"[{SHELL_WHITESPACE}]"
BLANKS = re.compile("[ \t]+")


def blank_separated_words(text: str) -> list[str]:
    """Return the words of `text` that spaces and tabs separate."""
    return [word for word in BLANKS.split(text) if word]
