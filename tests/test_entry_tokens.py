"""One grammar for naming an entry, whichever surface a user names it through.

``--entries g:seq`` on the command line and ``{"entries": ["g:seq"]}`` inside an
op's ``--params`` are the same value arriving two ways, and they used to be
parsed by two functions that disagreed: the CLI rejected a token with no ``:``
while the three tracker ops read it as a bare sequence in the empty group. Since
every dataset the control plane creates has ``group=""``, the rejected spelling
was the common case, and a user had to type a colon to say nothing.
"""

from __future__ import annotations

import pytest

from mosaic.cli._io import parse_entries
from mosaic.core.helpers import make_entry_key, parse_entry_tokens

_CASES: list[tuple[list[str], list[tuple[str, str]]]] = [
    (["g:seq"], [("g", "seq")]),
    ([":seq"], [("", "seq")]),
    # A bare token is a sequence in the empty group -- the spelling the CLI used
    # to reject and the ops accepted.
    (["seq"], [("", "seq")]),
    # Split on the first colon only, so a sequence name may contain one.
    (["g:a:b"], [("g", "a:b")]),
    (["g1:s1", "g2:s2"], [("g1", "s1"), ("g2", "s2")]),
    ([], []),
]


@pytest.mark.parametrize(("tokens", "expected"), _CASES)
def test_the_grammar(tokens: list[str], expected: list[tuple[str, str]]) -> None:
    assert parse_entry_tokens(tokens) == expected


@pytest.mark.parametrize(("tokens", "expected"), _CASES)
def test_the_cli_agrees_with_the_ops(
    tokens: list[str], expected: list[tuple[str, str]]
) -> None:
    """The regression the split guarded against: two surfaces, one answer."""
    assert parse_entries(tokens) == parse_entry_tokens(tokens) == expected


def test_none_is_empty_rather_than_an_error() -> None:
    """Unset scope means every indexed entry, not a malformed one."""
    assert parse_entry_tokens(None) == []
    assert parse_entries(None) == []


def test_a_bare_token_names_the_key_it_is_stored_under() -> None:
    """Why the permissive reading is the correct one, not merely the lenient one.

    ``make_entry_key("", "seq")`` is ``"seq"``, so a bare token already spells
    the directory and filename the entry lives at. Rejecting it would have meant
    the name a user reads off disk is not a name they may type.
    """
    [(group, sequence)] = parse_entry_tokens(["seq"])

    assert make_entry_key(group, sequence) == "seq"
