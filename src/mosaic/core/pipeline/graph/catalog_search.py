"""The step types of the catalog index that a query matches, most relevant first.

A query and a step type's name are compared as words: lowercase, with spaces,
hyphens and underscores alike, so ``speed_angvel``, ``speed angvel`` and
``Speed-Angvel`` name ``speed-angvel``. Every word of a query must occur in a
step type's name, summary or category for it to match.

The module imports nothing beyond the standard library at run time, so a search
loads neither the feature library nor pandas.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .wire import CatalogIndexEntry

_SEPARATORS: Final = re.compile(r"[\s_-]+")

_KEY_WIDTH: Final = 6
"""The digits of each number in a ranking key, so that keys compare as strings."""


@dataclass(frozen=True, slots=True)
class CatalogMatch:
    """One step type that a query matches, with its place in relevance order.

    Attributes:
        entry: The step type's index row.
        key: The tuple that orders the matches of one query, best first. Its
            parts are strings, so a page cursor can record it.
    """

    entry: CatalogIndexEntry
    key: tuple[str, ...]


def _search_words(text: str) -> tuple[str, ...]:
    """Return the lowercase words of *text*, split on spaces, hyphens and underscores."""
    return tuple(word for word in _SEPARATORS.split(text.lower()) if word)


def search_catalog(
    entries: Iterable[CatalogIndexEntry], query: str
) -> list[CatalogMatch]:
    """Return the entries that every word of *query* matches, most relevant first.

    The ranking tiers, best first:

    0. The query's words are the name's words.
    1. Each word equals or begins a word of the name.
    2. Each word occurs in the name.
    3. Some words occur in the name and the rest in the summary or category.
    4. The words occur in the summary or category only.

    Within a tier an entry ranks higher with more words that equal a word of the
    name or its singular, then with fewer words found only in the category,
    then with a shorter name, then by name. A query without words matches every
    entry, ordered by name. Every key has the same parts, whatever the query.
    """
    words = _search_words(query)
    matches = [
        match for entry in entries if (match := _match(entry, words)) is not None
    ]
    return sorted(matches, key=lambda match: match.key)


def _match(entry: CatalogIndexEntry, words: Sequence[str]) -> CatalogMatch | None:
    """Return *entry* ranked against *words*, ``None`` when a word misses it."""
    if not words:
        return CatalogMatch(entry=entry, key=_key(entry, (0, 0, 0, 0)))
    tokens = _search_words(entry.name)
    name = " ".join(tokens)
    summary = entry.summary.lower()
    category = entry.category.lower()
    in_name = [word in name for word in words]
    in_summary = [word in summary for word in words]
    in_category = [word in category for word in words]
    if not all(map(any, zip(in_name, in_summary, in_category, strict=True))):
        return None
    if " ".join(words) == name:
        tier = 0
    elif all(any(token.startswith(word) for token in tokens) for word in words):
        tier = 1
    elif all(in_name):
        tier = 2
    elif any(in_name):
        tier = 3
    else:
        tier = 4
    inexact = sum(word not in tokens and f"{word}s" not in tokens for word in words)
    category_only = sum(
        found_in_category and not (found_in_name or found_in_summary)
        for found_in_name, found_in_summary, found_in_category in zip(
            in_name, in_summary, in_category, strict=True
        )
    )
    numbers = (tier, inexact, category_only, len(entry.name))
    return CatalogMatch(entry=entry, key=_key(entry, numbers))


def _key(entry: CatalogIndexEntry, numbers: tuple[int, ...]) -> tuple[str, ...]:
    """Return the ranking key of *entry*: *numbers* zero-padded, then its name."""
    return (*(f"{number:0{_KEY_WIDTH}d}" for number in numbers), entry.name)
