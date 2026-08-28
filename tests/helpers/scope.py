"""A resolved scope, stated directly rather than read off a dataset."""

from __future__ import annotations

from mosaic.core.entry import Entry
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.scope import Scope

__all__ = ["resolved_scope", "scope_over"]


def resolved_scope(selector: Scope, *entries: Entry) -> ResolvedScope:
    """*selector* paired with what it resolved to.

    Stated rather than enumerated. A test therefore needs no media index to
    name a coverage. A selector names what it resolved to, the only pairing
    :meth:`~mosaic.core.dataset.Dataset.resolve_scope` produces. An unset
    selector resolves to no entry, and one naming entries resolves to those.
    A test pairing an unset selector with entries is describing a state the
    resolver does not produce, and says so where it does it.
    """
    return ResolvedScope(entries=set(entries), selector=selector)


def scope_over(*entries: Entry) -> ResolvedScope:
    """The scope a caller naming *entries* gets, with no entries meaning unset.

    The shorthand for the common case, where a test drives a run over named
    entries and the selector is the entry list itself. Called with nothing it
    gives an unset selector, covering every indexed entry. That differs from a
    selector that named entries and matched none, which is
    ``resolved_scope(Scope(entries=[]))``.
    """
    if not entries:
        return ResolvedScope()
    return resolved_scope(Scope(entries=list(entries)), *entries)
