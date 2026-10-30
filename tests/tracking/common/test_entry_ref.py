"""An entry reference names its ``(group, sequence)`` entry."""

from __future__ import annotations

from mosaic.tracking.common.entry_ref import EntryRef


def test_an_entry_ref_names_its_entry() -> None:
    assert EntryRef(group="g", sequence="s").entry == ("g", "s")
    assert EntryRef(sequence="s").entry == ("", "s")
