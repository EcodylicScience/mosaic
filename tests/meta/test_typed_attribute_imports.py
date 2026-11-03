"""The tag grammar loads the standard library and ``JsonValue`` alone.

mosaic-api shares the grammar in ``core/typed_attribute.py``, which imports only
the standard library and ``core/json_value.py``. A pydantic import in either
costs every importer the import and an adapter built at module load. The check
runs in a subprocess, because an earlier test of this process has already
imported pydantic.
"""

from __future__ import annotations

from tests.helpers import modules_loaded_by


def test_the_tag_grammar_loads_no_pydantic() -> None:
    assert modules_loaded_by("import mosaic.core.typed_attribute", ("pydantic",)) == []
