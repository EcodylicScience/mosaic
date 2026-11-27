"""One spelling of the bin column names, below both of their users.

``core/pipeline/types/data_config.py`` defines ``BIN_START`` and ``BIN_END``
beside ``SUMMARY_KEYS``. The aggregate features import them from there, since
``core`` cannot import ``behavior``. The scan reads each file's syntax tree.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Final

import pytest

from tests.helpers import SOURCE_ROOT, imports_in, is_third_party_source, source_tree

DECLARING_MODULE: Final = SOURCE_ROOT / "core" / "pipeline" / "types" / "data_config.py"
BIN_SPELLINGS: Final = frozenset({"bin_start", "bin_end"})
BIN_NAMES: Final = frozenset({"BIN_START", "BIN_END"})


def _spelled_out(tree: ast.Module) -> list[int]:
    """The lines of *tree* that hold a bin column name as a string literal."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in BIN_SPELLINGS
    ]


def _imported_from_aggregate_common(tree: ast.Module) -> list[int]:
    """The lines of *tree* that import a bin name through ``aggregate_common``."""
    return [
        imported.line
        for imported in imports_in(tree, scope="anywhere", relative=True)
        if imported.name in BIN_NAMES
        and imported.module.rsplit(".", 1)[-1] == "aggregate_common"
    ]


def _source_files() -> list[str]:
    return sorted(
        str(path.relative_to(SOURCE_ROOT))
        for path in SOURCE_ROOT.rglob("*.py")
        if not is_third_party_source(path, SOURCE_ROOT)
    )


def test_only_the_declaring_module_spells_a_bin_column_name() -> None:
    offenders = [
        f"{where}:{line}"
        for where in _source_files()
        if (path := SOURCE_ROOT / where) != DECLARING_MODULE
        for line in _spelled_out(source_tree(path))
    ]
    assert offenders == []


def test_no_module_imports_a_bin_name_through_aggregate_common() -> None:
    offenders = [
        f"{where}:{line}"
        for where in _source_files()
        for line in _imported_from_aggregate_common(source_tree(SOURCE_ROOT / where))
    ]
    assert offenders == []


def test_the_declaring_module_spells_both_names() -> None:
    assert len(_spelled_out(source_tree(DECLARING_MODULE))) == len(BIN_SPELLINGS)


@pytest.mark.parametrize(
    ("source", "scan"),
    [
        pytest.param('x = "bin_start"', _spelled_out, id="literal"),
        pytest.param(
            "from .aggregate_common import BIN_END, SUBSET",
            _imported_from_aggregate_common,
            id="relative-import",
        ),
        pytest.param(
            "from mosaic.behavior.feature_library.aggregate_common import BIN_START",
            _imported_from_aggregate_common,
            id="absolute-import",
        ),
    ],
)
def test_the_scan_catches_each_offense(
    source: str, scan: Callable[[ast.Module], list[int]]
) -> None:
    assert scan(ast.parse(source)) == [1]
