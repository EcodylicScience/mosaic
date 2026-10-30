"""The appearance modules of ``core`` import neither ``behavior`` nor ``tracking``.

``core`` is the foundation of both packages. ``test_behavior_tracking_layering``
checks ``behavior`` against ``tracking`` only, so this test covers the other two
edges for the modules that define the appearance contract. The scan reads each
file's syntax tree, so an import inside a function counts.
"""

from __future__ import annotations

import ast
from typing import Final

import pytest

from tests.helpers import SOURCE_ROOT, imports_reaching, package_of, source_tree

APPEARANCE_MODULES: Final = ("core/pipeline/appearance.py",)
"""The modules of the appearance contract, relative to ``src/mosaic``."""

HIGHER_PACKAGES: Final = ("mosaic.behavior", "mosaic.tracking")


def _offenders(where: str, tree: ast.Module, package: str) -> list[str]:
    """The imports of *tree* that reach ``behavior`` or ``tracking``."""
    return [
        f"{where}:{reached.line} imports {module} ({reached.statement})"
        for reached in imports_reaching(tree, package, literal_packages=HIGHER_PACKAGES)
        for higher in HIGHER_PACKAGES
        if (module := reached.into(higher))
    ]


@pytest.mark.parametrize("where", APPEARANCE_MODULES)
def test_an_appearance_module_imports_neither_analysis_package(where: str) -> None:
    path = SOURCE_ROOT / where
    assert path.is_file()
    assert _offenders(where, source_tree(path), package_of(path)) == []


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("from mosaic.tracking import ops", id="from-tracking"),
        pytest.param("import mosaic.behavior.feature_library", id="import-behavior"),
        pytest.param(
            "def read():\n    from ...tracking import common\n",
            id="relative-in-function",
        ),
    ],
)
def test_the_scan_catches_an_import_of_an_analysis_package(source: str) -> None:
    assert _offenders("synthetic.py", ast.parse(source), "mosaic.core.pipeline")
