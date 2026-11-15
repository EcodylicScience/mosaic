"""The definition store and sidecar load neither pandas nor the feature library.

``agent/mcp/spec.py`` imports the store's models, and its import budget excludes
the analysis stack. Planning reads sidecars on the same budget. The modules that
declare and load a definition name no layer above ``core``.
"""

from __future__ import annotations

import pytest

from tests.helpers import SOURCE_ROOT, modules_loaded_by

HEAVY_MODULES = (
    "pandas",
    "numpy",
    "mosaic.behavior",
    "mosaic.behavior.feature_library",
)


def test_importing_the_store_loads_neither_pandas_nor_the_feature_library() -> None:
    assert (
        modules_loaded_by(
            "import mosaic.core.pipeline.definitions.store", HEAVY_MODULES
        )
        == []
    )


def test_importing_the_sidecar_loads_neither_pandas_nor_the_feature_library() -> None:
    assert (
        modules_loaded_by(
            "import mosaic.core.pipeline.definitions.sidecar", HEAVY_MODULES
        )
        == []
    )


def test_reading_a_directory_and_rendering_an_error_load_neither() -> None:
    statement = (
        "import tempfile\n"
        "from pathlib import Path\n"
        "from mosaic.core.pipeline.definitions import store\n"
        "directory = Path(tempfile.mkdtemp()) / 'x'\n"
        "directory.mkdir()\n"
        "store.list_versions(directory)\n"
        "store.read_current(directory)\n"
        "store.parse_definition_slug('def:x')\n"
        "from mosaic.core.pipeline.error_envelope import error_envelope\n"
        "error_envelope(store.DefinitionNotFound('def:x'))\n"
    )
    assert modules_loaded_by(statement, HEAVY_MODULES) == []


def test_importing_the_scope_loads_neither_pandas_nor_the_feature_library() -> None:
    assert (
        modules_loaded_by(
            "import mosaic.core.pipeline.definitions.scope", HEAVY_MODULES
        )
        == []
    )


@pytest.mark.parametrize("module", ["declare.py", "loader.py", "scope.py"])
def test_the_declaring_and_loading_modules_name_neither_upper_layer(
    module: str,
) -> None:
    text = (SOURCE_ROOT / "core" / "pipeline" / "definitions" / module).read_text(
        encoding="utf-8"
    )
    assert "mosaic.behavior" not in text
    assert "mosaic.tracking" not in text
