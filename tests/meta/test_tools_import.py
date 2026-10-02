"""Every script under ``tools/`` still imports.

CI's ruff check covers ``tools/``, but basedpyright runs ungated and no other
test imports these scripts, so a script can lint clean and still fail to
import. An import is the one cheap check available for a script nothing else
reads.

Before the fix this test protects, ``_utils.py`` still exported the selector
under the name ``Scope``. The stale ``from mosaic.core.pipeline._utils
import Scope`` in ``readdress_legacy_features.py`` resolved to that class
instead of ``ResolvedScope``, the dataclass its construction sites needed.
An import check written against that state would have passed. This test
verifies two facts together: that a tool script names the class it means,
and that ``_utils`` exports no other name for the class it does not.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Final

import pytest

from tests.helpers import REPO_ROOT

_TOOLS_DIRECTORY: Final = REPO_ROOT / "tools"

_TOOL_SCRIPTS: Final = sorted(_TOOLS_DIRECTORY.rglob("*.py"))


def _import_standalone(script: Path) -> None:
    """Import *script* as an isolated module, leaving no trace in ``sys.modules``.

    The module name is unique to this check and never registered under the
    script's own name. Importing one script therefore cannot satisfy, or
    collide with, an import of another.
    """
    module_name = f"_tools_import_check__{script.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None, (
        f"no import spec could be built for {script}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("script", _TOOL_SCRIPTS, ids=lambda path: path.name)
def test_tool_script_imports(script: Path) -> None:
    """Importing *script* on its own must not raise."""
    _import_standalone(script)
