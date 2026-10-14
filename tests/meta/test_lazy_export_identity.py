"""A package that binds its names on access returns the object its table names.

Python binds a submodule as an attribute of its package the first time anything
imports it, and a package ``__getattr__`` is consulted only for an attribute the
package lacks. A public name equal to a submodule's basename therefore returns
the module once that submodule is imported, and the defining object before. The
answer depends on import order.

Each check runs in a fresh interpreter that imports every submodule first. The
packages are found by their call to ``lazy_export``, so a new one is checked
without an edit here.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.helpers import SOURCE_ROOT

LAZY_PACKAGES = tuple(
    sorted(
        ".".join(("mosaic", *init.parent.relative_to(SOURCE_ROOT).parts))
        for init in SOURCE_ROOT.rglob("__init__.py")
        if "lazy_export(__name__" in init.read_text()
    )
)
"""The packages whose ``__getattr__`` binds a public name through ``lazy_export``."""

_CHECK = """
import importlib
import pkgutil

package = importlib.import_module({package!r})
submodules = sorted(m.name for m in pkgutil.iter_modules(package.__path__))
for submodule in submodules:
    importlib.import_module(f"{{package.__name__}}.{{submodule}}")
shadowed = sorted(set(package._MODULE_OF) & set(submodules))
rebound = []
for name, source in package._MODULE_OF.items():
    module, _, attribute = source.partition(":")
    defined = getattr(importlib.import_module(module, package.__name__), attribute or name)
    if getattr(package, name) is not defined:
        rebound.append(name)
print(f"shadowed={{shadowed}} rebound={{rebound}}")
"""


def test_the_lazy_packages_are_found() -> None:
    assert "mosaic.core.pipeline.graph" in LAZY_PACKAGES


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_every_public_name_is_the_object_its_table_names(package: str) -> None:
    """No public name equals a submodule basename, and each resolves to its source."""
    completed = subprocess.run(
        [sys.executable, "-c", _CHECK.format(package=package)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "shadowed=[] rebound=[]", (
        f"{package}: a public name in 'shadowed' is also a submodule basename. "
        f"Rename the submodule. {completed.stdout.strip()}"
    )
