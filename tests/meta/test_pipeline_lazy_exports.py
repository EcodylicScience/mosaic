"""``mosaic.core.pipeline`` binds its public names on access, and importing it is cheap.

``core/pipeline/__init__.py`` used to import its public names and the media op
modules at module scope. Python executes a package's ``__init__`` before any
submodule of it, so reading a run-log or a recipe through ``graph`` or
``inventory`` loaded pandas, cv2 and av through the package alone.

The checks below run in a subprocess each. An in-process probe answers about
whatever an earlier test already imported.
"""

from __future__ import annotations

import subprocess
import sys

import mosaic.core.pipeline as pipeline_package

HEAVY_MODULES = ("cv2", "av", "pandas")
"""The modules that importing the package must not load."""

SUBMODULES_ONCE_BOUND = (
    "joined_export",
    "preprocess",
    "preprocess_index",
    "store_export",
    "transcode",
)
"""Submodules that the package imported for their registration side effect."""


def _run(code: str) -> str:
    """Run *code* in a fresh interpreter, returning its stdout."""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_every_exported_name_resolves_by_from_import() -> None:
    names = ", ".join(sorted(pipeline_package.__all__))
    assert _run(f"from mosaic.core.pipeline import {names}\nprint('ok')") == "ok"


def test_every_exported_name_resolves_as_an_attribute() -> None:
    names = sorted(pipeline_package.__all__)
    code = (
        "import mosaic.core.pipeline as package\n"
        f"missing = [n for n in {names!r} if getattr(package, n, None) is None]\n"
        "print(missing or 'ok')"
    )
    assert _run(code) == "ok"


def test_a_submodule_still_imports_from_the_package() -> None:
    """A submodule name falls through ``__getattr__`` to the submodule import."""
    names = ", ".join(SUBMODULES_ONCE_BOUND)
    assert _run(f"from mosaic.core.pipeline import {names}\nprint('ok')") == "ok"


def test_an_unknown_name_raises_attribute_error() -> None:
    code = (
        "import mosaic.core.pipeline as package\n"
        "try:\n"
        "    package.no_such_name\n"
        "except AttributeError:\n"
        "    print('raised')\n"
    )
    assert _run(code) == "raised"


def test_importing_the_package_loads_no_heavy_module() -> None:
    """One eager import added back to ``core/pipeline/__init__.py`` fails here."""
    code = (
        "import sys\n"
        "import mosaic.core.pipeline\n"
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules) or 'none')"
    )
    assert _run(code) == "none"


def test_the_package_body_imports_no_submodule() -> None:
    code = (
        "import sys\n"
        "import mosaic.core.pipeline\n"
        "loaded = [m for m in sys.modules if m.startswith('mosaic.core.pipeline.')]\n"
        "print(sorted(loaded) or 'none')"
    )
    assert _run(code) == "none"
