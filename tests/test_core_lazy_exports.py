"""``mosaic.core`` binds its public names on access, and a leaf import stays cheap.

``core/__init__.py`` used to import ``Dataset``, the name helpers and
``track_library`` at module scope. Python executes a package's ``__init__``
before any submodule of it, so every consumer of every leaf beneath ``core``
paid for pandas, and for h5py through the SLEAP converter -- an environment
without h5py met a ``ModuleNotFoundError`` while importing three field names off
``mosaic.core.scope``.

The checks below run in a subprocess each. An in-process probe answers about
whatever an earlier test already imported, which for pandas is always.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import mosaic.core as core_package

LEAF_IMPORT = "from mosaic.core.scope import SCOPE_PARAM_KEYS"
"""A leaf under ``core``, imported for a name that needs pydantic and nothing else."""


def _run(code: str) -> str:
    """Run *code* in a fresh interpreter, returning its stdout."""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


@pytest.mark.parametrize("name", sorted(core_package.__all__))
def test_every_exported_name_resolves(name: str) -> None:
    """Both spellings, in a fresh interpreter, for every name ``__all__`` claims.

    ``from mosaic.core import track_library`` raised ``RecursionError`` while
    type-checking clean: ``from . import X`` inside the module ``__getattr__``
    runs ``_handle_fromlist``, which probes the package with ``hasattr`` and
    re-enters ``__getattr__``. Only a name that is both a ``__getattr__`` arm and
    a submodule reaches it, so the two spellings are checked separately.
    """
    assert (
        _run(
            f"from mosaic.core import {name}\n"
            f"import mosaic.core\n"
            f"assert getattr(mosaic.core, {name!r}) is not None\n"
            f"print('ok')"
        )
        == "ok"
    )


def test_a_leaf_import_loads_neither_pandas_nor_h5py() -> None:
    """What the lazy ``__init__`` buys, asserted rather than measured once.

    One ``from .dataset import Dataset`` added back to ``core/__init__.py``
    reverts it, and nothing else would notice.
    """
    assert (
        _run(
            f"import sys\n"
            f"{LEAF_IMPORT}\n"
            f"print(sorted(m for m in ('pandas', 'h5py') if m in sys.modules) or 'neither')"
        )
        == "neither"
    )


def test_the_package_body_binds_no_name_from_mosaic() -> None:
    """The import that would revert it, refused at the source rather than the symptom."""
    assert (
        _run(
            "import sys\n"
            "import mosaic.core\n"
            "loaded = [m for m in sys.modules if m.startswith('mosaic.core.')]\n"
            "print(sorted(loaded) or 'none')"
        )
        == "none"
    )


def test_the_entry_token_parser_loads_no_dataframe_library() -> None:
    """mosaic-queue's submit commands parse tokens where pandas must not go.

    The grammar lived on core/helpers.py, which imports pandas and numpy at
    module scope. core/entry.py imports the standard library and nothing else.
    """
    assert (
        _run(
            "import sys\n"
            "from mosaic.core.entry import parse_entry_tokens\n"
            "print('pandas' if 'pandas' in sys.modules else 'clean')"
        )
        == "clean"
    )
