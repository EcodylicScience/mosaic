"""``import mosaic.behavior`` must not pull in PyTorch.

``model_library/__init__.py`` imports both T-Rex identity modules eagerly, and
``behavior/__init__.py`` imports ``model_library``. torch is an optional extra,
so a top-level ``import torch`` anywhere under ``model_library`` would break
every mosaic import -- not just the identity path -- for anyone without the
``[identity]`` extra, and take the whole test suite with it.

The modules keep torch behind a lazy ``_import_torch()`` and define their
``nn.Module`` subclasses inside factory functions. That is easy to undo by
accident while editing, and nothing else checks it, so this does.
"""

from __future__ import annotations

import subprocess
import sys

_BLOCK_TORCH_AND_IMPORT = """
import importlib.abc
import sys


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"torch is blocked for this test (asked for {fullname})")
        return None


sys.meta_path.insert(0, NoTorch())

import mosaic.behavior  # noqa: F401
import mosaic.behavior.model_library  # noqa: F401
from mosaic.behavior.model_library import (  # noqa: F401
    TRexIdentityNetwork,
    TRexV118_3IdentityNetwork,
)

assert "torch" not in sys.modules, "torch was imported despite the block"
print("OK")
"""


def test_behavior_imports_without_torch() -> None:
    """Importing the behavior package with torch blocked must still succeed."""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH_AND_IMPORT],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"importing mosaic.behavior requires torch.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout
