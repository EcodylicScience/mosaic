"""``import mosaic.behavior`` must not pull in PyTorch.

``behavior/__init__.py`` imports ``model_library``, and every network under it
needs torch. torch is an optional extra, so a top-level ``import torch`` anywhere
under ``model_library`` would break every mosaic import -- not just the identity
path -- for anyone without the ``[identity]`` extra, and take the whole test
suite with it.

Two habits hold that line: torch stays behind a lazy ``import_torch()``, and
every ``nn.Module`` subclass is defined inside a factory function. Both are easy
to undo by accident while editing, and nothing else checks them, so this does.

Each network module is imported by name rather than relying on
``model_library/__init__.py`` to reach them, because it deliberately imports
none of them -- so a module that started importing torch eagerly would go
unnoticed here until something else imported it.
"""

from __future__ import annotations

import subprocess
import sys

_BLOCK_TORCH_AND_IMPORT = """
import importlib.abc
import sys

import pytest

# Selected by CI's `identity` job with `-m identity` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.identity


class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError(f"torch is blocked for this test (asked for {fullname})")
        return None


sys.meta_path.insert(0, NoTorch())

import mosaic.behavior  # noqa: F401
import mosaic.behavior.model_library  # noqa: F401
import mosaic.behavior.model_library.dinov2_temporal_identity  # noqa: F401
import mosaic.behavior.model_library.identity_classifier  # noqa: F401
import mosaic.behavior.model_library.identity_common  # noqa: F401
import mosaic.behavior.model_library.identity_embedding  # noqa: F401
import mosaic.behavior.model_library.timm_backbone  # noqa: F401

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
