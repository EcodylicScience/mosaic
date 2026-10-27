"""``mosaic.core.media`` and its ``preprocess`` package bind their names on access.

Both ``__init__`` files used to import the frame readers and the step modules at
module scope. Python executes a package's ``__init__`` before any submodule of
it, so reading the media index or a recorded placement loaded cv2 and av, and
``Dataset`` and the inventory did through them.
"""

from __future__ import annotations

import pytest

import mosaic.core.media as media_package
import mosaic.core.media.preprocess as preprocess_package
from tests.helpers import modules_loaded_by, run_fresh

MEDIA_READERS = ("cv2", "av")
"""The modules that a leaf import must not load."""

LAZY_PACKAGES = {
    "mosaic.core.media": media_package.__all__,
    "mosaic.core.media.preprocess": preprocess_package.__all__,
}
"""Each lazy package and the names it exports."""

LEAF_IMPORTS = (
    "import mosaic.core.media",
    "import mosaic.core.media.preprocess",
    "from mosaic.core.media.preprocess.geometry import Placement",
    "from mosaic.core.media.facts_columns import row_to_facts",
    "import mosaic.core.media.reprobe, mosaic.core.media.uniformity",
    "import mosaic.core.media.store_division",
    "import mosaic.core.dataset",
)
"""The imports that a dataset read makes, each in its own interpreter."""


@pytest.mark.parametrize("statement", LEAF_IMPORTS)
def test_a_leaf_import_loads_no_media_reader(statement: str) -> None:
    assert modules_loaded_by(statement, MEDIA_READERS) == []


@pytest.mark.parametrize("package", sorted(LAZY_PACKAGES))
def test_every_exported_name_resolves(package: str) -> None:
    names = sorted(LAZY_PACKAGES[package])
    code = (
        f"from {package} import {', '.join(names)}\n"
        f"import {package} as package\n"
        f"missing = [n for n in {names!r} if getattr(package, n, None) is None]\n"
        "print(missing or 'ok')"
    )
    assert run_fresh(code) == "ok"


def test_reading_a_name_of_the_preprocess_package_registers_every_step() -> None:
    """``MEDIA_STEPS`` read first from the package holds every step of the union."""
    code = (
        "import typing\n"
        "from mosaic.core.media.preprocess import MEDIA_STEPS\n"
        "from mosaic.core.media.preprocess.specs import MediaStepSpec\n"
        "union = typing.get_args(MediaStepSpec.__value__)[0]\n"
        "names = sorted(member.name for member in typing.get_args(union))\n"
        "print(names == sorted(MEDIA_STEPS))"
    )
    assert run_fresh(code) == "True"
