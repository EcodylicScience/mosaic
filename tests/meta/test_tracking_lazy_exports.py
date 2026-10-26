"""``mosaic.tracking`` binds its public names on access, and importing it is cheap.

``tracking/__init__.py`` used to import every tracker subpackage at module scope.
Python executes a package's ``__init__`` before any submodule of it, so reading
one tracking module, the inventory's included, loaded every tracker, the
trainers, pandas, cv2 and av.

The checks below run in a subprocess each. An in-process probe answers about
whatever an earlier test already imported.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mosaic.tracking as tracking_package
from mosaic.tracking.common.index import (
    TrackerRunRowBase,
    tracker_index,
    tracker_index_path,
)
from tests.helpers import make_dataset, modules_loaded_by, run_fresh

HEAVY_MODULES = ("cv2", "av", "pandas")
"""The modules that importing the package must not load."""

TRACKER_SUBPACKAGES = (
    "mosaic.tracking.litpose",
    "mosaic.tracking.sleap",
    "mosaic.tracking.trex",
    "mosaic.tracking.ultralytics_track",
)
"""The subpackages that the package imported at module scope."""

EXPORTING_PACKAGES = (
    "mosaic.tracking",
    "mosaic.tracking.common",
    "mosaic.tracking.frame_extraction",
)
"""The packages under ``mosaic.tracking`` that re-export names from submodules."""

LAZY_PACKAGES = (
    *EXPORTING_PACKAGES,
    "mosaic.tracking.frame_predictions",
    "mosaic.tracking.ops",
)
"""The packages under ``mosaic.tracking`` whose ``__init__`` imports no submodule."""

PRODUCER_MODULES = (
    *TRACKER_SUBPACKAGES,
    "mosaic.tracking.ops.train",
    "mosaic.tracking.ops.predict_frames",
    "mosaic.tracking.ops.score_predictions",
    "mosaic.tracking.model_refs",
    "mosaic.tracking.common.bridge",
    "mosaic.tracking.common.scope",
    "mosaic.tracking.frame_extraction.dataset_runs",
    "mosaic.tracking.frame_extraction.extraction",
)
"""The trackers, trainers and frame extraction, which an inventory read leaves unloaded."""

INVENTORY_BUDGET = ("cv2", "av", "mosaic.behavior.feature_library")
"""The media readers and the feature library, which an inventory read leaves unloaded."""

PREDICTION_READ_MODULES = (
    "mosaic.tracking.frame_predictions.index",
    "mosaic.tracking.frame_predictions.read",
)
"""The modules that define the frame-predictions indexes and read them."""


@pytest.mark.parametrize("name", sorted(tracking_package.__all__))
def test_every_exported_name_resolves(name: str) -> None:
    """Both spellings, in a fresh interpreter, for every name ``__all__`` claims."""
    assert (
        run_fresh(
            f"from mosaic.tracking import {name}\n"
            f"import mosaic.tracking\n"
            f"assert getattr(mosaic.tracking, {name!r}) is not None\n"
            f"print('ok')"
        )
        == "ok"
    )


@pytest.mark.parametrize("package", EXPORTING_PACKAGES)
def test_every_name_of_a_lazy_package_resolves(package: str) -> None:
    code = (
        "import importlib\n"
        f"package = importlib.import_module({package!r})\n"
        "missing = [n for n in package.__all__ if getattr(package, n, None) is None]\n"
        "print(missing or 'ok')"
    )
    assert run_fresh(code) == "ok"


def test_a_submodule_still_imports_from_the_package() -> None:
    """A submodule name falls through ``__getattr__`` to the submodule import."""
    assert run_fresh("from mosaic.tracking import ops, trex\nprint('ok')") == "ok"


def test_an_unknown_name_raises_attribute_error() -> None:
    code = (
        "import mosaic.tracking as package\n"
        "try:\n"
        "    package.no_such_name\n"
        "except AttributeError:\n"
        "    print('raised')\n"
    )
    assert run_fresh(code) == "raised"


def test_importing_the_package_loads_no_heavy_module() -> None:
    """One eager import added back to ``tracking/__init__.py`` fails here."""
    code = (
        "import sys\n"
        "import mosaic.tracking\n"
        f"print(sorted(m for m in {HEAVY_MODULES!r} if m in sys.modules) or 'none')"
    )
    assert run_fresh(code) == "none"


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_a_package_body_imports_no_submodule(package: str) -> None:
    code = (
        "import sys\n"
        f"import {package}\n"
        f"loaded = [m for m in sys.modules if m.startswith({package + '.'!r})]\n"
        "print(sorted(loaded) or 'none')"
    )
    assert run_fresh(code) == "none"


def test_the_inventory_reports_a_tracker_run_without_loading_a_producer(
    tmp_path: Path,
) -> None:
    """A fresh process reads a tracker run with no tracker module imported.

    Opening the dataset, reading its inventory, listing it and summarizing it
    loads no tracker subpackage, no op module that runs a tool, no frame
    extraction, the imports deferred into the record builders included, and none
    of cv2, av and the feature library. pandas loads, because every index is read
    through it.
    """
    ds = make_dataset(tmp_path / "ds")
    run_id = "trex.1.0-aaaaaaaaaa"
    work = ds.get_root("trex") / run_id / "seq_a"
    work.mkdir(parents=True)
    index = tracker_index(tracker_index_path(ds, "trex"), TrackerRunRowBase)
    index.ensure()
    index.append(
        [
            TrackerRunRowBase(
                run_id=run_id,
                group="",
                sequence="seq_a",
                abs_path=Path(ds.relative_to_root(work)),
                video_abs_path="",
                params_hash="",
            )
        ]
    )
    index.mark_finished(run_id)
    code = (
        "import json, sys\n"
        "from mosaic.core.dataset import open_dataset\n"
        "from mosaic.core.dataset_summary import dataset_summary\n"
        "from mosaic.core.pipeline.inventory import TrackerRunRef, inventory_wire\n"
        "from mosaic.tracking import dataset_inventory\n"
        f"ds = open_dataset({str(ds.manifest_path)!r}, ensure_roots=False)\n"
        "found = dataset_inventory(ds)\n"
        "_ = inventory_wire(ds, found)\n"
        "_ = dataset_summary(ds, found=found)\n"
        f"record = found.record(TrackerRunRef(root_key='trex', run_id={run_id!r}))\n"
        f"modules = {(*PRODUCER_MODULES, *INVENTORY_BUDGET)!r}\n"
        "loaded = sorted(m for m in modules if m in sys.modules)\n"
        "print(json.dumps({'status': record and record.status, 'loaded': loaded}))"
    )

    assert json.loads(run_fresh(code)) == {"status": "complete", "loaded": []}


@pytest.mark.parametrize("module", PREDICTION_READ_MODULES)
def test_a_prediction_read_module_loads_no_producer(module: str) -> None:
    """The index and readers leave frame extraction and every tool unloaded."""
    loaded = modules_loaded_by(
        f"import {module}", (*PRODUCER_MODULES, *INVENTORY_BUDGET)
    )

    assert loaded == []
