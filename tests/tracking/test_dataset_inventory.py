"""``dataset_inventory``: an inventory that reports the tracking kinds."""

from __future__ import annotations

import ast
from pathlib import Path

from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.inventory import TrainedModelRef
from mosaic.tracking import dataset_inventory
from tests.helpers import REPO_ROOT, make_dataset, register_trained_model

TRACKING_KINDS = ("frame-run", "prepared-dataset", "trained-model", "tracker-run")


def test_every_tracking_kind_is_reported(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "d")
    run_root = model_run_root(ds, "train-pose", "train-pose.0.1-0123456789")
    run_root.mkdir(parents=True)
    weights = run_root / "best.pt"
    weights.write_bytes(b"weights")
    register_trained_model(ds, "train-pose", "train-pose.0.1-0123456789", weights)

    found = dataset_inventory(ds, kinds=TRACKING_KINDS)

    assert found.unavailable_kinds == frozenset()
    ref = TrainedModelRef(op_kind="train-pose", run_id="train-pose.0.1-0123456789")
    assert found.status(ref) == "complete"


def _registered_kinds(source: Path) -> set[str]:
    """Return the kinds that *source* passes to ``register_inventory_contributor``."""
    return {
        node.args[0].value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_inventory_contributor"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def test_one_function_registers_every_tracking_kind() -> None:
    """Read from the source: a registration at module scope elsewhere would make
    a kind depend on which tracking modules a process had imported."""
    package = REPO_ROOT / "src" / "mosaic" / "tracking"
    registering = {
        ".".join(source.relative_to(package.parent.parent).with_suffix("").parts): kinds
        for source in sorted(package.rglob("*.py"))
        if (kinds := _registered_kinds(source))
    }

    assert set(registering) == {"mosaic.tracking.inventory"}
    assert sorted(registering["mosaic.tracking.inventory"]) == sorted(TRACKING_KINDS)
