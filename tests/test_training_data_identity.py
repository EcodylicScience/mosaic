"""Nothing outside the data a one-shot op names may enter its identity.

A training or conversion op is named by its params plus a content fingerprint of
the data it reads. Each op chooses that fingerprint at its own call site, and
nothing checked the choice: ``train-sleap`` and ``convert-points`` handed a
single file to a fingerprint that listed every file under the file's parent
folder. With the labels at a dataset root, every run's own output moved the next
run's identifier, and an identical resubmission retrained instead of reusing.

So this reads the registry rather than naming ops. Every op whose identity is
its params plus the data it reads -- the ones declaring ``scope_takes = "none"``
-- is planned, then its own run root gains a file and every folder holding one
of its inputs gains an unrelated file, and it is planned again. The two
identifiers must agree. A new one-shot op fails the completeness check by name
until it has a builder here, which is the point: a pairing chosen per call site
is only safe if something asks each one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.models import model_run_root
from mosaic.core.pipeline.ops import OPS, Op
from mosaic.core.params import Params
from tests.helpers import make_dataset, minimal_op_params


def _file(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(content)
    return path


def _yolo(ds: Dataset, data: str) -> list[Path]:
    data_yaml = Path(ds.resolve_path(data))
    _ = _file(data_yaml.parent / "train" / "images" / "a.png", b"train-image")
    _ = _file(data_yaml.parent / "val" / "images" / "b.png", b"val-image")
    _ = _file(data_yaml, b"train: train/images\nval: val/images\nnc: 1\nnames: [a]\n")
    return [data_yaml]


def _sleap(ds: Dataset, params: dict[str, object]) -> list[Path]:
    return [_file(Path(ds.resolve_path(str(params["labels"]))), b"slp bytes")]


def _points_conversion(ds: Dataset, params: dict[str, object]) -> list[Path]:
    xml = _file(Path(ds.resolve_path(str(params["cvat_xml"]))), b"<annotations/>")
    images = Path(ds.resolve_path(str(params["images_dir"])))
    _ = _file(images / "a.png", b"image")
    return [xml, images]


def _directory(key: str) -> Callable[[Dataset, dict[str, object]], list[Path]]:
    def build(ds: Dataset, params: dict[str, object]) -> list[Path]:
        directory = Path(ds.resolve_path(str(params[key])))
        _ = _file(directory / "content.txt", b"content")
        return [directory]

    return build


_BUILDERS: dict[str, Callable[[Dataset, dict[str, object]], list[Path]]] = {
    "convert-points": _points_conversion,
    "train-litpose": _directory("project"),
    "train-localizer": _directory("dataset_dir"),
    "train-points": lambda ds, params: _yolo(ds, str(params["data"])),
    "train-pose": lambda ds, params: _yolo(ds, str(params["data"])),
    "train-sleap": _sleap,
}
"""What each one-shot op reads, laid out at the paths ``minimal_op_params`` names.

Each builder returns the input paths, so the test knows which folders to write
beside.
"""


def _one_shot_kinds() -> set[str]:
    return {kind for kind, op in OPS.items() if op.scope_takes == "none"}


def test_every_one_shot_op_has_a_builder() -> None:
    """A new one-shot op is asked the question the day it registers."""
    missing = _one_shot_kinds() - set(_BUILDERS)
    assert not missing, (
        f"{sorted(missing)} declare scope_takes = 'none' and have no builder in "
        f"_BUILDERS, so nothing checks that their data fingerprint ignores "
        f"what sits beside their data"
    )
    assert set(_BUILDERS) <= _one_shot_kinds(), "a builder names no one-shot op"


def _plan(op: Op[Any], ds: Dataset, params: Params) -> str:
    return op.plan_identity(ds, params, ResolvedScope()).run_id


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_what_sits_beside_the_data_does_not_move_the_identity(
    kind: str, tmp_path: Path
) -> None:
    """The run's own output, and an unrelated file beside each input, change nothing.

    Both perturbations are what a real dataset does between two submissions: a
    finished run leaves files under ``models/<kind>/<run_id>/``, and a dataset
    root keeps gaining run-logs, feature runs and other models.
    """
    ds = make_dataset(tmp_path / "ds")
    raw = minimal_op_params(kind)
    inputs = _BUILDERS[kind](ds, raw)
    op = OPS[kind]()
    params = op.Params.model_validate(raw)

    before = _plan(op, ds, params)
    _ = _file(model_run_root(ds, kind, before) / "training_log.csv", b"epoch\n0\n")
    for path in inputs:
        _ = _file(path.parent / "unrelated.txt", b"not this run's data")
        _ = _file(path.parent / "elsewhere" / "deeper.txt", b"nor this")
    after = _plan(op, ds, params)

    assert after == before, (
        f"{kind}'s identity moved when files it does not read appeared beside "
        f"its data, so an identical resubmission would train again"
    )


@pytest.mark.parametrize("kind", sorted(_BUILDERS))
def test_changed_data_does_move_the_identity(kind: str, tmp_path: Path) -> None:
    """The other direction, so the test above cannot pass by digesting nothing."""
    ds = make_dataset(tmp_path / "ds")
    raw = minimal_op_params(kind)
    inputs = _BUILDERS[kind](ds, raw)
    op = OPS[kind]()
    params = op.Params.model_validate(raw)

    before = _plan(op, ds, params)
    first = inputs[0]
    if first.is_dir():
        _ = _file(first / "added.txt", b"new content")
    elif first.name == "data.yaml":
        _ = _file(first.parent / "train" / "images" / "c.png", b"another image")
    else:
        _ = _file(first, first.read_bytes() + b" edited")
    assert _plan(op, ds, params) != before
