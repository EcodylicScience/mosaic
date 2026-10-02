"""The ``convert-points`` op, through the real CVAT converter.

Its run-log lifecycle and output tree, reuse and overwrite, and the refusal of
annotations that match no image.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.pipeline.ops import OPS, describe_op, run_op
from mosaic.core.pipeline.run_log import read_runs, run_log_dir

from tests.helpers import stub_media_dataset


def _write_cvat_points_fixture(root: Path, n_groups: int = 5, per_group: int = 2):
    """Write a tiny CVAT 'for Images 1.1' XML + matching (empty) image files.

    Returns (xml_path, images_dir). Filenames use the ``<stem>__frame_XXXXXX.png``
    convention so ``split_by='group'`` groups by video stem.
    """
    images_dir = root / "cvat" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<annotations>"]
    for g in range(n_groups):
        for f in range(per_group):
            name = f"v{g}__frame_{f:06d}.png"
            (images_dir / name).write_bytes(b"")  # existence only; dims come from XML
            lines.append(f'  <image name="{name}" width="640" height="480">')
            lines.append('    <points points="100.0,120.0">')
            lines.append('      <attribute name="class">UnmarkedBee</attribute>')
            lines.append("    </points>")
            lines.append("  </image>")
    lines.append("</annotations>")
    xml_path = root / "cvat" / "annotations.xml"
    xml_path.write_text("\n".join(lines))
    return xml_path, images_dir


def test_convert_points_registered():
    assert "convert-points" in OPS
    d = describe_op("convert-points")
    assert d["category"] == "convert"
    assert {"cvat_xml", "images_dir", "class_names", "radii"} <= set(
        d["params_schema"]["properties"]
    )


def test_convert_points_lifecycle(tmp_path):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    xml, images_dir = _write_cvat_points_fixture(ds.base_dir)

    params = {
        "cvat_xml": ds.relative_to_root(xml),
        "images_dir": ds.relative_to_root(images_dir),
        "class_names": ["UnmarkedBee"],
        "radii": {"UnmarkedBee": 100.0},
        "split_by": "group",
        "symlink_images": False,
    }
    run_id = run_op(ds, "convert-points", dict(params))
    assert run_id.startswith("convert-points.")

    # runs-row lifecycle
    runs = read_runs(run_log_dir(ds.base_dir), kind="convert-points")
    assert len(runs) == 1 and runs[0]["status"] == "finished"
    assert runs[0]["run_id"] == run_id

    # data.yaml + splits written under models/convert-points/<run_id>/
    from mosaic.core.pipeline.models import model_run_root

    out = model_run_root(ds, "convert-points", run_id)
    data_yaml = out / "data.yaml"
    assert data_yaml.exists()
    n_labels = sum(
        len(list((out / split / "labels").glob("*.txt")))
        for split in ("train", "valid", "test")
        if (out / split / "labels").exists()
    )
    assert n_labels == 10  # 5 groups x 2 frames

    # index row recorded + finished
    from mosaic.tracking.ops.convert import (
        converted_dataset_index,
    )
    from mosaic.core.pipeline.models import model_index_path

    idx = converted_dataset_index(model_index_path(ds, "convert-points"))
    df = idx.read(run_id=run_id)
    assert len(df) == 1
    assert df.iloc[0]["class_names"] == "UnmarkedBee"
    assert int(df.iloc[0]["n_train"]) >= 1

    # deterministic + cache hit: identical inputs -> same run_id, no error
    run_id2 = run_op(ds, "convert-points", dict(params))
    assert run_id2 == run_id


def test_convert_points_overwrite_rebuilds_the_dataset(tmp_path: Path) -> None:
    """The overwrite argument is read by a reuse gate that is not a training one.

    convert-points decides reuse from an existing data.yaml rather than from a
    completion row, and it clears the run root before rewriting. A marker
    dropped into that root measures whether the clear happened, which a
    timestamp on a deterministic converter would not.
    """
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    xml, images_dir = _write_cvat_points_fixture(ds.base_dir)
    from mosaic.core.pipeline.models import model_run_root

    params = {
        "cvat_xml": ds.relative_to_root(xml),
        "images_dir": ds.relative_to_root(images_dir),
        "class_names": ["UnmarkedBee"],
        "radii": {"UnmarkedBee": 100.0},
        "symlink_images": False,
    }
    run_id = run_op(ds, "convert-points", dict(params))
    marker = model_run_root(ds, "convert-points", run_id) / "marker.txt"
    marker.write_text("kept across a reuse")

    assert run_op(ds, "convert-points", dict(params)) == run_id
    assert marker.exists(), "a reuse must not clear the run root"

    assert run_op(ds, "convert-points", dict(params), overwrite=True) == run_id
    assert not marker.exists(), "overwrite must clear the run root and rebuild"
    assert (model_run_root(ds, "convert-points", run_id) / "data.yaml").exists()


def test_convert_points_no_matching_images_raises(tmp_path):
    ds = stub_media_dataset(tmp_path, ["vid1", "vid2"])
    xml, images_dir = _write_cvat_points_fixture(ds.base_dir)
    empty_dir = ds.base_dir / "cvat" / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="no training labels"):
        run_op(
            ds,
            "convert-points",
            {
                "cvat_xml": ds.relative_to_root(xml),
                "images_dir": ds.relative_to_root(empty_dir),
                "class_names": ["UnmarkedBee"],
                "radii": {"UnmarkedBee": 100.0},
            },
        )
