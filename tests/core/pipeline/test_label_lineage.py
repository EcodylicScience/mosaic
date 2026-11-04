"""The label controls a reader's inputs carry, through every run upstream of them.

A run carries controls when a top-level value of its recorded params is a
mapping holding a non-null ``remap`` or ``mask``. The walk takes their union
over the input runs and every feature run they read.
"""

from __future__ import annotations

from pathlib import Path

from mosaic.core.pipeline.label_controls import LabelControls
from mosaic.core.pipeline.label_lineage import lineage_label_controls
from tests.helpers import write_lineage_run


def test_an_unreadable_params_json_gives_no_controls(tmp_path: Path) -> None:
    root = tmp_path / "features" / "xgboost" / "0.4-aaaaaaaaaa"
    root.mkdir(parents=True)
    (root / "params.json").write_text("{not json")
    assert lineage_label_controls([root]) == LabelControls()


def test_an_absent_params_json_gives_no_controls(tmp_path: Path) -> None:
    root = tmp_path / "features" / "xgboost" / "0.4-aaaaaaaaaa"
    root.mkdir(parents=True)
    assert not lineage_label_controls([root])


def test_labels_holding_neither_field_give_none(tmp_path: Path) -> None:
    root = write_lineage_run(
        tmp_path / "features",
        "sequence-classifier",
        "0.1-aaaaaaaaaa",
        {"labels": {"kind": "behavior", "remap": None, "mask": None}},
    )
    assert not lineage_label_controls([root])


def test_a_mask_two_steps_up_is_found(tmp_path: Path) -> None:
    features = tmp_path / "features"
    _ = write_lineage_run(
        features,
        "extract-labeled-templates",
        "0.2-aaaaaaaaaa",
        {"labels": {"kind": "behavior", "mask": [2]}},
    )
    _ = write_lineage_run(
        features,
        "xgboost",
        "0.4-bbbbbbbbbb",
        {"templates": {"feature": "extract-labeled-templates"}},
        upstream=[("extract-labeled-templates", "0.2-aaaaaaaaaa")],
    )
    smooth = write_lineage_run(
        features,
        "smooth-probabilities",
        "0.1-cccccccccc",
        {"window": 5},
        upstream=[("xgboost", "0.4-bbbbbbbbbb")],
    )
    controls = lineage_label_controls([smooth])
    assert controls == LabelControls(mask=frozenset({2}))
    assert controls


def test_the_union_over_every_input(tmp_path: Path) -> None:
    features = tmp_path / "features"
    first = write_lineage_run(
        features,
        "sequence-classifier",
        "0.1-aaaaaaaaaa",
        {"labels": {"kind": "behavior", "remap": {"2": "0"}}},
    )
    second = write_lineage_run(
        features,
        "sequence-classifier",
        "0.1-bbbbbbbbbb",
        {"labels": {"kind": "behavior", "mask": [3]}},
    )
    controls = lineage_label_controls([first, second])
    assert controls == LabelControls(remap=frozenset({(2, 0)}), mask=frozenset({3}))


def test_no_inputs_give_no_controls() -> None:
    assert not lineage_label_controls([])
