"""The entries that an `extract-labeled-templates` run kept for training.

A model fitted from templates deals its folds over entries, and an entry that drew no
template row still counts as one the run saw. `train_entries_of_templates` reads them
from the provenance written beside the templates.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mosaic.behavior.feature_library.helpers import train_entries_of_templates


def _provenance(root: Path, splits: dict[tuple[str, str], str]) -> Path:
    root.mkdir(parents=True)
    pd.DataFrame(
        {
            "group": [g for g, _ in splits],
            "sequence": [s for _, s in splits],
            "split": list(splits.values()),
            "count": 0,
            "proportion": 0.0,
        }
    ).to_parquet(root / "template_provenance.parquet")
    return root / "templates.parquet"


def test_the_train_entries_are_those_the_run_assigned_to_train(tmp_path: Path) -> None:
    templates = _provenance(
        tmp_path / "run",
        {("g", "a"): "train", ("g", "b"): "test", ("h", "c"): "train"},
    )

    assert train_entries_of_templates(templates) == [("g", "a"), ("h", "c")]


def test_a_run_without_provenance_has_no_known_train_entries(tmp_path: Path) -> None:
    assert train_entries_of_templates(tmp_path / "templates.parquet") is None
