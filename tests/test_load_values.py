"""`load_values` must be able to answer the refusal it can provoke.

It threads ``tracks_run_id`` precisely so a dataset holding two tracks recipes for
one sequence can say which to read, and its docstring explains why. For labels it
accepted nothing: it called ``_build_labels_lookup(ds, kind)`` with the default,
while that function's own docstring instructs callers to pass the same selector they
gave ``resolve_labels_variants``. So on a dataset with two label recipes the
resolver refused and named a keyword ``load_values`` did not have.

The message compounded it by naming ``build_manifest``, which has no
``labels_run_id`` parameter either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.pipeline import run as run_module
from mosaic.core.pipeline.labels_index import _ambiguous_label_variant_message
from mosaic.core.pipeline.run import load_values
from mosaic.core.pipeline.types import GroundTruthLabelsSource, TracksColumn
from tests.mock_dataset import MockDataset


def test_the_refusal_names_a_keyword_the_caller_can_pass() -> None:
    """It named build_manifest, which takes no such argument."""
    message = _ambiguous_label_variant_message(("", "seq"), ["a.0.1-aaa", "b.0.1-bbb"])

    assert "labels_run_id" in message
    assert "load_values" in message
    assert "build_manifest" not in message


def test_load_values_threads_the_labels_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selector has to reach the lookup, which is where it stopped."""
    ds = MockDataset(tmp_path)
    tracks = ds.get_root("tracks")
    path = tracks / "g1__s1.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame": range(4),
            "time": [f / 30.0 for f in range(4)],
            "id": [0] * 4,
            "X": np.arange(4.0),
            "Y": np.arange(4.0),
        }
    ).to_parquet(path)
    pd.DataFrame([{"group": "g1", "sequence": "s1", "abs_path": str(path)}]).to_csv(
        tracks / "index.csv", index=False
    )

    seen: list[str | None] = []

    def _record(
        _ds: object, kind: str, labels_run_id: str | None = None
    ) -> dict[tuple[str, str], Path]:
        seen.append(labels_run_id)
        return {}

    monkeypatch.setattr(run_module, "_build_labels_lookup", _record)

    _ = load_values(
        ds,
        [TracksColumn(column="X"), GroundTruthLabelsSource()],
        labels_run_id="chosen.0.1-abc",
    )

    assert seen == ["chosen.0.1-abc"]
