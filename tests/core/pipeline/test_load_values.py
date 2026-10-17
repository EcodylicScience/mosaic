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
from mosaic.core.pipeline.loading import row_individual_pairs
from mosaic.core.pipeline.labels_index import _ambiguous_label_variant_message
from mosaic.core.pipeline.run import load_values
from mosaic.core.pipeline.types import GroundTruthLabelsSource, TracksColumn
from tests.helpers import MockDataset, add_track_sequences


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
    add_track_sequences(ds, ("g1", "s1"), n_rows=4)

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


def test_row_individual_pairs_prefers_pair_ids() -> None:
    """A pair row is aligned by its own direction, an individual row by its id."""
    pairs = pd.DataFrame({"frame": [0, 0], "id": [7, 7], "id1": [0, 1], "id2": [1, 0]})
    individuals = pd.DataFrame({"frame": [0, 0], "id": [0.0, np.nan]})

    pair_ids = row_individual_pairs(pairs)
    individual_ids = row_individual_pairs(individuals)

    assert pair_ids is not None and pair_ids.tolist() == [[0, 1], [1, 0]]
    assert individual_ids is not None
    assert individual_ids.tolist() == [[0, -1], [-1, -1]]
    assert row_individual_pairs(pd.DataFrame({"frame": [0]})) is None


def test_non_integer_ids_fall_back_to_frame_alignment() -> None:
    """Named individuals cannot be matched to an event's integer ids."""
    named = pd.DataFrame({"frame": [0, 1], "id": ["ant_a", "ant_b"]})
    fractional = pd.DataFrame({"frame": [0], "id1": [0.5], "id2": [1.0]})

    assert row_individual_pairs(named) is None
    assert row_individual_pairs(fractional) is None
