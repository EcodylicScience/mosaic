"""KpmsFeature through ``run_feature`` against a real keypoint-moseq install.

These skip unless both of its preconditions hold: an interpreter that has
keypoint-moseq, and the license acceptance that ``KpmsFeature`` requires before
it will spawn one. The acceptance is a skip condition rather than something the
suite sets on the user's behalf -- a test run that accepted a non-commercial
license for whoever happened to invoke it would be a hole in the very check
these tests exercise.

One dataset is fitted and applied in one run, run again as a cache hit, and
applied by a run that names the fitted model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.kpms import (
    KPMS_ENV,
    KPMS_LICENSE_ENV,
    KPMS_PYTHON_ENV,
    KpmsFeature,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.toolenv import missing_interpreter
from mosaic.core.pipeline.types import Result
from tests.helpers import (
    add_tracks_table,
    kpms_params,
    make_dataset,
    model_reference,
    result_outputs,
    result_run_root,
)


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        missing_interpreter(KPMS_ENV) is not None,
        reason=(
            f"no keypoint-moseq interpreter found; build the environment under "
            f"feature_library/external or set {KPMS_PYTHON_ENV}"
        ),
    ),
    pytest.mark.skipif(
        os.environ.get(KPMS_LICENSE_ENV, "").strip() != "1",
        reason=f"{KPMS_LICENSE_ENV}=1 not set; see NOTICE for the terms",
    ),
]

_VARIANT: Final = "convert-synthetic.0.1-aaaaaaaaaa"
"""The tracks variant that contains the synthetic tracks."""

_NUM_FRAMES: Final = 300

_PARAMS: Final = kpms_params(
    num_iters_ar=10, num_iters_full=10, num_iters_apply=10, remove_outliers=False
)


def _synthetic_tracks() -> list[pd.DataFrame]:
    """Six sequences, two motion modes, 3 keypoints (triangle), 300 frames.

    Three keypoints in an asymmetric triangle (nose at top, left/right at
    base). The asymmetry means heading direction shows up in egocentric
    coordinates after alignment, allowing the AR-HMM to separate modes.

    - "horizontal" (3 seqs): constant +x velocity
    - "vertical"   (3 seqs): constant +y velocity
    """
    offsets = [(0.0, 1.0), (-0.8, -0.5), (0.8, -0.5)]
    modes = [("horizontal", 1.0, 0.0), ("vertical", 0.0, 1.0)]
    tracks: list[pd.DataFrame] = []
    for mode_name, vx, vy in modes:
        for i in range(3):
            t = np.arange(_NUM_FRAMES, dtype=np.float64)
            data: dict[str, object] = {
                "frame": np.arange(_NUM_FRAMES),
                "group": "grp",
                "sequence": f"{mode_name}_{i}",
            }
            for k, (dx, dy) in enumerate(offsets):
                data[f"poseX{k}"] = vx * t + dx
                data[f"poseY{k}"] = vy * t + dy
            tracks.append(pd.DataFrame(data))
    return tracks


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    ds = make_dataset(tmp_path)
    for table in _synthetic_tracks():
        add_tracks_table(ds, _VARIANT, table)
    return ds


def _feature(**overrides: object) -> KpmsFeature:
    return KpmsFeature(
        inputs=KpmsFeature.Inputs(("tracks",)), params={**_PARAMS, **overrides}
    )


def _syllables(ds: Dataset, result: Result) -> dict[str, pd.DataFrame]:
    outputs = result_outputs(ds, result)
    assert len(outputs) == 6
    for name, table in outputs.items():
        assert len(table) == _NUM_FRAMES, name
        assert table["syllable"].dtype in (np.int32, np.int64), name
    return outputs


def test_fit_apply_rerun_and_apply_a_named_model(dataset: Dataset) -> None:
    fitted = run_feature(dataset, _feature())

    assert fitted.failed_entries == ()
    syllables = _syllables(dataset, fitted)
    run_root = result_run_root(dataset, fitted)
    assert (run_root / "kpms_model.joblib").is_file()
    assert (run_root / "kpms_server.log").read_text()
    print("\n--- Syllable diagnostics ---")
    for name, table in sorted(syllables.items()):
        unique, counts = np.unique(table["syllable"], return_counts=True)
        dist = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts, strict=True))
        print(f"  {name}: {dist}")

    again = run_feature(dataset, _feature())

    assert again.cache_hit is True
    assert again.run_id == fitted.run_id

    applied = run_feature(dataset, _feature(model=model_reference(fitted)))

    assert applied.failed_entries == ()
    assert applied.run_id != fitted.run_id
    _ = _syllables(dataset, applied)
