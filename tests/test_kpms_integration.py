"""Integration tests for KpmsFeature (requires external .venv with kpms)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_KPMS_PYTHON = Path(__file__).resolve().parent.parent / (
    "src/mosaic/behavior/feature_library/external/.venv/bin/python"
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _KPMS_PYTHON.exists(),
        reason=f"kpms .venv not found at {_KPMS_PYTHON}",
    ),
]


def _make_synthetic_tracks() -> list[tuple[str, pd.DataFrame]]:
    """Six sequences, two motion modes, 3 keypoints (triangle), 300 frames.

    Three keypoints in an asymmetric triangle (nose at top, left/right at
    base). The asymmetry means heading direction shows up in egocentric
    coordinates after alignment, allowing the AR-HMM to separate modes.

    - "horizontal" (3 seqs): constant +x velocity
    - "vertical"   (3 seqs): constant +y velocity
    """
    num_frames = 300
    tracks: list[tuple[str, pd.DataFrame]] = []

    # Triangle keypoint offsets from center of mass
    offsets = [(0.0, 1.0), (-0.8, -0.5), (0.8, -0.5)]

    modes = [
        ("horizontal", 1.0, 0.0),
        ("vertical", 0.0, 1.0),
    ]
    seqs_per_mode = 3

    for mode_name, vx, vy in modes:
        for i in range(seqs_per_mode):
            seq_name = f"{mode_name}_{i}"
            t = np.arange(num_frames, dtype=np.float64)
            cx = vx * t
            cy = vy * t
            data: dict[str, object] = {
                "frame": np.arange(num_frames),
                "group": "grp",
                "sequence": seq_name,
            }
            for k, (dx, dy) in enumerate(offsets):
                data[f"poseX{k}"] = cx + dx
                data[f"poseY{k}"] = cy + dy
            tracks.append((f"grp__{seq_name}", pd.DataFrame(data)))

    return tracks


def _make_feature(
    params: dict[str, object] | None = None,
) -> "KpmsFeature":
    from mosaic.behavior.feature_library.kpms import KpmsFeature

    inputs = KpmsFeature.Inputs(("tracks",))
    return KpmsFeature(inputs=inputs, params=params)


def _make_inputs(
    tracks: list[tuple[str, pd.DataFrame]],
):
    def inputs_fn():
        yield from tracks

    return inputs_fn


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    return tmp_path / "kpms_run"


@pytest.fixture
def feature_params() -> dict[str, object]:
    return {
        "num_iters_ar": 10,
        "num_iters_full": 10,
        "num_iters_apply": 10,
        "remove_outliers": False,
        "pose": {"pose_n": 3, "keypoint_names": ["nose", "left", "right"]},
        "anterior_bodyparts": ["nose"],
        "posterior_bodyparts": ["left", "right"],
        "latent_dim": 4,
    }


def test_fit_and_apply(
    run_root: Path,
    feature_params: dict[str, object],
) -> None:
    tracks = _make_synthetic_tracks()
    feat = _make_feature(feature_params)

    loaded = feat.load_state(run_root, {}, {})
    assert loaded is False

    feat.fit(_make_inputs(tracks))

    results: dict[str, np.ndarray] = {}
    for name, df in tracks:
        result = feat.apply(df)
        assert "frame" in result.columns
        assert "syllable" in result.columns
        assert len(result) == len(df)
        assert result["syllable"].dtype in (np.int32, np.int64)
        results[name] = result["syllable"].values

    # Diagnostic: syllable distributions per track
    print("\n--- Syllable diagnostics ---")
    for name, syllables in results.items():
        unique, counts = np.unique(syllables, return_counts=True)
        dist = ", ".join(f"{u}:{c}" for u, c in zip(unique, counts))
        print(f"  {name}: {dist}")

    names = list(results)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sa, sb = results[a], results[b]
            total = len(sa) + len(sb)
            # For each shared syllable, count frames in both tracks
            shared_labels = set(np.unique(sa)) & set(np.unique(sb))
            worst = 0
            worst_label = -1
            for label in shared_labels:
                shared_frames = int((sa == label).sum()) + int((sb == label).sum())
                if shared_frames > worst:
                    worst = shared_frames
                    worst_label = label
            pct = 100 * worst / total if total > 0 else 0
            print(f"  overlap({a}, {b}): {pct:.0f}% (syllable {worst_label})")

    feat.save_state(run_root)


def test_save_and_reload(
    run_root: Path,
    feature_params: dict[str, object],
) -> None:
    tracks = _make_synthetic_tracks()

    # Fit and save
    feat1 = _make_feature(feature_params)
    feat1.load_state(run_root, {}, {})
    feat1.fit(_make_inputs(tracks))
    feat1.save_state(run_root)

    # Reload
    feat2 = _make_feature(feature_params)
    loaded = feat2.load_state(run_root, {}, {})
    assert loaded is True

    _, df = tracks[0]
    result = feat2.apply(df)
    assert "frame" in result.columns
    assert "syllable" in result.columns
    assert len(result) == len(df)

    feat2.save_state(run_root)


def test_server_log_saved(
    run_root: Path,
    feature_params: dict[str, object],
) -> None:
    tracks = _make_synthetic_tracks()
    feat = _make_feature(feature_params)
    feat.load_state(run_root, {}, {})
    feat.fit(_make_inputs(tracks))
    feat.save_state(run_root)

    log_path = run_root / "kpms_server.log"
    assert log_path.exists()
    log_text = log_path.read_text()
    assert len(log_text) > 0
