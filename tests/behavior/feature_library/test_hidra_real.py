"""The hidra feature against the real ``hidra-predict``.

``test_hidra_feature.py`` fakes the program. Here the feature runs HiDRA on a short
synthetic two-mouse clip with one ensemble config. The test then runs
``hidra-predict`` on a long-format file it writes itself, and the feature's
probabilities must equal that run's: the mapping, the frame offset and the
id-to-mouse round trip are checked against the program, not against a mirror of it.

The test needs ``hidra-predict`` (``MOSAIC_HIDRA_BIN`` or ``$PATH``) and the weights of
the default revision (``MOSAIC_HIDRA_MODELS_ROOT``). CI's ``hidra`` job provides both
and sets ``MOSAIC_CI_HIDRA``, which makes their absence a failure rather than a skip.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest

from mosaic.behavior.feature_library.hidra_feature import (
    DEFAULT_WEIGHTS_REVISION,
    HIDRA_ENV,
    HidraFeature,
    HidraPredictAvailable,
    missing_weight_files,
    read_hidra_provenance,
    weights_dir,
)
from mosaic.core.pipeline.toolenv import locate_tool, subprocess_env

pytestmark = pytest.mark.hidra

_CONFIG: Final = "11fps_4bp"
_FPS: Final = 30.0
_FRAMES: Final = 240
_FIRST_FRAME: Final = 500
_PIX_PER_CM: Final = 20.0
_BODYPARTS: Final = ("nose", "ear_left", "ear_right", "neck", "tail_base")
_HEAD: Final = {"lab": "JovialSwallow", "action": "sniff"}


def _require_hidra() -> Path:
    """The weights directory, or skip (fail under ``MOSAIC_CI_HIDRA``) without HiDRA."""
    problems: list[str] = []
    unavailable = HidraPredictAvailable().missing()
    if unavailable is not None:
        problems.append(unavailable)
    models = weights_dir(None, DEFAULT_WEIGHTS_REVISION)
    if missing_weight_files(models, [_CONFIG]):
        problems.append(f"no HiDRA weights in {models}")
    if problems:
        if os.environ.get("MOSAIC_CI_HIDRA"):
            pytest.fail(" ".join(problems))
        pytest.skip(" ".join(problems))
    return models


def _clip() -> pd.DataFrame:
    """Two mice walking past each other, five keypoints each, from frame 500."""
    t = np.arange(_FRAMES) / _FPS
    rows: list[pd.DataFrame] = []
    for identity, (x0, direction) in enumerate(((200.0, 1.0), (600.0, -1.0))):
        cx = x0 + direction * 40.0 * t
        cy = 300.0 + 20.0 * np.sin(t + identity)
        heading = 0.0 if direction > 0 else np.pi
        offsets = {  # body frame, centimeters forward and left
            "nose": (4.0, 0.0),
            "ear_left": (2.8, 0.9),
            "ear_right": (2.8, -0.9),
            "neck": (2.0, 0.0),
            "tail_base": (-4.0, 0.0),
        }
        table = pd.DataFrame(
            {
                "frame": np.arange(_FIRST_FRAME, _FIRST_FRAME + _FRAMES),
                "time": (np.arange(_FRAMES) + _FIRST_FRAME) / _FPS,
                "id": identity + 7,
                "sequence": "clip",
                "X": cx,
                "Y": cy,
            }
        )
        for index, name in enumerate(_BODYPARTS):
            forward, left = offsets[name]
            dx = (forward * np.cos(heading) - left * np.sin(heading)) * _PIX_PER_CM
            dy = (forward * np.sin(heading) + left * np.cos(heading)) * _PIX_PER_CM
            table[f"poseX{index}"] = cx + dx
            table[f"poseY{index}"] = cy + dy
        rows.append(table)
    return pd.concat(rows, ignore_index=True).sort_values(["frame", "id"])


def _direct_run(clip: pd.DataFrame, models: Path, root: Path) -> pd.DataFrame:
    """``hidra-predict`` on a long-format file written here, independently of the feature."""
    folder = root / "direct"
    folder.mkdir()
    long = pd.concat(
        [
            pd.DataFrame(
                {
                    "video_frame": clip["frame"] - _FIRST_FRAME,
                    "mouse_id": clip["id"].map({7: "mouse1", 8: "mouse2"}),
                    "bodypart": name,
                    "x": clip[f"poseX{index}"].astype(np.float32),
                    "y": clip[f"poseY{index}"].astype(np.float32),
                }
            )
            for index, name in enumerate(_BODYPARTS)
        ],
        ignore_index=True,
    )
    long.to_parquet(folder / "clip.parquet", index=False)
    out = root / "direct-out"
    argv = [
        *locate_tool(HIDRA_ENV, executable="hidra-predict").argv,
        str(folder),
        "--labs",
        _HEAD["lab"],
        "--actions",
        _HEAD["action"],
        "--configs",
        _CONFIG,
        "--pix-per-cm",
        str(_PIX_PER_CM),
        "--fps",
        str(_FPS),
        "--output",
        "probs",
        "--out",
        str(out),
    ]
    _ = subprocess.run(
        argv,
        env=subprocess_env({"HIDRA_MODELS_DIR": str(models)}),
        check=True,
        capture_output=True,
    )
    return pd.read_parquet(out / "clip.frames.parquet")


def test_the_feature_reports_what_hidra_predict_reports(tmp_path: Path) -> None:
    models = _require_hidra()
    clip = _clip()
    feature = HidraFeature(
        params={
            "pose": {"pose_n": len(_BODYPARTS)},
            "keypoint_map": {f"kp{i}": name for i, name in enumerate(_BODYPARTS)},
            "heads": [_HEAD],
            "pix_per_cm": _PIX_PER_CM,
            "configs": [_CONFIG],
        }
    )

    table = feature.apply(clip)
    install = read_hidra_provenance(
        locate_tool(HIDRA_ENV, executable="hidra-predict").argv, subprocess_env()
    )
    direct = _direct_run(clip, models, tmp_path)

    assert '"version": "' in install
    assert '"thresholds_csv": "' in install
    assert "JovialSwallow" in install
    assert len(table) == 2 * _FRAMES
    column = "prob_sniff__JovialSwallow"
    assert table[column].between(0.0, 1.0).all()
    for (id1, id2), subject, target in (
        ((7, 8), "mouse1", "mouse2"),
        ((8, 7), "mouse2", "mouse1"),
    ):
        rows = table[(table["id1"] == id1) & (table["id2"] == id2)].sort_values("frame")
        expected = direct[
            (direct["subject"] == subject) & (direct["target"] == target)
        ].sort_values("frame")
        np.testing.assert_array_equal(
            rows["frame"].to_numpy(), expected["frame"].to_numpy() + _FIRST_FRAME
        )
        np.testing.assert_array_equal(
            rows[column].to_numpy(), expected["prob"].to_numpy(dtype=np.float32)
        )
