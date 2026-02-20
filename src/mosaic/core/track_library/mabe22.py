"""MABe22 .npy track converter.

Converts MABe22 multi-species multi-task benchmark files to the standardized
trex_v1 parquet schema. Handles mouse triplets, fly groups, and beetle-ant data.

MABe22 file structure:
    {vocabulary: [...], sequences: {seq_id: {keypoints: ..., annotations: ...}}}

Keypoint layouts per species:
    Mouse:  (T, 3, 12, 2) — 3 animals, 12 keypoints, xy   [int32 pixel]
    Fly:    (T, 11, 24, 2) — 11 animals, 24 keypoints, xy  [float32]
    Beetle: (T, 4)         — flat (beetle_x, beetle_y, ant_x, ant_y)  [float32 0-1]

Paper: https://arxiv.org/abs/2207.10553
Data:  https://doi.org/10.22002/rdsa8-rde65
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from mosaic.core.dataset import register_track_converter, register_track_seq_enumerator
from mosaic.core.track_library.helpers import angle_from_pca, norm_hint


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_mabe22(path: Path | str) -> dict:
    """Load a MABe22 .npy file and return the unwrapped dict."""
    p = Path(path)
    data = np.load(p, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected pickled dict in MABe22 file, got {type(data)}: {p}")
    return data


# ---------------------------------------------------------------------------
# Per-sequence converter
# ---------------------------------------------------------------------------

def _mabe22_seq_to_df(keypoints: np.ndarray,
                      annotations: Optional[np.ndarray],
                      seq_id: str,
                      groupname: str,
                      fps: float = 30.0) -> pd.DataFrame:
    """
    Convert one MABe22 sequence to a long-format DataFrame.

    Handles three keypoint shapes:
        4D (T, n_animals, n_kp, 2) — mouse / fly
        2D (T, n_cols)             — beetle (flat centroid pairs)
    """
    keypoints = np.asarray(keypoints, dtype=float)

    if keypoints.ndim == 4:
        return _convert_4d(keypoints, annotations, seq_id, groupname, fps)
    elif keypoints.ndim == 2:
        return _convert_2d(keypoints, annotations, seq_id, groupname, fps)
    else:
        raise ValueError(
            f"Unsupported MABe22 keypoint shape {keypoints.shape} for sequence {seq_id}. "
            f"Expected 4D (T, animals, kp, 2) or 2D (T, cols)."
        )


def _convert_4d(keypoints: np.ndarray,
                annotations: Optional[np.ndarray],
                seq_id: str,
                groupname: str,
                fps: float) -> pd.DataFrame:
    """Mouse / fly: keypoints shape (T, n_animals, n_kp, 2)."""
    T, n_anim, n_kp, _ = keypoints.shape

    rows = []
    for a in range(n_anim):
        X = keypoints[:, a, :, 0]  # (T, n_kp)
        Y = keypoints[:, a, :, 1]  # (T, n_kp)
        XY = keypoints[:, a, :, :]  # (T, n_kp, 2)

        cx = X.mean(axis=1)
        cy = Y.mean(axis=1)

        VX = np.gradient(cx) * fps
        VY = np.gradient(cy) * fps
        SPEED = np.hypot(VX, VY)
        AX = np.gradient(VX) * fps
        AY = np.gradient(VY) * fps
        ANGLE = angle_from_pca(XY)

        data = {
            "frame": np.arange(T, dtype=int),
            "time":  np.arange(T, dtype=float) / fps,
            "id":    np.full(T, a, dtype=int),
            "X": cx, "Y": cy,
            "X#wcentroid": cx, "Y#wcentroid": cy,
            "VX": VX, "VY": VY,
            "SPEED": SPEED, "AX": AX, "AY": AY,
            "ANGLE": ANGLE,
            "group": groupname,
            "sequence": seq_id,
        }

        for k in range(n_kp):
            data[f"poseX{k}"] = X[:, k]
            data[f"poseY{k}"] = Y[:, k]

        rows.append(pd.DataFrame(data))

    out = pd.concat(rows, ignore_index=True)
    _add_annotations(out, annotations, T, n_anim)
    _add_trex_placeholders(out)
    return out


def _convert_2d(keypoints: np.ndarray,
                annotations: Optional[np.ndarray],
                seq_id: str,
                groupname: str,
                fps: float) -> pd.DataFrame:
    """Beetle: keypoints shape (T, 4) — [beetle_x, beetle_y, ant_x, ant_y]."""
    T, n_cols = keypoints.shape
    # Split into pairs: each consecutive (x, y) is one animal
    n_anim = n_cols // 2

    rows = []
    for a in range(n_anim):
        cx = keypoints[:, 2 * a]
        cy = keypoints[:, 2 * a + 1]

        VX = np.gradient(cx) * fps
        VY = np.gradient(cy) * fps
        SPEED = np.hypot(VX, VY)
        AX = np.gradient(VX) * fps
        AY = np.gradient(VY) * fps
        # Only one point per animal — no PCA heading possible
        ANGLE = np.arctan2(VY, VX)

        data = {
            "frame": np.arange(T, dtype=int),
            "time":  np.arange(T, dtype=float) / fps,
            "id":    np.full(T, a, dtype=int),
            "X": cx, "Y": cy,
            "X#wcentroid": cx, "Y#wcentroid": cy,
            "VX": VX, "VY": VY,
            "SPEED": SPEED, "AX": AX, "AY": AY,
            "ANGLE": ANGLE,
            "group": groupname,
            "sequence": seq_id,
        }

        # Single-point pose
        data["poseX0"] = cx
        data["poseY0"] = cy

        rows.append(pd.DataFrame(data))

    out = pd.concat(rows, ignore_index=True)
    _add_annotations(out, annotations, T, n_anim)
    _add_trex_placeholders(out)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_annotations(df: pd.DataFrame,
                     annotations: Optional[np.ndarray],
                     T: int,
                     n_anim: int) -> None:
    """Add annotation columns to the DataFrame (in-place).

    MABe22 annotations are (n_labels, T) — same across all animals in a sequence.
    We broadcast each label row to all animals.
    """
    if annotations is None:
        return
    ann = np.asarray(annotations)
    if ann.ndim == 1:
        # Single label track
        tiled = np.tile(ann, n_anim)
        df["label"] = tiled
    elif ann.ndim == 2:
        # (n_labels, T) — add each as a separate column
        n_labels = ann.shape[0]
        for i in range(n_labels):
            tiled = np.tile(ann[i], n_anim)
            df[f"annotation_{i}"] = tiled


def _add_trex_placeholders(df: pd.DataFrame) -> None:
    """Add standard trex_v1 placeholder columns."""
    df["missing"] = False
    df["visual_identification_p"] = 1.0
    df["timestamp"] = df["time"]
    for col in ["SPEED#pcentroid", "SPEED#wcentroid", "midline_x", "midline_y",
                "midline_length", "midline_segment_length", "normalized_midline",
                "ANGULAR_V#centroid", "ANGULAR_A#centroid", "BORDER_DISTANCE#pcentroid",
                "MIDLINE_OFFSET", "num_pixels", "detection_p"]:
        if col not in df.columns:
            df[col] = np.nan


# ---------------------------------------------------------------------------
# Top-level converter (registered)
# ---------------------------------------------------------------------------

def _mabe22_converter(path: Path, params: dict) -> pd.DataFrame:
    """Convert a MABe22 .npy file to a trex_v1-compatible DataFrame."""
    raw = load_mabe22(path)

    # MABe22 files have {vocabulary, sequences: {seq_id: ...}} or
    # submission files have {sequences: {seq_id: ...}} directly
    if "sequences" in raw:
        sequences = raw["sequences"]
    else:
        # Fallback: treat all non-metadata keys as sequences
        sequences = {k: v for k, v in raw.items()
                     if isinstance(v, dict) and k not in ("vocabulary", "keypoint_vocabulary",
                                                           "frame_number_map", "task_type")}

    if not sequences:
        raise ValueError(f"No sequences found in MABe22 file: {path}")

    prefer_group = norm_hint(params.get("group"))
    prefer_seq = norm_hint(params.get("sequence"))
    fps = float(params.get("fps", params.get("fps_default", 30.0)))

    # Default group from filename stem (e.g., "mouse_triplet_train")
    group = prefer_group or path.stem

    if prefer_seq:
        if prefer_seq not in sequences:
            raise KeyError(f"Sequence {prefer_seq!r} not found in {path}. "
                           f"Available: {list(sequences.keys())[:10]}")
        seq_dict = sequences[prefer_seq]
        kp = np.asarray(seq_dict["keypoints"])
        ann = np.asarray(seq_dict["annotations"]) if "annotations" in seq_dict else None
        return _mabe22_seq_to_df(kp, ann, prefer_seq, group, fps)

    # Convert all sequences if no specific one requested — but for multi-sequence
    # files called from convert_one_track, we should only get one at a time
    if len(sequences) == 1:
        seq_id = next(iter(sequences))
        seq_dict = sequences[seq_id]
        kp = np.asarray(seq_dict["keypoints"])
        ann = np.asarray(seq_dict["annotations"]) if "annotations" in seq_dict else None
        return _mabe22_seq_to_df(kp, ann, seq_id, group, fps)

    raise ValueError(
        f"MABe22 file {path} contains {len(sequences)} sequences. "
        f"Pass params with 'sequence' to select one, or use index_tracks_raw + convert_all_tracks."
    )


def _enumerate_mabe22_sequences(path: Path) -> list[tuple[str, str]]:
    """Return (group, sequence) pairs from a MABe22 file."""
    raw = load_mabe22(path)
    group = path.stem

    if "sequences" in raw:
        sequences = raw["sequences"]
    else:
        sequences = {k: v for k, v in raw.items()
                     if isinstance(v, dict) and k not in ("vocabulary", "keypoint_vocabulary",
                                                           "frame_number_map", "task_type")}

    return [(group, str(s)) for s in sequences.keys()]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

register_track_converter("mabe22_npy", _mabe22_converter)
register_track_seq_enumerator("mabe22_npy", _enumerate_mabe22_sequences)
