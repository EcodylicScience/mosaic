"""CalMS21 .npy/.json track converter.

Converts CalMS21 multi-animal pose tracking files to the standardized
trex_v1 parquet schema. Handles task1/task2/task3 splits.

CalMS21 keypoint layout: (T, n_animals, xy=2, n_landmarks)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from mosaic.core.dataset import register_track_converter, register_track_seq_enumerator
from mosaic.core.track_library.helpers import (
    load_calms21, angle_from_two_points, angle_from_pca, norm_hint,
)


def _calms21_seq_to_trex_df(one_seq_dict: dict,
                            groupname: str,
                            seq_id: str,
                            neck_idx: Optional[int] = None,
                            tail_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Convert a single sequence dict to T-Rex-like long DataFrame (rows = frames x animals).
    """
    # Pick features: either 'features' present or 'keypoints'
    use_features = ("features" in one_seq_dict)
    if use_features:
        # not used in output columns; could be stored elsewhere if needed
        _ = np.asarray(one_seq_dict["features"])  # (T, K)
    keypoints = np.asarray(one_seq_dict["keypoints"])    # (T, 2, 2, L)
    scores    = np.asarray(one_seq_dict.get("scores", None))       # (T, 2, L) or None
    ann       = np.asarray(one_seq_dict["annotations"]) if "annotations" in one_seq_dict else None
    meta      = one_seq_dict.get("metadata", {})
    fps       = float(meta.get("fps", meta.get("frame_rate", 30.0)))

    T = keypoints.shape[0]
    n_anim = keypoints.shape[1]
    n_lm   = keypoints.shape[3]

    rows = []
    for a in range(n_anim):
        # Extract XY for this animal: (T, L, 2)
        X = keypoints[:, a, 0, :]  # (T, L)
        Y = keypoints[:, a, 1, :]  # (T, L)
        XY = np.stack([X, Y], axis=-1)  # (T, L, 2)

        # Centroid over landmarks
        cx = X.mean(axis=1)  # (T,)
        cy = Y.mean(axis=1)

        # Vel/acc (finite diff)
        VX = np.gradient(cx) * fps
        VY = np.gradient(cy) * fps
        SPEED = np.hypot(VX, VY)
        AX = np.gradient(VX) * fps
        AY = np.gradient(VY) * fps

        # Heading angle
        if (neck_idx is not None) and (tail_idx is not None) and 0 <= neck_idx < n_lm and 0 <= tail_idx < n_lm:
            neck = XY[:, neck_idx, :]  # (T,2)
            tail = XY[:, tail_idx, :]
            ANGLE = angle_from_two_points(neck, tail)
        else:
            ANGLE = angle_from_pca(XY)

        # Build a per-frame DataFrame
        data = {
            "frame": np.arange(T, dtype=int),
            "time":  np.arange(T, dtype=float) / fps,
            "id":    np.full(T, a, dtype=int),
            "X": cx,
            "Y": cy,
            "X#wcentroid": cx,
            "Y#wcentroid": cy,
            "VX": VX, "VY": VY,
            "SPEED": SPEED, "AX": AX, "AY": AY,
            "ANGLE": ANGLE,
            "group": np.full(T, groupname),
            "sequence": np.full(T, seq_id),
        }

        # Pose columns
        for k in range(n_lm):
            data[f"poseX{k}"] = X[:, k]
            data[f"poseY{k}"] = Y[:, k]

        # Optional: label per frame if present (flatten if multi-dim)
        if ann is not None:
            lbl = ann
            if lbl.ndim > 1:
                lbl = lbl[:, 0]
            data["label"] = lbl.astype(int, copy=False)

        # Optional: keypoint scores columns, if provided
        if scores is not None:
            S = np.asarray(scores)  # (T, 2, L)
            S_a = S[:, a, :]        # (T, L)
            for k in range(n_lm):
                data[f"poseP{k}"] = S_a[:, k]

        rows.append(pd.DataFrame(data))

    out = pd.concat(rows, ignore_index=True)
    # Add placeholders often present in T-Rex schema
    out["missing"] = False
    out["visual_identification_p"] = 1.0
    out["timestamp"] = out["time"]
    for col in ["X","Y","SPEED#pcentroid","SPEED#wcentroid","midline_x","midline_y",
                "midline_length","midline_segment_length","normalized_midline",
                "ANGULAR_V#centroid","ANGULAR_A#centroid","BORDER_DISTANCE#pcentroid",
                "MIDLINE_OFFSET","num_pixels","detection_p"]:
        if col not in out.columns:
            out[col] = np.nan
    return out


def calms21_to_trex_df(path: Path | str,
                       prefer_group: Optional[str] = None,
                       prefer_sequence: Optional[str] = None,
                       neck_idx: Optional[int] = None,
                       tail_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Load a CalMS21 .npy/.json and return a concatenated T-Rex-like DataFrame.
    Optionally filter to a specific (group, sequence).
    """
    nested = load_calms21(path)

    groups_present = set(nested.keys())
    seq_filter = None
    direct_group_match_only = True
    if prefer_group and prefer_group not in groups_present:
        # interpret dataset-level hint (e.g., calms21_task1_test)
        seq_filter = _calms21_make_seq_filter_from_hint(prefer_group)
        if seq_filter is not None:
            direct_group_match_only = False

    rows = []
    for groupname, group in nested.items():
        for seq_id, seq in group.items():
            # strict sequence filter (exact match) if requested
            if prefer_sequence and seq_id != prefer_sequence:
                continue
            # group filter: either exact top-level match, or sequence-path filter if hint provided
            if direct_group_match_only:
                if prefer_group and groupname != prefer_group:
                    continue
            else:
                if seq_filter and not seq_filter(groupname, seq_id):
                    continue

            # ensure arrays where needed
            seq = {k: (np.array(v) if isinstance(v, list) else v) for k, v in seq.items()}
            rows.append(_calms21_seq_to_trex_df(seq, groupname, seq_id,
                                                neck_idx=neck_idx, tail_idx=tail_idx))
    if not rows:
        if prefer_group or prefer_sequence:
            raise KeyError(f"Requested CalMS21 ({prefer_group}, {prefer_sequence}) not found in {path}")
        raise RuntimeError(f"No sequences found in CalMS21 file: {path}")
    return pd.concat(rows, ignore_index=True)


def _calms21_converter(path: Path, params: dict) -> pd.DataFrame:
    prefer_group   = norm_hint(params.get("group"))
    prefer_sequence= norm_hint(params.get("sequence"))
    neck_idx = params.get("neck_idx", None)
    tail_idx = params.get("tail_idx", None)
    debug = bool(params.get("debug", False))

    # quick inspect
    nested = load_calms21(path)
    if debug:
        pairs = [(g, s) for g, grp in nested.items() for s in grp.keys()]
        print(f"[calms21] in-file pairs ({len(pairs)}): {pairs[:10]}{' ...' if len(pairs)>10 else ''}")
        print(f"[calms21] prefer_group={prefer_group} prefer_sequence={prefer_sequence}")

    # if explicit selection given, return only that
    if prefer_group or prefer_sequence:
        return calms21_to_trex_df(
            path,
            prefer_group=prefer_group,
            prefer_sequence=prefer_sequence,
            neck_idx=neck_idx,
            tail_idx=tail_idx,
        )

    # else single-pair inference
    pairs = [(g, s) for g, grp in nested.items() for s in grp.keys()]
    if len(pairs) == 1:
        g, s = pairs[0]
        return calms21_to_trex_df(
            path,
            prefer_group=g,
            prefer_sequence=s,
            neck_idx=neck_idx,
            tail_idx=tail_idx,
        )
    raise ValueError(
        f"Ambiguous CalMS21 file {path}; contains multiple sequences {pairs}. "
        f"Pass params with group/sequence to disambiguate."
    )


def _calms21_make_seq_filter_from_hint(hint: Optional[str]):
    """
    Return a predicate f(groupname, seq_id)->bool for dataset-level hints like
    'calms21_task1_train', 'calms21_task1_test', 'calms21_task2_train/test',
    'calms21_task3_train/test'. If not applicable, return None.
    """
    if not hint:
        return None
    h = hint.strip().lower()

    def pred_task_split(task_prefix: str, split: str):
        def _pred(_g, _s):
            # matches path patterns like taskX/.../<split>/...
            return _s.startswith(task_prefix) and (f"/{split}/" in _s)
        return _pred

    # task1
    if h.startswith("calms21_task1_"):
        split = "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        if split:
            return pred_task_split("task1/", split)

    # task2 (note: has an annotator level 'task2/annotator1/<split>/...')
    if h.startswith("calms21_task2_"):
        split = "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        if split:
            def _pred(_g, _s):
                return _s.startswith("task2/") and (f"/{split}/" in _s)
            return _pred

    # task3 (behavior level: 'task3/<behavior>/<split>/...')
    if h.startswith("calms21_task3_"):
        split = "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        if split:
            return pred_task_split("task3/", split)

    return None


def _enumerate_calms21_sequences(path: Path) -> list[tuple[str, str]]:
    nested = load_calms21(path)
    pairs: list[tuple[str, str]] = []
    for g, grp in nested.items():
        for s in grp.keys():
            pairs.append((str(g), str(s)))
    return pairs


# Register converters for both .npy and .json sources (same structure)
register_track_converter("calms21_npy", _calms21_converter)
register_track_converter("calms21_json", _calms21_converter)

# Register sequence enumerators
register_track_seq_enumerator("calms21_npy", _enumerate_calms21_sequences)
register_track_seq_enumerator("calms21_json", _enumerate_calms21_sequences)
