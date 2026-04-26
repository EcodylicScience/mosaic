"""MABe22 behavior label converter.

Converts MABe22 .npy annotation tracks to dense per-frame behavior labels.

MABe22 annotations are stored alongside keypoints inside
``{sequences: {seq_id: {keypoints, annotations}}}``. The ``annotations``
array is either:

- 2D ``(n_labels, T)`` — binary tracks, one row per behavior in
  ``vocabulary``. Converted to multiclass via ``argmax(axis=0) + 1``,
  with 0 reserved for "no behavior active" frames.
- 1D ``(T,)`` — already-multiclass dense labels. Used directly.

Writes one dense-format NPZ per sequence, so each feature frame maps
to its label via ``frame`` index (no ``individual_ids`` — MABe22
labels are sequence-level, applied to every animal/pair).
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mosaic.core.helpers import make_entry_key, to_safe_name


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


def _load_mabe22(path: Path) -> dict:
    from mosaic.core.track_library.mabe22 import load_mabe22
    return load_mabe22(path)


class MABe22BehaviorConverter:
    """Convert MABe22 .npy annotation tracks to dense per-frame NPZ labels."""

    src_format = "mabe22_npy"
    label_kind = "behavior"
    label_format = "dense"

    _defaults = dict(
        group_from="filename",
        background_label="background",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.params = _merge_params(params, self._defaults)
        self.params.update(kwargs)

    def convert(self,
                src_path: Path,
                raw_row: pd.Series,
                labels_root: Path,
                params: dict,
                overwrite: bool,
                existing_pairs: set[tuple[str, str]]) -> list[dict]:
        raw = _load_mabe22(src_path)

        vocab = raw.get("vocabulary") or raw.get("keypoint_vocabulary") or []
        vocab = [str(v) for v in vocab]
        label_map = {0: self.params["background_label"]}
        for i, name in enumerate(vocab):
            label_map[i + 1] = name
        label_ids = np.array(list(label_map.keys()), dtype=int)
        label_names = np.array(list(label_map.values()), dtype=object)

        if "sequences" in raw:
            sequences = raw["sequences"]
        else:
            sequences = {k: v for k, v in raw.items()
                         if isinstance(v, dict)
                         and k not in ("vocabulary", "keypoint_vocabulary",
                                       "frame_number_map", "task_type")}

        raw_group_hint = str(raw_row.get("group", "") or "")
        group_val = raw_group_hint or src_path.stem

        rows_out: list[dict] = []
        for seq_key, seq_dict in sequences.items():
            if "annotations" not in seq_dict:
                continue

            ann = np.asarray(seq_dict["annotations"])
            if ann.ndim == 1:
                dense_labels = ann.astype(np.int32, copy=False)
            elif ann.ndim == 2:
                mask = ann.any(axis=0)
                argmax = ann.argmax(axis=0).astype(np.int32)
                dense_labels = np.where(mask, argmax + 1, 0).astype(np.int32)
            else:
                continue

            seq_val = str(seq_key)
            pair = (group_val, seq_val)
            safe_group = to_safe_name(group_val) if group_val else ""
            safe_seq = to_safe_name(seq_val)
            fname = f"{make_entry_key(group_val, seq_val)}.npz"
            out_path = labels_root / fname

            if not overwrite and pair in existing_pairs and out_path.exists():
                continue

            payload = {
                "group": group_val,
                "sequence": seq_val,
                "sequence_key": seq_val,
                "label_format": "dense",
                "labels": dense_labels,
                "label_ids": label_ids,
                "label_names": label_names,
            }
            np.savez_compressed(out_path, **payload)
            existing_pairs.add(pair)

            rows_out.append({
                "kind": "behavior",
                "label_format": "dense",
                "group": group_val,
                "sequence": seq_val,
                "group_safe": safe_group,
                "sequence_safe": safe_seq,
                "abs_path": str(out_path.resolve()),
                "source_abs_path": str(src_path.resolve()),
                "source_md5": raw_row.get("md5", ""),
                "n_frames": int(dense_labels.shape[0]),
                "label_ids": ",".join(map(str, label_map.keys())),
                "label_names": ",".join(label_map.values()),
            })

        return rows_out

    def get_metadata(self) -> dict:
        return {}
