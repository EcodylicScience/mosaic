"""MABe22 behavior label converter.

Converts MABe22 .npy annotation tracks to dense per-frame behavior labels.

MABe22 annotations are stored alongside keypoints inside
``{sequences: {seq_id: {keypoints, annotations}}}``. The ``annotations``
array is either:

- 2D ``(n_labels, T)`` -- binary tracks, one row per behavior in
  ``vocabulary``. Converted to multiclass via ``argmax(axis=0) + 1``,
  with 0 reserved for "no behavior active" frames.
- 1D ``(T,)`` -- already-multiclass dense labels. Used directly.

Emits one dense-format payload per sequence, so each feature frame maps
to its label via ``frame`` index (no ``individual_ids`` -- MABe22
labels are sequence-level, applied to every animal/pair).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
)


def _load_mabe22(path: Path) -> dict:
    """Load a MABe22 .npy annotation file."""
    from mosaic.core.track_library.mabe22 import load_mabe22

    return load_mabe22(path)


class MABe22BehaviorParams(LabelConvertParams):
    """Parameters for the MABe22 behavior converter.

    ``background_label`` names label 0 -- the "no behavior active" frame -- so it
    appears in the payload ``label_names`` and changes the labels; it is hashed.
    ``group_from`` (on the base) is entry policy and is not.
    """

    background_label: str = "background"


class MABe22BehaviorConverter(LabelConverter[MABe22BehaviorParams]):
    """Convert MABe22 .npy annotation tracks to dense per-frame NPZ labels."""

    src_format = "mabe22_npy"
    label_kind = "behavior"
    label_format = "dense"
    version = "0.1"
    Params = MABe22BehaviorParams

    def convert(
        self,
        src_path: Path,
        params: MABe22BehaviorParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read one MABe22 file into one :class:`LabelEntry` per sequence."""
        raw = _load_mabe22(src_path)

        vocab = raw.get("vocabulary") or raw.get("keypoint_vocabulary") or []
        vocab = [str(v) for v in vocab]
        label_map = {0: params.background_label}
        for i, name in enumerate(vocab):
            label_map[i + 1] = name
        label_ids = np.array(list(label_map.keys()), dtype=int)
        label_names = np.array(list(label_map.values()), dtype=object)

        if "sequences" in raw:
            sequences = raw["sequences"]
        else:
            sequences = {
                k: v
                for k, v in raw.items()
                if isinstance(v, dict)
                and k
                not in (
                    "vocabulary",
                    "keypoint_vocabulary",
                    "frame_number_map",
                    "task_type",
                )
            }

        raw_group_hint = str(raw_row.get("group", "") or "")
        group_val = raw_group_hint or src_path.stem

        entries: list[LabelEntry] = []
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
            payload: dict[str, object] = {
                "group": group_val,
                "sequence": seq_val,
                "sequence_key": seq_val,
                "label_format": self.label_format,
                "labels": dense_labels,
                "label_ids": label_ids,
                "label_names": label_names,
            }

            entries.append(
                LabelEntry(
                    group=group_val,
                    sequence=seq_val,
                    payload=payload,
                    n_frames=int(dense_labels.shape[0]),
                    label_ids=tuple(int(i) for i in label_map),
                    label_names=tuple(str(n) for n in label_map.values()),
                )
            )

        return entries

    def get_metadata(self) -> dict[str, object]:
        """MABe22-specific metadata for ``dataset.meta['labels'][kind]``."""
        return {}
