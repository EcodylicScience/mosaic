"""CalMS21 behavior label converter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
)
from mosaic.core.track_library.calms21 import calms21_entry_name

# The CalMS21 behavior vocabulary. Lived in ``dataset.py`` and was imported back
# here across the converter/dataset boundary; it is CalMS21-specific data, so it
# lives with the converter that owns it, and nothing imports ``dataset`` to read
# it any more.
BEHAVIOR_LABEL_MAP = {
    0: "attack",
    1: "investigation",
    2: "mount",
    3: "other_interaction",
}


def _load_calms21(path: Path):
    """Load a CalMS21 file (.npy or .json)."""
    from mosaic.core.track_library.helpers import load_calms21

    return load_calms21(path)


class CalMS21BehaviorParams(LabelConvertParams):
    """Parameters for the CalMS21 behavior converter.

    ``resident_id`` / ``intruder_id`` are the individual IDs the directed pair
    interaction is written between; they change the labels, so they are hashed.
    ``group_from`` (on the base) is entry policy and is not.
    """

    resident_id: int = 0
    intruder_id: int = 1


class CalMS21BehaviorConverter(LabelConverter[CalMS21BehaviorParams]):
    """Convert CalMS21 npy/json behavior annotations to the behavior format.

    CalMS21 files contain nested structure: group -> sequence -> annotations.
    This converter extracts per-frame behavior labels and converts them to
    ``individual_pair_v1`` format with explicit individual IDs. CalMS21 behaviors
    (attack, investigation, mount, other_interaction) are directed pair
    interactions between resident (ID=0) and intruder (ID=1), implicitly
    resident -> intruder.
    """

    src_format = "calms21_npy"  # Also handles calms21_json via load_calms21
    label_kind = "behavior"
    label_format = "individual_pair_v1"
    # 0.2: entry names are compound (``task1__test__m``) rather than slash paths,
    # matching what ``Calms21Converter`` has emitted since its own 0.2. The labels
    # are otherwise identical, but a variant's identity covers what it emits, and
    # this changed the entry keys and therefore the filenames.
    version = "0.2"
    Params = CalMS21BehaviorParams

    def convert(
        self,
        src_path: Path,
        params: CalMS21BehaviorParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read one CalMS21 file into one :class:`LabelEntry` per sequence."""
        nested = _load_calms21(src_path)
        entries: list[LabelEntry] = []

        label_ids = np.array(list(BEHAVIOR_LABEL_MAP.keys()), dtype=int)
        label_names = np.array(list(BEHAVIOR_LABEL_MAP.values()), dtype=object)

        raw_group_hint = str(raw_row.get("group", "") or "")
        group_from = params.group_from

        for group_name, seqs in nested.items():
            group_val_infile = str(group_name or "")

            if group_from in {"filename", "both"} and raw_group_hint:
                group_val = raw_group_hint
            else:
                group_val = group_val_infile

            for seq_key, seq_dict in seqs.items():
                if "annotations" not in seq_dict:
                    continue

                dense_labels = np.asarray(seq_dict["annotations"])
                if dense_labels.ndim > 1:
                    # Multi-annotator case: take the first column.
                    dense_labels = dense_labels[:, 0]
                dense_labels = dense_labels.astype(int, copy=False)

                # All CalMS21 labels are meaningful behaviors -- there is no
                # background label to exclude -- so every frame is an event.
                event_frames = np.arange(len(dense_labels), dtype=np.int32)
                event_labels = dense_labels.astype(np.int32)

                # Each event is a directed pair [resident, intruder].
                n_events = len(event_frames)
                individual_ids = np.full(
                    (n_events, 2),
                    [params.resident_id, params.intruder_id],
                    dtype=np.int32,
                )

                # The same flattening ``Calms21Converter.enumerate_sequences``
                # applies. Two reasons it is not optional: ``write_labels_row``
                # runs an entry name through ``validate_entry_name``, which
                # rejects the ``/`` a raw CalMS21 id carries -- so the raw form
                # did not merely mismatch, it raised -- and the tracks side has
                # emitted the compound spelling since its 0.2, so anything else
                # here is a label that no track table can be joined to.
                # ``sequence_key`` keeps the raw id, which is still the key
                # inside the source file.
                seq_val = calms21_entry_name(str(seq_key))
                payload: dict[str, object] = {
                    "group": group_val,
                    "sequence": seq_val,
                    "sequence_key": str(seq_key),
                    "label_format": self.label_format,
                    "frames": event_frames,
                    "labels": event_labels,
                    "individual_ids": individual_ids,
                    "label_ids": label_ids,
                    "label_names": label_names,
                }
                if (
                    group_from in {"filename", "both"}
                    and group_val_infile
                    and group_val_infile != group_val
                ):
                    payload["source_group"] = group_val_infile

                entries.append(
                    LabelEntry(
                        group=group_val,
                        sequence=seq_val,
                        payload=payload,
                        n_frames=int(dense_labels.shape[0]),
                        label_ids=tuple(int(i) for i in BEHAVIOR_LABEL_MAP),
                        label_names=tuple(BEHAVIOR_LABEL_MAP.values()),
                    )
                )

        return entries

    def get_metadata(self) -> dict[str, object]:
        """CalMS21-specific metadata for ``dataset.meta['labels'][kind]``."""
        return {
            "label_ids": list(BEHAVIOR_LABEL_MAP.keys()),
            "label_names": list(BEHAVIOR_LABEL_MAP.values()),
        }
