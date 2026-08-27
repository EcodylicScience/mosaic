"""Custom label converter template.

A minimal, well-commented starting point for a custom label converter. Good for a
one-off format, a modified annotation tool, or an experimental setup.

Use this template when:

- You have custom CSV / JSON / Excel annotation files.
- You need to add a time offset or adjust timestamps.
- You need to inject animal IDs not present in the original files.
- Your annotation format is project-specific.

Example use case:
    Video snippets annotated in BORIS, where you also need to add a time offset
    (the snippet's start time in the full video), inject an animal ID that was
    not recorded during annotation, and adjust timestamps for synchronization.

Steps to create a custom converter:

1. Copy this file to a new name (e.g. ``my_custom_labels.py``).
2. Rename the class and its ``Params`` model, and set the class variables
   (``src_format`` / ``label_kind`` / ``label_format`` / ``version`` / ``Params``).
3. Implement ``_load_source_file()`` for your file format.
4. Implement ``_build_label_map()`` for your behavior names.
5. Implement ``_extract_annotations()`` for your data structure.
6. Register it in ``label_library/__init__.py`` (see the bottom of this file).
   Before indexing, not after: ``index_labels_raw`` refuses a ``src_format`` no
   registered converter claims.

``convert()`` returns one :class:`~mosaic.core.label_converter.LabelEntry` per
sequence -- data, not files. The ``Dataset`` writes each entry's payload into
``labels/<kind>/<run_id>/`` and records the typed index row, so this class never
computes an output path, writes a ``.npz``, or handles overwrite.

See the bottom of this file for a complete working example.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
)
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)

_FPS_DESCRIPTION = (
    "Frame rate used to convert annotation times to frame indices, unless a "
    "sequence supplies its own."
)

_VERBOSE_DESCRIPTION = "Whether the converter reports its progress as it runs."

_BACKGROUND_LABEL_DESCRIPTION = "The label assigned when no behavior is annotated."


class CustomLabelParams(LabelConvertParams):
    """Parameters for :class:`CustomLabelConverter`.

    ``group_from`` and ``strict_schema`` are already declared on
    :class:`~mosaic.core.label_converter.LabelConvertParams` -- do not redeclare
    them. Fields that change the labels are plain (hashed) fields, part of the
    label variant identity; validation-only or throughput-only knobs are tagged
    ``HASH_EXCLUDE`` so retuning them never mints a new variant.
    """

    # Parameters that change the labels -- hashed.
    fps: Annotated[float, Declared(_FPS_DESCRIPTION)] = 30.0
    background_label: Annotated[str, Declared(_BACKGROUND_LABEL_DESCRIPTION)] = "none"

    # Add your own hashed parameters here (they change the labels):
    # time_offset: float = 0.0
    # animal_id: str = "unknown"

    # A knob that does not change the labels -- excluded from the identity.
    verbose: Annotated[bool, HASH_EXCLUDE, Declared(_VERBOSE_DESCRIPTION)] = False


class CustomLabelConverter(LabelConverter[CustomLabelParams]):
    """Minimal template for a custom label converter.

    REQUIRED: implement ``_load_source_file()``, ``_build_label_map()`` and
    ``_extract_annotations()``.

    REQUIRED: set the class variables ``src_format`` / ``label_kind`` /
    ``label_format`` / ``version`` / ``Params``.
    """

    # ============================================================
    # STEP 1: REGISTRATION -- set these to unique identifiers.
    # ============================================================

    src_format = "my_custom_format"  # Used in convert_all_labels(source_format=...).
    label_kind = "behavior"  # Usually "behavior" or "id_tags".
    label_format = "my_custom_v1"  # Payload format a reader dispatches on.
    version = "0.1"  # Bump by hand when output semantics change.
    Params = CustomLabelParams

    # ============================================================
    # STEP 2: MAIN CONVERT METHOD -- usually you do not modify this.
    # ============================================================

    def convert(
        self,
        src_path: Path,
        params: CustomLabelParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read one source file into one :class:`LabelEntry` per sequence.

        Usually you do NOT modify this; implement the three helper methods below
        instead. This method loads the file, builds the label map, extracts
        annotations, converts them to per-frame labels, and returns one
        :class:`LabelEntry` per sequence. Writing the ``.npz`` files and the
        index row is the ``Dataset``'s job.

        Args:
            src_path: The raw label file to read.
            params: Typed conversion parameters (the label variant identity).
            raw_row: The ``labels_raw/index.csv`` row (group hint, source md5).
                Never hashed.

        Returns:
            One :class:`LabelEntry` per sequence.
        """
        # 1. Load your custom file format.
        data = self._load_source_file(src_path)

        # 2. Build the behavior vocabulary {id: name}.
        label_map = self._build_label_map(data, params)
        label_name_to_id = {name: id_ for id_, name in label_map.items()}
        label_ids = np.array(list(label_map.keys()), dtype=int)
        label_names = np.array(list(label_map.values()), dtype=object)

        # 3. The group string honors ``group_from`` (entry policy, never hashed).
        raw_group_hint = str(raw_row.get("group", "") or "")
        group_val = self._determine_group("", raw_group_hint, params)

        # 4. Extract the sequences (one per subject, video, ...).
        sequences = self._extract_annotations(data, src_path, raw_row, params)

        entries: list[LabelEntry] = []
        for seq_info in sequences:
            seq_name = str(seq_info["sequence_name"])
            annotations = seq_info["annotations"]  # [(start, stop, behavior), ...]
            fps = float(seq_info.get("fps", params.fps))
            metadata = seq_info.get("metadata", {})

            # Convert (start, stop, behavior) spans to per-frame labels.
            labels, frames = self._convert_to_frame_labels(
                annotations, label_name_to_id, fps, params
            )

            # Build the .npz payload: the arrays and scalars a reader loads.
            payload: dict[str, object] = {
                "group": group_val,
                "sequence": seq_name,
                "sequence_key": seq_name,
                "frames": frames,
                "labels": labels,
                "label_ids": label_ids,
                "label_names": label_names,
                "fps": fps,
            }

            # Fold any extra per-sequence metadata into the payload.
            for key, value in metadata.items():
                payload[f"meta_{key}"] = value

            entries.append(
                LabelEntry(
                    group=group_val,
                    sequence=seq_name,
                    payload=payload,
                    n_frames=int(labels.shape[0]),
                    label_ids=tuple(int(i) for i in label_map),
                    label_names=tuple(label_map.values()),
                )
            )

        return entries

    # ============================================================
    # STEP 3: IMPLEMENT THESE METHODS FOR YOUR FORMAT.
    # ============================================================

    def _load_source_file(self, src_path: Path) -> object:
        """Load your custom annotation file.

        Args:
            src_path: Path to the annotation file.

        Returns:
            Your data structure (DataFrame, dict, list, ...).

        Examples:
            CSV: ``return pd.read_csv(src_path)``.

            JSON::

                import json
                with open(src_path) as f:
                    return json.load(f)

            Excel: ``return pd.read_excel(src_path, sheet_name="Annotations")``.
        """
        raise NotImplementedError(
            "Implement _load_source_file() to load your annotation file. "
            "See the docstring for examples."
        )

    def _build_label_map(
        self, data: object, params: CustomLabelParams
    ) -> dict[int, str]:
        """Build the mapping from behavior ID to behavior name.

        Args:
            data: Loaded data from :meth:`_load_source_file`.
            params: Converter parameters.

        Returns:
            Mapping from ID to behavior name. MUST include
            ``params.background_label`` (conventionally at ID 0).

        Examples:
            From a DataFrame column::

                behaviors = sorted(data["behavior"].unique())
                behaviors = [params.background_label] + behaviors
                return {i: name for i, name in enumerate(behaviors)}

            Hardcoded::

                return {0: "none", 1: "grooming", 2: "eating", 3: "resting"}
        """
        raise NotImplementedError(
            "Implement _build_label_map() to create the behavior ID mapping. "
            "See the docstring for examples."
        )

    def _extract_annotations(
        self,
        data: object,
        src_path: Path,
        raw_row: Mapping[str, object],
        params: CustomLabelParams,
    ) -> list[dict[str, object]]:
        """Extract annotations into a standardized structure.

        This is the KEY method to implement for your format.

        Args:
            data: Loaded data from :meth:`_load_source_file`.
            src_path: Source file path (for reference).
            raw_row: The ``labels_raw/index.csv`` row.
            params: Converter parameters.

        Returns:
            A list of sequences, each a dict with:

            - ``"sequence_name"`` (str, REQUIRED): a unique name for the sequence.
            - ``"annotations"`` (list, REQUIRED): ``[(start, stop, behavior)]``,
              times in seconds; use ``start == stop`` for a point event.
            - ``"fps"`` (float, optional): FPS for this sequence.
            - ``"metadata"`` (dict, optional): extra metadata to fold into the
              payload under a ``meta_`` prefix.

        Examples:
            Single sequence from a DataFrame::

                annotations = [
                    (row["start"], row["stop"], row["behavior"])
                    for _, row in data.iterrows()
                ]
                return [{
                    "sequence_name": "recording_001",
                    "annotations": annotations,
                    "fps": params.fps,
                }]

            One sequence per subject/video::

                sequences = []
                for video_name, group_df in data.groupby("video"):
                    annotations = [
                        (row["start"], row["stop"], row["behavior"])
                        for _, row in group_df.iterrows()
                    ]
                    sequences.append({
                        "sequence_name": video_name,
                        "annotations": annotations,
                        "fps": group_df["fps"].iloc[0],
                    })
                return sequences
        """
        raise NotImplementedError(
            "Implement _extract_annotations() to extract behavior events. "
            "See the docstring for examples."
        )

    # ============================================================
    # HELPER METHODS -- usually you do not need to modify these.
    # ============================================================

    def _determine_group(
        self,
        source_group: str,
        raw_group_hint: str,
        params: CustomLabelParams,
    ) -> str:
        """Pick the output group per ``group_from`` (entry policy, never hashed)."""
        if params.group_from in {"filename", "both"} and raw_group_hint:
            return raw_group_hint
        return str(source_group or "")

    def _convert_to_frame_labels(
        self,
        annotations: list[tuple[float, float, str]],
        label_map: dict[str, int],
        fps: float,
        params: CustomLabelParams,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert time-based annotations to per-frame labels.

        Usually you do NOT modify this -- it handles the standard conversion from
        ``(start, stop, behavior)`` tuples to per-frame labels.

        Args:
            annotations: ``[(start_time, stop_time, behavior_name)]``, in seconds.
            label_map: Behavior name to ID mapping.
            fps: Frames per second.
            params: Converter parameters.

        Returns:
            A ``(labels, frames)`` pair: per-frame behavior IDs of shape
            ``(n_frames,)`` and their frame indices.
        """
        background_id = label_map[params.background_label]

        # Determine total duration.
        if not annotations:
            # No annotations -- create a minimal array.
            labels = np.array([background_id], dtype=int)
            frames = np.array([0], dtype=np.int32)
            return labels, frames

        max_time = max(stop for _start, stop, _behavior in annotations)
        n_frames = int(np.ceil(max_time * fps)) + 1

        # Initialize with the background label.
        labels = np.full(n_frames, background_id, dtype=int)

        # Fill in behaviors.
        for start_time, stop_time, behavior_name in annotations:
            behavior_id = label_map.get(behavior_name, background_id)

            start_frame = int(start_time * fps)
            stop_frame = int(stop_time * fps)

            # A point event marks a single frame.
            is_point = abs(start_time - stop_time) < 1e-6
            if is_point:
                if 0 <= start_frame < n_frames:
                    labels[start_frame] = behavior_id
            else:
                start_frame = max(0, start_frame)
                stop_frame = min(n_frames - 1, stop_frame)
                labels[start_frame : stop_frame + 1] = behavior_id

        frames = np.arange(n_frames, dtype=np.int32)
        return labels, frames

    def get_metadata(self) -> dict[str, object]:
        """Optional format-specific metadata for ``dataset.meta['labels'][kind]``."""
        return {}


# ============================================================
# EXAMPLE: modified BORIS with time offset and animal IDs.
# ============================================================


_TIME_OFFSET_DESCRIPTION = "Added to all timestamps."

_ANIMAL_ID_DESCRIPTION = (
    "The animal ID injected into the sequence name and its metadata, since it "
    "is not present in the BORIS file."
)


class ModifiedBorisParams(CustomLabelParams):
    """Parameters for :class:`ModifiedBorisConverter`.

    Extends :class:`CustomLabelParams` with two hashed fields: both change the
    labels, so both are part of the variant identity.
    """

    time_offset: Annotated[float, Declared(_TIME_OFFSET_DESCRIPTION, unit="s")] = 0.0
    animal_id: Annotated[str, Declared(_ANIMAL_ID_DESCRIPTION)] = "unknown"


class ModifiedBorisConverter(CustomLabelConverter):
    """Example: BORIS annotations from video snippets.

    Handles a time offset (the snippet's start time in the full video), an
    injected animal ID (not in the BORIS file), and the standard BORIS CSV
    format.

    Usage:
        >>> dataset.convert_all_labels(
        ...     source_format="modified_boris",
        ...     time_offset=120.5,  # Snippet starts at 2:00.5 in the full video.
        ...     animal_id="mouse_A12",  # Not in the BORIS file.
        ...     fps=30.0,
        ... )
    """

    src_format = "modified_boris"
    label_kind = "behavior"
    label_format = "modified_boris_v1"
    version = "0.1"
    Params = ModifiedBorisParams

    def _load_source_file(self, src_path: Path) -> pd.DataFrame:
        """Load the BORIS CSV export."""
        return pd.read_csv(src_path)

    def _build_label_map(
        self, data: pd.DataFrame, params: ModifiedBorisParams
    ) -> dict[int, str]:
        """Build the label map from the ``Behavior`` column."""
        behaviors = sorted(data["Behavior"].dropna().unique())
        if params.background_label not in behaviors:
            behaviors = [params.background_label] + behaviors
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(
        self,
        data: pd.DataFrame,
        src_path: Path,
        raw_row: Mapping[str, object],
        params: ModifiedBorisParams,
    ) -> list[dict[str, object]]:
        """Extract annotations, applying the time offset and animal ID."""
        time_offset = params.time_offset
        animal_id = params.animal_id

        # Extract point-style rows and apply the time offset.
        annotations: list[tuple[float, float, str]] = []
        for _, row in data.iterrows():
            start = row["Time"] + time_offset
            # A row with a Status column belongs to a START/STOP pair; handle
            # those below rather than as point events here.
            if "Status" in data.columns and pd.notna(row.get("Status")):
                continue
            stop = row.get("Stop", row["Time"]) + time_offset
            annotations.append((start, stop, row["Behavior"]))

        # Pair START/STOP events into spans.
        if "Status" in data.columns:
            active: dict[str, float] = {}
            for _, row in data.iterrows():
                if pd.isna(row.get("Status")):
                    continue
                behavior = row["Behavior"]
                time = row["Time"] + time_offset
                if row["Status"] == "START":
                    active[behavior] = time
                elif row["Status"] == "STOP" and behavior in active:
                    start = active.pop(behavior)
                    annotations.append((start, time, behavior))

        # One sequence carrying the injected animal ID as metadata.
        return [
            {
                "sequence_name": f"snippet_{animal_id}",
                "annotations": annotations,
                "fps": params.fps,
                "metadata": {
                    "animal_id": animal_id,
                    "time_offset": time_offset,
                    "source_file": src_path.name,
                },
            }
        ]


# ============================================================
# To register this converter, in label_library/__init__.py:
#
# from mosaic.core.label_converter import register_label_converter
# from . import custom_label_template
#
# custom_label_template.ModifiedBorisConverter = register_label_converter(
#     custom_label_template.ModifiedBorisConverter
# )
# ============================================================
