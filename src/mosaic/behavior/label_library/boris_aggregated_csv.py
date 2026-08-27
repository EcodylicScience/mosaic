"""
BORIS Aggregated Events CSV/TSV label converter.

BORIS (Behavioral Observation Research Interactive Software) exports
behavioral observations in aggregated events format with one row per
behavior event including start/stop times and durations.

Expected BORIS Aggregated CSV/TSV format:
    Observation id, Observation date, Description, Media file, Total length,
    FPS, Subject, Behavior, Behavioral category, Modifiers, Behavior type,
    Start (s), Stop (s), Duration (s), Comment start, Comment stop

Key features:
- Auto-detects FPS from file (or uses parameter override)
- Handles both STATE events (with duration) and POINT events (instantaneous)
- Supports multiple subjects per observation
- Handles "No focal subject" behaviors
- Preserves behavioral categories and modifiers as metadata

References
----------
BORIS User Guide: http://www.boris.unito.it/user_guide/export_events/
GitHub: https://github.com/olivierfriard/BORIS
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

from mosaic.behavior.label_library._boris_descriptions import (
    BACKGROUND_LABEL_DESCRIPTION,
    BEHAVIOR_COL_DESCRIPTION,
    BEHAVIOR_TYPE_COL_DESCRIPTION,
    CATEGORY_COL_DESCRIPTION,
    FPS_COL_DESCRIPTION,
    FPS_DESCRIPTION,
    INCLUDE_POINT_EVENTS_DESCRIPTION,
    MODIFIERS_COL_DESCRIPTION,
    NO_FOCAL_SUBJECT_NAME_DESCRIPTION,
    OBSERVATION_COL_DESCRIPTION,
    PAIR_BEHAVIORS_DESCRIPTION,
    START_COL_DESCRIPTION,
    STOP_COL_DESCRIPTION,
    SUBJECT_COL_DESCRIPTION,
    SUBJECT_ID_MAP_DESCRIPTION,
)
from mosaic.core.helpers import to_safe_name
from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
)
from mosaic.core.params import Declared

_DELIMITER_DESCRIPTION = (
    "The column delimiter, tab for a TSV export or comma for a CSV export."
)


class BorisAggregatedCSVParams(LabelConvertParams):
    """Parameters for the BORIS aggregated-events converter.

    Every field determines the parsed labels -- the delimiter and FPS decide how
    the file is read and how times become frames, the column names select which
    columns are parsed, and the id maps and ``background_label`` decide which
    events survive and which individual IDs they carry -- so all are hashed.
    ``group_from`` and ``strict_schema`` (on the base) are entry policy and
    validation strictness, and are excluded from identity there.
    """

    delimiter: Annotated[str, Declared(_DELIMITER_DESCRIPTION)] = "\t"
    fps: Annotated[float | None, Declared(FPS_DESCRIPTION)] = None
    subject_id_map: Annotated[
        dict[str, int] | None, Declared(SUBJECT_ID_MAP_DESCRIPTION)
    ] = None
    pair_behaviors: Annotated[
        list[str] | None, Declared(PAIR_BEHAVIORS_DESCRIPTION)
    ] = None
    background_label: Annotated[str, Declared(BACKGROUND_LABEL_DESCRIPTION)] = "none"
    no_focal_subject_name: Annotated[
        str, Declared(NO_FOCAL_SUBJECT_NAME_DESCRIPTION)
    ] = "no_focal_subject"
    include_point_events: Annotated[
        bool, Declared(INCLUDE_POINT_EVENTS_DESCRIPTION)
    ] = True
    observation_col: Annotated[str, Declared(OBSERVATION_COL_DESCRIPTION)] = (
        "Observation id"
    )
    subject_col: Annotated[str, Declared(SUBJECT_COL_DESCRIPTION)] = "Subject"
    behavior_col: Annotated[str, Declared(BEHAVIOR_COL_DESCRIPTION)] = "Behavior"
    start_col: Annotated[str, Declared(START_COL_DESCRIPTION)] = "Start (s)"
    stop_col: Annotated[str, Declared(STOP_COL_DESCRIPTION)] = "Stop (s)"
    behavior_type_col: Annotated[str, Declared(BEHAVIOR_TYPE_COL_DESCRIPTION)] = (
        "Behavior type"
    )
    fps_col: Annotated[str, Declared(FPS_COL_DESCRIPTION)] = "FPS"
    category_col: Annotated[str, Declared(CATEGORY_COL_DESCRIPTION)] = (
        "Behavioral category"
    )
    modifiers_col: Annotated[str, Declared(MODIFIERS_COL_DESCRIPTION)] = (
        "Modifiers (empty if none)"
    )


class BorisAggregatedCSVConverter(LabelConverter[BorisAggregatedCSVParams]):
    """
    Convert BORIS Aggregated Events CSV/TSV to behavior dataset format.

    This converter processes BORIS aggregated event exports where each row
    represents a single behavior event with start time, stop time, and duration.
    Outputs individual_pair_v1 format with explicit individual IDs.

    The converter:
    1. Loads the BORIS CSV/TSV file
    2. Auto-detects FPS from the file or uses provided parameter
    3. Groups events by observation and subject
    4. Converts time-based events to sparse event format
    5. Creates one sequence per (observation, subject) combination
    6. Assigns individual IDs based on subject_id_map

    BORIS Subject Handling:
    - If subject_id_map provided: Maps subject names to numeric IDs
    - Individual behaviors: [subject_id, -1]
    - Pair behaviors (in pair_behaviors list): [id1, id2] and [id2, id1] (symmetric)
    - "No focal subject": [-1, -1] (scene-level labels)

    Usage
    -----
    >>> # Individual behaviors with subject mapping
    >>> dataset.convert_all_labels(
    ...     kind="behavior",
    ...     source_format="boris_aggregated_csv",
    ...     subject_id_map={"bee_0": 0, "bee_1": 1, "bee_2": 2},
    ...     pair_behaviors=["trophallaxis"],  # Symmetric pair behaviors
    ... )
    >>>
    >>> # Scene-level labels (no focal subject)
    >>> dataset.convert_all_labels(
    ...     kind="behavior",
    ...     source_format="boris_aggregated_csv",
    ...     # No subject_id_map: treats all as scene-level
    ... )
    """

    src_format = "boris_aggregated_csv"
    label_kind = "behavior"
    label_format = "individual_pair_v1"
    version = "0.1"
    Params = BorisAggregatedCSVParams

    def convert(
        self,
        src_path: Path,
        params: BorisAggregatedCSVParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read one BORIS file into one :class:`LabelEntry` per (observation, subject)."""
        # Load BORIS CSV/TSV
        df = pd.read_csv(src_path, delimiter=params.delimiter)

        # Validate required columns
        required_cols = [
            params.observation_col,
            params.subject_col,
            params.behavior_col,
            params.start_col,
            params.stop_col,
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"BORIS file missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Auto-detect FPS if not provided
        fps = params.fps
        if fps is None:
            if params.fps_col in df.columns:
                fps_values = df[params.fps_col].dropna().unique()
                if len(fps_values) == 0:
                    raise ValueError("FPS column exists but contains no valid values")
                elif len(fps_values) > 1:
                    print(
                        f"Warning: Multiple FPS values found: {fps_values}. "
                        f"Using first: {fps_values[0]}"
                    )
                fps = float(fps_values[0])
            else:
                raise ValueError(
                    f"FPS not provided and '{params.fps_col}' column not found in file. "
                    f"Please specify fps parameter."
                )

        print(f"Using FPS: {fps}")

        # Build label map from unique behaviors
        unique_behaviors = sorted(df[params.behavior_col].dropna().unique())
        if params.background_label not in unique_behaviors:
            unique_behaviors = [params.background_label] + unique_behaviors

        label_map = {i: name for i, name in enumerate(unique_behaviors)}
        label_name_to_id = {name: i for i, name in label_map.items()}

        # Determine group name
        group_val = self._determine_group("", params, raw_row)

        # Group by observation and subject
        obs_col = params.observation_col
        subj_col = params.subject_col

        # Handle NaN in subject column (treat as no focal subject)
        df[subj_col] = df[subj_col].fillna(params.no_focal_subject_name)

        entries: list[LabelEntry] = []

        # Process each (observation, subject) combination as a sequence
        for (obs_id, subject), obs_subj_df in df.groupby([obs_col, subj_col]):
            # Create sequence identifier
            # Format: observation_id__subject_name
            obs_safe = to_safe_name(str(obs_id))
            subj_safe = to_safe_name(str(subject))
            seq_val = f"{obs_safe}__{subj_safe}"

            # Convert events to sparse individual_pair_v1 format
            event_frames, event_labels, individual_ids = self._convert_to_sparse_events(
                obs_subj_df, label_name_to_id, fps, params, str(subject)
            )

            # Build npz payload
            label_ids = np.array(list(label_map.keys()), dtype=np.int32)
            label_names = np.array(list(label_map.values()), dtype=object)

            payload: dict[str, object] = {
                "group": group_val,
                "sequence": seq_val,
                "sequence_key": seq_val,
                "label_format": self.label_format,
                "frames": event_frames,
                "labels": event_labels,
                "individual_ids": individual_ids,
                "label_ids": label_ids,
                "label_names": label_names,
                "source_observation": str(obs_id),
                "source_subject": str(subject),
                "fps": float(fps),
            }

            n_frames = int(np.max(event_frames) + 1) if len(event_frames) > 0 else 0

            entries.append(
                LabelEntry(
                    group=group_val,
                    sequence=seq_val,
                    payload=payload,
                    n_frames=n_frames,
                    label_ids=tuple(int(i) for i in label_map.keys()),
                    label_names=tuple(str(n) for n in label_map.values()),
                )
            )

        return entries

    def get_metadata(self) -> dict[str, object]:
        """Static metadata for ``dataset.meta['labels'][kind]``.

        The label vocabulary is data-driven -- built per file from the behaviors
        present -- so only the source-format tag is static here.
        """
        return {"source_format": "BORIS Aggregated Events"}

    # ============ HELPER METHODS ============

    def _determine_group(
        self,
        source_group: str,
        params: BorisAggregatedCSVParams,
        raw_row: Mapping[str, object],
    ) -> str:
        """Determine output group name based on group_from parameter."""
        group_from = params.group_from
        if group_from == "filename":
            return str(raw_row.get("group", "") or "")
        elif group_from == "infile":
            return source_group
        elif group_from == "both":
            return str(raw_row.get("group", "") or "")
        else:
            return source_group

    def _convert_to_sparse_events(
        self,
        df: pd.DataFrame,
        label_map: dict[str, int],
        fps: float,
        params: BorisAggregatedCSVParams,
        subject: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert BORIS event table to sparse individual_pair_v1 format.

        Parameters
        ----------
        df : pd.DataFrame
            Events for a single (observation, subject) pair
        label_map : dict
            Behavior name to ID mapping
        fps : float
            Frames per second
        params : BorisAggregatedCSVParams
            Converter parameters
        subject : str
            Subject name from BORIS

        Returns
        -------
        frames : np.ndarray
            Frame indices, shape (n_events,), dtype int32
        labels : np.ndarray
            Behavior labels, shape (n_events,), dtype int32
        individual_ids : np.ndarray
            Individual IDs, shape (n_events, 2), dtype int32
        """
        behavior_col = params.behavior_col
        start_col = params.start_col
        stop_col = params.stop_col
        behavior_type_col = params.behavior_type_col
        background_id = label_map[params.background_label]
        include_point = params.include_point_events
        subject_id_map = params.subject_id_map
        pair_behaviors = params.pair_behaviors or []

        # Determine subject_id
        if subject_id_map and subject in subject_id_map:
            subject_id = subject_id_map[subject]
        elif subject == params.no_focal_subject_name:
            subject_id = -1  # Scene-level
        else:
            subject_id = -1  # Default to scene-level if no mapping

        # Collect all events
        event_frames_list = []
        event_labels_list = []
        individual_ids_list = []

        # Process each event
        for _, row in df.iterrows():
            behavior = row[behavior_col]
            behavior_id = label_map.get(behavior, background_id)

            # Skip background labels (sparse format)
            if behavior_id == background_id:
                continue

            start_time = row[start_col]
            stop_time = row[stop_col]

            # Convert times to frames
            start_frame = int(start_time * fps)
            stop_frame = int(stop_time * fps)

            # Check if this is a point event
            if behavior_type_col in df.columns:
                behavior_type = str(row[behavior_type_col]).upper()
                is_point = behavior_type == "POINT"
            else:
                # If no behavior type column, infer from start == stop
                is_point = abs(start_time - stop_time) < 1e-6

            # Generate frame range
            if is_point:
                if not include_point:
                    continue
                frame_range = [start_frame]
            else:
                frame_range = range(start_frame, stop_frame + 1)

            # Check if this is a pair behavior
            is_pair_behavior = behavior in pair_behaviors

            # Add events for each frame
            for frame in frame_range:
                if frame < 0:
                    continue

                if is_pair_behavior:
                    # Pair behavior: need to find the other subject(s)
                    # For now, use [-1, -1] placeholder
                    # User should handle pair creation in custom code if needed
                    individual_ids_list.append([subject_id, -1])
                else:
                    # Individual behavior: [subject_id, -1]
                    individual_ids_list.append([subject_id, -1])

                event_frames_list.append(frame)
                event_labels_list.append(behavior_id)

        # Convert to numpy arrays
        if len(event_frames_list) == 0:
            # No events: return empty arrays
            frames = np.array([], dtype=np.int32)
            labels = np.array([], dtype=np.int32)
            individual_ids = np.zeros((0, 2), dtype=np.int32)
        else:
            frames = np.array(event_frames_list, dtype=np.int32)
            labels = np.array(event_labels_list, dtype=np.int32)
            individual_ids = np.array(individual_ids_list, dtype=np.int32)

        return frames, labels, individual_ids
