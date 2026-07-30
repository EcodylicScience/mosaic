"""
BORIS Pandas DataFrame pickle label converter.

BORIS (Behavioral Observation Research Interactive Software) can export
behavioral observations as a Pandas DataFrame saved with pickle. This format
preserves all data types and structure from the aggregated events export.

The pickle format contains the same data as the aggregated CSV/TSV export,
but with proper Python data types (datetime, float, etc.) already applied.

Expected structure:
    DataFrame with columns including:
    - Observation id (str)
    - Observation date (datetime or str)
    - Description (str)
    - Media file (str)
    - Total length (float)
    - FPS (float)
    - Subject (str)
    - Behavior (str)
    - Behavioral category (str)
    - Modifiers (empty if none) (str)
    - Behavior type (str): "STATE" or "POINT"
    - Start (s) (float)
    - Stop (s) (float)
    - Duration (s) (float or NaN for point events)
    - Comment start (str)
    - Comment stop (str)
    - [Independent variables columns]

Loading:
    import pandas as pd
    df = pd.read_pickle('boris_export.pkl')

References
----------
BORIS User Guide: http://www.boris.unito.it/user_guide/export_events/
GitHub: https://github.com/olivierfriard/BORIS
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.helpers import to_safe_name
from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
)


class BorisPandasPickleParams(LabelConvertParams):
    """Parameters for the BORIS Pandas pickle converter.

    Every field here changes the parsed labels -- the frame rate, the subject
    to individual-ID mapping, which behaviors are treated as symmetric pairs,
    the background label, and the column names read from the export -- so they
    are all hashed. ``group_from`` and ``strict_schema`` (on the base) are entry
    policy and validation strictness and are excluded from identity.

    Attributes:
        fps: Frames per second. If ``None``, auto-detected from the FPS column
            in the DataFrame.
        subject_id_map: Mapping from BORIS subject names to numeric IDs, e.g.
            ``{"bee_0": 0, "bee_1": 1}``. If ``None``, all labels are treated as
            scene-level (``individual_ids=[-1, -1]``).
        pair_behaviors: Behavior names that are symmetric pair interactions.
            These create both ``[i, j]`` and ``[j, i]`` entries.
        background_label: Label used when no behavior is active (label ID 0).
        no_focal_subject_name: Name used for behaviors with "No focal subject".
        include_point_events: Whether to include POINT events (instantaneous
            behaviors).
        observation_col: Column name for the observation identifier.
        subject_col: Column name for the subject.
        behavior_col: Column name for the behavior.
        start_col: Column name for the start time.
        stop_col: Column name for the stop time.
        behavior_type_col: Column name for the behavior type (STATE/POINT).
        fps_col: Column name for the FPS value.
        category_col: Column name for the behavioral category.
        modifiers_col: Column name for the modifiers.
    """

    fps: float | None = None
    subject_id_map: dict[str, int] | None = None
    pair_behaviors: list[str] | None = None
    background_label: str = "none"
    no_focal_subject_name: str = "no_focal_subject"
    include_point_events: bool = True
    observation_col: str = "Observation id"
    subject_col: str = "Subject"
    behavior_col: str = "Behavior"
    start_col: str = "Start (s)"
    stop_col: str = "Stop (s)"
    behavior_type_col: str = "Behavior type"
    fps_col: str = "FPS"
    category_col: str = "Behavioral category"
    modifiers_col: str = "Modifiers (empty if none)"


class BorisPandasPickleConverter(LabelConverter[BorisPandasPickleParams]):
    """
    Convert BORIS Pandas DataFrame pickle to behavior dataset format.

    This converter processes BORIS pickle exports created with:
        df.to_pickle('filename.pkl')

    The pickle format contains the same aggregated event data as CSV/TSV
    exports but with preserved data types (datetime, float, etc.).
    Outputs individual_pair_v1 format with explicit individual IDs.

    The converter:
    1. Loads the Pandas DataFrame from pickle
    2. Auto-detects FPS from the DataFrame or uses provided parameter
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
    ...     source_format="boris_pandas_pickle",
    ...     subject_id_map={"bee_0": 0, "bee_1": 1, "bee_2": 2},
    ...     pair_behaviors=["trophallaxis"],  # Symmetric pair behaviors
    ... )
    >>>
    >>> # Scene-level labels (no focal subject)
    >>> dataset.convert_all_labels(
    ...     kind="behavior",
    ...     source_format="boris_pandas_pickle",
    ...     # No subject_id_map: treats all as scene-level
    ... )
    """

    # ============ REGISTRATION ATTRIBUTES ============
    src_format = "boris_pandas_pickle"
    label_kind = "behavior"
    label_format = "individual_pair_v1"
    version = "0.1"
    Params = BorisPandasPickleParams

    def convert(
        self,
        src_path: Path,
        params: BorisPandasPickleParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """
        Convert BORIS Pandas pickle to per-sequence behavior labels.

        Args:
            src_path: Path to the BORIS pickle file (.pkl).
            params: Typed conversion parameters.
            raw_row: The ``labels_raw/index.csv`` row, carrying the group hint
                and the source md5.

        Returns:
            One :class:`LabelEntry` per (observation, subject) sequence.
        """
        # Load BORIS DataFrame from pickle
        try:
            df = pd.read_pickle(src_path)
        except Exception as e:
            raise ValueError(f"Failed to load BORIS pickle file: {e}")

        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Expected DataFrame in pickle file, got {type(df)}. "
                f"File may not be a BORIS export."
            )

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
                f"BORIS DataFrame missing required columns: {missing}\n"
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
                        f"Warning: Multiple FPS values found: {fps_values}. Using first: {fps_values[0]}"
                    )
                fps = float(fps_values[0])
            else:
                raise ValueError(
                    f"FPS not provided and '{params.fps_col}' column not found in DataFrame. "
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
                "label_format": "individual_pair_v1",
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
        """Return metadata for dataset.meta."""
        return {
            "source_format": "BORIS Pandas Pickle",
        }

    # ============ HELPER METHODS ============

    def _determine_group(
        self,
        source_group: str,
        params: BorisPandasPickleParams,
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
        params: BorisPandasPickleParams,
        subject: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert BORIS event table to sparse individual_pair_v1 format.

        Args:
            df: Events for a single (observation, subject) pair.
            label_map: Behavior name to ID mapping.
            fps: Frames per second.
            params: Converter parameters.
            subject: Subject name from BORIS.

        Returns:
            A tuple ``(frames, labels, individual_ids)`` where ``frames`` has
            shape ``(n_events,)`` dtype int32, ``labels`` has shape
            ``(n_events,)`` dtype int32, and ``individual_ids`` has shape
            ``(n_events, 2)`` dtype int32.
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
