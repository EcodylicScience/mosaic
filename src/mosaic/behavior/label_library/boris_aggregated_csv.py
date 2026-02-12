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

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from mosaic.core.helpers import to_safe_name


def _merge_params(overrides: Optional[Dict[str, Any]], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to merge user params with defaults."""
    if not overrides:
        return dict(defaults)
    out = dict(defaults)
    out.update({k: v for k, v in overrides.items() if v is not None})
    return out


class BorisAggregatedCSVConverter:
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

    Parameters
    ----------
    group_from : str, default="filename"
        How to determine group name
    delimiter : str, default="\\t"
        CSV delimiter (use "\\t" for TSV, "," for CSV)
    fps : float or None, default=None
        Frames per second. If None, auto-detected from FPS column in file
    subject_id_map : dict or None, default=None
        Mapping from BORIS subject names to numeric IDs.
        Example: {"bee_0": 0, "bee_1": 1}
        If None, all labels treated as scene-level (individual_ids=[-1, -1])
    pair_behaviors : list[str] or None, default=None
        List of behavior names that are symmetric pair interactions.
        These will create both [i, j] and [j, i] entries.
        Requires exactly 2 subjects per observation.
    background_label : str, default="none"
        Label to use when no behavior is active (label ID 0)
    no_focal_subject_name : str, default="no_focal_subject"
        Name to use for behaviors with "No focal subject"
    include_point_events : bool, default=True
        Whether to include POINT events (instantaneous behaviors)
    observation_col : str, default="Observation id"
        Column name for observation identifier
    subject_col : str, default="Subject"
        Column name for subject
    behavior_col : str, default="Behavior"
        Column name for behavior
    start_col : str, default="Start (s)"
        Column name for start time
    stop_col : str, default="Stop (s)"
        Column name for stop time
    behavior_type_col : str, default="Behavior type"
        Column name for behavior type (STATE/POINT)
    fps_col : str, default="FPS"
        Column name for FPS value
    """

    # ============ REGISTRATION ATTRIBUTES ============
    src_format = "boris_aggregated_csv"
    label_kind = "behavior"
    label_format = "individual_pair_v1"

    # ============ DEFAULT PARAMETERS ============
    _defaults = dict(
        group_from="filename",
        delimiter="\t",  # TSV by default, use "," for CSV
        fps=None,  # Auto-detect from file
        subject_id_map=None,  # Map subject names to numeric IDs
        pair_behaviors=None,  # List of symmetric pair behavior names
        background_label="none",
        no_focal_subject_name="no_focal_subject",
        include_point_events=True,
        # Column names (can be overridden if BORIS export format changes)
        observation_col="Observation id",
        subject_col="Subject",
        behavior_col="Behavior",
        start_col="Start (s)",
        stop_col="Stop (s)",
        behavior_type_col="Behavior type",
        fps_col="FPS",
        category_col="Behavioral category",
        modifiers_col="Modifiers (empty if none)",
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the converter with parameters."""
        self.params = _merge_params(params, self._defaults)
        self.params.update(kwargs)

    def convert(self,
                src_path: Path,
                raw_row: pd.Series,
                labels_root: Path,
                params: dict,
                overwrite: bool,
                existing_pairs: set[tuple[str, str]]) -> list[dict]:
        """
        Convert BORIS aggregated CSV/TSV to per-sequence behavior labels.

        Parameters
        ----------
        src_path : Path
            Path to BORIS aggregated events CSV/TSV file
        raw_row : pd.Series
            Row from tracks_raw/index.csv
        labels_root : Path
            Output directory for label files
        params : dict
            Conversion parameters
        overwrite : bool
            Whether to overwrite existing files
        existing_pairs : set[tuple[str, str]]
            Already converted (group, sequence) pairs

        Returns
        -------
        list[dict]
            Index rows for labels/index.csv
        """
        p = self.params

        # Load BORIS CSV/TSV
        df = pd.read_csv(src_path, delimiter=p["delimiter"])

        # Validate required columns
        required_cols = [
            p["observation_col"],
            p["subject_col"],
            p["behavior_col"],
            p["start_col"],
            p["stop_col"],
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"BORIS file missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Auto-detect FPS if not provided
        fps = p["fps"]
        if fps is None:
            if p["fps_col"] in df.columns:
                fps_values = df[p["fps_col"]].dropna().unique()
                if len(fps_values) == 0:
                    raise ValueError("FPS column exists but contains no valid values")
                elif len(fps_values) > 1:
                    print(f"Warning: Multiple FPS values found: {fps_values}. Using first: {fps_values[0]}")
                fps = float(fps_values[0])
            else:
                raise ValueError(
                    f"FPS not provided and '{p['fps_col']}' column not found in file. "
                    f"Please specify fps parameter."
                )

        print(f"Using FPS: {fps}")

        # Build label map from unique behaviors
        unique_behaviors = sorted(df[p["behavior_col"]].dropna().unique())
        if p["background_label"] not in unique_behaviors:
            unique_behaviors = [p["background_label"]] + unique_behaviors

        label_map = {i: name for i, name in enumerate(unique_behaviors)}
        label_name_to_id = {name: i for i, name in label_map.items()}

        # Determine group name
        group_val = self._determine_group("", raw_row)

        # Group by observation and subject
        obs_col = p["observation_col"]
        subj_col = p["subject_col"]

        # Handle NaN in subject column (treat as no focal subject)
        df[subj_col] = df[subj_col].fillna(p["no_focal_subject_name"])

        rows_out = []

        # Process each (observation, subject) combination as a sequence
        for (obs_id, subject), obs_subj_df in df.groupby([obs_col, subj_col]):
            # Create sequence identifier
            # Format: observation_id__subject_name
            obs_safe = to_safe_name(str(obs_id))
            subj_safe = to_safe_name(str(subject))
            seq_val = f"{obs_safe}__{subj_safe}"

            pair = (group_val, seq_val)

            # Skip if exists
            if not overwrite and pair in existing_pairs:
                continue

            # Convert events to sparse individual_pair_v1 format
            event_frames, event_labels, individual_ids = self._convert_to_sparse_events(
                obs_subj_df, label_name_to_id, fps, p, str(subject)
            )

            # Create safe names and output path
            safe_group = to_safe_name(group_val) if group_val else ""
            safe_seq = to_safe_name(seq_val)
            fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
            out_path = labels_root / fname

            # Build npz payload
            label_ids = np.array(list(label_map.keys()), dtype=np.int32)
            label_names = np.array(list(label_map.values()), dtype=object)

            payload = {
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

            # Save npz
            np.savez_compressed(out_path, **payload)
            existing_pairs.add(pair)

            # Build index row
            index_row = {
                "kind": self.label_kind,
                "label_format": self.label_format,
                "group": group_val,
                "sequence": seq_val,
                "group_safe": safe_group,
                "sequence_safe": safe_seq,
                "abs_path": str(out_path.resolve()),
                "source_abs_path": str(src_path.resolve()),
                "source_md5": raw_row.get("md5", ""),
                "n_frames": int(np.max(event_frames) + 1) if len(event_frames) > 0 else 0,
                "n_events": int(len(event_frames)),
                "label_ids": ",".join(map(str, label_map.keys())),
                "label_names": ",".join(label_map.values()),
                "source_observation": str(obs_id),
                "source_subject": str(subject),
                "fps": str(fps),
            }
            rows_out.append(index_row)

        return rows_out

    def get_metadata(self) -> dict:
        """Return metadata for dataset.meta."""
        return {
            "fps": self.params.get("fps"),
            "source_format": "BORIS Aggregated Events",
        }

    # ============ HELPER METHODS ============

    def _determine_group(self, source_group: str, raw_row: pd.Series) -> str:
        """Determine output group name based on group_from parameter."""
        group_from = self.params.get("group_from", "filename")
        if group_from == "filename":
            return str(raw_row.get("group", "") or "")
        elif group_from == "infile":
            return source_group
        elif group_from == "both":
            return str(raw_row.get("group", "") or "")
        else:
            return source_group

    def _convert_to_sparse_events(self,
                                   df: pd.DataFrame,
                                   label_map: dict,
                                   fps: float,
                                   params: dict,
                                   subject: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        params : dict
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
        behavior_col = params["behavior_col"]
        start_col = params["start_col"]
        stop_col = params["stop_col"]
        behavior_type_col = params["behavior_type_col"]
        background_id = label_map[params["background_label"]]
        include_point = params["include_point_events"]
        subject_id_map = params.get("subject_id_map")
        pair_behaviors = params.get("pair_behaviors") or []

        # Determine subject_id
        if subject_id_map and subject in subject_id_map:
            subject_id = subject_id_map[subject]
        elif subject == params["no_focal_subject_name"]:
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
