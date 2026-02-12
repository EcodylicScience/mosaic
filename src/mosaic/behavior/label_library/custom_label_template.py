"""
Custom Label Converter Template

This is a minimal, well-commented template for creating custom label converters.
Perfect for one-off formats, modified annotation tools, or experimental setups.

Use this template when:
- You have custom CSV/JSON/Excel annotation files
- You need to add time offsets or adjust timestamps
- You need to inject animal IDs not in the original files
- Your annotation format is project-specific

Example use case:
    Video snippets annotated in BORIS, but need to:
    - Add time offset (snippet start time in full video)
    - Add animal IDs (not recorded during annotation)
    - Adjust timestamps for synchronization

Steps to create a custom converter:
1. Copy this file to a new name (e.g., `my_custom_labels.py`)
2. Update the registration attributes (src_format, label_format)
3. Implement _load_source_file() for your file format
4. Implement _build_label_map() for your behavior names
5. Implement _extract_annotations() for your data structure
6. Test with a small file
7. Import in label_library/__init__.py to register

See the bottom of this file for a complete working example.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
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


class CustomLabelConverter:
    """
    Minimal template for custom label converters.

    REQUIRED: Implement these methods:
    - _load_source_file()
    - _build_label_map()
    - _extract_annotations()

    REQUIRED: Set these class attributes:
    - src_format: str
    - label_kind: str
    - label_format: str
    """

    # ============================================================
    # STEP 1: REGISTRATION - Set these to unique identifiers
    # ============================================================

    src_format = "my_custom_format"      # Used in convert_all_labels(source_format="...")
    label_kind = "behavior"              # Usually "behavior" or "id_tags"
    label_format = "my_custom_v1"        # Version identifier for this format

    # ============================================================
    # STEP 2: PARAMETERS - Define your converter's parameters
    # ============================================================

    _defaults = dict(
        group_from="filename",           # Standard: how to determine group name
        fps=30.0,                        # Frames per second (if not in file)
        background_label="none",         # Label when no behavior is active

        # Add your custom parameters here:
        # time_offset=0.0,               # Add to all timestamps (seconds)
        # animal_id_file=None,           # Path to file mapping videos to animal IDs
        # id_column="animal_id",         # Column name for IDs
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize converter with parameters."""
        self.params = _merge_params(params, self._defaults)
        self.params.update(kwargs)

    # ============================================================
    # STEP 3: MAIN CONVERT METHOD - Usually you don't need to modify this
    # ============================================================

    def convert(self,
                src_path: Path,
                raw_row: pd.Series,
                labels_root: Path,
                params: dict,
                overwrite: bool,
                existing_pairs: set[tuple[str, str]]) -> list[dict]:
        """
        Main conversion method. Usually you DON'T need to modify this.
        Instead, implement the helper methods below.

        This method:
        1. Loads your file
        2. Builds label map
        3. Extracts annotations
        4. Converts to frame-by-frame labels
        5. Saves NPZ files
        6. Returns index rows
        """
        p = self.params

        # 1. Load your custom file format
        data = self._load_source_file(src_path)

        # 2. Build behavior label map
        label_map = self._build_label_map(data, p)
        label_name_to_id = {name: id_ for id_, name in label_map.items()}

        # 3. Determine group name
        group_val = self._determine_group("", raw_row)

        # 4. Extract annotations (list of sequences)
        sequences = self._extract_annotations(data, src_path, raw_row, p)

        rows_out = []

        # 5. Process each sequence
        for seq_info in sequences:
            seq_name = seq_info["sequence_name"]
            annotations = seq_info["annotations"]  # List of (start, stop, behavior_name)
            fps = seq_info.get("fps", p["fps"])
            metadata = seq_info.get("metadata", {})

            pair = (group_val, seq_name)

            # Skip if exists
            if not overwrite and pair in existing_pairs:
                continue

            # Convert to frame-by-frame labels
            labels, frames = self._convert_to_frame_labels(
                annotations, label_name_to_id, fps, p
            )

            # Create safe names and output path
            safe_group = to_safe_name(group_val) if group_val else ""
            safe_seq = to_safe_name(seq_name)
            fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
            out_path = labels_root / fname

            # Build NPZ payload
            label_ids = np.array(list(label_map.keys()), dtype=int)
            label_names = np.array(list(label_map.values()), dtype=object)

            payload = {
                "group": group_val,
                "sequence": seq_name,
                "sequence_key": seq_name,
                "frames": frames,
                "labels": labels,
                "label_ids": label_ids,
                "label_names": label_names,
                "fps": float(fps),
            }

            # Add any custom metadata from seq_info
            for key, value in metadata.items():
                payload[f"meta_{key}"] = value

            # Save NPZ
            np.savez_compressed(out_path, **payload)
            existing_pairs.add(pair)

            # Build index row
            index_row = {
                "kind": self.label_kind,
                "label_format": self.label_format,
                "group": group_val,
                "sequence": seq_name,
                "group_safe": safe_group,
                "sequence_safe": safe_seq,
                "abs_path": str(out_path.resolve()),
                "source_abs_path": str(src_path.resolve()),
                "source_md5": raw_row.get("md5", ""),
                "n_frames": int(labels.shape[0]),
                "label_ids": ",".join(map(str, label_map.keys())),
                "label_names": ",".join(label_map.values()),
                "fps": str(fps),
            }

            # Add metadata to index row
            for key, value in metadata.items():
                index_row[f"meta_{key}"] = str(value)

            rows_out.append(index_row)

        return rows_out

    # ============================================================
    # STEP 4: IMPLEMENT THESE METHODS FOR YOUR FORMAT
    # ============================================================

    def _load_source_file(self, src_path: Path) -> Any:
        """
        Load your custom annotation file.

        Parameters
        ----------
        src_path : Path
            Path to annotation file

        Returns
        -------
        Any
            Your data structure (DataFrame, dict, list, etc.)

        Examples
        --------
        CSV:
        >>> return pd.read_csv(src_path)

        JSON:
        >>> import json
        >>> with open(src_path) as f:
        ...     return json.load(f)

        Excel:
        >>> return pd.read_excel(src_path, sheet_name="Annotations")

        Multiple files (e.g., CSV + metadata JSON):
        >>> df = pd.read_csv(src_path)
        >>> meta_path = src_path.with_suffix('.json')
        >>> with open(meta_path) as f:
        ...     metadata = json.load(f)
        >>> return {"annotations": df, "metadata": metadata}
        """
        # IMPLEMENT THIS FOR YOUR FILE FORMAT
        raise NotImplementedError(
            "Implement _load_source_file() to load your annotation file. "
            "See docstring for examples."
        )

    def _build_label_map(self, data: Any, params: dict) -> dict[int, str]:
        """
        Build mapping from behavior ID to behavior name.

        Parameters
        ----------
        data : Any
            Loaded data from _load_source_file()
        params : dict
            Converter parameters

        Returns
        -------
        dict[int, str]
            Mapping from ID to behavior name
            MUST include background_label at ID 0

        Examples
        --------
        From DataFrame column:
        >>> behaviors = sorted(data["behavior"].unique())
        >>> behaviors = [params["background_label"]] + behaviors
        >>> return {i: name for i, name in enumerate(behaviors)}

        From metadata:
        >>> behaviors = data["metadata"]["behavior_names"]
        >>> return {i: name for i, name in enumerate(behaviors)}

        Hardcoded:
        >>> return {
        ...     0: "none",
        ...     1: "grooming",
        ...     2: "eating",
        ...     3: "resting",
        ... }
        """
        # IMPLEMENT THIS TO BUILD YOUR LABEL MAP
        raise NotImplementedError(
            "Implement _build_label_map() to create behavior ID mapping. "
            "See docstring for examples."
        )

    def _extract_annotations(self,
                            data: Any,
                            src_path: Path,
                            raw_row: pd.Series,
                            params: dict) -> List[dict]:
        """
        Extract annotations into a standardized structure.

        This is the KEY method to implement for your format.

        Parameters
        ----------
        data : Any
            Loaded data from _load_source_file()
        src_path : Path
            Source file path (for reference)
        raw_row : pd.Series
            Row from tracks_raw/index.csv (may contain metadata)
        params : dict
            Converter parameters

        Returns
        -------
        List[dict]
            List of sequences, each with:
            {
                "sequence_name": str,           # REQUIRED: Unique name for this sequence
                "annotations": List[Tuple],     # REQUIRED: [(start, stop, behavior_name), ...]
                "fps": float,                   # OPTIONAL: FPS for this sequence
                "metadata": dict,               # OPTIONAL: Extra metadata to save
            }

        Notes
        -----
        Each annotation is a tuple: (start_time, stop_time, behavior_name)
        - start_time: float, in seconds
        - stop_time: float, in seconds (use start_time for point events)
        - behavior_name: str, must be in label_map

        Examples
        --------
        Single sequence from DataFrame:
        >>> df = data
        >>> annotations = [
        ...     (row["start"], row["stop"], row["behavior"])
        ...     for _, row in df.iterrows()
        ... ]
        >>> return [{
        ...     "sequence_name": "recording_001",
        ...     "annotations": annotations,
        ...     "fps": params["fps"],
        ... }]

        Multiple subjects/videos:
        >>> sequences = []
        >>> for video_name, group_df in data.groupby("video"):
        ...     annotations = [
        ...         (row["start"], row["stop"], row["behavior"])
        ...         for _, row in group_df.iterrows()
        ...     ]
        ...     sequences.append({
        ...         "sequence_name": video_name,
        ...         "annotations": annotations,
        ...         "fps": group_df["fps"].iloc[0],
        ...     })
        >>> return sequences

        With time offset and animal ID injection:
        >>> time_offset = params.get("time_offset", 0.0)
        >>> animal_id = params.get("animal_id", "unknown")
        >>> annotations = [
        ...     (row["start"] + time_offset,
        ...      row["stop"] + time_offset,
        ...      row["behavior"])
        ...     for _, row in data.iterrows()
        ... ]
        >>> return [{
        ...     "sequence_name": f"video_{animal_id}",
        ...     "annotations": annotations,
        ...     "metadata": {"animal_id": animal_id, "time_offset": time_offset},
        ... }]
        """
        # IMPLEMENT THIS TO EXTRACT YOUR ANNOTATIONS
        raise NotImplementedError(
            "Implement _extract_annotations() to extract behavior events. "
            "See docstring for examples."
        )

    # ============================================================
    # HELPER METHODS - Usually you don't need to modify these
    # ============================================================

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

    def _convert_to_frame_labels(self,
                                 annotations: List[Tuple],
                                 label_map: dict,
                                 fps: float,
                                 params: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert time-based annotations to frame-by-frame labels.

        Usually you DON'T need to modify this - it handles the standard
        conversion from (start, stop, behavior) tuples to per-frame labels.

        Parameters
        ----------
        annotations : List[Tuple]
            List of (start_time, stop_time, behavior_name) tuples
        label_map : dict
            Behavior name to ID mapping
        fps : float
            Frames per second
        params : dict
            Converter parameters

        Returns
        -------
        labels : np.ndarray
            Per-frame behavior IDs, shape (n_frames,)
        frames : np.ndarray
            Frame indices, shape (n_frames,)
        """
        background_id = label_map[params["background_label"]]

        # Determine total duration
        if not annotations:
            # No annotations - create minimal array
            labels = np.array([background_id], dtype=int)
            frames = np.array([0], dtype=np.int32)
            return labels, frames

        max_time = max(stop for start, stop, behavior in annotations)
        n_frames = int(np.ceil(max_time * fps)) + 1

        # Initialize with background
        labels = np.full(n_frames, background_id, dtype=int)

        # Fill in behaviors
        for start_time, stop_time, behavior_name in annotations:
            behavior_id = label_map.get(behavior_name, background_id)

            start_frame = int(start_time * fps)
            stop_frame = int(stop_time * fps)

            # Check if point event
            is_point = abs(start_time - stop_time) < 1e-6

            if is_point:
                # Mark single frame
                if 0 <= start_frame < n_frames:
                    labels[start_frame] = behavior_id
            else:
                # Fill range
                start_frame = max(0, start_frame)
                stop_frame = min(n_frames - 1, stop_frame)
                labels[start_frame:stop_frame + 1] = behavior_id

        frames = np.arange(n_frames, dtype=np.int32)
        return labels, frames

    def get_metadata(self) -> dict:
        """
        Optional: Return format-specific metadata.

        This metadata gets added to dataset.meta['labels'][kind].
        """
        return {}


# ============================================================
# EXAMPLE: Modified BORIS with Time Offset and Animal IDs
# ============================================================

class ModifiedBorisConverter(CustomLabelConverter):
    """
    Example: BORIS annotations from video snippets.

    Handles:
    - Time offset (snippet start time in full video)
    - Animal ID injection (not in BORIS file)
    - Standard BORIS CSV format

    Usage:
    >>> dataset.convert_all_labels(
    ...     source_format="modified_boris",
    ...     time_offset=120.5,      # Snippet starts at 2:00.5 in full video
    ...     animal_id="mouse_A12",  # Not in BORIS file
    ...     fps=30.0,
    ... )
    """

    src_format = "modified_boris"
    label_kind = "behavior"
    label_format = "modified_boris_v1"

    _defaults = dict(
        group_from="filename",
        fps=30.0,
        background_label="none",
        time_offset=0.0,          # Add to all timestamps
        animal_id="unknown",      # Inject animal ID
    )

    def _load_source_file(self, src_path: Path) -> pd.DataFrame:
        """Load BORIS CSV export."""
        return pd.read_csv(src_path)

    def _build_label_map(self, data: pd.DataFrame, params: dict) -> dict:
        """Build label map from Behavior column."""
        behaviors = sorted(data["Behavior"].dropna().unique())
        if params["background_label"] not in behaviors:
            behaviors = [params["background_label"]] + behaviors
        return {i: name for i, name in enumerate(behaviors)}

    def _extract_annotations(self,
                            data: pd.DataFrame,
                            src_path: Path,
                            raw_row: pd.Series,
                            params: dict) -> List[dict]:
        """Extract annotations with time offset and animal ID."""
        time_offset = params["time_offset"]
        animal_id = params["animal_id"]

        # Extract annotations and apply time offset
        annotations = []
        for _, row in data.iterrows():
            start = row["Time"] + time_offset
            # Check if this is a state event (has Status column)
            if "Status" in data.columns and pd.notna(row.get("Status")):
                # Skip for now, we'll pair START/STOP later
                continue
            else:
                # Point event or duration in separate column
                stop = row.get("Stop", row["Time"]) + time_offset
                annotations.append((start, stop, row["Behavior"]))

        # If we have START/STOP events, pair them
        if "Status" in data.columns:
            active = {}
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

        # Create sequence with animal ID
        return [{
            "sequence_name": f"snippet_{animal_id}",
            "annotations": annotations,
            "fps": params["fps"],
            "metadata": {
                "animal_id": animal_id,
                "time_offset": time_offset,
                "source_file": src_path.name,
            },
        }]


# ============================================================
# To use this converter, add to label_library/__init__.py:
#
# from . import custom_label_template
# custom_label_template.ModifiedBorisConverter = register_label_converter(
#     custom_label_template.ModifiedBorisConverter
# )
# ============================================================
