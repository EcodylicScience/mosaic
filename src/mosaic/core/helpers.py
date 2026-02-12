from urllib.parse import quote, unquote
from typing import Optional, Tuple
import numpy as np
import pandas as pd

def to_safe_name(s: str) -> str:
    # encode EVERYTHING that could be problematic on any OS
    return quote(s, safe="")   # e.g. "task1/train/m1" -> "task1%2Ftrain%2Fm1"

def from_safe_name(safe: str) -> str:
    return unquote(safe)


# =============================================================================
# Label Format Helpers
# =============================================================================

def detect_label_format(npz_data: dict) -> str:
    """
    Detect the label format from an NPZ file's contents.

    Parameters
    ----------
    npz_data : dict or np.lib.npyio.NpzFile
        Loaded NPZ data (from np.load())

    Returns
    -------
    str
        One of: "individual_pair_v1", "dense", "unknown"

    Examples
    --------
    >>> with np.load("labels.npz", allow_pickle=True) as npz:
    ...     fmt = detect_label_format(npz)
    """
    # Check for explicit label_format key
    if "label_format" in npz_data.files if hasattr(npz_data, 'files') else "label_format" in npz_data:
        fmt = str(npz_data["label_format"])
        if fmt:
            return fmt

    # Heuristic detection based on keys present
    keys = set(npz_data.files if hasattr(npz_data, 'files') else npz_data.keys())

    # individual_pair_v1: has frames, labels, individual_ids arrays
    if {"frames", "labels", "individual_ids"}.issubset(keys):
        return "individual_pair_v1"

    # Dense format: just has labels array (and it's likely 1D with length = n_frames)
    if "labels" in keys:
        labels = np.asarray(npz_data["labels"])
        # If labels is 1D and there's no frames array, assume dense
        if labels.ndim == 1 and "frames" not in keys:
            return "dense"

    return "unknown"


def expand_labels_to_dense(
    frames: np.ndarray,
    labels: np.ndarray,
    individual_ids: Optional[np.ndarray] = None,
    n_frames: Optional[int] = None,
    default_label: int = 0,
    individual_filter: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Expand sparse event-based labels to a dense per-frame array.

    Converts from individual_pair_v1 format (sparse events) to a dense array
    where labels[i] is the label at frame i.

    Parameters
    ----------
    frames : np.ndarray
        1D array of frame indices for each event, shape (n_events,)
    labels : np.ndarray
        1D array of label IDs for each event, shape (n_events,)
    individual_ids : np.ndarray, optional
        2D array of [id1, id2] for each event, shape (n_events, 2).
        If provided with individual_filter, only events matching the filter
        are included.
    n_frames : int, optional
        Total number of frames in the dense output. If None, uses max(frames) + 1.
    default_label : int, default=0
        Label value for frames without events (typically 0 = "none"/"background")
    individual_filter : tuple of (int, int), optional
        If provided, only include events where individual_ids matches this pair.
        For symmetric behaviors, you may want to filter for a specific direction.
        Use (-1, -1) for scene-level labels, (id, -1) for individual labels.

    Returns
    -------
    np.ndarray
        Dense 1D array of shape (n_frames,) where output[i] is the label at frame i.
        If multiple events occur at the same frame, the last one wins.

    Examples
    --------
    >>> frames = np.array([10, 11, 12, 50, 51])
    >>> labels = np.array([1, 1, 1, 2, 2])
    >>> dense = expand_labels_to_dense(frames, labels, n_frames=100)
    >>> dense[10:13]  # [1, 1, 1]
    >>> dense[0]      # 0 (default)

    With individual filtering:
    >>> individual_ids = np.array([[0, 1], [0, 1], [0, 1], [1, 0], [1, 0]])
    >>> dense_01 = expand_labels_to_dense(frames, labels, individual_ids,
    ...                                    individual_filter=(0, 1))
    >>> # Only includes events where individual_ids == [0, 1]
    """
    frames = np.asarray(frames, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()

    if frames.shape[0] != labels.shape[0]:
        raise ValueError(f"frames and labels must have same length, got {frames.shape[0]} vs {labels.shape[0]}")

    if frames.shape[0] == 0:
        return np.full(n_frames or 1, default_label, dtype=np.int64)

    # Apply individual filter if specified
    if individual_filter is not None and individual_ids is not None:
        individual_ids = np.asarray(individual_ids)
        if individual_ids.ndim == 1:
            individual_ids = individual_ids.reshape(-1, 2)

        id1, id2 = individual_filter
        mask = (individual_ids[:, 0] == id1) & (individual_ids[:, 1] == id2)
        frames = frames[mask]
        labels = labels[mask]

    # Determine output size
    if n_frames is None:
        n_frames = int(frames.max()) + 1 if frames.size > 0 else 1

    # Create dense array with default label
    dense = np.full(n_frames, default_label, dtype=np.int64)

    # Fill in labeled frames (last event wins if duplicates)
    valid_mask = (frames >= 0) & (frames < n_frames)
    dense[frames[valid_mask]] = labels[valid_mask]

    return dense


def load_labels_auto(
    path,
    n_frames: Optional[int] = None,
    default_label: int = 0,
    individual_filter: Optional[Tuple[int, int]] = None,
    return_format: str = "dense",
) -> np.ndarray:
    """
    Load labels from NPZ file, auto-detecting format and converting as needed.

    Supports both dense (legacy) and individual_pair_v1 (sparse) formats.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ label file
    n_frames : int, optional
        For sparse formats, the total number of frames to expand to.
        If None, uses max(frames) + 1 from the file.
    default_label : int, default=0
        Label for unlabeled frames when expanding sparse to dense
    individual_filter : tuple of (int, int), optional
        For individual_pair_v1 format, filter to specific individual pair
    return_format : str, default="dense"
        Output format: "dense" returns per-frame array, "sparse" returns
        (frames, labels, individual_ids) tuple for individual_pair_v1

    Returns
    -------
    np.ndarray or tuple
        If return_format="dense": 1D array of shape (n_frames,)
        If return_format="sparse": tuple of (frames, labels, individual_ids)

    Examples
    --------
    >>> labels = load_labels_auto("behavior/hex_03.npz")
    >>> labels.shape  # (n_frames,)

    >>> frames, labels, ids = load_labels_auto("behavior/hex_03.npz",
    ...                                         return_format="sparse")
    """
    import numpy as np
    from pathlib import Path

    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        fmt = detect_label_format(npz)

        if fmt == "individual_pair_v1":
            frames = np.asarray(npz["frames"], dtype=np.int64).ravel()
            labels = np.asarray(npz["labels"], dtype=np.int64).ravel()
            individual_ids = np.asarray(npz["individual_ids"])
            if individual_ids.ndim == 1:
                individual_ids = individual_ids.reshape(-1, 2)

            if return_format == "sparse":
                return frames, labels, individual_ids

            # Expand to dense
            return expand_labels_to_dense(
                frames, labels, individual_ids,
                n_frames=n_frames,
                default_label=default_label,
                individual_filter=individual_filter,
            )

        elif fmt == "dense" or "labels" in (npz.files if hasattr(npz, 'files') else npz):
            labels = np.asarray(npz["labels"], dtype=np.int64).ravel()

            if return_format == "sparse":
                # Convert dense to sparse format
                frames = np.arange(len(labels), dtype=np.int64)
                individual_ids = np.full((len(labels), 2), -1, dtype=np.int64)
                return frames, labels, individual_ids

            return labels

        else:
            raise ValueError(f"Cannot load labels from {path}: unknown format '{fmt}'")


def load_labels_for_feature_frames(
    path,
    feature_frames: np.ndarray,
    default_label: int = 0,
    deduplicate_symmetric: bool = True,
    individual_filter: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load labels from NPZ file and align to specific feature frame indices.

    This is the key function for aligning sparse event-based labels (like
    individual_pair_v1 format) with row-indexed feature data. Rather than
    expanding to a full dense array, it looks up the label for each
    specific frame in feature_frames.

    Parameters
    ----------
    path : str or Path
        Path to the NPZ label file
    feature_frames : np.ndarray
        1D array of frame indices from the feature data. Each element
        specifies which video frame that feature row corresponds to.
        The output will have one label per element in feature_frames.
    default_label : int, default=0
        Label for frames that don't have labeled events (typically 0 = "none")
    deduplicate_symmetric : bool, default=True
        For individual_pair_v1 format with symmetric storage (both [i,j] and
        [j,i] stored), deduplicate by keeping only id1 <= id2 events.
        Ignored when individual_filter is set (filtering is more specific).
    individual_filter : tuple of (int, int), optional
        For individual_pair_v1 format, only include events matching this
        specific (id1, id2) pair. When set, deduplicate_symmetric is skipped
        since the filter is already pair-specific.

    Returns
    -------
    np.ndarray
        1D array of labels with shape (len(feature_frames),).
        labels[i] is the label for frame feature_frames[i].

    Examples
    --------
    >>> # Feature data has 1000 rows covering frames 5000-6000
    >>> feature_frames = np.array([5000, 5001, 5002, ...])  # from parquet
    >>> labels = load_labels_for_feature_frames("behavior.npz", feature_frames)
    >>> labels.shape  # (1000,) - one label per feature row

    Notes
    -----
    This function solves the frame coordinate alignment problem that occurs
    when:
    - Behavior labels are stored with original video frame indices (e.g., 15002-65927)
    - Feature data is row-indexed (0, 1, 2, ...) but each row corresponds to
      a specific video frame stored in a 'frame' column
    - The feature frame range may not fully overlap with labeled frames

    For frames without labeled events, default_label is returned.
    For frames with multiple labeled events, the last one wins.
    """
    from pathlib import Path

    path = Path(path)
    feature_frames = np.asarray(feature_frames, dtype=np.int64).ravel()

    with np.load(path, allow_pickle=True) as npz:
        fmt = detect_label_format(npz)

        if fmt == "individual_pair_v1":
            frames = np.asarray(npz["frames"], dtype=np.int64).ravel()
            labels = np.asarray(npz["labels"], dtype=np.int64).ravel()
            individual_ids = np.asarray(npz["individual_ids"])
            if individual_ids.ndim == 1:
                individual_ids = individual_ids.reshape(-1, 2)

            if individual_filter is not None:
                # Filter to specific pair — check both orderings for symmetric labels
                id1, id2 = individual_filter
                mask_fwd = (individual_ids[:, 0] == id1) & (individual_ids[:, 1] == id2)
                mask_rev = (individual_ids[:, 0] == id2) & (individual_ids[:, 1] == id1)
                mask = mask_fwd | mask_rev
                frames = frames[mask]
                labels = labels[mask]
            elif deduplicate_symmetric:
                # Deduplicate symmetric pairs if requested
                mask = individual_ids[:, 0] <= individual_ids[:, 1]
                frames = frames[mask]
                labels = labels[mask]

            # Build frame -> label mapping (last event wins if multiple per frame)
            frame_to_label = dict(zip(frames, labels))

            # Look up labels for each feature frame
            result = np.array(
                [frame_to_label.get(f, default_label) for f in feature_frames],
                dtype=np.int64,
            )
            return result

        elif "labels" in (npz.files if hasattr(npz, 'files') else npz):
            # Dense format - direct indexing
            dense = np.asarray(npz["labels"], dtype=np.int64).ravel()

            # Handle out-of-bounds frames with default label
            result = np.full(len(feature_frames), default_label, dtype=np.int64)
            valid_mask = (feature_frames >= 0) & (feature_frames < len(dense))
            result[valid_mask] = dense[feature_frames[valid_mask]]
            return result

        else:
            raise ValueError(f"Cannot load labels from {path}: unknown format '{fmt}'")


def chunk_sequence(df: pd.DataFrame,
                   time_chunk_sec: float | None = None,
                   frame_chunk: int | None = None):
    """
    Yield (chunk_id, df_chunk, meta) from a per-sequence DataFrame.
    If time_chunk_sec is provided and 'time' exists, chunk by time.
    Else if frame_chunk is provided and 'frame' exists, chunk by frame.
    Else yield the whole sequence as a single chunk.
    meta contains start/end frame/time if available.
    """
    frame_key = "frame" if "frame" in df.columns else None
    time_key = "time" if "time" in df.columns else None

    if time_chunk_sec and time_key in df.columns:
        starts = np.arange(df[time_key].min(), df[time_key].max() + time_chunk_sec, time_chunk_sec)
        for idx, start in enumerate(starts):
            end = start + time_chunk_sec
            mask = (df[time_key] >= start) & (df[time_key] < end)
            sub = df[mask]
            if sub.empty:
                continue
            yield idx, sub, {
                "start_time": float(start),
                "end_time": float(end),
                "start_frame": int(sub[frame_key].iloc[0]) if frame_key else None,
                "end_frame": int(sub[frame_key].iloc[-1]) if frame_key else None,
            }
    elif frame_chunk and frame_key in df.columns:
        frames = df[frame_key].to_numpy()
        start_frame = frames.min()
        end_frame = frames.max()
        for idx, start in enumerate(range(start_frame, end_frame + 1, int(frame_chunk))):
            end = start + int(frame_chunk)
            mask = (df[frame_key] >= start) & (df[frame_key] < end)
            sub = df[mask]
            if sub.empty:
                continue
            yield idx, sub, {
                "start_frame": int(start),
                "end_frame": int(end),
                "start_time": float(sub[time_key].iloc[0]) if time_key else None,
                "end_time": float(sub[time_key].iloc[-1]) if time_key else None,
            }
    else:
        meta = {}
        if frame_key:
            meta["start_frame"] = int(df[frame_key].iloc[0])
            meta["end_frame"] = int(df[frame_key].iloc[-1])
        if time_key:
            meta["start_time"] = float(df[time_key].iloc[0])
            meta["end_time"] = float(df[time_key].iloc[-1])
        yield 0, df, meta


# =============================================================================
# Time/Frame Range Filtering
# =============================================================================

def filter_time_range(
    df: pd.DataFrame,
    filter_start_frame: Optional[int] = None,
    filter_end_frame: Optional[int] = None,
    filter_start_time: Optional[float] = None,
    filter_end_time: Optional[float] = None,
    frame_col: str = "frame",
    time_col: str = "time",
) -> pd.DataFrame:
    """
    Filter DataFrame to a time/frame range.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with frame and/or time columns
    filter_start_frame : int, optional
        Discard frames < this value
    filter_end_frame : int, optional
        Discard frames >= this value
    filter_start_time : float, optional
        Discard rows where time < this value (seconds)
    filter_end_time : float, optional
        Discard rows where time >= this value (seconds)
    frame_col : str, default "frame"
        Name of the frame column
    time_col : str, default "time"
        Name of the time column

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with index reset
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    mask = pd.Series(True, index=df.index)

    if filter_start_frame is not None and frame_col in df.columns:
        mask &= df[frame_col] >= filter_start_frame
    if filter_end_frame is not None and frame_col in df.columns:
        mask &= df[frame_col] < filter_end_frame
    if filter_start_time is not None and time_col in df.columns:
        mask &= df[time_col] >= filter_start_time
    if filter_end_time is not None and time_col in df.columns:
        mask &= df[time_col] < filter_end_time

    return df.loc[mask].reset_index(drop=True)


# =============================================================================
# Hierarchical Naming Helpers
# =============================================================================

def parse_compound_name(name: str, separator: str = "__") -> list[str]:
    """
    Split a compound hierarchical name into its components.

    Supports arbitrary depths (2, 3, 4+ levels).

    Parameters
    ----------
    name : str
        Compound name like "fish_01__speed_3__loop_1"
    separator : str, default "__"
        The separator between hierarchy levels

    Returns
    -------
    list[str]
        List of components, e.g. ["fish_01", "speed_3", "loop_1"]

    Examples
    --------
    >>> parse_compound_name("fish_01__speed_3__loop_1")
    ['fish_01', 'speed_3', 'loop_1']

    >>> parse_compound_name("arena_1__day_015__hour_14")
    ['arena_1', 'day_015', 'hour_14']

    >>> parse_compound_name("simple_name")
    ['simple_name']
    """
    if not name:
        return []
    return name.split(separator)


def build_compound_name(*parts: str, separator: str = "__") -> str:
    """
    Join hierarchy components into a compound name.

    Supports any number of parts.

    Parameters
    ----------
    *parts : str
        Hierarchy components to join, e.g. "fish_01", "speed_3", "loop_1"
    separator : str, default "__"
        The separator between hierarchy levels

    Returns
    -------
    str
        Compound name, e.g. "fish_01__speed_3__loop_1"

    Examples
    --------
    >>> build_compound_name("fish_01", "speed_3", "loop_1")
    'fish_01__speed_3__loop_1'

    >>> build_compound_name("arena_1", "day_015", "hour_14")
    'arena_1__day_015__hour_14'

    >>> build_compound_name("single")
    'single'
    """
    # Filter out None and empty strings
    valid_parts = [p for p in parts if p]
    return separator.join(valid_parts)


def parse_hierarchy(
    group: str,
    sequence: str,
    level_names: list[str],
    separator: str = "__",
) -> dict[str, str | None]:
    """
    Parse group and sequence into named hierarchy levels.

    The full hierarchy is constructed by concatenating group and sequence
    components, then mapping them to the provided level names.

    Parameters
    ----------
    group : str
        The group name (may be compound, e.g. "experiment_A__arena_1")
    sequence : str
        The sequence name (may be compound, e.g. "day_015__hour_14")
    level_names : list[str]
        Names for each hierarchy level, e.g. ["experiment", "arena", "day", "hour"]
    separator : str, default "__"
        The separator between hierarchy levels

    Returns
    -------
    dict[str, str | None]
        Dictionary mapping level names to values. Missing levels are None.

    Examples
    --------
    >>> parse_hierarchy("fish_01", "speed_3__loop_1",
    ...                 level_names=["fish", "speed", "loop"])
    {'fish': 'fish_01', 'speed': 'speed_3', 'loop': 'loop_1'}

    >>> parse_hierarchy("experiment_A__arena_1", "day_015__hour_14",
    ...                 level_names=["experiment", "arena", "day", "hour"])
    {'experiment': 'experiment_A', 'arena': 'arena_1', 'day': 'day_015', 'hour': 'hour_14'}

    >>> # Handles fewer parts than names (missing levels are None)
    >>> parse_hierarchy("fish_01", "loop_1", level_names=["fish", "speed", "loop"])
    {'fish': 'fish_01', 'speed': 'loop_1', 'loop': None}
    """
    # Combine group and sequence parts
    group_parts = parse_compound_name(group, separator) if group else []
    seq_parts = parse_compound_name(sequence, separator) if sequence else []
    all_parts = group_parts + seq_parts

    # Map to level names
    result = {}
    for i, name in enumerate(level_names):
        result[name] = all_parts[i] if i < len(all_parts) else None

    return result
