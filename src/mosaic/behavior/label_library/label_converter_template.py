"""
Template for creating new label converters.

To create a new label converter:
1. Copy this file to a new name (e.g., boris_behavior.py)
2. Update the class name and registration attributes
3. Implement the convert() method with your format-specific logic
4. Import in label_library/__init__.py to register it
5. Test with dataset.convert_all_labels(source_format="your_format")

The converter will be automatically registered and available for use.
"""

from pathlib import Path
from typing import Optional, Dict, Any
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


class MyLabelConverter:
    """
    Template for a label converter.

    This converter reads labels from [describe source format] and converts
    them to per-sequence npz files in the behavior dataset format.

    Usage
    -----
    After implementing this converter and importing it in __init__.py:

    >>> dataset.convert_all_labels(
    ...     kind="behavior",
    ...     source_format="my_format",
    ...     # Add any format-specific parameters here
    ... )
    """

    # ============ REGISTRATION ATTRIBUTES (REQUIRED) ============
    # These MUST be class attributes for registration to work

    src_format = "my_format"           # Must match tracks_raw/index.csv src_format column
    label_kind = "behavior"            # e.g., "behavior", "id_tags", "poses"
    label_format = "my_format_v1"      # Unique identifier for this format version

    # ============ DEFAULT PARAMETERS ============

    _defaults = dict(
        group_from="filename",          # How to determine group name: 'filename', 'infile', or 'both'
        # Add format-specific parameters here:
        # fps=30.0,
        # delimiter=",",
        # time_col="Time",
        # etc.
    )

    def __init__(self, params: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the converter with parameters.

        Parameters
        ----------
        params : dict, optional
            Configuration parameters (merged with _defaults)
        **kwargs : additional keyword arguments
            Can override params values
        """
        self.params = _merge_params(params, self._defaults)
        self.params.update(kwargs)

    # ============ MAIN CONVERSION METHOD (REQUIRED) ============

    def convert(self,
                src_path: Path,
                raw_row: pd.Series,
                labels_root: Path,
                params: dict,
                overwrite: bool,
                existing_pairs: set[tuple[str, str]]) -> list[dict]:
        """
        Convert one source file to per-sequence label npz files.

        This is the main method that does the conversion work. Implement your
        format-specific logic here.

        Parameters
        ----------
        src_path : Path
            Path to source label file
        raw_row : pd.Series
            Row from tracks_raw/index.csv with metadata (group, md5, etc.)
        labels_root : Path
            Output directory (e.g., dataset_root/labels/behavior/)
        params : dict
            Conversion parameters (combined with self.params)
        overwrite : bool
            Whether to overwrite existing files
        existing_pairs : set[tuple[str, str]]
            Set of (group, sequence) pairs already converted

        Returns
        -------
        list[dict]
            List of index row dicts to append to labels/index.csv

        Notes
        -----
        Each index row dict should contain:
        - kind: str (e.g., "behavior")
        - label_format: str (e.g., "my_format_v1")
        - group: str
        - sequence: str
        - group_safe: str (safe filename version of group)
        - sequence_safe: str (safe filename version of sequence)
        - abs_path: str (path to output npz file)
        - source_abs_path: str (path to input file)
        - source_md5: str (MD5 hash of source file)
        - n_frames: int (number of frames in this sequence)
        - label_ids: str (comma-separated label IDs, e.g., "0,1,2,3")
        - label_names: str (comma-separated label names, e.g., "attack,investigate,mount,other")
        - (any additional metadata fields your format needs)

        Each npz file should contain:
        - group: str
        - sequence: str
        - sequence_key: str (usually same as sequence)
        - frames: np.ndarray (shape=(T,), dtype=int32) - frame indices
        - labels: np.ndarray (shape=(T,), dtype=int) - per-frame label IDs
        - label_ids: np.ndarray (dtype=int) - array of valid label IDs
        - label_names: np.ndarray (dtype=object) - array of label names
        - (any additional metadata your format needs)
        """

        # -------- STEP 1: Load the source file --------
        data = self._load_source_file(src_path)

        # -------- STEP 2: Extract label mapping (if applicable) --------
        label_map = self._get_label_map(data)  # e.g., {0: "class_a", 1: "class_b"}

        # -------- STEP 3: Process each sequence --------
        rows_out: list[dict] = []

        # Example structure (adapt to your format):
        # Structure could be:
        # - Single sequence per file
        # - Multiple sequences in nested structure
        # - Table format with sequence column

        # Example for nested structure:
        for group_name, sequences in self._extract_sequences(data).items():
            for seq_id, seq_data in sequences.items():
                # Extract labels array
                labels = self._extract_labels(seq_data)
                frames = np.arange(len(labels), dtype=np.int32)

                # Determine output group/sequence names
                group_val = self._determine_group(group_name, raw_row)
                seq_val = str(seq_id)

                # Skip if already exists
                pair = (group_val, seq_val)
                if not overwrite and pair in existing_pairs and self._output_exists(labels_root, group_val, seq_val):
                    continue

                # Create safe filenames
                safe_group = to_safe_name(group_val) if group_val else ""
                safe_seq = to_safe_name(seq_val)
                fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
                out_path = labels_root / fname

                # Build npz payload
                payload = self._build_npz_payload(
                    group_val, seq_val, frames, labels, label_map
                )

                # Save npz
                np.savez_compressed(out_path, **payload)
                existing_pairs.add(pair)

                # Build index row
                index_row = self._build_index_row(
                    group_val, seq_val, safe_group, safe_seq,
                    out_path, src_path, raw_row, labels, label_map
                )
                rows_out.append(index_row)

        return rows_out

    # ============ OPTIONAL METADATA METHOD ============

    def get_metadata(self) -> dict:
        """
        Optional: return format-specific metadata for dataset.meta['labels'][kind].

        This metadata will be merged into dataset.meta['labels']['behavior'] (or other kind).
        Useful for storing label mappings, format version info, etc.

        Returns
        -------
        dict
            Metadata to merge into dataset.meta['labels'][kind]
            Common keys: label_ids, label_names, format_version, etc.
        """
        return {}

    # ============ HELPER METHODS (CUSTOMIZE THESE) ============

    def _load_source_file(self, src_path: Path) -> Any:
        """
        Load and parse the source label file.

        Parameters
        ----------
        src_path : Path
            Path to source file

        Returns
        -------
        Any
            Parsed data structure (format-dependent)

        Examples
        --------
        For CSV:
        >>> return pd.read_csv(src_path)

        For JSON:
        >>> import json
        >>> with open(src_path) as f:
        ...     return json.load(f)

        For NPY:
        >>> return np.load(src_path, allow_pickle=True)
        """
        raise NotImplementedError("Implement _load_source_file for your format")

    def _get_label_map(self, data: Any) -> dict[int, str]:
        """
        Extract or define the label mapping.

        Parameters
        ----------
        data : Any
            Loaded data from _load_source_file

        Returns
        -------
        dict[int, str]
            Mapping from label ID to label name
            Example: {0: "attack", 1: "investigation", 2: "mount"}
        """
        raise NotImplementedError("Implement _get_label_map for your format")

    def _extract_sequences(self, data: Any) -> dict:
        """
        Extract sequences from the loaded data.

        Parameters
        ----------
        data : Any
            Loaded data from _load_source_file

        Returns
        -------
        dict
            Nested structure: {group_name: {seq_id: seq_data, ...}, ...}
            Or single level: {"default_group": {seq_id: seq_data, ...}}
        """
        raise NotImplementedError("Implement _extract_sequences for your format")

    def _extract_labels(self, seq_data: Any) -> np.ndarray:
        """
        Extract labels array from sequence data.

        Parameters
        ----------
        seq_data : Any
            Data for a single sequence

        Returns
        -------
        np.ndarray
            Labels array, shape=(T,), dtype=int
            Each element is a label ID corresponding to the label_map
        """
        raise NotImplementedError("Implement _extract_labels for your format")

    def _determine_group(self, source_group: str, raw_row: pd.Series) -> str:
        """
        Determine output group name based on group_from parameter.

        Parameters
        ----------
        source_group : str
            Group name from inside the source file
        raw_row : pd.Series
            Row from tracks_raw/index.csv

        Returns
        -------
        str
            Group name to use for output
        """
        group_from = self.params.get("group_from", "filename")
        if group_from == "filename":
            return str(raw_row.get("group", "") or "")
        elif group_from == "infile":
            return source_group
        elif group_from == "both":
            return str(raw_row.get("group", "") or "")
        else:
            return source_group

    def _output_exists(self, labels_root: Path, group: str, sequence: str) -> bool:
        """Check if output file already exists."""
        safe_group = to_safe_name(group) if group else ""
        safe_seq = to_safe_name(sequence)
        fname = f"{safe_group + '__' if safe_group else ''}{safe_seq}.npz"
        return (labels_root / fname).exists()

    def _build_npz_payload(self,
                          group: str,
                          sequence: str,
                          frames: np.ndarray,
                          labels: np.ndarray,
                          label_map: dict) -> dict:
        """
        Build the npz file payload.

        Standard keys (required):
        - group: str
        - sequence: str
        - sequence_key: str (usually same as sequence)
        - frames: np.ndarray (int32, shape=(T,))
        - labels: np.ndarray (int, shape=(T,))
        - label_ids: np.ndarray (int)
        - label_names: np.ndarray (object)

        Parameters
        ----------
        group : str
            Group name
        sequence : str
            Sequence name
        frames : np.ndarray
            Frame indices
        labels : np.ndarray
            Per-frame labels
        label_map : dict
            Label ID to name mapping

        Returns
        -------
        dict
            Payload for np.savez_compressed
        """
        label_ids = np.array(list(label_map.keys()), dtype=int)
        label_names = np.array(list(label_map.values()), dtype=object)

        return {
            "group": group,
            "sequence": sequence,
            "sequence_key": sequence,
            "frames": frames,
            "labels": labels,
            "label_ids": label_ids,
            "label_names": label_names,
        }

    def _build_index_row(self,
                        group: str,
                        sequence: str,
                        group_safe: str,
                        sequence_safe: str,
                        out_path: Path,
                        src_path: Path,
                        raw_row: pd.Series,
                        labels: np.ndarray,
                        label_map: dict) -> dict:
        """
        Build an index row dict for labels/index.csv.

        Parameters
        ----------
        group : str
            Group name
        sequence : str
            Sequence name
        group_safe : str
            Safe filename version of group
        sequence_safe : str
            Safe filename version of sequence
        out_path : Path
            Path to output npz file
        src_path : Path
            Path to source file
        raw_row : pd.Series
            Row from tracks_raw/index.csv
        labels : np.ndarray
            Labels array
        label_map : dict
            Label ID to name mapping

        Returns
        -------
        dict
            Index row for labels/index.csv
        """
        return {
            "kind": self.label_kind,
            "label_format": self.label_format,
            "group": group,
            "sequence": sequence,
            "group_safe": group_safe,
            "sequence_safe": sequence_safe,
            "abs_path": str(out_path.resolve()),
            "source_abs_path": str(src_path.resolve()),
            "source_md5": raw_row.get("md5", ""),
            "n_frames": int(labels.shape[0]),
            "label_ids": ",".join(map(str, label_map.keys())),
            "label_names": ",".join(label_map.values()),
        }
