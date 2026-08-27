"""Template for creating new label converters.

To create a new label converter:

1. Copy this file to a new name (e.g. ``boris_behavior.py``).
2. Rename the class and its ``Params`` model, and set the class variables
   (``src_format`` / ``label_kind`` / ``label_format`` / ``version`` / ``Params``).
3. Add any format-specific fields to the ``Params`` model.
4. Implement ``convert()`` -- it returns one :class:`LabelEntry` per sequence and
   never touches the filesystem; the ``Dataset`` writes the ``.npz`` files and the
   typed index row.
5. Register it in ``label_library/__init__.py`` with
   :func:`~mosaic.core.label_converter.register_label_converter`. Before
   indexing, not after: ``index_labels_raw`` refuses a ``src_format`` no
   registered converter claims, because an index written under one is an index
   whose rows every later conversion skips without saying so.

Usage from a ``Dataset`` (unchanged by the new contract):

    >>> from mosaic.core import open_dataset
    >>> dataset = open_dataset("/path/to/dataset")
    >>> # Index the raw annotation files into labels_raw/index.csv. This is the
    >>> # label sibling of index_tracks_raw -- raw labels no longer share the
    >>> # tracks_raw root.
    >>> dataset.index_labels_raw(
    ...     ["/path/to/annotations"],
    ...     patterns="*.csv",
    ...     src_format="my_format",  # Must match this converter's src_format.
    ... )
    >>> # Convert. The Dataset reads labels_raw, dispatches to this converter,
    >>> # and writes labels/<kind>/<run_id>/*.npz plus the typed index row.
    >>> dataset.convert_all_labels(
    ...     kind="behavior",
    ...     source_format="my_format",
    ...     # Any Params field can be passed as a keyword here:
    ...     default_fps=30.0,
    ... )
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated

import numpy as np

from mosaic.core.label_converter import (
    LabelConvertParams,
    LabelConverter,
    LabelEntry,
)
from mosaic.core.params import (
    HASH_EXCLUDE,
    NEEDS_DESCRIPTION,
    Declared,
)

_DEFAULT_FPS_DESCRIPTION = "Frame rate recorded in each sequence's payload."


class MyLabelParams(LabelConvertParams):
    """Parameters for :class:`MyLabelConverter`.

    ``group_from`` and ``strict_schema`` are already declared on
    :class:`~mosaic.core.label_converter.LabelConvertParams` -- do not redeclare
    them. Add only what this format needs.

    A field that changes the labels is a plain (hashed) field: it is part of the
    label variant identity, so two conversions that differ in it mint two
    variants. A validation-only or throughput-only knob is tagged
    ``HASH_EXCLUDE`` so retuning it never busts the cache, even though it still
    appears in ``params.json`` and reaches the converter.
    """

    # A parameter that changes the labels -- hashed. Written into the payload,
    # so it is part of the output and belongs in the identity.
    default_fps: Annotated[float, Declared(_DEFAULT_FPS_DESCRIPTION)] = 30.0

    # A knob that does not change the labels -- excluded from the identity.
    verbose: Annotated[bool, HASH_EXCLUDE, Declared(NEEDS_DESCRIPTION)] = False


class MyLabelConverter(LabelConverter[MyLabelParams]):
    """Template for a label converter.

    Reads labels from [describe your source format] and returns one
    :class:`~mosaic.core.label_converter.LabelEntry` per sequence. It returns
    *data*, never files: the ``Dataset`` writes each entry's ``payload`` into
    ``labels/<kind>/<run_id>/`` and records the typed index row, so this class
    must not compute output paths, write ``.npz`` files, or handle overwrite.

    Class variables:
        src_format: Matched against the ``labels_raw/index.csv`` ``src_format``
            column, and passed as ``source_format`` to ``convert_all_labels``.
        label_kind: The kind produced (``"behavior"``, ``"id_tags"``, ...), i.e.
            the ``labels/<kind>/`` subdirectory.
        label_format: The on-disk payload format a reader dispatches on.
        version: Declared compatibility version, a visible segment of the label
            variant identity. Bump it by hand when the output semantics change.
        Params: The typed parameter model above.
    """

    src_format = "my_format"
    label_kind = "behavior"
    label_format = "my_format_v1"
    version = "0.1"
    Params = MyLabelParams

    def convert(
        self,
        src_path: Path,
        params: MyLabelParams,
        raw_row: Mapping[str, object],
    ) -> list[LabelEntry]:
        """Read one source file into one :class:`LabelEntry` per sequence.

        Args:
            src_path: The raw label file to read.
            params: Typed conversion parameters. These, plus ``src_format``,
                ``label_kind`` and ``version``, are what the label variant
                identity is made of.
            raw_row: The ``labels_raw/index.csv`` row, carrying the group hint
                and the source ``md5``. Never hashed.

        Returns:
            One :class:`LabelEntry` per sequence found in the file. Each entry's
            ``payload`` is the exact dict of arrays and scalars a reader loads
            from the ``.npz`` (the dict the old contract passed to
            ``np.savez_compressed``).
        """
        # -------- STEP 1: Load the source file. --------
        data = self._load_source_file(src_path)

        # -------- STEP 2: Extract the label vocabulary. --------
        # e.g. {0: "class_a", 1: "class_b"}. Recorded on the index row so a
        # listing need not open every file.
        label_map = self._get_label_map(data)
        label_ids = np.array(list(label_map.keys()), dtype=int)
        label_names = np.array(list(label_map.values()), dtype=object)

        # The group hint from labels_raw/index.csv, honored per ``group_from``.
        raw_group_hint = str(raw_row.get("group", "") or "")

        # -------- STEP 3: One LabelEntry per sequence. --------
        # The source structure could be a single sequence per file, a nested
        # {group: {sequence: data}} mapping, or a table with a sequence column.
        # Adapt _extract_sequences to your format; the loop stays the same.
        entries: list[LabelEntry] = []
        for group_name, sequences in self._extract_sequences(data).items():
            for seq_id, seq_data in sequences.items():
                # Per-frame label IDs and their frame indices.
                labels = self._extract_labels(seq_data).astype(np.int32, copy=False)
                frames = np.arange(len(labels), dtype=np.int32)

                group_val = self._determine_group(group_name, raw_group_hint, params)
                seq_val = str(seq_id)

                # Build the .npz payload: the arrays and scalars a reader loads.
                payload: dict[str, object] = {
                    "group": group_val,
                    "sequence": seq_val,
                    "sequence_key": seq_val,
                    "label_format": self.label_format,
                    "frames": frames,
                    "labels": labels,
                    "label_ids": label_ids,
                    "label_names": label_names,
                    "fps": float(params.default_fps),
                }

                entries.append(
                    LabelEntry(
                        group=group_val,
                        sequence=seq_val,
                        payload=payload,
                        n_frames=int(labels.shape[0]),
                        label_ids=tuple(int(i) for i in label_map),
                        label_names=tuple(label_map.values()),
                    )
                )

        return entries

    def get_metadata(self) -> dict[str, object]:
        """Optional format-specific metadata for ``dataset.meta['labels'][kind]``.

        Merged into ``dataset.meta['labels'][label_kind]``. Useful for a label
        mapping or a format-version note. Return an empty dict when there is
        nothing to record.
        """
        return {}

    # ============ HELPER METHODS (CUSTOMIZE THESE) ============

    def _determine_group(
        self,
        source_group: str,
        raw_group_hint: str,
        params: MyLabelParams,
    ) -> str:
        """Pick the output group per ``group_from`` (entry policy, never hashed).

        ``group_from`` selects which group string to assign; it is not a property
        of the labels, so it is excluded from the variant identity.
        """
        if params.group_from in {"filename", "both"} and raw_group_hint:
            return raw_group_hint
        return str(source_group or "")

    def _load_source_file(self, src_path: Path) -> object:
        """Load and parse the source label file.

        Returns:
            The parsed data structure (format-dependent).

        Examples:
            For CSV: ``return pd.read_csv(src_path)``.

            For JSON::

                import json
                with open(src_path) as f:
                    return json.load(f)

            For NPY: ``return np.load(src_path, allow_pickle=True)``.
        """
        raise NotImplementedError("Implement _load_source_file for your format")

    def _get_label_map(self, data: object) -> dict[int, str]:
        """Extract or define the label mapping.

        Args:
            data: Loaded data from :meth:`_load_source_file`.

        Returns:
            Mapping from label ID to label name, e.g.
            ``{0: "attack", 1: "investigation", 2: "mount"}``.
        """
        raise NotImplementedError("Implement _get_label_map for your format")

    def _extract_sequences(self, data: object) -> dict[str, dict[str, object]]:
        """Extract sequences from the loaded data.

        Args:
            data: Loaded data from :meth:`_load_source_file`.

        Returns:
            A nested ``{group_name: {seq_id: seq_data}}`` mapping. For a
            single-group file, use one key, e.g. ``{"": {seq_id: seq_data}}``.
        """
        raise NotImplementedError("Implement _extract_sequences for your format")

    def _extract_labels(self, seq_data: object) -> np.ndarray:
        """Extract the per-frame label array from one sequence's data.

        Args:
            seq_data: Data for a single sequence.

        Returns:
            An integer array of shape ``(T,)``; each element is a label ID from
            the label map.
        """
        raise NotImplementedError("Implement _extract_labels for your format")
