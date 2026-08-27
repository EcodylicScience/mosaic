"""CalMS21 .npy/.json track converter.

Converts CalMS21 multi-animal pose tracking files to the standardized
``mosaic_v1`` parquet schema: pixels, keypoints, and the body centre.
Handles task1/task2/task3 splits.

CalMS21 keypoint layout: (T, n_animals, xy=2, n_landmarks)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from typing import Annotated

from mosaic.core.helpers import build_compound_name
from mosaic.core.params import (
    HASH_EXCLUDE,
    Declared,
)
from mosaic.core.track_converter import (
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    register_track_converter,
)
from mosaic.core.track_library.helpers import load_calms21, norm_hint


def _calms21_seq_to_trex_df(
    one_seq_dict: dict,
    groupname: str,
    seq_id: str,
) -> pd.DataFrame:
    """
    Convert a single sequence dict to T-Rex-like long DataFrame (rows = frames x animals).
    """
    # Pick features: either 'features' present or 'keypoints'
    use_features = "features" in one_seq_dict
    if use_features:
        # not used in output columns; could be stored elsewhere if needed
        _ = np.asarray(one_seq_dict["features"])  # (T, K)
    keypoints = np.asarray(one_seq_dict["keypoints"])  # (T, 2, 2, L)
    # Guarded on the key, not on the result. ``np.asarray(None)`` is a 0-d object
    # array rather than None, so the old ``scores is not None`` check downstream
    # was always true and a file without scores -- which the schema permits --
    # raised IndexError on the first ``scores[:, a, :]``.
    scores = (
        np.asarray(one_seq_dict["scores"]) if "scores" in one_seq_dict else None
    )  # (T, 2, L) or None
    ann = (
        np.asarray(one_seq_dict["annotations"])
        if "annotations" in one_seq_dict
        else None
    )
    meta = one_seq_dict.get("metadata", {})
    fps = float(meta.get("fps", meta.get("frame_rate", 30.0)))

    T = keypoints.shape[0]
    n_anim = keypoints.shape[1]
    n_lm = keypoints.shape[3]

    rows = []
    for a in range(n_anim):
        # Extract XY for this animal
        X = keypoints[:, a, 0, :]  # (T, L)
        Y = keypoints[:, a, 1, :]  # (T, L)

        # Centroid over landmarks
        cx = X.mean(axis=1)  # (T,)
        cy = Y.mean(axis=1)

        # Build a per-frame DataFrame
        data = {
            "frame": np.arange(T, dtype=int),
            "time": np.arange(T, dtype=float) / fps,
            "id": np.full(T, a, dtype=int),
            "X": cx,
            "Y": cy,
            "group": np.full(T, groupname),
            "sequence": np.full(T, seq_id),
        }

        # Pose columns
        for k in range(n_lm):
            data[f"poseX{k}"] = X[:, k]
            data[f"poseY{k}"] = Y[:, k]

        # Optional: label per frame if present (flatten if multi-dim)
        if ann is not None:
            lbl = ann
            if lbl.ndim > 1:
                lbl = lbl[:, 0]
            data["label"] = lbl.astype(int, copy=False)

        # Optional: keypoint scores columns, if provided
        if scores is not None:
            S = np.asarray(scores)  # (T, 2, L)
            S_a = S[:, a, :]  # (T, L)
            for k in range(n_lm):
                data[f"poseP{k}"] = S_a[:, k]

        rows.append(pd.DataFrame(data))

    # No TRex-shaped placeholders. Eighteen columns used to be added here so the
    # output "looked like" a TRex table -- fifteen of them all-NaN floats, which
    # `feature_columns()` then dragged into every template matrix, scaler and
    # embedding built from CalMS21. A column nobody measured is not a column.
    return pd.concat(rows, ignore_index=True)


def calms21_to_trex_df(
    path: Path | str,
    prefer_group: Optional[str] = None,
    prefer_sequence: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a CalMS21 .npy/.json and return a concatenated T-Rex-like DataFrame.
    Optionally filter to a specific (group, sequence).
    """
    nested = load_calms21(path)

    groups_present = set(nested.keys())
    seq_filter = None
    direct_group_match_only = True
    if prefer_group and prefer_group not in groups_present:
        # interpret dataset-level hint (e.g., calms21_task1_test)
        seq_filter = _calms21_make_seq_filter_from_hint(prefer_group)
        if seq_filter is not None:
            direct_group_match_only = False

    rows = []
    for groupname, group in nested.items():
        for seq_id, seq in group.items():
            # The hint arrives in the flattened spelling enumerate_sequences
            # emits, so compare in that spelling. seq_id itself stays raw --
            # the group-hint filter below reads the slashes.
            entry_name = calms21_entry_name(str(seq_id))
            if prefer_sequence and entry_name != prefer_sequence:
                continue
            # group filter: either exact top-level match, or sequence-path filter if hint provided
            if direct_group_match_only:
                if prefer_group and groupname != prefer_group:
                    continue
            else:
                if seq_filter and not seq_filter(groupname, seq_id):
                    continue

            # ensure arrays where needed
            seq = {
                k: (np.array(v) if isinstance(v, list) else v) for k, v in seq.items()
            }
            rows.append(_calms21_seq_to_trex_df(seq, groupname, entry_name))
    if not rows:
        if prefer_group or prefer_sequence:
            raise KeyError(
                f"Requested CalMS21 ({prefer_group}, {prefer_sequence}) not found in {path}"
            )
        raise RuntimeError(f"No sequences found in CalMS21 file: {path}")
    return pd.concat(rows, ignore_index=True)


_DEBUG_DESCRIPTION = "Print the in-file (group, sequence) pairs."


class Calms21Params(TrackConvertParams):
    """Parameters for the CalMS21 converters."""

    # Diagnostics only, so excluded from identity -- it changes what is
    # printed, never what is written.
    debug: Annotated[bool, HASH_EXCLUDE, Declared(_DEBUG_DESCRIPTION)] = False


def calms21_entry_name(seq_id: str) -> str:
    """The mosaic entry name for a CalMS21 in-file sequence id.

    CalMS21 spells its ids as slash paths -- ``task1/test/mouse075_task1_annotator1``
    -- read verbatim out of the source file. mosaic percent-encodes a ``/`` for
    filenames and always has, so this worked; but an entry name doubles as a
    filesystem path component in the control plane, where ``sequence_of()`` splits
    on the first ``/`` and the media directory interpolates the name into a path.
    A name that cannot round-trip there is not a name mosaic should mint.

    So the levels are joined with the repo's own compound-name separator instead.
    ``task1/test/mouse075`` becomes ``task1__test__mouse075``, which
    ``parse_hierarchy`` reads with its *default* separator -- CalMS21 gains
    ``get_sequence_metadata(level_names=["task", "split", "mouse"])`` rather than
    needing ``separator="/"`` for it.

    Applied where the name is *emitted*, never before the group-hint filter:
    ``_calms21_make_seq_filter_from_hint`` matches on ``f"/{split}/"`` inside the
    raw id, so flattening first would silently match nothing.
    """
    return build_compound_name(*seq_id.split("/"))


class Calms21Converter(TrackConverter[Calms21Params]):
    """CalMS21 -> a ``mosaic_v1`` table, one ``(group, sequence)`` at a time."""

    src_format = "calms21_npy"
    # 0.2: entry names are compound (``task1__test__m``) rather than slash paths.
    # The tables are otherwise identical, but the identity of a *variant* covers
    # what it emits, and this changed the entry keys and therefore the filenames.
    # 0.3: derived columns are gone, and so are the eighteen TRex-shaped
    # placeholders -- fifteen of them all-NaN floats that every template matrix
    # built from CalMS21 was carrying.
    version = "0.3"
    enumerable = True
    output_schema = "mosaic_v1"
    Params = Calms21Params

    def convert(
        self, path: Path, params: Calms21Params, hints: EntryHints
    ) -> pd.DataFrame:
        prefer_group = norm_hint(hints.group)
        prefer_sequence = norm_hint(hints.sequence)

        nested = load_calms21(path)
        if params.debug:
            pairs = [(g, s) for g, grp in nested.items() for s in grp.keys()]
            print(
                f"[calms21] in-file pairs ({len(pairs)}): {pairs[:10]}"
                f"{' ...' if len(pairs) > 10 else ''}"
            )
            print(
                f"[calms21] prefer_group={prefer_group} "
                f"prefer_sequence={prefer_sequence}"
            )

        # if explicit selection given, return only that
        if prefer_group or prefer_sequence:
            return calms21_to_trex_df(
                path,
                prefer_group=prefer_group,
                prefer_sequence=prefer_sequence,
            )

        # else single-pair inference
        pairs = [(g, s) for g, grp in nested.items() for s in grp.keys()]
        if len(pairs) == 1:
            g, s = pairs[0]
            return calms21_to_trex_df(
                path,
                prefer_group=g,
                prefer_sequence=s,
            )
        raise ValueError(
            f"Ambiguous CalMS21 file {path}; contains multiple sequences {pairs}. "
            f"Pass hints with group/sequence to disambiguate."
        )

    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        # Flattened here as well as at emission, so a hint round-tripped through
        # this list matches the name ``convert`` actually writes.
        nested = load_calms21(path)
        return [
            (str(g), calms21_entry_name(str(s)))
            for g, grp in nested.items()
            for s in grp.keys()
        ]


def _calms21_make_seq_filter_from_hint(hint: Optional[str]):
    """
    Return a predicate f(groupname, seq_id)->bool for dataset-level hints like
    'calms21_task1_train', 'calms21_task1_test', 'calms21_task2_train/test',
    'calms21_task3_train/test'. If not applicable, return None.
    """
    if not hint:
        return None
    h = hint.strip().lower()

    def pred_task_split(task_prefix: str, split: str):
        def _pred(_g, _s):
            # matches path patterns like taskX/.../<split>/...
            return _s.startswith(task_prefix) and (f"/{split}/" in _s)

        return _pred

    # task1
    if h.startswith("calms21_task1_"):
        split = (
            "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        )
        if split:
            return pred_task_split("task1/", split)

    # task2 (note: has an annotator level 'task2/annotator1/<split>/...')
    if h.startswith("calms21_task2_"):
        split = (
            "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        )
        if split:

            def _pred(_g, _s):
                return _s.startswith("task2/") and (f"/{split}/" in _s)

            return _pred

    # task3 (behavior level: 'task3/<behavior>/<split>/...')
    if h.startswith("calms21_task3_"):
        split = (
            "train" if h.endswith("train") else ("test" if h.endswith("test") else None)
        )
        if split:
            return pred_task_split("task3/", split)

    return None


# One class per source format, rather than one class registered twice. The two
# file formats hold the same structure and convert identically, but a tracks
# variant identity names exactly one producer, so the format it read has to be
# part of that name rather than an ambiguity inside it.
_ = register_track_converter(Calms21Converter)


@register_track_converter
class Calms21JsonConverter(Calms21Converter):
    """The same conversion, reading the ``.json`` spelling of a CalMS21 file."""

    src_format = "calms21_json"
