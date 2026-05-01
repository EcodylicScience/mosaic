"""Shared helpers for embedding-based identity models.

The V200 :class:`~mosaic.behavior.feature_library.identity_model.GlobalIdentityModel`
keeps its crop-loading and label-mapping logic as private methods. The
embedding-based identity plugins (MegaDescriptor, DINOv2 + temporal) reuse
the same pattern, so the relevant logic is lifted here and shared.

Two extra utilities are provided that V200 doesn't need:

* :func:`compute_prototypes` -- mean-pool L2-normalized embeddings per
  identity to produce a per-class prototype matrix.
* :func:`knn_predict` -- cosine-similarity k-NN against prototypes,
  returned as a softmax over identities so the output shape matches V200's
  ``(N, num_classes)`` probability convention.

V200 itself is intentionally left untouched; these helpers are only used by
the new plugins.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from mosaic.core.helpers import make_entry_key

if TYPE_CHECKING:
    import pandas as pd

    from mosaic.core.pipeline.types import InputStream
    from mosaic.core.pipeline.types.params import Params


def build_label_mapping(
    params: Params,
    inputs: InputStream,
    *,
    identities_attr: str = "identities",
    group_as_identity_attr: str = "group_as_identity",
) -> tuple[dict[str, int], list[str]]:
    """Build a mapping from pipeline ``entry_key`` to integer label.

    Mirrors ``GlobalIdentityModel._build_label_mapping``: users specify
    ``identities`` as ``{"name": ["group/sequence", ...]}`` (readable form),
    but the InputStream yields canonical ``entry_key = make_entry_key(group,
    sequence)``. We translate every key into the canonical form before
    storing.

    Args:
        params: Feature params object exposing ``identities`` and
            ``group_as_identity`` fields.
        inputs: Pipeline input stream (only consumed if
            ``group_as_identity=True``).
        identities_attr: Attribute name for the identities dict on
            ``params``. Override only if your params model uses a different
            name.
        group_as_identity_attr: Attribute name for the group-as-identity
            shortcut on ``params``.

    Returns:
        (entry_key -> label index, sorted identity names).
    """
    identities: dict[str, list[str]] | None = getattr(params, identities_attr, None)
    group_as_identity: bool = bool(getattr(params, group_as_identity_attr, False))

    seq_to_label: dict[str, int] = {}

    def _normalize(seq_key: str) -> str:
        if "/" in seq_key:
            group, sequence = seq_key.split("/", 1)
            return make_entry_key(group, sequence)
        return seq_key

    if identities is not None:
        identity_names = sorted(identities.keys())
        name_to_label = {name: i for i, name in enumerate(identity_names)}
        for name, seqs in identities.items():
            label = name_to_label[name]
            for seq_key in seqs:
                seq_to_label[_normalize(seq_key)] = label
        return seq_to_label, identity_names

    if group_as_identity:
        entry_keys = iter_entry_keys(inputs)
        group_set: set[str] = set()
        for entry_key in entry_keys:
            group = entry_key.split("__", 1)[0] if "__" in entry_key else entry_key
            group_set.add(group)
        identity_names = sorted(group_set)
        name_to_label = {name: i for i, name in enumerate(identity_names)}
        for entry_key in entry_keys:
            group = entry_key.split("__", 1)[0] if "__" in entry_key else entry_key
            if group in name_to_label:
                seq_to_label[entry_key] = name_to_label[group]
        return seq_to_label, identity_names

    msg = (
        "[identity-model] Either 'identities' dict or "
        "'group_as_identity=True' must be provided."
    )
    raise ValueError(msg)


def iter_entry_keys(inputs: InputStream) -> list[str]:
    """Collect all entry keys from the input stream."""
    return [entry_key for entry_key, _df in inputs()]


def load_crop_frames(
    entry_key: str,
    df: pd.DataFrame,
    *,
    crop_root: str | Path | None,
    channels: int,
    max_frames: int,
) -> list[np.ndarray]:
    """Load EgocentricCrop frame images for one sequence.

    Resolves the per-sequence directory and reads every PNG under
    ``<crop_root>/<group>__<sequence>/frames_id*/frame_*.png``. Returns
    color images as RGB ``(H, W, 3)`` (OpenCV's default BGR is converted),
    grayscale as ``(H, W, 1)``.

    Resolution order for the crop root:

    1. ``crop_root`` argument (explicit override, recommended).
    2. ``df._source_dir`` if set by the pipeline runner.
    3. Empty list if neither is available.

    Args:
        entry_key: Canonical ``"safe_group__safe_sequence"`` identifier.
        df: DataFrame from the input stream. May carry raw ``group`` /
            ``sequence`` columns and a ``_source_dir`` attribute.
        crop_root: EgocentricCrop output root containing
            ``<group>__<sequence>/`` subdirs.
        channels: 1 = grayscale, 3 = RGB.
        max_frames: Stop after this many frames have been collected.

    Returns:
        List of ``(H, W, channels)`` uint8 arrays, capped at ``max_frames``.
    """
    frames: list[np.ndarray] = []

    seq_dir: Path | None = None

    if crop_root is not None:
        root_path = Path(crop_root)
        if (
            df is not None
            and not df.empty
            and "group" in df.columns
            and "sequence" in df.columns
        ):
            group = str(df["group"].iloc[0])
            sequence = str(df["sequence"].iloc[0])
            seq_dir = root_path / f"{group}__{sequence}"
        else:
            seq_dir = root_path / entry_key

    if seq_dir is None or not seq_dir.is_dir():
        source_dir = getattr(df, "_source_dir", None)
        if source_dir is not None and Path(source_dir).is_dir():
            seq_dir = Path(source_dir)

    if seq_dir is None or not seq_dir.is_dir():
        return frames

    for frames_subdir in sorted(seq_dir.glob("frames_id*")):
        if not frames_subdir.is_dir():
            continue
        for img_path in sorted(frames_subdir.glob("frame_*.png")):
            img = _load_image(img_path, channels=channels)
            if img is not None:
                frames.append(img)
                if len(frames) >= max_frames:
                    return frames
    return frames


def load_crop_clips(
    entry_key: str,
    df: pd.DataFrame,
    *,
    crop_root: str | Path | None,
    channels: int,
    clip_len: int,
    max_clips: int,
    stride: int | None = None,
) -> list[np.ndarray]:
    """Load EgocentricCrop frames as a list of fixed-length clips.

    For temporal identity models. Frames within one ``frames_id*/``
    subdirectory are consecutive in time, so clips are built per subdir and
    not allowed to span across subdirs.

    Args:
        entry_key: Canonical ``"safe_group__safe_sequence"`` identifier.
        df: DataFrame from the input stream.
        crop_root: EgocentricCrop output root.
        channels: 1 = grayscale, 3 = RGB.
        clip_len: Number of frames per clip.
        max_clips: Cap on returned clips.
        stride: Step between clip start indices. Defaults to ``clip_len``
            (non-overlapping).

    Returns:
        List of ``(clip_len, H, W, channels)`` uint8 arrays.
    """
    if stride is None:
        stride = clip_len

    clips: list[np.ndarray] = []

    seq_dir = _resolve_sequence_dir(entry_key, df, crop_root)
    if seq_dir is None:
        return clips

    for frames_subdir in sorted(seq_dir.glob("frames_id*")):
        if not frames_subdir.is_dir():
            continue
        frame_paths = sorted(frames_subdir.glob("frame_*.png"))
        if len(frame_paths) < clip_len:
            continue
        for start in range(0, len(frame_paths) - clip_len + 1, stride):
            window = frame_paths[start : start + clip_len]
            imgs: list[np.ndarray] = []
            ok = True
            for p in window:
                img = _load_image(p, channels=channels)
                if img is None:
                    ok = False
                    break
                imgs.append(img)
            if ok:
                clips.append(np.stack(imgs, axis=0))
                if len(clips) >= max_clips:
                    return clips
    return clips


def compute_prototypes(
    embeddings: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Mean-pool embeddings per class and L2-normalize the result.

    Args:
        embeddings: ``(N, D)`` float array of per-sample embeddings.
        labels: ``(N,)`` integer class labels in ``[0, num_classes)``.
        num_classes: Total number of identities.

    Returns:
        ``(num_classes, D)`` float32 prototype matrix. Empty classes are
        zero-rows and will yield zero similarity at inference.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    dim = embeddings.shape[1]
    prototypes = np.zeros((num_classes, dim), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        if not np.any(mask):
            continue
        proto = embeddings[mask].mean(axis=0)
        norm = float(np.linalg.norm(proto))
        if norm > 0:
            proto = proto / norm
        prototypes[c] = proto
    return prototypes


def knn_predict(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    *,
    temperature: float = 0.07,
) -> np.ndarray:
    """Cosine-similarity k-NN against prototypes, returned as a softmax.

    L2-normalizes ``embeddings`` (prototypes are assumed already unit-norm),
    computes cosine similarity, and softmaxes with the given temperature
    so the result is a proper probability distribution over identities --
    matching V200's ``predict()`` output shape.

    Args:
        embeddings: ``(N, D)`` query embeddings.
        prototypes: ``(num_classes, D)`` unit-norm prototypes.
        temperature: Softmax temperature. Lower = sharper. Default 0.07
            is the standard for cosine-similarity classification.

    Returns:
        ``(N, num_classes)`` float32 probability array.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    sims = embeddings @ prototypes.T
    logits = sims / temperature
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)


# --- Internal ---


def _resolve_sequence_dir(
    entry_key: str, df: pd.DataFrame, crop_root: str | Path | None
) -> Path | None:
    if crop_root is not None:
        root_path = Path(crop_root)
        if (
            df is not None
            and not df.empty
            and "group" in df.columns
            and "sequence" in df.columns
        ):
            group = str(df["group"].iloc[0])
            sequence = str(df["sequence"].iloc[0])
            candidate = root_path / f"{group}__{sequence}"
        else:
            candidate = root_path / entry_key
        if candidate.is_dir():
            return candidate

    source_dir = getattr(df, "_source_dir", None)
    if source_dir is not None and Path(source_dir).is_dir():
        return Path(source_dir)
    return None


def _load_image(path: Path, *, channels: int) -> np.ndarray | None:
    """Load one image as ``(H, W, channels)`` uint8.

    Color images are converted from OpenCV's BGR to RGB.
    """
    if channels == 1:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        return img[:, :, np.newaxis]

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
