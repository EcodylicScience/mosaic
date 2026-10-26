"""Frame extraction methods for single- and multi-video workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any, Final, Optional, get_args
import json
import uuid

import numpy as np
from mosaic_media import MediaFacts, probe_media
from mosaic.user_paths import user_path

from mosaic.core.json_value import JsonValue
from mosaic.core.media.imgstore_io import is_imgstore
from mosaic.core.media.video_io import (
    MultiVideoReader,
    MultiVideoReaderLike,
    extract_candidate_features,
    extract_candidate_features_multi,
    frame_feature,
    normalize_crop_rect,
    normalize_frame_range,
    read_frames_at,
    read_frames_at_multi,
    save_frames_as_png,
    save_frames_as_png_multi,
    video_metadata_or_probe,
)

from .sampling import (
    Exclusion,
    ExtractionMethod,
    KmeansSettings,
    SamplingMethod,
    select_frame_indices,
)


CropSpec = tuple[int, int, int, int] | dict[str, Any]


def _features_in_order(
    frames: Iterable[tuple[int, np.ndarray]],
    indices: np.ndarray,
    *,
    crop_rect: tuple[int, int, int, int] | None,
    resize: tuple[int, int],
    grayscale: bool,
) -> np.ndarray:
    """Return the feature of each frame of *indices*, in the order of *indices*.

    Args:
        frames: The ``(frame, image)`` pairs read for *indices*.
        indices: The frame indices, one or more.
        crop_rect: The (x, y, w, h) crop applied before the resize, or ``None``.
        resize: The (width, height) that each frame is resized to.
        grayscale: Whether frames are converted to grayscale.

    Returns:
        The ``(M, D)`` features, one row per index.
    """
    features = {
        frame: frame_feature(image, crop_rect, resize, grayscale)
        for frame, image in frames
    }
    return np.vstack([features[int(index)] for index in indices.tolist()])


@dataclass(frozen=True, slots=True)
class SingleFileFeatures:
    """The candidate frames of one video file or imgstore, with their features.

    Attributes:
        path: The video file or imgstore directory.
        facts: The stored media facts of *path*, or ``None`` to probe it.
        resize: The (width, height) that each frame is resized to.
        grayscale: Whether frames are converted to grayscale.
        crop_rect: The (x, y, w, h) crop applied before the resize, or ``None``.
    """

    path: Path
    facts: MediaFacts | None
    resize: tuple[int, int]
    grayscale: bool
    crop_rect: tuple[int, int, int, int] | None

    def candidates(
        self, start: int, end: int, step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode every *step*-th frame from *start* to *end* inclusive.

        Args:
            start: The first frame of the range.
            end: The last frame of the range.
            step: The stride between candidates.

        Returns:
            The candidate indices, ``(N,)``, and their features, ``(N, D)``.
        """
        return extract_candidate_features(
            video_path=self.path,
            start_frame=start,
            end_frame=end,
            candidate_step=step,
            resize=self.resize,
            grayscale=self.grayscale,
            crop_rect=self.crop_rect,
            max_candidates=None,
            facts=self.facts,
        )

    def at(self, indices: np.ndarray) -> np.ndarray:
        """Decode the frames at *indices*, through the seek of each file type.

        Args:
            indices: One or more frame indices, in any order.

        Returns:
            The ``(M, D)`` features, one row per index in the order of *indices*.
        """
        return _features_in_order(
            read_frames_at(self.path, indices, self.facts),
            indices,
            crop_rect=self.crop_rect,
            resize=self.resize,
            grayscale=self.grayscale,
        )


@dataclass(frozen=True, slots=True)
class MultiClipFeatures:
    """The candidate frames of a sequence of files, indexed across them.

    Attributes:
        reader: The open reader over the sequence.
        resize: The (width, height) that each frame is resized to.
        grayscale: Whether frames are converted to grayscale.
        crop_rect: The (x, y, w, h) crop applied before the resize, or ``None``.
    """

    reader: MultiVideoReaderLike
    resize: tuple[int, int]
    grayscale: bool
    crop_rect: tuple[int, int, int, int] | None

    def candidates(
        self, start: int, end: int, step: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode every *step*-th frame from *start* to *end* inclusive.

        Args:
            start: The first frame of the range.
            end: The last frame of the range.
            step: The stride between candidates.

        Returns:
            The candidate indices, ``(N,)``, and their features, ``(N, D)``.
        """
        return extract_candidate_features_multi(
            reader=self.reader,
            start_frame=start,
            end_frame=end,
            candidate_step=step,
            resize=self.resize,
            grayscale=self.grayscale,
            crop_rect=self.crop_rect,
            max_candidates=None,
        )

    def at(self, indices: np.ndarray) -> np.ndarray:
        """Decode the frames at the global *indices*, by a seek to each.

        Args:
            indices: One or more global frame indices, in any order.

        Returns:
            The ``(M, D)`` features, one row per index in the order of *indices*.
        """
        return _features_in_order(
            read_frames_at_multi(self.reader, indices),
            indices,
            crop_rect=self.crop_rect,
            resize=self.resize,
            grayscale=self.grayscale,
        )


@dataclass(frozen=True)
class FrameExtractionResult:
    """Result metadata for a frame extraction run."""

    run_id: str
    method: str
    video_path: str
    output_dir: str
    manifest_path: str
    n_requested: int
    n_extracted: int
    selected_frame_indices: list[int]
    start_frame: int
    end_frame: int
    candidate_step: int
    crop: Optional[dict[str, int]]
    created_utc: str
    files: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _make_run_id() -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{now}_{suffix}"


_METHODS: Final = frozenset(get_args(ExtractionMethod))

_SHOWN_OUT_OF_RANGE: Final = 5
"""How many out-of-range indices a refusal names before summarising the rest."""


def _requested_count(n_frames: int | None) -> int:
    """How many frames a sampling method is asked for, refused unless positive."""
    if n_frames is None or int(n_frames) <= 0:
        raise ValueError("n_frames must be > 0")
    return int(n_frames)


def _checked_method(
    method: str,
    n_frames: int | None,
    candidate_step: int,
    frame_indices: Sequence[int] | None,
) -> str:
    """The method, normalised, once the arguments it reads agree with it.

    ``frame_indices`` is given exactly when the method is ``"list"``, so a
    caller can branch on it and the checker narrows it there.
    """
    method_norm = str(method).strip().lower()
    if method_norm not in _METHODS:
        allowed = ", ".join(repr(name) for name in get_args(ExtractionMethod))
        raise ValueError(f"method must be one of: {allowed}")
    if method_norm == "list":
        if frame_indices is None:
            raise ValueError("method 'list' needs frame_indices")
    else:
        if frame_indices is not None:
            message = (
                f"frame_indices is read only under method 'list', not {method_norm!r}"
            )
            raise ValueError(message)
        _ = _requested_count(n_frames)
    if int(candidate_step) <= 0:
        raise ValueError("candidate_step must be > 0")
    return method_norm


def _sampling_method(method: str) -> SamplingMethod:
    """Return *method* as a sampling method, refused when it names another method."""
    match method:
        case "uniform" | "kmeans":
            return method
        case _:
            raise ValueError(f"method {method!r} is not a sampling method")


def _listed_selection(frame_indices: Sequence[int], total_frames: int) -> np.ndarray:
    """The listed frames, sorted and de-duplicated, refused past either end.

    Sorted because a frame's file is named by its index and the order carries
    no meaning, and because the readers decode forward.

    Raises:
        ValueError: If an index is negative or not below *total_frames*.
    """
    selected = sorted({int(index) for index in frame_indices})
    outside = [index for index in selected if index < 0 or index >= total_frames]
    if outside:
        shown = ", ".join(str(index) for index in outside[:_SHOWN_OUT_OF_RANGE])
        rest = len(outside) - _SHOWN_OUT_OF_RANGE
        more = f" and {rest} more" if rest > 0 else ""
        message = (
            f"frame indices {shown}{more} fall outside the video's "
            f"{total_frames} frames"
        )
        raise ValueError(message)
    return np.asarray(selected, dtype=np.int32)


def _crop_to_dict(
    crop_rect: Optional[tuple[int, int, int, int]],
) -> Optional[dict[str, int]]:
    if crop_rect is None:
        return None
    x, y, w, h = [int(v) for v in crop_rect]
    return {"x": x, "y": y, "w": w, "h": h}


def extract_frames(
    video_path: Path | str,
    output_root: Path | str | None = None,
    n_frames: int | None = 50,
    method: str = "uniform",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    candidate_step: int = 1,
    crop: Optional[CropSpec] = None,
    kmeans_resize: tuple[int, int] = (64, 64),
    kmeans_grayscale: bool = True,
    kmeans_max_candidates: Optional[int] = 5000,
    kmeans_batch_size: int = 1024,
    kmeans_max_iter: int = 100,
    kmeans_n_init: int | str = "auto",
    random_state: int = 42,
    run_id: Optional[str] = None,
    output_dir: Optional[Path | str] = None,
    facts: MediaFacts | None = None,
    frame_indices: Sequence[int] | None = None,
    excluded: Exclusion | None = None,
) -> FrameExtractionResult:
    """
    Extract representative frames from a single video.

    Parameters
    ----------
    video_path
        Path to source video.
    output_root
        Root directory where run outputs are created.
    n_frames
        Number of frames to extract under "uniform" or "kmeans". Not read under
        "list".
    method
        "uniform", "kmeans" or "list".
    start_frame, end_frame
        Optional inclusive frame range; defaults to full video.
    candidate_step
        Candidate downsampling stride in frames (>=1).
    crop
        Optional crop rectangle as (x, y, w, h) or {"x","y","w","h"}.
    kmeans_resize
        Feature image size (width, height) for k-means.
    kmeans_grayscale
        If True, convert candidate frames to grayscale before feature flattening.
    kmeans_max_candidates
        Optional cap on candidate frames decoded for k-means.
    random_state
        Random seed for k-means and tie-breaking.
    run_id
        Optional explicit run id. If omitted, generated automatically.
    output_dir
        Optional explicit output directory. When provided, frames are written
        directly into this directory instead of the auto-computed path under
        output_root. Useful for Dataset integration where the caller controls
        the directory layout. If set, output_root is ignored.
    facts
        Stored media facts for *video_path*, injected into the candidate-frame
        reader so it does not re-probe. ``None`` for bare-path callers.
    frame_indices
        The frames to write under "list", and given only then. Sorted and
        de-duplicated; an index outside the video raises ``ValueError``.
    excluded
        The frames that "uniform" or "kmeans" sampling avoids and samples around,
        or ``None``. Its keys extend the ``sampling`` record. Not read under
        "list".
    """
    method_norm = _checked_method(method, n_frames, candidate_step, frame_indices)

    # Resolve the measurement once, for the plain-video case only: a store is a
    # directory with no elementary stream, so probe_media cannot measure it, and
    # video_metadata_or_probe already routes it to imgstore_metadata instead.
    # Resolving here means the candidate-feature and PNG-save reads below reuse
    # this one measurement instead of each probing the file again.
    resolved_facts = facts
    if resolved_facts is None and not is_imgstore(video_path):
        resolved_facts = probe_media(user_path(video_path).resolve())
    meta = video_metadata_or_probe(video_path, resolved_facts)
    start, end = normalize_frame_range(meta.frame_count, start_frame, end_frame)
    crop_rect = normalize_crop_rect(crop, meta.width, meta.height)

    if frame_indices is not None:
        selected = _listed_selection(frame_indices, meta.frame_count)
        n_requested = int(selected.size)
        sampling_details: dict[str, JsonValue] = {}
    else:
        kmeans = KmeansSettings(
            resize=kmeans_resize,
            grayscale=kmeans_grayscale,
            max_candidates=kmeans_max_candidates,
            batch_size=kmeans_batch_size,
            max_iter=kmeans_max_iter,
            n_init=kmeans_n_init,
        )
        n_requested = _requested_count(n_frames)
        selected, sampling_details = select_frame_indices(
            _sampling_method(method_norm),
            start=start,
            end=end,
            candidate_step=int(candidate_step),
            n_frames=n_requested,
            source=SingleFileFeatures(
                path=meta.path,
                facts=resolved_facts,
                resize=kmeans.resize,
                grayscale=kmeans.grayscale,
                crop_rect=crop_rect,
            ),
            kmeans=kmeans,
            random_state=int(random_state),
            excluded=excluded,
        )

    run = run_id or _make_run_id()
    if output_dir is not None:
        out_dir = user_path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if output_root is None:
            raise ValueError("Either output_root or output_dir must be provided.")
        out_dir = user_path(output_root).resolve() / meta.path.stem / method_norm / run
        out_dir.mkdir(parents=True, exist_ok=False)

    file_records = save_frames_as_png(
        video_path=meta.path,
        frame_indices=selected,
        output_dir=out_dir,
        crop_rect=crop_rect,
        facts=resolved_facts,
    )

    created_utc = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    result = FrameExtractionResult(
        run_id=run,
        method=method_norm,
        video_path=str(meta.path),
        output_dir=str(out_dir),
        manifest_path=str(out_dir / "run_info.json"),
        n_requested=n_requested,
        n_extracted=int(len(file_records)),
        selected_frame_indices=[int(i) for i in selected.tolist()],
        start_frame=int(start),
        end_frame=int(end),
        candidate_step=int(candidate_step),
        crop=_crop_to_dict(crop_rect),
        created_utc=created_utc,
        files=file_records,
    )

    manifest = result.to_dict()
    manifest["video_meta"] = {
        "width": int(meta.width),
        "height": int(meta.height),
        "fps": float(meta.fps),
        "frame_count": int(meta.frame_count),
    }
    manifest["sampling"] = sampling_details
    manifest["random_state"] = int(random_state)
    (out_dir / "run_info.json").write_text(json.dumps(manifest, indent=2))
    return result


def extract_frames_multi(
    video_paths: Sequence[Path | str],
    output_root: Path | str | None = None,
    n_frames: int | None = 50,
    method: str = "uniform",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    candidate_step: int = 1,
    crop: Optional[CropSpec] = None,
    kmeans_resize: tuple[int, int] = (64, 64),
    kmeans_grayscale: bool = True,
    kmeans_max_candidates: Optional[int] = 5000,
    kmeans_batch_size: int = 1024,
    kmeans_max_iter: int = 100,
    kmeans_n_init: int | str = "auto",
    random_state: int = 42,
    run_id: Optional[str] = None,
    output_dir: Optional[Path | str] = None,
    facts: Sequence[MediaFacts] | None = None,
    frame_indices: Sequence[int] | None = None,
    excluded: Exclusion | None = None,
) -> FrameExtractionResult:
    """
    Extract representative frames from a multi-video sequence.

    Candidates are pooled across all videos using virtual (global) frame
    indices. The selected frames are then saved as PNGs.

    Parameters are the same as :func:`extract_frames` except:

    Parameters
    ----------
    video_paths : list[Path | str]
        Ordered list of video file paths forming the sequence.
    facts : sequence of MediaFacts, optional
        Stored media facts parallel to *video_paths*, injected into the reader
        so it does not re-probe. ``None`` for bare-path callers.
    frame_indices : sequence of int, optional
        Global frame indices to write under "list", counted across the clips.
    excluded : Exclusion, optional
        The global frames that sampling avoids and samples around.
    """
    method_norm = _checked_method(method, n_frames, candidate_step, frame_indices)

    # Extracted PNGs are addressed by global frame index and become
    # pose-annotation input, so a misindexed frame poisons the annotation set:
    # an analysis read.
    reader = MultiVideoReader(
        [Path(p) for p in video_paths], facts=facts, target="analysis"
    )
    total_frames = reader.total_frames
    start, end = normalize_frame_range(total_frames, start_frame, end_frame)
    crop_rect = normalize_crop_rect(crop, reader.width, reader.height)

    if frame_indices is not None:
        selected = _listed_selection(frame_indices, total_frames)
        n_requested = int(selected.size)
        sampling_details: dict[str, JsonValue] = {}
    else:
        kmeans = KmeansSettings(
            resize=kmeans_resize,
            grayscale=kmeans_grayscale,
            max_candidates=kmeans_max_candidates,
            batch_size=kmeans_batch_size,
            max_iter=kmeans_max_iter,
            n_init=kmeans_n_init,
        )
        n_requested = _requested_count(n_frames)
        selected, sampling_details = select_frame_indices(
            _sampling_method(method_norm),
            start=start,
            end=end,
            candidate_step=int(candidate_step),
            n_frames=n_requested,
            source=MultiClipFeatures(
                reader=reader,
                resize=kmeans.resize,
                grayscale=kmeans.grayscale,
                crop_rect=crop_rect,
            ),
            kmeans=kmeans,
            random_state=int(random_state),
            excluded=excluded,
        )

    run = run_id or _make_run_id()
    if output_dir is not None:
        out_dir = user_path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        if output_root is None:
            raise ValueError("Either output_root or output_dir must be provided.")
        out_dir = user_path(output_root).resolve() / "multi" / method_norm / run
        out_dir.mkdir(parents=True, exist_ok=False)

    # save_frames_as_png_multi seeks internally; no need to reopen
    file_records = save_frames_as_png_multi(
        reader=reader,
        frame_indices=selected,
        output_dir=out_dir,
        crop_rect=crop_rect,
    )
    reader.close()

    created_utc = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    result = FrameExtractionResult(
        run_id=run,
        method=method_norm,
        video_path=json.dumps([str(p) for p in video_paths]),
        output_dir=str(out_dir),
        manifest_path=str(out_dir / "run_info.json"),
        n_requested=n_requested,
        n_extracted=int(len(file_records)),
        selected_frame_indices=[int(i) for i in selected.tolist()],
        start_frame=int(start),
        end_frame=int(end),
        candidate_step=int(candidate_step),
        crop=_crop_to_dict(crop_rect),
        created_utc=created_utc,
        files=file_records,
    )

    manifest = result.to_dict()
    manifest["video_meta"] = {
        "width": reader.width,
        "height": reader.height,
        "fps": float(reader.fps),
        "total_frames": total_frames,
        "video_count": len(video_paths),
        "video_paths": [str(p) for p in video_paths],
    }
    manifest["sampling"] = sampling_details
    manifest["random_state"] = int(random_state)
    (out_dir / "run_info.json").write_text(json.dumps(manifest, indent=2))
    return result


def load_extraction_manifest(path: Path | str) -> dict[str, Any]:
    """Load a saved JSON manifest from a previous extraction run."""
    p = user_path(path).resolve()
    return json.loads(p.read_text())
