"""Frame extraction methods for single-video workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import json
import math
import uuid

import numpy as np

from .sampling import select_kmeans_frames, select_uniform_frames
from .video_io import (
    extract_candidate_features,
    get_video_metadata,
    normalize_crop_rect,
    normalize_frame_range,
    save_frames_as_png,
)


CropSpec = tuple[int, int, int, int] | dict[str, Any]


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


def _crop_to_dict(crop_rect: Optional[tuple[int, int, int, int]]) -> Optional[dict[str, int]]:
    if crop_rect is None:
        return None
    x, y, w, h = [int(v) for v in crop_rect]
    return {"x": x, "y": y, "w": w, "h": h}


def extract_frames(
    video_path: Path | str,
    output_root: Path | str,
    n_frames: int,
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
        Number of frames to extract.
    method
        "uniform" or "kmeans".
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
    """
    method_norm = str(method).strip().lower()
    if method_norm not in {"uniform", "kmeans"}:
        raise ValueError("method must be one of: 'uniform', 'kmeans'")
    if int(n_frames) <= 0:
        raise ValueError("n_frames must be > 0")
    if int(candidate_step) <= 0:
        raise ValueError("candidate_step must be > 0")

    meta = get_video_metadata(video_path)
    start, end = normalize_frame_range(meta.frame_count, start_frame, end_frame)
    crop_rect = normalize_crop_rect(crop, meta.width, meta.height)

    if method_norm == "uniform":
        candidates = np.arange(start, end + 1, int(candidate_step), dtype=np.int32)
        selected = select_uniform_frames(candidates, int(n_frames))
        sampling_details: dict[str, Any] = {}
    else:
        effective_step = int(candidate_step)
        if kmeans_max_candidates is not None and int(kmeans_max_candidates) > 0:
            approx_candidates = ((int(end) - int(start)) // int(candidate_step)) + 1
            if approx_candidates > int(kmeans_max_candidates):
                stride_mult = int(math.ceil(approx_candidates / float(kmeans_max_candidates)))
                effective_step = int(candidate_step) * max(1, stride_mult)

        candidates, features = extract_candidate_features(
            video_path=meta.path,
            start_frame=start,
            end_frame=end,
            candidate_step=int(effective_step),
            resize=(int(kmeans_resize[0]), int(kmeans_resize[1])),
            grayscale=bool(kmeans_grayscale),
            crop_rect=crop_rect,
            max_candidates=None,
        )
        selected = select_kmeans_frames(
            candidate_indices=candidates,
            features=features,
            n_frames=int(n_frames),
            random_state=int(random_state),
            batch_size=int(kmeans_batch_size),
            max_iter=int(kmeans_max_iter),
            n_init=kmeans_n_init,
        )
        sampling_details = {
            "kmeans_resize": [int(kmeans_resize[0]), int(kmeans_resize[1])],
            "kmeans_grayscale": bool(kmeans_grayscale),
            "kmeans_max_candidates": None if kmeans_max_candidates is None else int(kmeans_max_candidates),
            "kmeans_effective_candidate_step": int(effective_step),
            "kmeans_batch_size": int(kmeans_batch_size),
            "kmeans_max_iter": int(kmeans_max_iter),
            "kmeans_n_init": kmeans_n_init,
            "candidate_count": int(candidates.size),
        }

    run = run_id or _make_run_id()
    out_dir = Path(output_root).expanduser().resolve() / meta.path.stem / method_norm / run
    out_dir.mkdir(parents=True, exist_ok=False)

    file_records = save_frames_as_png(
        video_path=meta.path,
        frame_indices=selected,
        output_dir=out_dir,
        crop_rect=crop_rect,
    )

    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    result = FrameExtractionResult(
        run_id=run,
        method=method_norm,
        video_path=str(meta.path),
        output_dir=str(out_dir),
        manifest_path=str(out_dir / "run_info.json"),
        n_requested=int(n_frames),
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


def load_extraction_manifest(path: Path | str) -> dict[str, Any]:
    """Load a saved JSON manifest from a previous extraction run."""
    p = Path(path).expanduser().resolve()
    return json.loads(p.read_text())
