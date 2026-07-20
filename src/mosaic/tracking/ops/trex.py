"""TREx as a tracking op -- ``mosaic run --kind trex``.

Wraps :func:`mosaic.tracking.trex.run_trex` as a registered ``TrackingOp`` so TREx rides the
schema-driven runner and every execution backend (local / rq / k8s) with Pydantic param
validation + discovery -- the same one-contract path SLEAP / DeepLabCut will adopt. The
implementation is unchanged: ``run_trex`` still shells out to the ``trex`` binary in its conda
env and still hashes its *internal settings dict* for the ``run_id``, so existing TREx tracks
stay cache-valid (the op only re-routes the same call through a ``JobContext``).

``resource_class = "gpu"`` because TREx needs the GPU for YOLO detection -- its ``category``
of ``"convert"`` would not imply that, so it declares the class explicitly, and the execution
router then sends it to the GPU lane / k8s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from mosaic.core.pipeline.types import HASH_EXCLUDE, Params
from mosaic.tracking.registry import TrackingOp, register_tracking_op

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


class TrexParams(Params):
    """Parameters for the ``trex`` tracking op (mirrors :func:`run_trex`'s settings + scope)."""

    # scope (empty -> all indexed media)
    groups: list[str] | None = None
    sequences: list[str] | None = None
    entries: list[str] | None = None  # "group:sequence" pairs (":seq" or "seq" == empty group)
    # detection / conversion (part of the run_id identity)
    detect_model: str | None = None
    detect_type: str = "yolo"
    detect_conf_threshold: float = 0.5
    detect_iou_threshold: float = 0.1
    cm_per_pixel: float = 1.0
    meta_encoding: str = "gray"
    convert_extra_settings: dict[str, object] | None = None
    # tracking (part of the run_id identity)
    track_max_individuals: int = 1
    track_max_speed: float = 80.0
    track_max_reassign_time: float = 2.0
    track_trusted_probability: float = 0.1
    analysis_range: tuple[int, int] | None = None
    visual_identification_model_path: str | None = None
    auto_train: bool = False
    track_extra_settings: dict[str, object] | None = None
    # execution knobs -- throughput/behavior only, excluded from the run_id (and TREx's own
    # settings-dict hash already omits them, so this keeps params.json <-> run_id consistent).
    convert_to_tracks: Annotated[bool, HASH_EXCLUDE] = True
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    timeout: Annotated[int, HASH_EXCLUDE] = 600


@register_tracking_op
class TrexOp(TrackingOp[TrexParams]):
    """Run TREx (convert + track) over scoped videos, bridging results into ``tracks/``."""

    kind = "trex"
    category = "convert"
    resource_class: ClassVar[str] = "gpu"
    version = "0.1"
    Params = TrexParams

    def target(self, params: TrexParams) -> str:
        return "trex-track"

    def run(self, ds: Dataset, params: TrexParams, ctx: JobContext) -> str:
        # Heavy TREx imports (subprocess/opencv) stay inside run() so registration is light.
        from mosaic.tracking.trex.dataset_runs import run_trex

        entry_pairs = _parse_entries(params.entries)
        return run_trex(
            ds,
            ctx=ctx,  # run within the op's Job Contract -- no double-wrapping
            groups=params.groups,
            sequences=params.sequences,
            entries=entry_pairs or None,
            detect_model=params.detect_model,
            detect_type=params.detect_type,
            detect_conf_threshold=params.detect_conf_threshold,
            detect_iou_threshold=params.detect_iou_threshold,
            cm_per_pixel=params.cm_per_pixel,
            meta_encoding=params.meta_encoding,
            convert_extra_settings=params.convert_extra_settings,
            track_max_individuals=params.track_max_individuals,
            track_max_speed=params.track_max_speed,
            track_max_reassign_time=params.track_max_reassign_time,
            track_trusted_probability=params.track_trusted_probability,
            analysis_range=params.analysis_range,
            visual_identification_model_path=params.visual_identification_model_path,
            auto_train=params.auto_train,
            track_extra_settings=params.track_extra_settings,
            overwrite=params.overwrite,
            convert_to_tracks=params.convert_to_tracks,
            timeout=params.timeout,
            # conda-env / bin / display are environment (image) concerns, so left unset here;
            # the trex runner resolves them from MOSAIC_TREX_CONDA_ENV / _BIN / _DISPLAY. This
            # keeps the run_id independent of *where* it ran.
        )


def _parse_entries(entries: list[str] | None) -> list[tuple[str, str]]:
    """Parse ``["group:sequence", ...]`` into ``[(group, sequence), ...]`` (empty group ok)."""
    if not entries:
        return []
    pairs: list[tuple[str, str]] = []
    for item in entries:
        group, sep, sequence = item.partition(":")
        pairs.append((group, sequence) if sep else ("", group))
    return pairs
