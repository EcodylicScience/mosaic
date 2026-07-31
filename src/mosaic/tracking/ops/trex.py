"""TREx as a tracking op -- ``mosaic run --kind trex``.

Wraps :func:`mosaic.tracking.trex.run_trex` as a registered ``Op`` so TREx rides the
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

from typing import TYPE_CHECKING, ClassVar

from mosaic.core.pipeline.ops import Op, register_op
from mosaic.tracking.common.params import TrackerOpParams
from mosaic.core.pipeline.types import JsonValue
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


class TrexParams(TrackerOpParams):
    """Parameters for the ``trex`` tracking op (mirrors :func:`run_trex`'s settings + scope)."""

    # detection / conversion (part of the run_id identity)
    detect_model: str | None = None
    detect_type: str = "yolo"
    detect_conf_threshold: float = 0.5
    detect_iou_threshold: float = 0.1
    cm_per_pixel: float = 1.0
    meta_encoding: str = "gray"
    convert_extra_settings: dict[str, JsonValue] | None = None
    # tracking (part of the run_id identity)
    track_max_individuals: int = 1
    track_max_speed: float = 80.0
    track_max_reassign_time: float = 2.0
    track_trusted_probability: float = 0.1
    analysis_range: tuple[int, int] | None = None
    visual_identification_model_path: str | None = None
    auto_train: bool = False
    # JsonValue rather than object on both pass-through dictionaries: these are
    # the only params carrying arbitrary user values into the run_id, so an
    # unrepresentable value here is what identity_ready now rejects. Typing them
    # moves that failure to params construction, where pydantic names the field,
    # instead of deep inside hash_params with only a type name. Every value that
    # worked before still validates -- JsonValue is recursive, so nested dicts
    # and lists are fine -- and none of them changes the digest.
    track_extra_settings: dict[str, JsonValue] | None = None
    # execution knobs -- throughput/behavior only, excluded from the run_id (and TREx's own
    # settings-dict hash already omits them, so this keeps params.json <-> run_id consistent).


@register_op
class TrexOp(Op[TrexParams]):
    """Run TREx (convert + track) over scoped videos, bridging results into ``tracks/``."""

    kind = TREX_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    # Read from the integration rather than restated, so the op and the
    # standalone run_trex cannot drift into naming the same run two ways.
    version = TREX_VERSION
    Params = TrexParams

    def target(self, params: TrexParams) -> str:
        return "trex-track"

    def run(self, ds: Dataset, params: TrexParams, ctx: JobContext) -> str:
        # Heavy TREx imports (subprocess/opencv) stay inside run() so registration is light.
        from mosaic.tracking.trex.dataset_runs import run_trex

        entry_pairs = params.entry_pairs()
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
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            # conda-env / bin / display are environment (image) concerns, so left unset here;
            # the trex runner resolves them from MOSAIC_TREX_CONDA_ENV / _BIN / _DISPLAY. This
            # keeps the run_id independent of *where* it ran.
        )
