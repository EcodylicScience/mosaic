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

from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.common.params import TrackerOpParams
from mosaic.core.pipeline.types import JsonValue
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


class TrexParams(TrackerOpParams):
    """Parameters for the ``trex`` tracking op (mirrors :func:`run_trex`'s settings + scope).

    **Every tool-facing parameter defaults to ``None``, meaning "do not send it".**
    mosaic declares no opinion about how TREx behaves; an unset parameter leaves
    TREx's own default in force, and setting one is unchanged.

    This is not cosmetic. Each of these used to carry a mosaic default, and *not
    one of them matched TREx's* -- so a caller who set nothing silently got
    mosaic's opinion with no way to decline it. ``detect_conf_threshold`` was
    five times stricter than TREx's 0.1; ``meta_encoding`` forced a grayscale
    ``.pv`` where TREx writes ``rgb8``; ``track_max_individuals`` tracked one
    animal against TREx's 1024. Two mattered beyond tuning:
    ``track_trusted_probability`` decides where a *tracklet ends*, and
    ``detect_iou_threshold`` has no numeric default at all -- TREx documents
    unset as preserving "the upstream model's default postprocessing behaviour"
    and set as possibly disabling end-to-end NMS-free inference, so a number
    here can move a YOLO26 detector off the inference path it was trained for.

    ``None`` reaches the argv builder, which omits the flag, so "unset" is
    expressed the whole way down rather than translated into a stand-in value.
    """

    # detection / conversion (part of the run_id identity)
    detect_model: str | None = None
    detect_type: str | None = None
    detect_conf_threshold: float | None = None
    detect_iou_threshold: float | None = None
    cm_per_pixel: float | None = None
    meta_encoding: str | None = None
    convert_extra_settings: dict[str, JsonValue] | None = None
    # tracking (part of the run_id identity)
    track_max_individuals: int | None = None
    track_max_speed: float | None = None
    track_max_reassign_time: float | None = None
    track_trusted_probability: float | None = None
    analysis_range: tuple[int, int] | None = None
    visual_identification_model_path: str | None = None
    auto_train: bool = False
    detect_keypoint_count: int | None = None
    """How many keypoints ``detect_model`` reports.

    Set it whenever that model is a **pose** model, or the tracks it produces
    carry no keypoints and nothing says so. TREx names ``poseX<i>``/``poseY<i>``
    itself from ``detect_keypoint_format``, which it learns by loading a model --
    and mosaic runs convert and track as two separate invocations, so the
    process doing the exporting has none. The value is refused from both the
    command line and a settings file, so naming the columns in ``output_fields``
    is the only route, and that is what mosaic does with this number.

    A *track* key, so setting it re-tracks and reuses the conversion.
    """
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

    def plan_identity(self, ds: Dataset, params: TrexParams) -> OpIdentity:
        """What a TREx run with these settings will be called.

        Both model references are resolved to an identity first, which is the
        same ordering ``run_trex`` follows and for the same reason: what the
        settings must carry is what a model *is*, because a bare weights path is
        a mutable key and swapping the file in place would let two different runs
        share one identifier.
        """
        from mosaic.tracking.common.mint import planned_model_id, tracker_identity
        from mosaic.tracking.trex.dataset_runs import trex_settings

        detect_model_id = (
            planned_model_id(
                ds,
                self.kind,
                [params.detect_model],
                _detect_model_kind(params.detect_model),
            )
            if params.detect_model is not None
            else None
        )
        vi_model_id = (
            planned_model_id(
                ds,
                self.kind,
                [params.visual_identification_model_path],
                "train-identity",
            )
            if params.visual_identification_model_path is not None
            else None
        )
        settings = trex_settings(
            detect_model=detect_model_id,
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
            visual_identification_model_path=vi_model_id,
            auto_train=params.auto_train,
            detect_keypoint_count=params.detect_keypoint_count,
            track_extra_settings=params.track_extra_settings,
        )
        return tracker_identity(self.kind, self.version, settings)

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
            detect_keypoint_count=params.detect_keypoint_count,
            track_extra_settings=params.track_extra_settings,
            overwrite=params.overwrite,
            convert_to_tracks=params.convert_to_tracks,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            # conda-env / bin / display are environment (image) concerns, so left unset here;
            # the trex runner resolves them from MOSAIC_TREX_CONDA_ENV / _BIN / _DISPLAY. This
            # keeps the run_id independent of *where* it ran.
        )


def _detect_model_kind(ref: str | None) -> str:
    """Which training op's index a detection-model reference resolves against.

    The kind comes from the reference itself, matching ``run_trex``: both
    ``train-pose`` and ``train-points`` produce runnable detection weights, and a
    run identifier resolves against the index its own training op wrote. A
    reference that is not a run identifier at all -- a bare weights path -- falls
    back rather than guessing, because the fallback only decides which spec reads
    the artifact and a path is read the same way either way.
    """
    from mosaic.core.pipeline.op_identity import parse_op_run_id

    parsed = parse_op_run_id(str(ref)) if ref else None
    return parsed.kind if parsed is not None else "train-points"
