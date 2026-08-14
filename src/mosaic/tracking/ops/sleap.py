"""SLEAP as a tracking op -- ``mosaic run --kind sleap``.

Wraps :func:`mosaic.tracking.sleap.run_sleap` as a registered ``Op`` so SLEAP rides
the schema-driven runner and every execution backend (local / rq / k8s) with
Pydantic param validation + discovery -- the same one-contract path TREx uses. The
implementation is unchanged: ``run_sleap`` still shells out to the ``sleap-track`` /
``sleap-convert`` console scripts in their own environment and hashes its *internal
settings dict* for the ``run_id`` (the op only re-routes the same call through a
``JobContext``).

``resource_class = "gpu"`` because SLEAP inference wants the GPU -- its ``category``
of ``"convert"`` would not imply that, so it declares the class explicitly and the
execution router sends it to the GPU lane / k8s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from mosaic.core.pipeline.ops import Op, OpIdentity, register_op
from mosaic.tracking.common.params import TrackerOpParams
from mosaic.core.pipeline.types import HASH_EXCLUDE, JsonValue
from mosaic.tracking.sleap.version import SLEAP_KIND, SLEAP_VERSION

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext


class SleapParams(TrackerOpParams):
    """Parameters for the ``sleap`` tracking op (mirrors :func:`run_sleap`'s settings + scope)."""

    # model: one external model directory, or two for top-down (centroid, then
    # centered-instance). Part of the run_id identity -- via a content digest of
    # the weights, never the paths themselves.
    model_paths: list[str]
    # tracking (part of the run_id identity)
    tracking: bool = True
    tracker: str = "flow"
    similarity: str = "instance"
    match: str = "hungarian"
    track_window: int = 5
    max_instances: int | None = None
    max_tracking: int | None = None
    peak_threshold: float = 0.2
    analysis_range: tuple[int, int] | None = None
    # JsonValue rather than object, so an unrepresentable value is rejected at
    # params construction (where pydantic names the field) instead of deep inside
    # hash_params. Every representable value still validates and none changes the
    # digest.
    sleap_extra_settings: dict[str, JsonValue] | None = None
    # execution knobs -- throughput/environment only, excluded from the run_id.
    batch_size: Annotated[int, HASH_EXCLUDE] = 4
    # cpu / cuda / a gpu index / None (auto). Where it ran, not what it produced.
    device: Annotated[str | None, HASH_EXCLUDE] = None


@register_op
class SleapOp(Op[SleapParams]):
    """Run SLEAP (infer + track) over scoped videos, bridging results into ``tracks/``."""

    kind = SLEAP_KIND
    category = "convert"
    domain = "tracking"
    resource_class: ClassVar[str] = "gpu"
    # Read from the integration rather than restated, so the op and the standalone
    # run_sleap cannot drift into naming the same run two ways.
    version = SLEAP_VERSION
    Params = SleapParams

    def target(self, params: SleapParams) -> str:
        return "sleap-track"

    def plan_identity(self, ds: Dataset, params: SleapParams) -> OpIdentity:
        """What a SLEAP run with these settings will be called.

        The model set resolves under the *training* kind rather than this
        tracker's, matching ``run_sleap``: a registered reference resolves
        against the index the training op wrote, and ``MODEL_KINDS`` declares
        ``train-sleap`` as SLEAP's own artifact shape for exactly this.
        """
        from mosaic.tracking.common.mint import planned_model_id, tracker_identity
        from mosaic.tracking.sleap.dataset_runs import sleap_settings
        from mosaic.tracking.sleap.version import TRAIN_SLEAP_KIND

        settings = sleap_settings(
            model_id=planned_model_id(
                ds, self.kind, list(params.model_paths), TRAIN_SLEAP_KIND
            ),
            tracking=params.tracking,
            tracker=params.tracker,
            similarity=params.similarity,
            match=params.match,
            track_window=params.track_window,
            max_instances=params.max_instances,
            max_tracking=params.max_tracking,
            peak_threshold=params.peak_threshold,
            analysis_range=params.analysis_range,
            sleap_extra_settings=params.sleap_extra_settings,
        )
        return tracker_identity(self.kind, self.version, settings)

    def run(self, ds: Dataset, params: SleapParams, ctx: JobContext) -> str:
        # Heavy SLEAP imports (subprocess/h5py) stay inside run() so registration is light.
        from mosaic.tracking.sleap.dataset_runs import run_sleap

        entry_pairs = params.entry_pairs()
        return run_sleap(
            ds,
            ctx=ctx,  # run within the op's Job Contract -- no double-wrapping
            model_paths=params.model_paths,
            groups=params.groups,
            sequences=params.sequences,
            entries=entry_pairs or None,
            tracking=params.tracking,
            tracker=params.tracker,
            similarity=params.similarity,
            match=params.match,
            track_window=params.track_window,
            max_instances=params.max_instances,
            max_tracking=params.max_tracking,
            peak_threshold=params.peak_threshold,
            analysis_range=params.analysis_range,
            sleap_extra_settings=params.sleap_extra_settings,
            batch_size=params.batch_size,
            device=params.device,
            convert_to_tracks=params.convert_to_tracks,
            overwrite=params.overwrite,
            idle_timeout=params.idle_timeout,
            max_runtime=params.max_runtime,
            # conda-env / bin are environment (image) concerns, left unset so the
            # runner resolves them from MOSAIC_SLEAP_CONDA_ENV / _BIN -- the run_id
            # stays independent of *where* it ran. ``device`` is passed but is
            # HASH_EXCLUDE, so cpu-vs-gpu selection likewise never enters the run_id.
        )
