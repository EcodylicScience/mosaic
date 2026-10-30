"""Golden corpus for the identifiers minted by ``hash_params`` directly.

``test_feature_run_id_golden.py`` routes every case through ``compute_run_id``, so it
pins *feature* identity and nothing else. Six further identifier families call
``hash_params`` straight -- frame extraction, TREx, transcode, and the train /
convert / infer ops -- and were pinned by nothing at all. A change to ``hash_params``
or the serializer beneath it could move all six with a green suite, which is the
exact failure this corpus exists to prevent.

Organised by family rather than as a flat list, because the families do not
share a freeze rule:

**frames is FROZEN, permanently, in algorithm, width AND payload.** It is the
only mosaic identifier pinned outside mosaic: ``mosaic-api`` writes it to
``AnnotationFrame.run_id`` *and embeds it mid-string* in ``image_path``, on
Dolt-tracked rows carrying keypoint annotation labor, where ``image_path`` is
additionally a restorable value column. Moving it -- by widening the digest or
by adding a field to ``ExtractFramesParams`` that reaches ``identity_dump()`` --
orphans every annotated frame path, recoverable only by re-annotating. Item 1.2
folds ``Op.version`` into op run identity and must carve this op out.

The other families stay at 40 bits today and may be widened later, but only
inside their own minter, behind a scheme-marker bump, and riding an identity
shift that is already scheduled.

**What this pins, and what it does not.** Each case builds a *real* op
``Params`` object, so a change to the digest, to ``identity_dump()``, or to any
params field moves a line here. It does **not** pin the dict literal each op
wraps its params in at the mint site (``{"params": ..., "data": ..., "base":
...}`` in ``ops/train.py:150-158``, and the equivalents in ``convert.py`` and
``infer.py``): those are built inline inside op run bodies rather than in a
named function, so there is nothing to call. Extracting them is follow-up work,
and item 1.2 needs it anyway.

Regenerating after a deliberate change::

    MOSAIC_UPDATE_GOLDEN=1 pytest tests/core/pipeline/test_op_identity_golden.py

Then read the diff: every moved line must be explained by the change that moved
it, and a moved ``frames/`` line is a bug, not a diff to accept.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from mosaic_media import CHROME_149
from mosaic_media.transcode import ANALYSIS_ENCODING

from mosaic.core.annotations.keypoint_map import KeypointMap, resolve_keypoint_map
from mosaic.core.annotations.model import KeypointSchema
from mosaic.core.pipeline._utils import hash_params
from mosaic.tracking.frame_extraction.dataset_runs import (
    ExtractFramesParams,
    frames_run_id,
)
from mosaic.tracking.frame_extraction.sampling import SamplingMethod
from mosaic.core.pipeline.composition import (
    MediaMember,
    SourceMember,
    labels_raw_composition,
    media_composition,
    tracks_raw_composition,
)
from mosaic.core.pipeline.labels_identity import (
    label_convert_variant_payload,
    label_converter_op,
    labels_run_id,
)
from mosaic.core.media.preprocess import ClaheStep, CropStep, TrimStep
from mosaic.core.pipeline.media_input import MediaInputParams
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.preprocess import PreprocessParams, preprocess_identity
from mosaic.core.pipeline.transcode import (
    TranscodeParams,
    transcode_recipe_hash,
    transcode_run_id,
)
from mosaic.core.pipeline.tracks_identity import (
    convert_variant_payload,
    converter_op,
    infer_variant_payload,
    tracks_run_id,
    tracker_variant_payload,
)
from mosaic.core.params import Params
from mosaic.media_probe_config import media_thresholds
from mosaic.tracking import register_ops
from mosaic.tracking.frame_predictions.params import PredictFramesParams
from mosaic.tracking.keypoint_sets import ResolvedSet
from mosaic.tracking.litpose.version import LITPOSE_KIND, LITPOSE_VERSION
from mosaic.tracking.ops.convert import convert_points_run_id
from mosaic.tracking.ops.infer import infer_run_id
from mosaic.tracking.ops.predict_frames import predict_frames_run_id
from mosaic.tracking.ops.prepare import (
    PrepareTrainingDataParams,
    prepare_training_data_run_id,
)
from mosaic.tracking.ops.prepare_embedding import (
    PrepareEmbeddingDataParams,
    prepare_embedding_data_run_id,
)
from mosaic.tracking.ops.score_predictions import (
    ScorePredictionsParams,
    score_predictions_run_id,
)
from mosaic.tracking.ops.train import train_run_id
from mosaic.tracking.sleap.version import SLEAP_KIND, SLEAP_VERSION
from mosaic.tracking.trex.version import TREX_KIND, TREX_TRACK_VERSION
from mosaic.tracking.ultralytics_track.version import (
    ULTRALYTICS_KIND,
    ULTRALYTICS_VERSION,
)

from tests.helpers import (
    UPDATE_GOLDEN_ENV,
    read_string_golden,
    regenerate_command,
    updating_golden,
    write_golden,
)

GOLDEN = "op_identity_golden.json"
"""``case id -> digest``. A plain map so the diff stays readable."""

# The op registry is populated by an explicit call, not by importing the
# package, so the params classes below are unreachable without it.
register_ops()


@dataclass(frozen=True)
class OpCase:
    """One op-params digest, fully specified.

    Attributes:
        case_id: Stable key into the golden file, prefixed with its family.
        params: A real op ``Params`` instance. Values are fixed literals and
            every path is relative, so the digest does not vary by machine.
        frozen: True when this family's identifier may never move. Asserted
            separately, so the reason survives as an executable claim rather
            than a comment.
    """

    case_id: str
    params: Params
    frozen: bool = False


def _op_params(kind: str, /, **values: object) -> Params:
    """Build the registered ``Params`` for *kind*.

    Going through the registry rather than importing each class keeps the corpus
    honest: a params class that moves module is still found, and an op removed
    from the registry fails here instead of silently losing coverage.
    """
    return OPS[kind].Params(**values)


def _media_op_params(kind: str, /, **values: object) -> MediaInputParams:
    """Build the registered ``Params`` for *kind*, an op that takes ``media``."""
    params = _op_params(kind, **values)
    assert isinstance(params, MediaInputParams)
    return params


MEDIA_VARIANT = "preprocess.0.1-aaaaaaaaaa"
"""The media variant that every ``*-media`` case reads.

``media`` is ``HASH_EXCLUDE``, and an ``OpCase`` digest over ``identity_dump()``
omits it. Each consumer folds it in where it mints, and the function cases below
call those minters.
"""


_PREPARE_EMBEDDING_PARAMS = _op_params(
    "prepare-embedding-data",
    sleap_run_id="sleap.1.6-abcdef0123",
    train_entries=[{"group": "g", "sequence": "s2"}, {"group": "g", "sequence": "s1"}],
    val_entries=[{"group": "g", "sequence": "s3"}],
)
"""The prepare-embedding-data params of the digest case and the run-identifier case.

The training entries are given out of order, and the params store them sorted.
"""


_CONVERT_POINTS_PARAMS = _op_params(
    "convert-points",
    cvat_xml="annotations/points.xml",
    images_dir="annotations/images",
    class_names=["bee", "feeder"],
    radii={"bee": 10.0, "feeder": 30.0},
)
"""The convert-points params of both the digest case and the run-identifier case."""


OP_CASES: tuple[OpCase, ...] = (
    # --- frames: FROZEN, see the module docstring -----------------------------
    OpCase(
        case_id="frames/uniform-100",
        params=_op_params("extract-frames", n_frames=100, method="uniform"),
        frozen=True,
    ),
    OpCase(
        case_id="frames/kmeans-20",
        params=_op_params("extract-frames", n_frames=20, method="kmeans"),
        frozen=True,
    ),
    # --- trex -----------------------------------------------------------------
    OpCase(case_id="trex/defaults", params=_op_params("trex")),
    OpCase(
        case_id="trex/max-individuals",
        params=_op_params("trex", track_max_individuals=4, cm_per_pixel=0.05),
    ),
    # --- sleap ----------------------------------------------------------------
    OpCase(
        case_id="sleap/single-model",
        params=_op_params("sleap", model_paths=["models/sleap_bottomup"]),
    ),
    OpCase(
        case_id="sleap/top-down",
        params=_op_params(
            "sleap",
            model_paths=["models/sleap_centroid", "models/sleap_instance"],
            use_flow=False,
            peak_threshold=0.3,
        ),
    ),
    # --- litpose --------------------------------------------------------------
    OpCase(
        case_id="litpose/single-model",
        params=_op_params("litpose", model_path="models/litpose_model"),
    ),
    OpCase(
        case_id="litpose/with-overrides",
        params=_op_params(
            "litpose",
            model_path="models/litpose_model",
            litpose_overrides={"data.image_resize_dims.height": 256},
        ),
    ),
    # --- transcode params (the recipe itself is pinned below) -----------------
    OpCase(
        case_id="transcode/analysis-entry",
        params=_op_params("transcode", target="analysis"),
    ),
    # --- train ----------------------------------------------------------------
    OpCase(
        case_id="train-pose/data",
        params=_op_params("train-pose", data="datasets/pose/data.yaml", epochs=10),
    ),
    OpCase(
        case_id="train-points/data",
        params=_op_params("train-points", data="datasets/points/data.yaml", epochs=10),
    ),
    OpCase(
        case_id="train-localizer/dataset-dir",
        params=_op_params("train-localizer", dataset_dir="datasets/localizer"),
    ),
    OpCase(
        case_id="train-sleap/labels",
        params=_op_params(
            "train-sleap",
            labels="labels/session.slp",
            head="centered_instance",
            backbone="unet",
            max_epochs=10,
            seed=7,
            validation_fraction=0.2,
        ),
    ),
    OpCase(
        case_id="train-sleap/embedding",
        params=_op_params(
            "train-sleap",
            labels="prepare-embedding-data.0.1-0123456789",
            head="embedding",
            backbone="convnext",
            max_epochs=10,
            seed=7,
            validation_fraction=0.2,
        ),
    ),
    OpCase(
        case_id="train-litpose/project",
        params=_op_params(
            "train-litpose",
            project="projects/mice",
            base_config="configs/litpose_default.yaml",
            model_type="heatmap",
            backbone="resnet50_animal_ap10k",
            max_epochs=10,
        ),
    ),
    OpCase(
        case_id="train-sleap/with-overrides",
        params=_op_params(
            "train-sleap",
            labels="labels/session.slp",
            head="centroid",
            backbone="convnext",
            max_epochs=10,
            seed=7,
            validation_fraction=0.2,
            sleap_overrides={"trainer_config.optimizer.lr": 0.0005},
        ),
    ),
    # --- convert --------------------------------------------------------------
    OpCase(case_id="convert-points/cvat", params=_CONVERT_POINTS_PARAMS),
    OpCase(case_id="prepare-embedding-data/default", params=_PREPARE_EMBEDDING_PARAMS),
    OpCase(
        case_id="resample-tracks/30fps",
        params=_op_params("resample-tracks", target_fps=30.0),
    ),
    OpCase(
        case_id="resample-tracks/30fps-prefiltered",
        params=_op_params("resample-tracks", target_fps=30.0, prefilter=40.0),
    ),
    # --- infer ----------------------------------------------------------------
    OpCase(
        case_id="infer-pose/model",
        params=_op_params("infer-pose", model="models/pose/best.pt"),
    ),
    OpCase(
        case_id="infer-points/model",
        params=_op_params("infer-points", model="models/points/best.pt"),
    ),
    OpCase(
        case_id="infer-localizer/model",
        params=_op_params("infer-localizer", model="models/localizer/best.pt"),
    ),
    # --- ultralytics ----------------------------------------------------------
    # The settings-based cases below pin what ``ultralytics_settings`` builds.
    # These two pin the params model's own hashed field set, which is what an
    # added or retyped field moves.
    OpCase(
        case_id="ultralytics/model",
        params=_op_params("ultralytics", model_path="models/pose/best.pt"),
    ),
    # An unset ``tracker_overrides`` hashes as null whatever its declared type,
    # so the case above cannot see its shape. This one hashes the resolved
    # table -- every setting of the chosen backend, not the two words the caller
    # wrote -- which is what ``_op_digest`` hashes through the generic
    # ``identity_dump()``.
    OpCase(
        case_id="ultralytics/tracker-overrides",
        params=_op_params(
            "ultralytics",
            model_path="models/pose/best.pt",
            tracker="botsort",
            tracker_overrides={"track_buffer": 90},
        ),
    ),
    # --- select-frames --------------------------------------------------------
    OpCase(
        case_id="select-frames/default",
        params=_op_params("select-frames", source_tracks_run_id="trex.0.1-abcdef0123"),
    ),
    OpCase(
        case_id="select-frames/exclude",
        params=_op_params(
            "select-frames",
            source_tracks_run_id="trex.0.1-abcdef0123",
            exclude=[
                {"group": "g", "sequence": "s1", "indices": [10, 20]},
                {"group": "g", "sequence": "s2", "indices": [5]},
            ],
        ),
    ),
    OpCase(
        case_id="select-frames/thresholds",
        params=_op_params(
            "select-frames",
            source_tracks_run_id="trex.0.1-abcdef0123",
            jump={"body_lengths": 1.2},
            low_score={"threshold": 0.3},
        ),
    ),
)

# The two transcode identifiers are minted by named, importable functions, so
# unlike the op ids above these pin more than the digest: the recipe hash pins
# the payload construction behind it, and the run id pins that the recipe is
# namespaced rather than hashed a second time.
_RECIPE_PARAMS = TranscodeParams(target="analysis")


def _recipe_hash() -> str:
    return transcode_recipe_hash(
        _RECIPE_PARAMS, ANALYSIS_ENCODING, CHROME_149, media_thresholds()
    )


def _run_id() -> str:
    # The recipe namespaced, with no second digest over it: a regression that
    # re-hashed the recipe, or folded a term back in, would move this line.
    return transcode_run_id(_recipe_hash())


# The three tracks-variant identifiers. Function cases rather than OpCases,
# because a track converter lives in TRACK_CONVERTERS rather than OPS and because
# what has to be pinned is the *payload wrapper* each producer builds, not just
# the digest: renaming a key inside one of these would move every variant
# directory on disk, and a corpus that called ``tracks_run_id`` with a hand-built
# payload would stay green through it.


def _convert_variant() -> str:
    """A conversion variant, minted from the converter's own version and params.

    Both read from the class rather than written here as literals. A hard-coded
    ``"0.1"`` and a hand-built ``{"neck_idx": 3, "tail_idx": 6}`` meant this case
    pinned a recipe that could not occur -- those params no longer exist -- and,
    worse, that a converter version bump moved nothing anybody would notice. A
    version is a visible segment of the identifier, so bumping one *should* show
    up here.
    """
    from mosaic.core.track_library.calms21 import Calms21Converter

    return tracks_run_id(
        converter_op("calms21_npy"),
        Calms21Converter.version,
        convert_variant_payload(Calms21Converter.Params().identity_dump()),
    )


def _trex_variant() -> str:
    # The tracker's own settings, passed through unwrapped -- so this value is
    # byte-identical to trex_run_id(settings) for the same settings.
    return tracks_run_id(
        TREX_KIND,
        TREX_TRACK_VERSION,
        tracker_variant_payload({"track_max_individuals": 4, "cm_per_pixel": 0.5}),
    )


def _infer_variant() -> str:
    return tracks_run_id(
        "infer-points",
        "0.1",
        infer_variant_payload({"conf": 0.5}, "train-points.0.1-aaaaaaaaaa"),
    )


def _infer_variant_on_media() -> str:
    return tracks_run_id(
        "infer-points",
        "0.1",
        infer_variant_payload(
            {"conf": 0.5}, "train-points.0.1-aaaaaaaaaa", media=MEDIA_VARIANT
        ),
    )


def _resample_variant() -> str:
    """A re-gridded variant: the resampling params, plus the source it chains from.

    The only production caller of ``tracks_run_id``'s ``upstream`` term, so this
    is what pins that the term is *added* rather than folded into the payload.
    The two are not interchangeable: an ``upstream`` inside the payload would
    make one resampling recipe unrecognisable as itself across two sources, and a
    corpus built from a hand-written payload would not notice the difference.
    """
    from mosaic.core.pipeline.tracks_identity import chained_variant_payload
    from mosaic.tracking.ops.resample import ResampleTracksOp

    return tracks_run_id(
        ResampleTracksOp.kind,
        ResampleTracksOp.version,
        chained_variant_payload(
            ResampleTracksOp.Params(target_fps=30.0).identity_dump()
        ),
        upstream="convert-trex_npz.0.2-aaaaaaaaaa",
    )


def _resample_variant_other_upstream() -> str:
    """The same recipe over a different source, which must be a different variant.

    Asserted as a distinct golden entry rather than as an inequality, so the pair
    also pins *which* of the two moves if the term is ever dropped.
    """
    from mosaic.core.pipeline.tracks_identity import chained_variant_payload
    from mosaic.tracking.ops.resample import ResampleTracksOp

    return tracks_run_id(
        ResampleTracksOp.kind,
        ResampleTracksOp.version,
        chained_variant_payload(
            ResampleTracksOp.Params(target_fps=30.0).identity_dump()
        ),
        upstream="trex.0.4-bbbbbbbbbb",
    )


def _assign_arenas_variant() -> str:
    """An arena-assigned variant: the op's params, plus the source it chains from.

    The layouts are not in it. Each row records its layout's digest in its
    composition, so a changed layout recomputes under the same variant.
    """
    from mosaic.core.pipeline.tracks_identity import chained_variant_payload
    from mosaic.tracking.ops.assign_arenas import AssignArenasOp

    return tracks_run_id(
        AssignArenasOp.kind,
        AssignArenasOp.version,
        chained_variant_payload(AssignArenasOp.Params().identity_dump()),
        upstream="convert-trex_npz.0.2-aaaaaaaaaa",
    )


def _trex_run_id_settings(media: str = "") -> str:
    """The identifier ``trex_settings``' *key set* mints, pinned.

    Nothing else pins it. The ``trex/*`` op cases come from ``TrexParams``, and
    the variant case above passes a hand-built two-key dict -- so a rename
    inside ``trex_settings`` moves every TREx run root and tracks variant on
    disk with a fully green suite. This is the case that fails instead.
    """
    from mosaic.tracking.trex.dataset_runs import trex_run_id

    return trex_run_id(_trex_settings(media))


def _trex_conversion_run_id() -> str:
    """The conversion cache's identifier for the settings of ``trex/run-id-settings``.

    It addresses every cached ``.pv``. Its version is apart from the tracking
    run's, so a change to tracking or publication re-tracks from the cached
    conversions, and this line does not move with the tracking run's.
    """
    from mosaic.tracking.trex.conversion_cache import conversion_run_id
    from mosaic.tracking.trex.dataset_runs import phase_settings

    return conversion_run_id(phase_settings(_trex_settings(), "convert"))


def _trex_settings(media: str = "") -> dict[str, object]:
    """The settings ``trex_settings`` builds from a fully stated ``TrexParams``.

    Every field is set explicitly rather than relying on defaults, because a
    default that changed would move the identifiers minted from these for a
    reason unrelated to the key set they guard. Both model references arrive as
    the identities the caller resolved them to, which the settings record.
    """
    from mosaic.tracking.trex.dataset_runs import trex_settings
    from mosaic.tracking.trex.params import TrexParams

    params = TrexParams(
        detect_model="train-points.0.1-aaaaaaaaaa",
        detect_type="points",
        detect_conf_threshold=0.25,
        detect_iou_threshold=0.7,
        cm_per_pixel=0.5,
        meta_encoding="gray",
        convert_extra_settings=None,
        track_max_individuals=4,
        track_max_speed=50.0,
        track_max_reassign_time=0.5,
        track_trusted_probability=0.5,
        analysis_range=None,
        visual_identification_model_path="train-identity.0.1-bbbbbbbbbb",
        auto_train=False,
        detect_keypoint_count=7,
        track_extra_settings=None,
        media=media,
    )
    return trex_settings(
        params,
        detect_model_id="train-points.0.1-aaaaaaaaaa",
        vi_model_id="train-identity.0.1-bbbbbbbbbb",
    )


def _sleap_variant() -> str:
    # The tracker's own settings, passed through unwrapped -- so this value is
    # byte-identical to sleap_run_id(settings) for the same settings. The model
    # term is a content digest (a literal here), never a path.
    return tracks_run_id(
        SLEAP_KIND,
        SLEAP_VERSION,
        tracker_variant_payload(
            {"model": "0123456789abcdef", "tracker": "flow", "peak_threshold": 0.2}
        ),
    )


def _litpose_variant() -> str:
    # The integration's own settings, passed through unwrapped -- so this value is
    # byte-identical to litpose_run_id(settings) for the same settings. The model
    # term is a content digest (a literal here), never a path.
    return tracks_run_id(
        LITPOSE_KIND,
        LITPOSE_VERSION,
        tracker_variant_payload(
            {"model": "0123456789abcdef", "litpose_overrides": None}
        ),
    )


def _sleap_run_id_settings(media: str = "") -> str:
    """The identifier ``sleap_settings``' *key set* mints, pinned.

    The SLEAP counterpart of :func:`_trex_run_id_settings`, and it exists for the
    same reason: the ``sleap/*`` op cases come from ``SleapParams``, and
    ``_sleap_variant`` passes a hand-built three-key dict, so a rename inside
    ``sleap_settings`` moves every SLEAP run root and tracks variant on disk with
    a fully green suite.

    ``tracking`` is True because the tracker knobs are dropped from identity when
    it is False -- the True case is the one that carries every key. The model
    reference on the params is not read here: the payload records the identity
    the caller resolved it to, passed beside it.
    """
    from mosaic.tracking.sleap.dataset_runs import sleap_run_id, sleap_settings
    from mosaic.tracking.sleap.params import SleapParams

    params = SleapParams(
        model_paths=["models/centroid"],
        tracking=True,
        use_flow=True,
        candidates_method="local_queues",
        features="keypoints",
        scoring_method="oks",
        track_matching_method="hungarian",
        tracking_window_size=5,
        max_tracks=4,
        max_instances=4,
        peak_threshold=0.2,
        analysis_range=None,
        sleap_extra_settings=None,
        media=media,
    )
    return sleap_run_id(sleap_settings(params, model_id="0123456789abcdef"))


def _sleap_run_id_settings_embedding() -> str:
    """The identifier that ``sleap_settings`` mints for an embedding model set.

    An embedding run passes the same keys with its tracker values, so this pins
    those values as :func:`_sleap_run_id_settings` pins the key set.
    """
    from mosaic.tracking.sleap.dataset_runs import sleap_run_id, sleap_settings
    from mosaic.tracking.sleap.params import SleapParams

    params = SleapParams(
        model_paths=["models/bottomup", "models/embedding"],
        tracking=True,
        use_flow=False,
        candidates_method="local_queues",
        features="embeddings",
        scoring_method="cosine_sim",
        track_matching_method="hungarian",
        tracking_window_size=5,
        max_tracks=4,
        max_instances=4,
        peak_threshold=0.2,
        analysis_range=None,
        sleap_extra_settings={"link_tracklets": True},
    )
    return sleap_run_id(sleap_settings(params, model_id="0123456789abcdef"))


def _litpose_run_id_settings(media: str = "") -> str:
    """The identifier ``litpose_settings``' *key set* mints, pinned.

    The Lightning Pose counterpart of :func:`_trex_run_id_settings`. Its settings
    dict is the smallest of the three, which makes it the easiest to rename a key
    in without noticing.
    """
    from mosaic.tracking.litpose.dataset_runs import litpose_run_id, litpose_settings
    from mosaic.tracking.litpose.params import LitposeParams

    params = LitposeParams(
        model_path="models/litpose_model", litpose_overrides=None, media=media
    )
    return litpose_run_id(litpose_settings(params, model_id="0123456789abcdef"))


# The per-sequence composition hashes (item 4.4). Function cases for the same
# reason the tracks variants above are: what has to be pinned is the payload
# *wrapper*, because renaming a key inside one would move every stored
# composition and a corpus that called ``hash_params`` with a hand-built dict
# would stay green straight through it. The reordered case is beside the ordered
# one deliberately -- a pair that agreed would be the defect, visible in the data
# file rather than only in a test.


def _media_single_camera() -> str:
    return media_composition(
        [
            MediaMember(camera="", video_order=0, uid="uid-a"),
            MediaMember(camera="", video_order=1, uid="uid-b"),
        ]
    ).digest


def _media_reordered() -> str:
    return media_composition(
        [
            MediaMember(camera="", video_order=0, uid="uid-b"),
            MediaMember(camera="", video_order=1, uid="uid-a"),
        ]
    ).digest


def _media_two_cameras() -> str:
    return media_composition(
        [
            MediaMember(camera="camB", video_order=0, uid="uid-b"),
            MediaMember(camera="camA", video_order=0, uid="uid-a"),
        ]
    ).digest


def _media_empty() -> str:
    return media_composition([]).digest


# train_run_id, infer_run_id and trex_settings mint identifiers that nothing in
# either corpus pinned -- the module docstring already confessed the gap. Item
# 4.6 changes what their model term *means*, which is the moment to close it.
# Literal model ids, so these stay filesystem-free.


def _train_run_id() -> str:
    # The version comes from the registry for the reason ``_op_params`` does: it
    # is a visible segment of the identifier, so a bump belongs in this file's
    # diff rather than leaving a literal here describing what the op used to mint.
    # The *base* stays a literal 0.1 id -- fine-tuning from an older model is a
    # real case, and the term is opaque to the minter either way.
    return train_run_id(
        "train-pose",
        OPS["train-pose"].version,
        _op_params("train-pose", data="d.yaml", epochs=3),
        "deadbeefcafe",
        "train-pose.0.1-aaaaaaaaaa",
    )


def _train_run_id_from_a_bare_path() -> str:
    # The population item 4.6 moves: a base with no run to name it now
    # contributes its weights digest where it used to contribute "".
    return train_run_id(
        "train-pose",
        OPS["train-pose"].version,
        _op_params("train-pose", data="d.yaml", epochs=3),
        "deadbeefcafe",
        "0123456789abcdef",
    )


def _train_sleap_run_id() -> str:
    # The registered version, for the reason ``_train_run_id`` gives: a version
    # bump changes this golden line.
    return train_run_id(
        "train-sleap",
        OPS["train-sleap"].version,
        _op_params("train-sleap", labels="labels/session.slp", max_epochs=10),
        "deadbeefcafe",
        "",
    )


def _prepared_sets() -> tuple[ResolvedSet, ...]:
    # Two sets from two datasets, given out of order on purpose: the minter sorts
    # them, so the order sets are named in is not a difference.
    return (
        ResolvedSet(
            origin_uuid="uuid-b",
            set_key="21-rats",
            revision=4,
            digest="bbbbbbbbbbbbbbbb",
            payload=Path("labels_raw/keypoints/21-rats/rev4/annotations.mosaic.json"),
        ),
        ResolvedSet(
            origin_uuid="uuid-a",
            set_key="17-mice",
            revision=12,
            digest="aaaaaaaaaaaaaaaa",
            payload=Path("labels_raw/keypoints/17-mice/rev12/annotations.mosaic.json"),
        ),
    )


def _prepare_run_id() -> str:
    # The references name no revision here and one in the case below. Both select
    # the same content, and a selector is not content, so the two are one run.
    return prepare_training_data_run_id(
        "prepare-training-data",
        OPS["prepare-training-data"].version,
        _op_params(
            "prepare-training-data",
            sets=[{"set_key": "17-mice"}, {"set_key": "21-rats"}],
        ),
        _prepared_sets(),
    )


def _prepare_run_id_with_revisions_named() -> str:
    return prepare_training_data_run_id(
        "prepare-training-data",
        OPS["prepare-training-data"].version,
        _op_params(
            "prepare-training-data",
            sets=[
                {"set_key": "21-rats", "revision": 4},
                {"set_key": "17-mice", "revision": 12},
            ],
        ),
        tuple(reversed(_prepared_sets())),
    )


def _prepare_run_id_with_a_holdout() -> str:
    # The hold-out enters by its split units, given out of order: the minter
    # sorts them, and the references in holdout leave the parameter dump.
    params = _op_params(
        "prepare-training-data",
        sets=[{"set_key": "17-mice"}, {"set_key": "21-rats"}],
        holdout=[{"set_key": "33-held"}],
    )
    assert isinstance(params, PrepareTrainingDataParams)
    return prepare_training_data_run_id(
        "prepare-training-data",
        OPS["prepare-training-data"].version,
        params,
        _prepared_sets(),
        holdout_units=("uuid-a::m07", "uuid-a::m03"),
    )


def _convert_points_run_id() -> str:
    # The digest case above omits the version. This one mints through the op's
    # minter with the registered version. A version bump changes its golden line.
    return convert_points_run_id(
        "convert-points",
        OPS["convert-points"].version,
        _CONVERT_POINTS_PARAMS,
        "deadbeefcafe",
        "0123456789abcdef",
    )


def _prepare_embedding_run_id() -> str:
    assert isinstance(_PREPARE_EMBEDDING_PARAMS, PrepareEmbeddingDataParams)
    return prepare_embedding_data_run_id(_PREPARE_EMBEDDING_PARAMS)


def _infer_run_id() -> str:
    return infer_run_id(
        "infer-points",
        "0.1",
        _media_op_params("infer-points", model="m.pt"),
        "train-points.0.1-aaaaaaaaaa",
    )


def _infer_run_id_on_media(kind: str, version: str, model_id: str) -> Callable[[], str]:
    """Return the identifier that inference op *kind* at *version* mints over a variant.

    *version* is a literal, as in :func:`_infer_run_id`. Bumping an op's version
    cannot change the digest that this pins.
    """

    def mint() -> str:
        return infer_run_id(
            kind,
            version,
            _media_op_params(kind, model="m.pt", media=MEDIA_VARIANT),
            model_id,
        )

    return mint


def _tracks_raw_two_files() -> str:
    return tracks_raw_composition(
        [
            SourceMember(name="a.npy", digest="digest-a", algo="md5"),
            SourceMember(name="b.npy", digest="digest-b", algo="md5"),
        ]
    ).digest


def _labels_convert_variant() -> str:
    # The label variant payload wrapper, pinned for the same reason the tracks one
    # is: renaming "kind" or "params" here would move every label variant on disk.
    return labels_run_id(
        label_converter_op("calms21_npy"),
        "0.1",
        label_convert_variant_payload("behavior", {"resident_id": 0, "intruder_id": 1}),
    )


def _labels_raw_two_files() -> str:
    # Byte-identical members to composition/tracks-raw-two-files, and it MUST mint
    # a different digest: source_composition_payload separates the two roots by
    # kind, so a change under one root cannot read as the other.
    return labels_raw_composition(
        [
            SourceMember(name="a.npy", digest="digest-a", algo="md5"),
            SourceMember(name="b.npy", digest="digest-b", algo="md5"),
        ]
    ).digest


# The frame-extraction identifier, minted through its own function rather than
# recomputed here. The OpCase above pins the *payload* -- a field added to
# ``ExtractFramesParams`` moves it -- but nothing pinned ``frames_run_id``
# itself, so a term added inside the minter moved the one identifier this file
# calls permanently frozen while every line stayed green. Item 6.4's revision
# term is exactly such a term, which is what made the gap worth closing.
#
# The revision case sits beside the default one deliberately: a pair that agreed
# would be the defect, and it is visible in the data file rather than only here.


def _frames_run_id() -> str:
    return frames_run_id("uniform", ExtractFramesParams(n_frames=100))


def _frames_run_id_revised() -> str:
    return frames_run_id("uniform", ExtractFramesParams(n_frames=100, revision=1))


def _frames_run_id_list() -> str:
    """A ``list`` run is named by its frames, through the same minter."""
    params = ExtractFramesParams.model_validate(
        {
            "method": "list",
            "frames": [
                {"sequence": "seq_b", "indices": [40, 7]},
                {"group": "g", "sequence": "seq_a", "indices": [0, 1745, 3487]},
            ],
        }
    )
    return frames_run_id(params.method, params)


def _frames_run_id_exclude(method: SamplingMethod) -> str:
    """A sampling run that excludes frames is named by its exclude and spacing."""
    params = ExtractFramesParams.model_validate(
        {
            "method": method,
            "n_frames": 20,
            "exclude": [
                {"sequence": "seq_b", "indices": [40, 7]},
                {"group": "g", "sequence": "seq_a", "indices": [0, 1745, 3487]},
            ],
            "exclude_spacing": 5,
        }
    )
    return frames_run_id(params.method, params)


def _frames_run_id_exclude_kmeans() -> str:
    return _frames_run_id_exclude("kmeans")


def _frames_run_id_exclude_uniform() -> str:
    return _frames_run_id_exclude("uniform")


def _ultralytics_settings_case(media: str = "") -> dict[str, object]:
    """The one settings dict both Ultralytics golden cases are built from.

    Shared rather than spelled twice, so the run-identifier case and the tracks
    variant case are provably one payload -- which is the property the two are
    there to pin.
    """
    from mosaic.tracking.ultralytics_track.dataset_runs import ultralytics_settings
    from mosaic.tracking.ultralytics_track.params import UltralyticsParams
    from mosaic.tracking.ultralytics_track.tracker_defaults import (
        BotsortConfig,
        resolve_tracker_config,
    )

    params = UltralyticsParams(
        # Every field ``ultralytics_settings`` reads is explicit rather than
        # left to a default, so a changed default moves nothing here for a
        # reason unrelated to the key set this case exists to guard.
        model_path="models/pose/best.pt",
        task="pose",
        tracker="botsort",
        tracker_overrides=BotsortConfig(track_buffer=90),
        conf=0.1,
        iou=0.7,
        imgsz=640,
        max_det=300,
        classes=None,
        agnostic_nms=False,
        start_frame=0,
        end_frame=None,
        frame_step=1,
        media=media,
    )
    return ultralytics_settings(
        params,
        model_id="0123456789abcdef",
        # Built rather than spelled as a literal, so this pins the merge and the
        # declared defaults as well as the key set.
        tracker_config=resolve_tracker_config("botsort", {"track_buffer": 90}),
    )


def _ultralytics_run_id_settings(media: str = "") -> str:
    """The identifier ``ultralytics_settings``' *key set* mints, pinned."""
    from mosaic.tracking.ultralytics_track.dataset_runs import ultralytics_run_id

    return ultralytics_run_id(_ultralytics_settings_case(media))


def _ultralytics_variant() -> str:
    # Passed through unwrapped, so this is byte-identical to
    # ultralytics_run_id(settings) for the same settings.
    return tracks_run_id(
        ULTRALYTICS_KIND,
        ULTRALYTICS_VERSION,
        tracker_variant_payload(_ultralytics_settings_case()),
    )


def _ultralytics_tracker_defaults() -> str:
    """All six declared default tables in one digest.

    ``run-id-settings`` pins the merge for one backend only, so a transcription
    slip in the other five would be invisible to the corpus. This is what makes
    the whole table a deliberate edit.
    """
    from mosaic.core.pipeline._utils import hash_params
    from mosaic.tracking.ultralytics_track.tracker_defaults import (
        TRACKER_NAMES,
        resolve_tracker_config,
    )

    return hash_params(
        {name: resolve_tracker_config(name, None) for name in TRACKER_NAMES}
    )


# The media variant identifier is minted by the function that the op plans with.
# Each case names its steps with explicit values, so that a changed step default
# does not change a digest here for a reason unrelated to the payload.
_PREPROCESS_CROP = CropStep(x=120, y=40, width=320, height=240)
_PREPROCESS_TRIM = TrimStep(start=100, stop=400)
_PREPROCESS_CLAHE = ClaheStep(clip_limit=2.0, tile_grid_size=8)


def _preprocess_one_step() -> str:
    return preprocess_identity(PreprocessParams(steps=[_PREPROCESS_CROP])).run_id


def _preprocess_three_steps() -> str:
    params = PreprocessParams(
        steps=[_PREPROCESS_CROP, _PREPROCESS_TRIM, _PREPROCESS_CLAHE]
    )
    return preprocess_identity(params).run_id


def _preprocess_upstream() -> str:
    params = PreprocessParams(
        steps=[_PREPROCESS_CLAHE], media="preprocess.0.1-aaaaaaaaaa"
    )
    return preprocess_identity(params).run_id


def _preprocess_labeled_rate() -> str:
    params = PreprocessParams(steps=[_PREPROCESS_CROP], fps=15.0)
    return preprocess_identity(params).run_id


def _preprocess_h264() -> str:
    params = PreprocessParams(steps=[_PREPROCESS_CROP], codec="h264")
    return preprocess_identity(params).run_id


def _predict_frames_run_id(
    model_kind: str, model_id: str, model: list[str], **values: object
) -> str:
    """The identifier of a predict-frames run of *model* on one extraction run."""
    params = _op_params(
        "predict-frames", model=model, frames_run_id="kmeans-0123456789", **values
    )
    assert isinstance(params, PredictFramesParams)
    return predict_frames_run_id(
        OPS["predict-frames"].version,
        params,
        model_id=model_id,
        model_kind=model_kind,
    )


def _predict_frames_single() -> str:
    # One run id is the model's identity.
    run = "train-pose.0.2-aaaaaaaaaa"
    return _predict_frames_run_id("train-pose", run, [run])


def _predict_frames_pair() -> str:
    # A SLEAP pair is named by the content digest of both trainings.
    pair = ["train-sleap.0.3-aaaaaaaaaa", "train-sleap.0.3-bbbbbbbbbb"]
    return _predict_frames_run_id("train-sleap", "0123abcdef", pair)


def _predict_frames_non_default() -> str:
    run = "train-points.0.2-aaaaaaaaaa"
    return _predict_frames_run_id(
        "train-points", run, [run], dor=2.0, conf_threshold=0.1
    )


_SCORED_POSE = KeypointSchema(names=("nose", "tail"))
_SCORED_RUN = "predict-frames.0.1-aaaaaaaaaa"
_SCORED_READ = (_SCORED_RUN, "g", "s1", "", "a" * 16)
"""The one predictions file of :data:`_SCORED_RUN` that a score reads.

Its digest has the width of a ``file_digest``.
"""


def _score_predictions_run_id(
    prediction_runs: Sequence[str] = (_SCORED_RUN,),
    *,
    resolved: KeypointMap | None = None,
    seen_digest: str | None = None,
    sources: Collection[str] = (),
    read: Collection[tuple[str, str, str, str, str]] = (_SCORED_READ,),
    **values: object,
) -> str:
    """The identifier of a score of *prediction_runs* against one revision.

    The seen frames are excluded when *seen_digest* is given, and the map is
    the identity unless *resolved* is given.
    """
    params = _op_params(
        "score-predictions",
        prediction_runs=list(prediction_runs),
        labels=[{"set_key": "17-mice"}],
        exclude_seen=seen_digest is not None,
        **values,
    )
    assert isinstance(params, ScorePredictionsParams)
    return score_predictions_run_id(
        OPS["score-predictions"].version,
        params,
        labels=[("uuid-a", "17-mice", "aaaaaaaaaaaaaaaa")],
        keypoint_map=resolved or resolve_keypoint_map(_SCORED_POSE, _SCORED_POSE.names),
        seen_digest=seen_digest,
        sources=sources,
        read=read,
    )


def _score_predictions_map() -> str:
    # The model names its keypoints by index, and an override table maps them.
    overrides = {"nose": "kp0", "tail": "kp1"}
    return _score_predictions_run_id(
        resolved=resolve_keypoint_map(
            _SCORED_POSE, ("kp0", "kp1"), overrides=overrides
        ),
        keypoint_map={"keypoints": overrides},
    )


def _score_predictions_sources_read() -> str:
    # A label object accepted from another run, whose file the score also reads.
    source = "predict-frames.0.1-cccccccccc"
    return _score_predictions_run_id(
        sources=[source],
        read=[_SCORED_READ, (source, "g", "s1", "", "c" * 16)],
    )


def _score_predictions_pool() -> str:
    pooled = "predict-frames.0.1-bbbbbbbbbb"
    return _score_predictions_run_id(
        [_SCORED_RUN, pooled],
        read=[_SCORED_READ, (pooled, "g", "s2", "", "b" * 16)],
    )


FUNCTION_CASES: dict[str, Callable[[], str]] = {
    "frames/run-id": _frames_run_id,
    "frames/run-id-revision-1": _frames_run_id_revised,
    "frames/run-id-list": _frames_run_id_list,
    "frames/run-id-exclude-kmeans": _frames_run_id_exclude_kmeans,
    "frames/run-id-exclude-uniform": _frames_run_id_exclude_uniform,
    "transcode/recipe-hash": _recipe_hash,
    "transcode/run-id": _run_id,
    "tracks/convert-variant": _convert_variant,
    "tracks/trex-variant": _trex_variant,
    "tracks/sleap-variant": _sleap_variant,
    "tracks/litpose-variant": _litpose_variant,
    "tracks/ultralytics-variant": _ultralytics_variant,
    "tracks/infer-variant": _infer_variant,
    "tracks/resample-variant": _resample_variant,
    "tracks/resample-variant-other-upstream": _resample_variant_other_upstream,
    "tracks/assign-arenas-variant": _assign_arenas_variant,
    "labels/convert-variant": _labels_convert_variant,
    "trex/run-id-settings": _trex_run_id_settings,
    "trex/conversion-run-id": _trex_conversion_run_id,
    "sleap/run-id-settings": _sleap_run_id_settings,
    "sleap/run-id-settings-embedding": _sleap_run_id_settings_embedding,
    "litpose/run-id-settings": _litpose_run_id_settings,
    "ultralytics/run-id-settings": _ultralytics_run_id_settings,
    "ultralytics/tracker-defaults": _ultralytics_tracker_defaults,
    "composition/media-single-camera": _media_single_camera,
    "composition/media-reordered": _media_reordered,
    "composition/media-two-cameras": _media_two_cameras,
    "composition/media-empty": _media_empty,
    "composition/tracks-raw-two-files": _tracks_raw_two_files,
    "composition/labels-raw-two-files": _labels_raw_two_files,
    "train-pose/run-id": _train_run_id,
    "train-pose/run-id-bare-base": _train_run_id_from_a_bare_path,
    "train-sleap/run-id": _train_sleap_run_id,
    "infer-points/run-id": _infer_run_id,
    "convert-points/run-id": _convert_points_run_id,
    "prepare-training-data/run-id": _prepare_run_id,
    "prepare-training-data/run-id-revisions-named": (
        _prepare_run_id_with_revisions_named
    ),
    "prepare-training-data/holdout": _prepare_run_id_with_a_holdout,
    "prepare-embedding-data/run-id": _prepare_embedding_run_id,
    "preprocess/one-step": _preprocess_one_step,
    "preprocess/three-steps": _preprocess_three_steps,
    "preprocess/upstream": _preprocess_upstream,
    "preprocess/labeled-rate": _preprocess_labeled_rate,
    "preprocess/h264": _preprocess_h264,
    "tracks/infer-variant-media": _infer_variant_on_media,
    "trex/run-id-settings-media": lambda: _trex_run_id_settings(MEDIA_VARIANT),
    "sleap/run-id-settings-media": lambda: _sleap_run_id_settings(MEDIA_VARIANT),
    "litpose/run-id-settings-media": lambda: _litpose_run_id_settings(MEDIA_VARIANT),
    "ultralytics/run-id-settings-media": lambda: _ultralytics_run_id_settings(
        MEDIA_VARIANT
    ),
    "infer-pose/run-id-media": _infer_run_id_on_media(
        "infer-pose", "0.5", "train-pose.0.1-aaaaaaaaaa"
    ),
    "infer-points/run-id-media": _infer_run_id_on_media(
        "infer-points", "0.5", "train-points.0.1-aaaaaaaaaa"
    ),
    "infer-localizer/run-id-media": _infer_run_id_on_media(
        "infer-localizer", "0.3", "train-localizer.0.1-aaaaaaaaaa"
    ),
    "predict-frames/single": _predict_frames_single,
    "predict-frames/pair": _predict_frames_pair,
    "predict-frames/non-default": _predict_frames_non_default,
    "score-predictions/default": _score_predictions_run_id,
    "score-predictions/map": _score_predictions_map,
    "score-predictions/seen": lambda: _score_predictions_run_id(
        seen_digest="0123456789"
    ),
    "score-predictions/sources-read": _score_predictions_sources_read,
    "score-predictions/pool": _score_predictions_pool,
}


def _op_digest(case: OpCase) -> str:
    """The digest an op mints over its params, without running the op."""
    return hash_params(case.params.identity_dump())


def _all_digests() -> dict[str, str]:
    fresh = {case.case_id: _op_digest(case) for case in OP_CASES}
    for case_id, fn in FUNCTION_CASES.items():
        fresh[case_id] = fn()
    return fresh


def test_case_ids_are_unique() -> None:
    """A duplicated case id would silently drop coverage from the golden file."""
    ids = [case.case_id for case in OP_CASES] + list(FUNCTION_CASES)
    assert len(ids) == len(set(ids)), "duplicate case ids"


def test_every_family_is_covered() -> None:
    """Each identifier family minting through ``hash_params`` has a case.

    The list is explicit rather than swept, so a new op that mints an identifier
    is a deliberate corpus change. If a family disappears from here, coverage
    was lost rather than the op.
    """
    families = {case_id.split("/", 1)[0] for case_id in _all_digests()}
    expected = {
        "frames",
        "trex",
        "sleap",
        "litpose",
        "ultralytics",
        "transcode",
        "train-pose",
        "train-points",
        "train-localizer",
        "train-sleap",
        "train-litpose",
        "convert-points",
        "prepare-training-data",
        "prepare-embedding-data",
        "preprocess",
        "resample-tracks",
        "infer-pose",
        "infer-points",
        "infer-localizer",
        "predict-frames",
        "score-predictions",
        "select-frames",
        "composition",
        "tracks",
        "labels",
    }
    assert families == expected, f"family coverage changed: {families ^ expected}"


def test_labels_and_tracks_roots_do_not_collide() -> None:
    """A label source and a track source of the same bytes stay distinct.

    ``calms21_npy`` is registered as both a track and a label converter, so the
    same physical file can sit in both raw roots. The composition kind term and
    the ``convert-labels-`` op prefix are what keep their identifiers apart -- a
    change under one root must never read as the other.
    """
    assert _labels_raw_two_files() != _tracks_raw_two_files()
    assert label_converter_op("calms21_npy") != converter_op("calms21_npy")


@pytest.mark.parametrize("case", OP_CASES, ids=lambda c: c.case_id)
def test_op_params_digest_matches_golden(case: OpCase) -> None:
    """The literal digest for *case* is unchanged since the file was written."""
    if updating_golden():
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: regenerating, see test_regenerate_golden")

    golden = read_string_golden(GOLDEN)
    if case.case_id not in golden:
        pytest.fail(
            f"No golden digest for '{case.case_id}'. If this case is new, run "
            f"`{regenerate_command(__name__)}` and review "
            f"the diff."
        )
    frozen_note = (
        " This family is FROZEN: mosaic-api embeds it in annotated frame paths "
        "on Dolt-tracked rows, so a moved digest orphans annotation labor. Do "
        "not regenerate -- revert the change that moved it."
        if case.frozen
        else ""
    )
    assert _op_digest(case) == golden[case.case_id], (
        f"Digest for '{case.case_id}' changed.{frozen_note}"
    )


@pytest.mark.parametrize("case_id", sorted(FUNCTION_CASES), ids=lambda c: str(c))
def test_function_digest_matches_golden(case_id: str) -> None:
    """The transcode minters pin payload construction as well as the digest."""
    if updating_golden():
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: regenerating, see test_regenerate_golden")

    golden = read_string_golden(GOLDEN)
    if case_id not in golden:
        pytest.fail(
            f"No golden digest for '{case_id}'. Run `{regenerate_command(__name__)}`."
        )
    assert FUNCTION_CASES[case_id]() == golden[case_id], (
        f"Digest for '{case_id}' changed."
    )


def test_regenerate_golden() -> None:
    """Rewrite the golden file. Runs only under the update environment variable."""
    if not updating_golden():
        pytest.skip(f"set {UPDATE_GOLDEN_ENV}=1 to regenerate")
    fresh = _all_digests()
    write_golden(GOLDEN, fresh)
    assert len(fresh) == len(OP_CASES) + len(FUNCTION_CASES)


def test_golden_file_has_no_stale_entries() -> None:
    """A golden entry with no matching case is dead weight that hides removals."""
    if updating_golden():
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: regenerating")
    known = {case.case_id for case in OP_CASES} | set(FUNCTION_CASES)
    stale = set(read_string_golden(GOLDEN)) - known
    assert not stale, f"golden file has entries with no case: {sorted(stale)}"
