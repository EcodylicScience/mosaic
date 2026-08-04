"""Golden corpus for the identifiers minted by ``hash_params`` directly.

``test_identity_golden.py`` routes every case through ``compute_run_id``, so it
pins *feature* identity and nothing else. Six further identifier families call
``hash_params`` straight -- frame extraction, TREx, transcode, and the train /
convert / infer ops -- and were pinned by nothing at all. A change to
``hash_params`` or the serializer beneath it could move all six with a green
suite, which is the exact failure this corpus exists to prevent.

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

    MOSAIC_UPDATE_GOLDEN=1 pytest tests/test_op_identity_golden.py

Then read the diff: every moved line must be explained by the change that moved
it, and a moved ``frames/`` line is a bug, not a diff to accept.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from mosaic_media import CHROME_149
from mosaic_media.transcode import ANALYSIS_ENCODING
from pydantic import RootModel

from mosaic.core.pipeline._utils import hash_params
from mosaic.tracking.frame_extraction.dataset_runs import (
    ExtractFramesParams,
    frames_run_id,
)
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
from mosaic.core.pipeline.ops import OPS
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
from mosaic.core.pipeline.types import Params
from mosaic.media_probe_config import media_thresholds
from mosaic.tracking import register_ops
from mosaic.tracking.litpose.version import LITPOSE_KIND, LITPOSE_VERSION
from mosaic.tracking.ops.infer import infer_run_id
from mosaic.tracking.ops.train import train_run_id
from mosaic.tracking.sleap.version import SLEAP_KIND, SLEAP_VERSION
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

GOLDEN_PATH = Path(__file__).parent / "data" / "op_identity_golden.json"
UPDATE_ENV = "MOSAIC_UPDATE_GOLDEN"

# The op registry is populated by an explicit call, not by importing the
# package, so the params classes below are unreachable without it.
register_ops()


class GoldenFile(RootModel[dict[str, str]]):
    """``case id -> digest``. A plain map so the diff stays readable."""


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
            tracker="simple",
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
        params=_op_params("transcode", entry=("g", "s"), target="analysis"),
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
    OpCase(
        case_id="convert-points/cvat",
        params=_op_params(
            "convert-points",
            cvat_xml="annotations/points.xml",
            images_dir="annotations/images",
            class_names=["bee", "feeder"],
            radii={"bee": 10.0, "feeder": 30.0},
        ),
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
)

# The two transcode identifiers are minted by named, importable functions, so
# unlike the op ids above these pin the real payload construction as well as
# the digest.
_RECIPE_PARAMS = TranscodeParams(entry=("g", "s"), target="analysis")


def _recipe_hash() -> str:
    return transcode_recipe_hash(
        _RECIPE_PARAMS, ANALYSIS_ENCODING, CHROME_149, media_thresholds()
    )


def _run_id() -> str:
    # Deliberately unsorted, so a regression that stopped sorting the sources
    # would move this line rather than pass.
    return transcode_run_id(_recipe_hash(), ["uuid-b", "uuid-a", "uuid-c"])


# The three tracks-variant identifiers. Function cases rather than OpCases,
# because a track converter lives in TRACK_CONVERTERS rather than OPS and because
# what has to be pinned is the *payload wrapper* each producer builds, not just
# the digest: renaming a key inside one of these would move every variant
# directory on disk, and a corpus that called ``tracks_run_id`` with a hand-built
# payload would stay green through it.


def _convert_variant() -> str:
    return tracks_run_id(
        converter_op("calms21_npy"),
        "0.1",
        convert_variant_payload({"neck_idx": 3, "tail_idx": 6}),
    )


def _trex_variant() -> str:
    # The tracker's own settings, passed through unwrapped -- so this value is
    # byte-identical to trex_run_id(settings) for the same settings.
    return tracks_run_id(
        TREX_KIND,
        TREX_VERSION,
        tracker_variant_payload({"track_max_individuals": 4, "cm_per_pixel": 0.5}),
    )


def _infer_variant() -> str:
    return tracks_run_id(
        "infer-points",
        "0.1",
        infer_variant_payload({"conf": 0.5}, "train-points.0.1-aaaaaaaaaa"),
    )


def _trex_run_id_settings() -> str:
    """The identifier ``trex_settings``' *key set* mints, pinned.

    Nothing else pins it. The ``trex/*`` op cases come from ``TrexParams``, and
    the variant case above passes a hand-built two-key dict -- so a rename
    inside ``trex_settings`` moves every TREx run root and tracks variant on
    disk with a fully green suite. This is the case that fails instead.

    Every argument is given explicitly rather than relying on defaults, because
    a default that changed would move this line for a reason unrelated to the
    key set it exists to guard.
    """
    from mosaic.tracking.trex.dataset_runs import trex_run_id, trex_settings

    return trex_run_id(
        trex_settings(
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
            track_extra_settings=None,
        )
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


def _sleap_run_id_settings() -> str:
    """The identifier ``sleap_settings``' *key set* mints, pinned.

    The SLEAP counterpart of :func:`_trex_run_id_settings`, and it exists for the
    same reason: the ``sleap/*`` op cases come from ``SleapParams``, and
    ``_sleap_variant`` passes a hand-built three-key dict, so a rename inside
    ``sleap_settings`` moves every SLEAP run root and tracks variant on disk with
    a fully green suite.

    ``tracking`` is True because the tracker knobs are dropped from identity when
    it is False -- the True case is the one that carries every key.
    """
    from mosaic.tracking.sleap.dataset_runs import sleap_run_id, sleap_settings

    return sleap_run_id(
        sleap_settings(
            model_id="0123456789abcdef",
            tracking=True,
            tracker="flow",
            similarity="instance",
            match="hungarian",
            track_window=5,
            max_instances=4,
            max_tracking=4,
            peak_threshold=0.2,
            analysis_range=None,
            sleap_extra_settings=None,
        )
    )


def _litpose_run_id_settings() -> str:
    """The identifier ``litpose_settings``' *key set* mints, pinned.

    The Lightning Pose counterpart of :func:`_trex_run_id_settings`. Its settings
    dict is the smallest of the three, which makes it the easiest to rename a key
    in without noticing.
    """
    from mosaic.tracking.litpose.dataset_runs import litpose_run_id, litpose_settings

    return litpose_run_id(
        litpose_settings(
            model_id="0123456789abcdef",
            litpose_overrides=None,
        )
    )


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
    return train_run_id(
        "train-pose",
        "0.1",
        _op_params("train-pose", data="d.yaml", epochs=3),
        "deadbeefcafe",
        "train-pose.0.1-aaaaaaaaaa",
    )


def _train_run_id_from_a_bare_path() -> str:
    # The population item 4.6 moves: a base with no run to name it now
    # contributes its weights digest where it used to contribute "".
    return train_run_id(
        "train-pose",
        "0.1",
        _op_params("train-pose", data="d.yaml", epochs=3),
        "deadbeefcafe",
        "0123456789abcdef",
    )


def _infer_run_id() -> str:
    return infer_run_id(
        "infer-points",
        "0.1",
        _op_params("infer-points", model="m.pt"),
        "train-points.0.1-aaaaaaaaaa",
    )


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


FUNCTION_CASES: dict[str, Callable[[], str]] = {
    "frames/run-id": _frames_run_id,
    "frames/run-id-revision-1": _frames_run_id_revised,
    "transcode/recipe-hash": _recipe_hash,
    "transcode/run-id": _run_id,
    "tracks/convert-variant": _convert_variant,
    "tracks/trex-variant": _trex_variant,
    "tracks/sleap-variant": _sleap_variant,
    "tracks/litpose-variant": _litpose_variant,
    "tracks/infer-variant": _infer_variant,
    "labels/convert-variant": _labels_convert_variant,
    "trex/run-id-settings": _trex_run_id_settings,
    "sleap/run-id-settings": _sleap_run_id_settings,
    "litpose/run-id-settings": _litpose_run_id_settings,
    "composition/media-single-camera": _media_single_camera,
    "composition/media-reordered": _media_reordered,
    "composition/media-two-cameras": _media_two_cameras,
    "composition/media-empty": _media_empty,
    "composition/tracks-raw-two-files": _tracks_raw_two_files,
    "composition/labels-raw-two-files": _labels_raw_two_files,
    "train-pose/run-id": _train_run_id,
    "train-pose/run-id-bare-base": _train_run_id_from_a_bare_path,
    "infer-points/run-id": _infer_run_id,
}


def _op_digest(case: OpCase) -> str:
    """The digest an op mints over its params, without running the op."""
    return hash_params(case.params.identity_dump())


def _all_digests() -> dict[str, str]:
    fresh = {case.case_id: _op_digest(case) for case in OP_CASES}
    for case_id, fn in FUNCTION_CASES.items():
        fresh[case_id] = fn()
    return fresh


def _load_golden() -> dict[str, str]:
    if not GOLDEN_PATH.exists():
        return {}
    return GoldenFile.model_validate_json(GOLDEN_PATH.read_text()).root


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
        "transcode",
        "train-pose",
        "train-points",
        "train-localizer",
        "train-sleap",
        "train-litpose",
        "convert-points",
        "infer-pose",
        "infer-points",
        "infer-localizer",
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
    if os.environ.get(UPDATE_ENV) == "1":
        pytest.skip(f"{UPDATE_ENV}=1: regenerating, see test_regenerate_golden")

    golden = _load_golden()
    if case.case_id not in golden:
        pytest.fail(
            f"No golden digest for '{case.case_id}'. If this case is new, run "
            f"`{UPDATE_ENV}=1 pytest tests/test_op_identity_golden.py` and review "
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
    if os.environ.get(UPDATE_ENV) == "1":
        pytest.skip(f"{UPDATE_ENV}=1: regenerating, see test_regenerate_golden")

    golden = _load_golden()
    if case_id not in golden:
        pytest.fail(
            f"No golden digest for '{case_id}'. Run "
            f"`{UPDATE_ENV}=1 pytest tests/test_op_identity_golden.py`."
        )
    assert FUNCTION_CASES[case_id]() == golden[case_id], (
        f"Digest for '{case_id}' changed."
    )


def test_regenerate_golden() -> None:
    """Rewrite the golden file. Runs only under the update environment variable."""
    if os.environ.get(UPDATE_ENV) != "1":
        pytest.skip(f"set {UPDATE_ENV}=1 to regenerate")
    fresh = _all_digests()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    assert len(fresh) == len(OP_CASES) + len(FUNCTION_CASES)


def test_golden_file_has_no_stale_entries() -> None:
    """A golden entry with no matching case is dead weight that hides removals."""
    if os.environ.get(UPDATE_ENV) == "1":
        pytest.skip(f"{UPDATE_ENV}=1: regenerating")
    known = {case.case_id for case in OP_CASES} | set(FUNCTION_CASES)
    stale = set(_load_golden()) - known
    assert not stale, f"golden file has entries with no case: {sorted(stale)}"
