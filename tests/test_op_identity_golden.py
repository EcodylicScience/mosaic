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
from mosaic.core.pipeline.ops import OPS
from mosaic.core.pipeline.transcode import (
    TranscodeParams,
    transcode_recipe_hash,
    transcode_run_id,
)
from mosaic.core.pipeline.types import Params
from mosaic.media_probe_config import media_thresholds
from mosaic.tracking import register_ops

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


FUNCTION_CASES: dict[str, Callable[[], str]] = {
    "transcode/recipe-hash": _recipe_hash,
    "transcode/run-id": _run_id,
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
        "transcode",
        "train-pose",
        "train-points",
        "train-localizer",
        "convert-points",
        "infer-pose",
        "infer-points",
        "infer-localizer",
    }
    assert families == expected, f"family coverage changed: {families ^ expected}"


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
