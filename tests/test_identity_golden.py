"""Golden corpus pinning literal feature identifiers.

Every other identity test in the suite asserts identifiers *relative to each
other* (``r2.run_id == r1.run_id``), which passes for any change to the hash
payload as long as the change is self-consistent. This module pins the literal
strings, so an unintended shift fails loudly and an intended one shows up as a
reviewable diff in a single data file.

The split is deliberate: the **matrix lives in code** (``CASES`` below, so adding
coverage is a reviewed change) and the **identifiers live in data**
(``data/identity_golden.json``, so a shift is one diff).

Regenerating after a deliberate identity change::

    MOSAIC_UPDATE_GOLDEN=1 pytest tests/test_identity_golden.py

Then read the resulting diff: every moved line must be explained by the change
that moved it. A line you cannot explain is a bug caught before it reached a
dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import RootModel

from mosaic.cli._features import build_feature
from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.run import compute_run_id

GOLDEN_PATH = Path(__file__).parent / "data" / "identity_golden.json"
UPDATE_ENV = "MOSAIC_UPDATE_GOLDEN"


class GoldenFile(RootModel[dict[str, str]]):
    """``case id -> run_id``. A plain map so the diff stays readable."""


@dataclass(frozen=True)
class Case:
    """One identity computation, fully specified.

    Attributes:
        case_id: Stable key into the golden file. Never reuse one for different
            content -- a renamed case reads as a moved identifier.
        feature: Registered feature slug.
        inputs: JSON inputs payload, or None for the default ``["tracks"]``.
        params: Params overrides, or None for defaults.
        frame_start: Start of the frame range term.
        frame_end: End of the frame range term.
        scope: Resolved scope entries. Only affects ``scope_dependent`` features.
        tracks_variants: Resolved tracks recipes. Affects any feature reading
            ``tracks``, scope-dependent or not, and is omitted from the payload
            when empty -- which is why every pre-existing case is unaffected.
        compositions: Per-entry ``(group, sequence, ((root, digest), ...))``
            source compositions. **Literals only.** This corpus is deliberately
            filesystem-free, so a value here stands in for what
            ``build_manifest`` would have read, never for a real one. Affects a
            ``scope_dependent`` feature that declares the matching root, and is
            omitted from the payload otherwise -- which is why every pre-existing
            case is unaffected by item 4.4.
    """

    case_id: str
    feature: str
    inputs: list[object] | None = None
    params: dict[str, object] | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    scope: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    tracks_variants: tuple[str, ...] = field(default_factory=tuple)
    labels_variants: tuple[str, ...] = field(default_factory=tuple)
    compositions: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = field(
        default_factory=tuple
    )


# Features that construct from ``["tracks"]`` with default params. Listed
# explicitly rather than swept from the registry so that adding a feature is a
# deliberate corpus change; ``test_every_constructible_feature_is_covered``
# below is what stops one being forgotten.
_TRACKS_DEFAULT_FEATURES = (
    "approach-avoidance",
    "body-scale",
    "collective-motion-metrics",
    "egocentric-crop",
    "ffgroups",
    "ffgroups-metrics",
    "id-tag-columns",
    "interaction-crop-pipeline",
    "movement-filter-interpolate",
    "movement-smooth",
    "nearest-neighbor",
    "nn-delta-bins",
    "nn-delta-response",
    "orientation-rel",
    "pair-egocentric",
    "pair-facing",
    "pair-interaction-filter",
    "pair-posedistance-pca",
    "pair-position",
    "pair-wavelet",
    "social-motion-summary",
    "speed-angvel",
    "track-subsample",
    "trajectory-smooth",
)

_TEMPLATES_REF: dict[str, object] = {
    "feature": "extract-templates",
    "pattern": "templates.parquet",
    "load": {"kind": "parquet"},
}

CASES: tuple[Case, ...] = (
    # --- every default-constructible feature, default everything ---
    *(
        Case(case_id=f"{slug}/default", feature=slug)
        for slug in _TRACKS_DEFAULT_FEATURES
    ),
    # --- params participate in identity ---
    Case(
        case_id="speed-angvel/params-smooth",
        feature="speed-angvel",
        params={"smooth_window": 11},
    ),
    Case(
        case_id="pair-wavelet/params-nfreq",
        feature="pair-wavelet",
        params={"n_freq": 20},
    ),
    Case(
        case_id="frame-aggregate/column-speed",
        feature="frame-aggregate",
        params={"column": "SPEED"},
    ),
    Case(
        case_id="collective-motion-metrics/subgroups-alpha",
        feature="collective-motion-metrics",
        params={"subgroup_col": "event", "area_method": "alpha_shape", "alpha": 60.0},
    ),
    # local-order-metrics has a required ``radius`` and so is not
    # default-constructible; it is covered by these bespoke cases instead.
    Case(
        case_id="local-order-metrics/radius-position",
        feature="local-order-metrics",
        params={"radius": 150.0},
    ),
    Case(
        case_id="local-order-metrics/params-shells",
        feature="local-order-metrics",
        params={"radius": 150.0, "n_shells": 4},
    ),
    Case(
        case_id="local-order-metrics/radius-body-scale",
        feature="local-order-metrics",
        params={
            "radius": 3.0,
            "radius_units": "body_scale",
            "body_scale": {"feature": "body-scale__from__tracks"},
        },
    ),
    # --- the frame range participates in identity ---
    Case(
        case_id="speed-angvel/frames-0-100",
        feature="speed-angvel",
        frame_start=0,
        frame_end=100,
    ),
    Case(
        case_id="speed-angvel/frames-0-200",
        feature="speed-angvel",
        frame_start=0,
        frame_end=200,
    ),
    # --- scope participates only for scope_dependent features (P2d) ---
    Case(
        case_id="speed-angvel/scope-a", feature="speed-angvel", scope=(("", "seq_a"),)
    ),
    Case(
        case_id="speed-angvel/scope-ab",
        feature="speed-angvel",
        scope=(("", "seq_a"), ("", "seq_b")),
    ),
    Case(
        case_id="pair-posedistance-pca/scope-a",
        feature="pair-posedistance-pca",
        scope=(("", "seq_a"),),
    ),
    Case(
        case_id="pair-posedistance-pca/scope-ab",
        feature="pair-posedistance-pca",
        scope=(("", "seq_a"), ("", "seq_b")),
    ),
    # The item 4.4 guard, and it is a guard rather than coverage: this feature
    # declares consumed_roots = (), so a composition in scope must reach nothing.
    # Its identifier must equal `pair-posedistance-pca/scope-ab` exactly -- if it
    # ever differs, a composition leaked into a feature that does not read the
    # root, which is the false invalidation the media-storage note warns about
    # and which would couple every table-only feature to media it never opened.
    Case(
        case_id="pair-posedistance-pca/scope-ab-composition-ignored",
        feature="pair-posedistance-pca",
        scope=(("", "seq_a"), ("", "seq_b")),
        compositions=(
            ("", "seq_a", (("media_raw", "aaaaaaaaaa"),)),
            ("", "seq_b", (("tracks_raw", "bbbbbbbbbb"),)),
        ),
    ),
    # --- feature-to-feature inputs ---
    Case(
        case_id="temporal-stack/from-speed-angvel",
        feature="temporal-stack",
        inputs=[{"feature": "speed-angvel"}],
    ),
    Case(
        case_id="temporal-stack/from-speed-angvel-pinned",
        feature="temporal-stack",
        inputs=[{"feature": "speed-angvel", "run_id": "0.1-deadbeef01"}],
    ),
    Case(
        case_id="extract-templates/from-pair-wavelet",
        feature="extract-templates",
        inputs=[{"feature": "pair-wavelet"}],
        params={"n_templates": 500},
    ),
    # --- scope_dependent global fitters ---
    Case(
        case_id="arhmm/from-pair-wavelet/scope-a",
        feature="arhmm",
        inputs=[{"feature": "pair-wavelet"}],
        scope=(("", "seq_a"),),
    ),
    Case(
        case_id="arhmm/from-pair-wavelet/scope-ab",
        feature="arhmm",
        inputs=[{"feature": "pair-wavelet"}],
        scope=(("", "seq_a"), ("", "seq_b")),
    ),
    # --- identity models: stream fitters, so the scope IS the training set ---
    # These three take egocentric crops, not tracks, so they are unreachable by
    # test_every_constructible_feature_is_covered (which probes with the default
    # ``["tracks"]`` payload and skips anything that will not construct from it).
    # Each is paired scope-a/scope-ab: the two entries of a pair must differ, and
    # a pair that agrees is the P2f defect visible in the data file.
    Case(
        case_id="global-identity-model/from-egocentric-crop/scope-a",
        feature="global-identity-model",
        inputs=[{"feature": "egocentric-crop"}],
        scope=(("", "seq_a"),),
    ),
    Case(
        case_id="global-identity-model/from-egocentric-crop/scope-ab",
        feature="global-identity-model",
        inputs=[{"feature": "egocentric-crop"}],
        scope=(("", "seq_a"), ("", "seq_b")),
    ),
    Case(
        case_id="global-identity-embedding/from-egocentric-crop/scope-a",
        feature="global-identity-embedding",
        inputs=[{"feature": "egocentric-crop"}],
        scope=(("", "seq_a"),),
    ),
    Case(
        case_id="global-identity-embedding/from-egocentric-crop/scope-ab",
        feature="global-identity-embedding",
        inputs=[{"feature": "egocentric-crop"}],
        scope=(("", "seq_a"), ("", "seq_b")),
    ),
    # The rename from ``global-identity-megadescriptor`` was identity-neutral.
    # A feature's own slug is not hashed and every Params field name survived,
    # so pinning the three defaults that moved -- model_name, image_size,
    # weights_name -- reproduces the digest that feature minted under its old
    # name, ``0.1-8aebe700d2``. If this line moves, the rename changed a recipe
    # and not just a name.
    Case(
        case_id="global-identity-embedding/from-egocentric-crop/megadescriptor-pinned",
        feature="global-identity-embedding",
        inputs=[{"feature": "egocentric-crop"}],
        params={
            "model_name": "BVRA/MegaDescriptor-L-384",
            "image_size": (384, 384),
            "weights_name": "megadescriptor_identity",
        },
        scope=(("", "seq_a"),),
    ),
    Case(
        case_id="global-identity-dinov2-temporal/from-egocentric-crop/scope-a",
        feature="global-identity-dinov2-temporal",
        inputs=[{"feature": "egocentric-crop"}],
        scope=(("", "seq_a"),),
    ),
    Case(
        case_id="global-identity-dinov2-temporal/from-egocentric-crop/scope-ab",
        feature="global-identity-dinov2-temporal",
        inputs=[{"feature": "egocentric-crop"}],
        scope=(("", "seq_a"), ("", "seq_b")),
    ),
    # --- params-level global fitters: the templates ref is the training set ---
    Case(
        case_id="global-scaler/templates",
        feature="global-scaler",
        inputs=[{"feature": "extract-templates"}],
        params={"templates": _TEMPLATES_REF},
    ),
    Case(
        case_id="global-kmeans/templates",
        feature="global-kmeans",
        inputs=[{"feature": "global-tsne"}],
        params={"templates": _TEMPLATES_REF},
    ),
    Case(
        case_id="xgboost/templates",
        feature="xgboost",
        inputs=[{"feature": "global-scaler"}],
        params={"templates": _TEMPLATES_REF, "default_class": 0},
    ),
    # --- the resolved tracks variant (item 3.3) ---
    #
    # ``speed-angvel/default`` above is the same feature with no variants
    # resolved, so the four together pin the whole rule: absent differs from
    # present, one variant differs from two, and two orderings of one pair agree.
    Case(
        case_id="speed-angvel/tracks-one-variant",
        feature="speed-angvel",
        tracks_variants=("convert-trex_npz.0.1-aaaaaaaaaa",),
    ),
    Case(
        case_id="speed-angvel/tracks-two-variants",
        feature="speed-angvel",
        tracks_variants=("convert-trex_npz.0.1-aaaaaaaaaa", "trex.0.1-bbbbbbbbbb"),
    ),
    # Deliberately equal to the line above: which recipes a run read is a set,
    # so the two spellings must digest alike. The equality in the data file is
    # the assertion.
    Case(
        case_id="speed-angvel/tracks-two-variants-reversed",
        feature="speed-angvel",
        tracks_variants=("trex.0.1-bbbbbbbbbb", "convert-trex_npz.0.1-aaaaaaaaaa"),
    ),
    # --- the resolved labels variant (item 9.3) ---
    #
    # id-tag-columns reads labels, so a resolved label recipe must move its
    # identifier. The pair pins the whole rule: the two are the same feature and
    # params, differing only in labels_variants, so present must differ from
    # absent -- and the ``_labels`` term is omitted when empty, which is why every
    # pre-existing case (none of which resolves a label variant) is unaffected.
    # Adjacent, so an accidental agreement would be visible in the data file.
    Case(
        case_id="id-tag-columns/no-labels-variant",
        feature="id-tag-columns",
        params={
            "labels": {"kind": "id_tags"},
            "label_kind": "id_tags",
            "fields": ["focal"],
        },
    ),
    Case(
        case_id="id-tag-columns/labels-one-variant",
        feature="id-tag-columns",
        params={
            "labels": {"kind": "id_tags"},
            "label_kind": "id_tags",
            "fields": ["focal"],
        },
        labels_variants=("convert-labels-boris_aggregated_csv.0.1-cccccccccc",),
    ),
    # --- archived analyses: identifiers that must never move ---
    #
    # The delivery document owes a manual re-run of the guppies analysis once per
    # identity-shifting milestone, to confirm a track-only dataset keeps
    # bit-identical identifiers. These three cases are that check, automated.
    #
    # **Their golden lines were transcribed from disk, not generated.** They are
    # the directory names under ``features/`` in the archived guppies dataset on
    # JD-SSD, together with the ``_params``/``_inputs`` recorded in each run's
    # ``params.json``. Regenerating them would pin whatever the code currently
    # produces, which is exactly the thing under test -- so if one of these moves
    # under ``MOSAIC_UPDATE_GOLDEN=1``, the change that moved it broke an archived
    # analysis and the right response is to fix the change, not the file.
    #
    # Why they are expected to hold: neither per-frame nor summary identity ever
    # contained a sequence name, and the archived ``tracks/index.csv`` predates
    # the ``run_id`` column entirely -- so the tracks term is *absent* rather than
    # empty, and an absent key digests differently from an empty one.
    Case(
        case_id="archive/guppies/trajectory-smooth",
        feature="trajectory-smooth",
        params={
            "speed_threshold": 40.0,
            "fps": 30.0,
            "interpolate_centroid": True,
            "interpolate_pose": False,
            "expand_frames": 8,
            "savgol_window": None,
            "savgol_polyorder": 1,
        },
    ),
    Case(
        case_id="archive/guppies/id-tag-columns",
        feature="id-tag-columns",
        params={
            "labels": {"kind": "id_tags"},
            "label_kind": "id_tags",
            "fields": ["focal"],
            "field_renames": {"focal": "Focal_fish"},
        },
    ),
    Case(
        case_id="archive/guppies/speed-angvel-from-smooth",
        feature="speed-angvel",
        inputs=[
            {
                "feature": "trajectory-smooth__from__tracks",
                "run_id": "0.1-990067d93d",
            }
        ],
        params={"step_size": 4, "smooth_window": 5, "fps": 30.0},
    ),
)


def _identifier(case: Case) -> str:
    """Compute the identifier for *case*, with no filesystem involved."""
    feature = build_feature(case.feature, case.inputs, case.params)
    scope = Scope(
        entries=set(case.scope),
        tracks_variants=case.tracks_variants,
        labels_variants=case.labels_variants,
        compositions={
            (group, sequence): dict(pairs)
            for group, sequence, pairs in case.compositions
        },
    )
    run_id, _ = compute_run_id(feature, case.frame_start, case.frame_end, scope)
    return run_id


def _load_golden() -> dict[str, str]:
    if not GOLDEN_PATH.exists():
        return {}
    return GoldenFile.model_validate_json(GOLDEN_PATH.read_text()).root


def _regenerate() -> dict[str, str]:
    """Recompute every case and rewrite the golden file."""
    fresh = {case.case_id: _identifier(case) for case in CASES}
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
    return fresh


def test_case_ids_are_unique() -> None:
    """A duplicated case id would silently drop coverage from the golden file."""
    ids = [case.case_id for case in CASES]
    assert len(ids) == len(set(ids)), "duplicate case ids in CASES"


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.case_id)
def test_identifier_matches_golden(case: Case) -> None:
    """The literal identifier for *case* is unchanged since the file was written."""
    if os.environ.get(UPDATE_ENV) == "1":
        pytest.skip(f"{UPDATE_ENV}=1: regenerating, see test_regenerate_golden")

    golden = _load_golden()
    if case.case_id not in golden:
        pytest.fail(
            f"No golden identifier for '{case.case_id}'. If this case is new, run "
            f"`{UPDATE_ENV}=1 pytest tests/test_identity_golden.py` and review the diff."
        )
    assert _identifier(case) == golden[case.case_id], (
        f"Identifier for '{case.case_id}' changed. If this shift is intended, run "
        f"`{UPDATE_ENV}=1 pytest tests/test_identity_golden.py` and explain every "
        f"moved line in the commit message."
    )


def test_regenerate_golden() -> None:
    """Rewrite the golden file. Runs only under the update environment variable."""
    if os.environ.get(UPDATE_ENV) != "1":
        pytest.skip(f"set {UPDATE_ENV}=1 to regenerate")
    fresh = _regenerate()
    assert len(fresh) == len(CASES)


def test_golden_file_has_no_stale_entries() -> None:
    """A golden entry with no matching case is dead weight that hides removals."""
    if os.environ.get(UPDATE_ENV) == "1":
        pytest.skip(f"{UPDATE_ENV}=1: regenerating")
    stale = set(_load_golden()) - {case.case_id for case in CASES}
    assert not stale, f"golden file has entries with no case: {sorted(stale)}"


def test_every_constructible_feature_is_covered() -> None:
    """A feature that constructs from defaults has at least one golden case.

    This is what stops a newly registered feature from shipping with no identity
    coverage. If it fires, add the slug to ``_TRACKS_DEFAULT_FEATURES`` (or a
    bespoke ``Case`` if it needs explicit inputs) and regenerate.
    """
    from mosaic.cli._features import available_slugs

    covered = {case.feature for case in CASES}
    missing: list[str] = []
    for slug in available_slugs():
        if slug in covered:
            continue
        try:
            # typer.Exit subclasses RuntimeError, so this catches "needs explicit
            # inputs or params" alongside any construction error.
            _ = build_feature(slug, None, None)
        except Exception:  # noqa: BLE001 - probing constructibility, not running it
            continue  # cover it with a bespoke Case instead
        missing.append(slug)
    assert not missing, (
        f"features construct from defaults but have no golden case: {missing}. "
        f"Add them to _TRACKS_DEFAULT_FEATURES and regenerate."
    )
