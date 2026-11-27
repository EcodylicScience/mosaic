"""Which steps may be wired to which, checked with no dataset in hand.

The claim under test is not only that a bad edge is refused, but that it is
refused **without a dataset** -- because a canvas asks this while a wire is being
drawn, before its user has chosen anything to run it against. So the sharpest
case here runs in a subprocess where constructing a dataset raises.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Collection

import pytest

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.feature_levels import EmitsLevel
from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
from mosaic.core.pipeline.graph import (
    UNKNOWN_COLUMNS,
    ColumnSet,
    TRACKS_DECLARATION,
    ConsumerDecl,
    Declaration,
    DeclarationCatalog,
    EditorCatalog,
    GpuWhen,
    ProducerDecl,
    can_connect,
    can_join,
    compatible_consumers,
    compatible_producers,
    declaration_catalog,
    describe_step_type,
    editor_catalog,
    possible_connections,
    resolve_emits,
)
from mosaic.core.pipeline.graph.columns import tracks_columns
from mosaic.core.pipeline.graph.validate import (
    JoinInput,
    join_problem,
    level_caution,
    subgroup_cautions,
)
from tests.helpers import run_child

# --- hand-built declarations, so the unit tests need no registry -------------

TRACKS = ProducerDecl(name="tracks", kind="tracks", level="individual")
SPEED = ProducerDecl(name="speed-angvel", kind="feature", level="individual")
PAIRWISE = ProducerDecl(name="pair-position", kind="feature", level="pair")
COLLECTIVE = ProducerDecl(
    name="collective-motion-metrics", kind="feature", level="global"
)
TREX = ProducerDecl(name="trex", kind="op", level="individual", writes_tracks=True)

JOINER = ConsumerDecl(
    name="temporal-stack", kind="feature", accepts_tracks=True, accepts_features=True
)
TRACK_SHAPED = ConsumerDecl(
    name="speed-angvel",
    kind="feature",
    accepts_tracks=True,
    accepts_features=True,
    requires_track_shape=True,
)
FEATURES_ONLY = ConsumerDecl(name="global-tsne", kind="feature", accepts_features=True)
SELF_LOADING = ConsumerDecl(name="kpms", kind="feature", takes_no_inputs=True)
CROSS = ConsumerDecl(
    name="interaction-crop-pipeline",
    kind="feature",
    accepts_tracks=True,
    accepts_features=True,
    cross_joins=True,
)


# --- one wire ----------------------------------------------------------------


def test_a_feature_may_read_tracks_and_another_feature() -> None:
    assert can_connect(TRACKS, JOINER)
    assert can_connect(SPEED, JOINER)


def test_an_op_is_not_a_feature_input() -> None:
    """What a feature reads from a tracker is its tracks variant, not the run."""
    verdict = can_connect(TREX, JOINER)
    assert not verdict
    assert "tracks field" in verdict.reason


def test_a_consumer_taking_no_inputs_refuses_everything() -> None:
    verdict = can_connect(TRACKS, SELF_LOADING)
    assert not verdict
    assert "no pipeline inputs" in verdict.reason


def test_a_track_input_refuses_a_producer_that_drops_the_track_frame() -> None:
    """Decided from the producer's declared output columns.

    A producer passing its input through, or writing ``X`` and ``Y``, keeps the
    track frame. A producer whose declared output lacks both does not. A producer
    with undeclared output columns is allowed, and input resolution checks its
    resolved output at run time. Refusing it would refuse every undeclared track
    producer, ``movement-smooth -> speed-angvel`` among them.
    """
    keeps = ProducerDecl(
        name="trajectory-smooth",
        kind="feature",
        level="individual",
        keeps_track_frame=True,
    )
    drops = ProducerDecl(
        name="nearest-neighbor",
        kind="feature",
        level="individual",
        keeps_track_frame=False,
    )
    undeclared = ProducerDecl(
        name="movement-smooth", kind="feature", level="individual"
    )

    assert can_connect(TRACKS, TRACK_SHAPED)
    assert can_connect(keeps, TRACK_SHAPED)
    assert can_connect(undeclared, TRACK_SHAPED)
    refused = can_connect(drops, TRACK_SHAPED)
    assert not refused
    assert "neither passes its input through nor writes X and Y" in refused.reason
    # A consumer of any table takes it.
    assert can_connect(drops, JOINER)


def test_a_features_only_consumer_refuses_raw_tracks() -> None:
    """``global-tsne`` fits on feature columns; there are none in a tracks table."""
    assert can_connect(SPEED, FEATURES_ONLY)
    verdict = can_connect(TRACKS, FEATURES_ONLY)
    assert not verdict
    assert "does not read tracks" in verdict.reason


def test_a_verdict_is_truthy_and_carries_its_reason() -> None:
    """A greyed-out option a user cannot explain is worse than an absent one."""
    refused = can_connect(TREX, JOINER)
    assert bool(refused) is False
    assert refused.reason
    assert can_connect(SPEED, JOINER).reason == ""


# --- the set ------------------------------------------------------------------


def test_one_input_is_always_alignable() -> None:
    assert can_join([SPEED], JOINER)
    assert can_join([PAIRWISE], JOINER)


def test_joining_an_individual_to_a_pair_is_refused() -> None:
    """The silent cartesian product, refused before it can be produced."""
    verdict = can_join([SPEED, PAIRWISE], JOINER)
    assert not verdict
    assert "pair every row" in verdict.reason


def test_joining_two_inputs_at_one_level_is_allowed() -> None:
    other = ProducerDecl(name="heading", kind="feature", level="individual")
    assert can_join([SPEED, other], JOINER)


def test_an_unidentified_aggregate_broadcasts_rather_than_fans_out() -> None:
    """A per-frame aggregate legitimately joins onto per-id rows on frame."""
    assert can_join([SPEED, COLLECTIVE], JOINER)


def test_the_cross_join_escape_is_honoured() -> None:
    """One consumer joins across levels on purpose and pays in memory."""
    assert not can_join([SPEED, PAIRWISE], JOINER)
    assert can_join([SPEED, PAIRWISE], CROSS)


WIDENING = ConsumerDecl(
    name="xgboost",
    kind="feature",
    accepts_tracks=True,
    accepts_features=True,
    expands_individuals=True,
)


def test_a_widening_consumer_takes_individual_inputs_after_a_pair_input() -> None:
    """Each individual input is joined onto the pair rows as ``id1`` and ``id2``."""
    assert can_join([PAIRWISE, SPEED], WIDENING)
    assert can_join([PAIRWISE, SPEED, COLLECTIVE], WIDENING)
    assert not can_join([PAIRWISE, SPEED], JOINER)


def test_a_widening_consumer_refuses_an_individual_input_first() -> None:
    """Folding pair rows onto individuals needs a choice among partners."""
    verdict = can_join([SPEED, PAIRWISE], WIDENING)
    assert not verdict
    assert "pair-level input first" in verdict.reason


@pytest.mark.parametrize(
    ("slug", "widens"),
    [
        ("overlay", False),
        ("scale-to-cm", False),
        ("frame-aggregate", False),
        ("interaction-crop-pipeline", False),
        ("xgboost", True),
        ("sequence-classifier", True),
        ("pair-wavelet", True),
    ],
)
def test_which_features_widen_individual_inputs(slug: str, widens: bool) -> None:
    """A consumer emitting ``as-input`` or ``pair`` widens; the others refuse."""
    declared = declaration_catalog().get(slug)
    assert declared is not None
    assert declared.consumes.expands_individuals is widens
    verdict = can_join([PAIRWISE, SPEED], declared.consumes)
    if slug == "interaction-crop-pipeline":
        # It cross-joins on the frame instead, which can_join also allows.
        assert verdict
        return
    assert bool(verdict) is widens
    if not widens:
        assert "pair every row" in verdict.reason


STACKS = ConsumerDecl(
    name="evaluate-labels",
    kind="feature",
    accepts_tracks=True,
    accepts_features=True,
    stacks_inputs=True,
)


def test_a_stacking_consumer_takes_inputs_at_one_level() -> None:
    """Rows are unioned and not joined, so identity columns need not be shared."""
    other = ProducerDecl(name="heading", kind="feature", level="individual")
    assert can_join([SPEED, other], STACKS)


def test_a_stacking_consumer_refuses_inputs_at_different_levels() -> None:
    verdict = can_join([SPEED, PAIRWISE], STACKS)
    assert not verdict
    assert "stacks its inputs" in verdict.reason


def test_evaluate_labels_declares_that_it_stacks_and_no_other_feature_does() -> None:
    stacking = {
        name
        for name, declared in declaration_catalog().entries.items()
        if declared.consumes.stacks_inputs
    }
    assert stacking == {"evaluate-labels"}


# --- enumeration --------------------------------------------------------------


@pytest.fixture
def small_catalog() -> DeclarationCatalog:
    def entry(producer: ProducerDecl) -> Declaration:
        return Declaration(
            produces=producer,
            consumes=ConsumerDecl(name=producer.name, kind="feature"),
            emits="individual",
        )

    return DeclarationCatalog(
        entries={p.name: entry(p) for p in (TRACKS, SPEED, PAIRWISE, COLLECTIVE)}
    )


def test_possible_connections_narrows_once_a_wire_is_taken(
    small_catalog: DeclarationCatalog,
) -> None:
    """The case that proves ``existing`` is load-bearing rather than decoration.

    With nothing wired in, a pair-level producer is a valid first input. Once an
    individual-level one is taken, it stops being one -- and a palette computed
    from the consumer alone would keep offering it.
    """
    empty = possible_connections(JOINER, small_catalog)
    assert empty["pair-position"]

    taken = possible_connections(JOINER, small_catalog, existing=[SPEED])
    assert not taken["pair-position"]
    assert taken["speed-angvel"]


def test_possible_connections_answers_for_every_candidate(
    small_catalog: DeclarationCatalog,
) -> None:
    """Each candidate is answered, a refused one with the reason ``can_join`` gives."""
    answers = possible_connections(JOINER, small_catalog, existing=[PAIRWISE])
    assert set(answers) == set(small_catalog.names())
    assert not all(answers.values())
    for name, verdict in answers.items():
        declared = small_catalog.get(name)
        assert declared is not None
        assert verdict == can_join([PAIRWISE, declared.produces], JOINER), name


def test_an_undeclared_candidate_is_refused_by_name(
    small_catalog: DeclarationCatalog,
) -> None:
    answers = possible_connections(JOINER, small_catalog, candidates=["nope"])
    assert not answers["nope"]
    assert "nothing declares" in answers["nope"].reason


def test_compatible_producers_lists_the_allowed_ones(
    small_catalog: DeclarationCatalog,
) -> None:
    assert compatible_producers(JOINER, small_catalog) == (
        "collective-motion-metrics",
        "pair-position",
        "speed-angvel",
        "tracks",
    )


def test_forward_and_reverse_listings_agree() -> None:
    """``p`` feeds ``c`` in one listing iff ``c`` reads ``p`` in the other."""
    catalog = declaration_catalog()
    names = catalog.names()
    for name in names:
        declared = catalog.get(name)
        assert declared is not None
        forward = set(compatible_consumers(declared.produces, catalog))
        for consumer_name in names:
            consumer = catalog.get(consumer_name)
            assert consumer is not None
            reverse = set(compatible_producers(consumer.consumes, catalog))
            assert (consumer_name in forward) == (name in reverse), (
                f"{name} -> {consumer_name} disagrees between the two listings"
            )


# --- passthrough resolution ---------------------------------------------------


def test_a_passthrough_takes_its_upstream_level() -> None:
    assert resolve_emits("as-input", ["pair"]) == "pair"
    assert resolve_emits("as-input", ["individual"]) == "individual"


def test_a_passthrough_with_no_upstream_reads_tracks() -> None:
    """The only thing a feature with no feature inputs can be reading."""
    assert resolve_emits("as-input") == "individual"


def test_unidentified_is_the_declaration_side_spelling_of_global() -> None:
    assert resolve_emits("unidentified") == "global"
    assert resolve_emits("pair") == "pair"


# --- the catalog --------------------------------------------------------------


def test_the_catalog_covers_every_feature_and_op() -> None:
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.ops import OPS

    catalog = declaration_catalog()
    for cls in FEATURES.values():
        assert getattr(cls, "name") in catalog
    for kind in OPS:
        assert kind in catalog
    assert "tracks" in catalog


def test_the_catalog_round_trips_through_json() -> None:
    """An API serves the declarations, and a client answers with the same code path.

    Each one is served inside its editor-catalog entry, and reads back from that
    JSON equal to the declaration validation reads, column contract included.
    """
    served = EditorCatalog.model_validate_json(editor_catalog().model_dump_json())
    declarations = declaration_catalog()
    for entry in served.entries:
        assert entry.declaration == declarations.get(entry.name), entry.name


def test_declarations_are_read_off_what_a_class_declares() -> None:
    """Spot-checks that the reader is reading, not guessing from a name."""
    catalog = declaration_catalog()

    tracks_reader = catalog.get("speed-angvel")
    assert tracks_reader is not None
    assert tracks_reader.consumes.accepts_tracks
    assert tracks_reader.consumes.requires_track_shape

    pairwise = catalog.get("pair-position")
    assert pairwise is not None
    assert pairwise.produces.level == "pair"

    tracker = catalog.get("trex")
    assert tracker is not None
    assert tracker.produces.writes_tracks
    assert tracker.consumes.reads_media
    assert not tracker.produces.writes_media

    encoder = catalog.get("transcode")
    assert encoder is not None
    assert encoder.produces.writes_media
    assert not encoder.produces.writes_tracks

    trainer = catalog.get("train-pose")
    assert trainer is not None
    assert not trainer.produces.writes_tracks
    assert not trainer.consumes.reads_media


def test_an_op_reads_media_unless_its_root_reads_none() -> None:
    """``reads_media`` follows the root's ``reads``, and the rebuild op reads none."""
    catalog = declaration_catalog()

    rebuild = catalog.entries["rebuild-lmt-events"]
    assert not rebuild.consumes.reads_media
    assert not rebuild.produces.writes_tracks
    others = [key for key in TRACKING_ROOTS if key in catalog.entries]
    assert "trex" in others
    for key in others:
        assert catalog.entries[key].consumes.reads_media == (
            TRACKING_ROOTS[key].reads != "none"
        ), key
    assert {key for key in others if TRACKING_ROOTS[key].reads == "none"} == {
        "rebuild-lmt-events"
    }


def test_the_catalog_lists_the_track_spaces_each_feature_computes_in() -> None:
    """Read off the feature's ``track_spaces``, sorted; an op and tracks list none."""
    catalog = declaration_catalog()

    assert catalog.entries["keypoint-kinematics"].track_spaces == (
        "image_px",
        "plane_cm",
        "world_m",
    )
    assert catalog.entries["speed-angvel"].track_spaces == ("image_px", "plane_cm")
    assert catalog.entries["convert-tracks"].track_spaces == ()
    assert TRACKS_DECLARATION.track_spaces == ()


def test_the_editor_catalog_serializes_the_track_spaces() -> None:
    [entry] = [
        entry
        for entry in editor_catalog().entries
        if entry.name == "keypoint-kinematics"
    ]

    served = json.loads(entry.model_dump_json())

    assert served["declaration"]["track_spaces"] == ["image_px", "plane_cm", "world_m"]


def test_convert_labels_declares_writes_labels() -> None:
    """The one op a labels field may reference, and it writes no tracks."""
    catalog = declaration_catalog()

    writers = {
        name
        for name, declared in catalog.entries.items()
        if declared.produces.writes_labels
    }

    assert writers == {"convert-labels"}
    assert not catalog.entries["convert-labels"].produces.writes_tracks
    converter = catalog.entries["convert-tracks"]
    assert converter.produces.writes_tracks
    assert not converter.consumes.reads_media
    assert converter.scope_takes == "none"


def test_the_media_domain_ops_are_the_ones_that_write_media() -> None:
    """The answer is read off each op's declared domain, without a name list.

    The three media ops that wrote media before a variant existed keep their
    answer, and ``preprocess``, which declares the media domain under a separate
    category, writes media as well.
    """
    catalog = declaration_catalog()

    writers = {
        name
        for name, declared in catalog.entries.items()
        if declared.produces.kind == "op" and declared.produces.writes_media
    }

    assert writers == {"transcode", "export-store", "export-joined", "preprocess"}
    assert not any(catalog.entries[name].produces.writes_tracks for name in writers)


# --- no dataset ---------------------------------------------------------------

_REFUSE_WITH_NO_DATASET = """
import sys

from mosaic.core.dataset import Dataset


def refuse(*args, **kwargs):
    raise AssertionError("a Dataset was constructed to answer a connection question")


Dataset.__init__ = refuse

from mosaic.core.pipeline.graph import ConsumerDecl, ProducerDecl, can_join

individual = ProducerDecl(name="speed-angvel", kind="feature", level="individual")
pairwise = ProducerDecl(name="pair-position", kind="feature", level="pair")
consumer = ConsumerDecl(
    name="temporal-stack", kind="feature", accepts_tracks=True, accepts_features=True
)

verdict = can_join([individual, pairwise], consumer)
assert not verdict, "a mismatched join must be refused"
assert "pair every row" in verdict.reason
print("OK")
"""


def test_a_mismatched_join_is_refused_with_no_dataset_constructed() -> None:
    """The whole point of keeping this dataset-independent, asserted rather than said.

    Constructing a ``Dataset`` is made to raise rather than blocking the module,
    because ``mosaic.core`` imports it unconditionally at package import -- so a
    blocked module would prove only that the package was imported, which is not
    the claim. The claim is that no dataset is *opened* to answer this.
    """
    proc = run_child([sys.executable, "-c", _REFUSE_WITH_NO_DATASET])
    assert proc.returncode == 0, (
        f"answering a connection question opened a dataset.\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK" in proc.stdout


# --- lanes --------------------------------------------------------------------


def test_a_lane_is_decided_from_what_a_step_declares() -> None:
    """GPU work splits into fits and the rest, ffmpeg work has its own lane, and
    everything else takes the default one."""
    from mosaic.core.pipeline.graph import lane_for

    catalog = declaration_catalog()
    expected = {
        "speed-angvel": "feature-compute",
        "prepare-training-data": "feature-compute",
        "lightning-action": "gpu-train",  # a global fitter that declares a GPU, whatever its device
        "transcode": "transcode",
        "export-joined": "transcode",
        "export-store": "transcode",
        "train-pose": "gpu-train",
        "train-sleap": "gpu-train",
        "feral": "gpu-train",
        "sequence-classifier": "gpu-train",
        "kpms": "gpu-train",
        "vame": "gpu-train",
        "global-identity-model": "gpu-train",
        "infer-pose": "gpu-infer",
        "trex": "gpu-infer",
    }
    for name, lane in expected.items():
        declared = catalog.get(name)
        assert declared is not None
        assert lane_for(declared) == lane, name


def test_a_submitted_job_finds_its_lane_by_name() -> None:
    """What a control plane holds is a feature slug or an op kind, not a declaration."""
    from mosaic.core.pipeline.graph import lane_for_step

    assert lane_for_step("train-pose") == "gpu-train"
    assert lane_for_step("speed-angvel") == "feature-compute"
    with pytest.raises(KeyError, match="train-poze"):
        _ = lane_for_step("train-poze")


@pytest.mark.parametrize("name", ["xgboost", "global-kmeans"])
@pytest.mark.parametrize(
    ("params", "lane"),
    [
        ({"device": "cpu"}, "feature-compute"),
        ({"device": "cuda"}, "gpu-train"),
        ({}, "feature-compute"),
    ],
    ids=["cpu", "cuda", "default"],
)
def test_a_lane_follows_the_param_that_a_declaration_names(
    name: str, params: dict[str, str], lane: str
) -> None:
    from mosaic.core.pipeline.graph import lane_for

    declared = declaration_catalog().get(name)
    assert declared is not None
    assert declared.resource_class == "cpu"
    assert lane_for(declared, params) == lane


def test_a_lane_without_params_is_the_lane_of_the_defaults() -> None:
    from mosaic.core.pipeline.graph import lane_for

    declared = declaration_catalog().get("xgboost")
    assert declared is not None
    assert declared.param_defaults["device"] == "cpu"
    assert lane_for(declared) == "feature-compute"


def test_a_submitted_job_finds_its_lane_by_name_and_params() -> None:
    from mosaic.core.pipeline.graph import lane_for_step

    assert lane_for_step("xgboost", {"device": "cuda"}) == "gpu-train"
    assert lane_for_step("xgboost", {"device": "cpu"}) == "feature-compute"
    assert lane_for_step("xgboost") == "feature-compute"
    assert lane_for_step("global-kmeans", {"device": "cuda"}) == "gpu-train"


def test_a_param_changes_the_lane_only_where_a_declaration_names_it() -> None:
    from mosaic.core.pipeline.graph import lane_for

    catalog = declaration_catalog()
    for name in catalog.names():
        declared = catalog.get(name)
        assert declared is not None
        if declared.gpu_when is None:
            assert lane_for(declared, {"device": "cuda"}) == lane_for(declared), name


def test_xgboost_and_global_kmeans_are_the_declarations_with_a_gpu_param() -> None:
    catalog = declaration_catalog()
    named = {
        name: declared.gpu_when
        for name in catalog.names()
        if (declared := catalog.get(name)) is not None and declared.gpu_when
    }

    assert named == {
        "xgboost": GpuWhen(field="device", values=("cuda",)),
        "global-kmeans": GpuWhen(field="device", values=("cuda",)),
    }


def test_a_gpu_param_is_a_field_of_its_features_params() -> None:
    catalog = declaration_catalog()
    for name in catalog.names():
        declared = catalog.get(name)
        assert declared is not None
        if declared.gpu_when is None:
            continue
        schema = describe_step_type(name).params_schema
        assert schema is not None
        properties = schema["properties"]
        assert isinstance(properties, dict)
        assert declared.gpu_when.field in properties, name
        assert declared.gpu_when.field in declared.param_defaults, name


def test_trex_takes_a_gpu_lane_from_its_own_declaration() -> None:
    """Its category is 'convert', which would not imply a GPU; it says so itself."""
    from mosaic.core.pipeline.graph import resource_class_of

    declared = declaration_catalog().get("trex")
    assert declared is not None
    assert declared.category == "convert"
    assert resource_class_of(declared) == "gpu"


def test_every_declaration_resolves_to_a_lane() -> None:
    """A step planned without a lane has nowhere to be offered."""
    from mosaic.core.pipeline.graph import lane_for

    catalog = declaration_catalog()
    for name in catalog.names():
        declared = catalog.get(name)
        assert declared is not None
        assert lane_for(declared)


# --- the join, read from the columns each input declares ----------------------


def _columns(*guaranteed: str, possible: Collection[str] = ()) -> ColumnSet:
    """A declared set: the *guaranteed* columns, and the *possible* ones beside them."""
    sure = frozenset(guaranteed)
    return ColumnSet(guaranteed=sure, possible=sure | frozenset(possible))


PER_FRAME_ROWS = _columns("group", "sequence", "frame", "time", "id", "speed")
PER_BIN = _columns("group", "sequence", "bin_start", "bin_end", "speed_mean")
PER_ARENA = _columns("group", "sequence", "arena", "n_frames")
ONE_ROW = _columns("group", "sequence", "density")
"""The outputs of a step that names nobody, with and without a summary key."""


def _declaration(
    consumes: ConsumerDecl = JOINER, emits: EmitsLevel = "individual"
) -> Declaration:
    return Declaration(
        produces=ProducerDecl(name=consumes.name, kind="feature", level="individual"),
        consumes=consumes,
        emits=emits,
    )


def _joined(
    first: ColumnSet,
    second: ColumnSet,
    *,
    consumer: ConsumerDecl = JOINER,
    first_values: dict[str, JsonValue] | None = None,
    second_values: dict[str, JsonValue] | None = None,
) -> str | None:
    return join_problem(
        "step",
        [
            JoinInput("first", first, first_values),
            JoinInput("second", second, second_values),
        ],
        _declaration(consumer),
    )


def _subgroup_cautions(
    first: ColumnSet,
    second: ColumnSet,
    *,
    consumer: ConsumerDecl = JOINER,
    first_values: dict[str, JsonValue] | None = None,
    second_values: dict[str, JsonValue] | None = None,
) -> list[str]:
    return subgroup_cautions(
        "step",
        [
            JoinInput("first", first, first_values),
            JoinInput("second", second, second_values),
        ],
        _declaration(consumer),
    )


def test_an_unidentified_summary_key_that_the_rows_cannot_have_is_refused() -> None:
    expected = (
        "step cannot read second beside first: second's output names nobody and "
        "has bin_start, which first cannot have"
    )

    assert _joined(PER_FRAME_ROWS, PER_BIN) == expected


def test_the_unidentified_side_may_be_the_one_merged_first() -> None:
    expected = (
        "step cannot read first beside second: first's output names nobody and "
        "has bin_start, which second cannot have"
    )

    assert _joined(PER_BIN, PER_FRAME_ROWS) == expected


def test_a_refusal_names_the_inputs_merged_before_the_one_that_is_read() -> None:
    other_rows = _columns("group", "sequence", "frame", "id", "heading")

    message = join_problem(
        "step",
        [
            JoinInput("first", PER_FRAME_ROWS),
            JoinInput("second", other_rows),
            JoinInput("third", PER_BIN),
        ],
        _declaration(),
    )

    assert message == (
        "step cannot read third beside the merge of first and second: third's "
        "output names nobody and has bin_start, which the merge of first and "
        "second cannot have"
    )


def test_every_unshared_key_is_checked_and_the_refusal_names_the_one_that_cannot_be_had() -> (
    None
):
    """The rows may have ``bin_start``, so the key the refusal names is ``arena``."""
    unidentified = _columns("group", "sequence", "bin_start", "arena", "n")
    rows = _columns("group", "sequence", "frame", "id", possible={"bin_start"})

    message = _joined(rows, unidentified)

    assert message is not None
    assert "and has arena, which first cannot have" in message


def test_a_key_the_other_side_may_have_is_left_to_the_run() -> None:
    rows = _columns(
        "group", "sequence", "frame", "id", possible={"bin_start", "bin_end"}
    )

    assert _joined(rows, PER_BIN) is None


def test_an_eligible_pair_is_not_refused() -> None:
    """A per-arena collective beside a one-row collective without summary keys."""
    assert _joined(PER_ARENA, ONE_ROW) is None
    assert _joined(ONE_ROW, PER_ARENA) is None


def test_a_side_that_only_may_be_ineligible_is_left_to_the_run() -> None:
    may_name_individuals = _columns("group", "sequence", "density", possible={"id"})
    may_be_identified = _columns("group", "sequence", "arena", possible={"id"})

    assert _joined(PER_ARENA, may_name_individuals) is None
    assert _joined(may_be_identified, PER_FRAME_ROWS) is None


def test_a_subgroup_table_joined_by_the_entry_is_cautioned_and_not_refused() -> None:
    """The run accepts the join where the other side has one row."""
    expected = (
        "step reads first beside second by the entry, and first sets subgroup_col to "
        "size, with one row per subgroup. The run refuses the join where both have "
        "more than one row. A subgroup does not join"
    )
    values: dict[str, JsonValue] = {"subgroup_col": "size"}

    assert _joined(ONE_ROW, PER_FRAME_ROWS, first_values=values) is None
    assert _subgroup_cautions(ONE_ROW, PER_FRAME_ROWS, first_values=values) == [
        expected
    ]


def test_a_subgroup_producer_is_named_wherever_it_stands_and_a_list_is_spelled_out() -> (
    None
):
    values: dict[str, JsonValue] = {"subgroup_col": ["size", "event"]}

    assert _subgroup_cautions(PER_FRAME_ROWS, ONE_ROW, second_values=values) == [
        "step reads second beside first by the entry, and second sets subgroup_col "
        "to size, event, with one row per subgroup. The run refuses the join where "
        "both have more than one row. A subgroup does not join"
    ]


def test_a_subgroup_table_on_a_position_handed_to_the_entry_join_is_cautioned() -> None:
    values: dict[str, JsonValue] = {"subgroup_col": "size"}

    (message,) = _subgroup_cautions(PER_ARENA, ONE_ROW, first_values=values)

    assert "first sets subgroup_col to size" in message
    assert _joined(PER_ARENA, ONE_ROW, first_values=values) is None


def test_every_subgroup_producer_of_the_inputs_is_cautioned() -> None:
    inputs = [
        JoinInput("first", ONE_ROW, {"subgroup_col": "size"}),
        JoinInput("second", PER_FRAME_ROWS),
        JoinInput("third", _columns("group", "sequence", "n"), {"subgroup_col": "x"}),
    ]

    messages = subgroup_cautions("step", inputs, _declaration())

    assert [message.split(" by the entry")[0] for message in messages] == [
        "step reads first beside second",
        "step reads first beside third",
        "step reads third beside the merge of first and second",
    ]


@pytest.mark.parametrize("unset", [None, "", [], False])
def test_an_unset_subgroup_column_is_no_caution(unset: JsonValue) -> None:
    assert not _subgroup_cautions(
        ONE_ROW, PER_FRAME_ROWS, first_values={"subgroup_col": unset}
    )
    assert not _subgroup_cautions(ONE_ROW, PER_FRAME_ROWS, first_values={})


def test_a_subgroup_table_that_names_individuals_is_no_caution() -> None:
    """Its rows are per individual, and the entry's one row repeats onto each."""
    per_individual = _columns("group", "sequence", "id", "speed_mean")

    assert not _subgroup_cautions(
        per_individual, ONE_ROW, first_values={"subgroup_col": "size"}
    )


def test_an_unbounded_input_is_never_refused() -> None:
    assert _joined(PER_FRAME_ROWS, UNKNOWN_COLUMNS) is None
    assert _joined(UNKNOWN_COLUMNS, PER_BIN) is None
    values: dict[str, JsonValue] = {"subgroup_col": "a"}
    assert _joined(UNKNOWN_COLUMNS, ONE_ROW, second_values=values) is None
    assert not _subgroup_cautions(UNKNOWN_COLUMNS, ONE_ROW, second_values=values)
    assert not _subgroup_cautions(ONE_ROW, UNKNOWN_COLUMNS, first_values=values)


def test_a_shared_alignment_column_leaves_the_join_to_the_run() -> None:
    """A per-frame collective with ``subgroup_col`` beside rows on ``frame`` plans as today."""
    per_frame = _columns("group", "sequence", "frame", "time", "n")

    values: dict[str, JsonValue] = {"subgroup_col": "a"}
    assert _joined(PER_FRAME_ROWS, per_frame, second_values=values) is None
    assert not _subgroup_cautions(PER_FRAME_ROWS, per_frame, second_values=values)
    assert _joined(PER_FRAME_ROWS, PER_BIN) is not None


def test_a_possible_shared_alignment_column_leaves_the_join_to_the_run() -> None:
    """A ``kept(frame)`` behind an undecided condition may be the join key."""
    maybe_per_frame = _columns("group", "sequence", "n", possible={"frame"})

    values: dict[str, JsonValue] = {"subgroup_col": "a"}

    assert _joined(PER_FRAME_ROWS, maybe_per_frame, second_values=values) is None
    assert not _subgroup_cautions(PER_FRAME_ROWS, maybe_per_frame, second_values=values)
    assert (
        _joined(PER_FRAME_ROWS, _columns("bin_start", "n", possible={"frame"})) is None
    )


def test_a_possible_summary_key_leaves_the_join_to_the_run() -> None:
    maybe_arena = _columns("group", "sequence", "n", possible={"arena"})

    values: dict[str, JsonValue] = {"subgroup_col": "a"}

    assert _joined(maybe_arena, PER_ARENA, first_values=values) is None
    assert not _subgroup_cautions(maybe_arena, PER_ARENA, first_values=values)


def test_a_join_on_summary_keys_that_both_guarantee_is_not_refused() -> None:
    per_id_bins = _columns("group", "sequence", "id", "bin_start", "bin_end", "speed")

    values: dict[str, JsonValue] = {"subgroup_col": "a"}

    assert _joined(per_id_bins, PER_BIN, second_values=values) is None
    assert not _subgroup_cautions(per_id_bins, PER_BIN, second_values=values)


@pytest.mark.parametrize("consumer", [CROSS, STACKS])
def test_a_consumer_that_cross_joins_or_stacks_is_not_checked(
    consumer: ConsumerDecl,
) -> None:
    assert _joined(PER_FRAME_ROWS, PER_BIN, consumer=consumer) is None


def test_a_position_that_widens_is_not_checked() -> None:
    pairs = _columns("group", "sequence", "frame", "id1", "id2", "dist")
    individuals = _columns("group", "sequence", "id", "arena", "n")

    assert _joined(pairs, individuals, consumer=WIDENING) is None


def test_one_input_is_no_join() -> None:
    assert join_problem("step", [JoinInput("first", PER_BIN)], _declaration()) is None


def test_the_tracks_may_hold_an_arena_and_no_bin() -> None:
    """An arena summary reads the tracks like any other input that may carry it."""
    tracks = tracks_columns(declaration_catalog())

    assert _joined(tracks, PER_ARENA) is None
    refusal = _joined(tracks, PER_BIN)
    assert refusal is not None
    assert "and has bin_start, which first cannot have" in refusal


# --- the caution ------------------------------------------------------------


def _level_caution(
    first: ColumnSet, *later: ColumnSet, emits: EmitsLevel = "as-input"
) -> str | None:
    inputs = [JoinInput("first", first)]
    inputs.extend(JoinInput(f"later{n}", one) for n, one in enumerate(later, start=1))
    return level_caution("step", inputs, _declaration(emits=emits))


def test_an_as_input_consumer_whose_first_input_names_nobody_is_cautioned() -> None:
    expected = (
        "step emits as-input and takes its level from input 0, first, which names "
        "nobody. Put the input with the finest rows first"
    )

    assert _level_caution(ONE_ROW, PER_FRAME_ROWS) == expected
    assert _level_caution(PER_BIN, ONE_ROW, PER_FRAME_ROWS) == expected


@pytest.mark.parametrize(
    "inputs",
    [
        (PER_FRAME_ROWS, ONE_ROW),
        (ONE_ROW,),
        (ONE_ROW, PER_ARENA),
        (ONE_ROW, _columns("group", "sequence", "n", possible={"id"})),
        (ONE_ROW, UNKNOWN_COLUMNS),
        (UNKNOWN_COLUMNS, PER_FRAME_ROWS),
    ],
)
def test_no_other_order_of_inputs_is_cautioned(inputs: tuple[ColumnSet, ...]) -> None:
    assert _level_caution(*inputs) is None


def test_a_consumer_that_does_not_take_its_level_from_input_zero_is_not_cautioned() -> (
    None
):
    assert _level_caution(ONE_ROW, PER_FRAME_ROWS, emits="individual") is None
