"""Every op says how much scope it takes and whether coverage names its output.

Both are declared on the class rather than inferred from its params model. The
four ops that spell a scope four ways -- a required list, a singular field, a
nullable default, an absence -- gave inference four answers to one question.
"""

import inspect
import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.entry import Entry
from mosaic.core.params import Params
from mosaic.core.pipeline.graph.compatibility import TRACKS_DECLARATION
from mosaic.core.pipeline.graph.resolve import declaration_catalog
from mosaic.core.pipeline.ops import (
    OPS,
    SCOPE_TAKES_VALUES,
    IdentityDeferred,
    Op,
    OpIdentity,
    ScopeTakes,
    describe_op,
    list_ops,
    register_op,
)
from mosaic.core.scope import Scope
from mosaic.tracking import register_ops
from tests.helpers import add_tracks_variant, make_dataset, minimal_op_params

register_ops()

LEGAL_VALUES = frozenset(["none", "any", "at-least-one", "exactly-one"])
"""The vocabulary, restated so widening it is a decision rather than a drift."""


class TestTheVocabulary:
    def test_the_declared_values_are_the_ones_ops_may_use(self) -> None:
        assert SCOPE_TAKES_VALUES == LEGAL_VALUES


class TestEveryOpDeclares:
    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_scope_takes_is_declared_and_legal(self, kind: str) -> None:
        assert OPS[kind].scope_takes in LEGAL_VALUES

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_scope_dependent_is_declared(self, kind: str) -> None:
        assert isinstance(OPS[kind].scope_dependent, bool)


class TestRegistrationRefuses:
    def test_an_op_without_scope_takes_is_refused(self) -> None:
        class Undeclared(Op[Params]):
            kind = "undeclared-scope-takes"
            category = "convert"
            domain = "tracking"
            scope_dependent = False
            Params = Params

        with pytest.raises(TypeError, match="scope_takes"):
            _ = register_op(Undeclared)

    def test_an_op_without_scope_dependent_is_refused(self) -> None:
        class Undeclared(Op[Params]):
            kind = "undeclared-scope-dependent"
            category = "convert"
            domain = "tracking"
            scope_takes = "any"
            Params = Params

        with pytest.raises(TypeError, match="scope_dependent"):
            _ = register_op(Undeclared)

    def test_an_illegal_scope_takes_is_refused(self) -> None:
        """A typo the annotation cannot catch, because a payload can carry one."""

        class Illegal(Op[Params]):
            kind = "illegal-scope-takes"
            category = "convert"
            domain = "tracking"
            scope_dependent = False
            Params = Params

        # Set after the class body. Inside it the declared Literal narrows the
        # assignment and the checker rejects the typo, which is the case a
        # checker already covers. Registration is what catches the rest.
        setattr(Illegal, "scope_takes", "some")
        with pytest.raises(TypeError, match="some"):
            _ = register_op(Illegal)

    def test_a_refused_op_does_not_enter_the_registry(self) -> None:
        """A refusal leaves the registry as it was."""
        assert "undeclared-scope-takes" not in OPS
        assert "undeclared-scope-dependent" not in OPS
        assert "illegal-scope-takes" not in OPS


class TestTheDeclarationsAreWhatWeIntend:
    """Pinned so a change to one is deliberate and shows in a diff."""

    def test_the_scope_free_ops(self) -> None:
        free = {kind for kind, op in OPS.items() if op.scope_takes == "none"}
        assert free == {
            "convert-points",
            "prepare-training-data",
            "train-pose",
            "train-points",
            "train-localizer",
            "train-sleap",
            "train-litpose",
        }

    def test_the_scope_dependent_ops(self) -> None:
        dependent = {kind for kind, op in OPS.items() if op.scope_dependent}
        assert dependent == {"resample-tracks"}

    def test_the_arity_constrained_ops(self) -> None:
        assert OPS["transcode"].scope_takes == "at-least-one"
        assert OPS["export-store"].scope_takes == "exactly-one"

    def test_the_ops_the_scope_gates_cover(self) -> None:
        """Pinned so the gates below cannot quietly cover more ops or fewer."""
        assert _gated_kinds() == GATED_OPS


ENTRY_SETS: tuple[list[Entry] | None, ...] = (
    None,
    [("A", "one")],
    [("A", "two"), ("B", "one")],
)
"""Three scopes over one dataset: unset, one entry, and two others."""

NARROWED_ENTRY_SETS: tuple[list[Entry] | None, ...] = ENTRY_SETS[1:]
"""The two of :data:`ENTRY_SETS` an op refusing an unset selector can take."""

SINGLE_ENTRY_SETS: tuple[list[Entry] | None, ...] = (
    [("A", "one")],
    [("A", "two")],
)
"""Two scopes an ``exactly-one`` op can receive: one entry, then another.

Unreached today. ``export-store`` is the only ``exactly-one`` op and this
dataset cannot feed it, so the gate skips it before these are chosen. Kept so
the selection below stays total over ``SCOPE_TAKES_VALUES``, which is what the
next such op will need.
"""


def _entry_sets_for(scope_takes: ScopeTakes) -> tuple[list[Entry] | None, ...]:
    """The scopes the gate below asks an op declaring *scope_takes* about.

    Chosen from the declaration, so an op is only ever handed a scope its own
    declaration admits. ``check_scope_takes`` refuses the rest before any op
    body runs, in ``run_op`` and in the planner alike, so a gate handing an op
    a wider scope than it accepts tests a path production cannot reach -- and
    what gets bent to survive such a test is the op.

    An unset selector resolves to no entries and carries "every indexed entry"
    in the selector, so ``at-least-one`` and ``exactly-one`` both refuse it and
    both drop it here. ``any`` and ``none`` keep all three: an op declaring it
    reads no scope still takes the argument and can still read it, which is
    the defect this gate is here to find.

    The ``exactly-one`` branch answers for no op today -- ``export-store`` is
    the only one and :data:`OPS_THE_FIXTURE_CANNOT_FEED` skips it first -- and
    is written rather than raised because the day such an op can be fed here,
    the question it must be asked is :data:`SINGLE_ENTRY_SETS`: one entry,
    then a different one. That is the sharper form of the same question, since
    an identity moving with *which* entry was named survives a comparison
    against a wider scope.
    """
    if scope_takes == "exactly-one":
        return SINGLE_ENTRY_SETS
    if scope_takes == "at-least-one":
        return NARROWED_ENTRY_SETS
    return ENTRY_SETS


def _selector(entries: list[Entry] | None) -> Scope:
    """*entries* as the selector a caller writes.

    ``None`` gives an unset selector, which covers every indexed entry. That
    is the first member of :data:`ENTRY_SETS` and the one an op accepting it
    must name the same run under as the others.
    """
    return Scope(entries=entries)


OPS_THE_FIXTURE_CANNOT_FEED = frozenset(["export-store"])
"""Ops the dataset gate leaves out, holding no input this dataset can feed.

``export-store`` reads imgstore recordings and ``three_entry_dataset`` holds
plain videos, so ``_stores_for`` refuses every scope the op admits before an
identity exists. Twelve of the gated ops sit out for the same kind of reason --
no weights, no ``data.yaml`` -- and say so through ``IdentityDeferred``, which
this one cannot: nothing upstream writes a recording, so an entry holding no
store is a refusal rather than a deferral. Its invariance is measured against
a store dataset instead, by
``tests/test_store_export.py::test_two_single_entry_scopes_name_one_run``.
"""


GATED_OPS = frozenset(
    [
        "convert-points",
        "extract-frames",
        "infer-localizer",
        "infer-points",
        "infer-pose",
        "litpose",
        "prepare-training-data",
        "sleap",
        "train-litpose",
        "train-localizer",
        "train-points",
        "train-pose",
        "train-sleap",
        "transcode",
        "trex",
        "ultralytics",
    ]
)
"""The ops the dataset gate below covers.

Every op declaring independence, less the one
:data:`OPS_THE_FIXTURE_CANNOT_FEED` names. The payload gate runs over all
eighteen and needs no population.
"""


def _gated_kinds() -> frozenset[str]:
    """The op kinds whose declarations the gates below check.

    Two axes, and they are about different things. The population is
    selected from ``scope_dependent`` rather than from a params field name:
    reading it off a field spelled ``entries`` is the inference these
    declarations replace, and it answers wrongly for an op spelling its scope
    otherwise -- ``export-store`` spelled one ``entry``. Then the ops
    :data:`OPS_THE_FIXTURE_CANNOT_FEED` names are dropped, which is a
    statement about this dataset rather than about any declaration, and each
    is named individually so that a second one is a deliberate edit.

    ``scope_takes = "none"`` does not exclude an op. Every op takes the scope
    as an argument to ``plan_identity`` and ``run``, and one that declares it
    reads no scope can read the argument anyway. That is the same defect as a
    scoped op reading it, and the same gate finds both. The exclusion held only
    while a scope arrived through a params field a scope-free op did not
    declare.
    """
    return frozenset(
        kind
        for kind, op in OPS.items()
        if not op.scope_dependent and kind not in OPS_THE_FIXTURE_CANNOT_FEED
    )


def _op_these_gates_cover(kind: str) -> type[Op[Params]]:
    """Op *kind*, or a skip when its declaration puts it outside these gates.

    An op declaring ``scope_dependent`` claims the dependence these gates
    refuse to find. It is skipped by name rather than passed, which keeps the
    count of ops the gates exercise visible in the report. So is one this
    dataset cannot feed, under its own reason: the two exclusions are not the
    same statement.

    The population is pinned by
    :meth:`TestTheDeclarationsAreWhatWeIntend.test_the_ops_the_scope_gates_cover`,
    so a skip cannot shrink what these gates exercise.
    """
    op = OPS[kind]
    if kind in OPS_THE_FIXTURE_CANNOT_FEED:
        pytest.skip(f"{kind}: this dataset holds no input it can read")
    if kind not in _gated_kinds():
        pytest.skip(f"{kind} declares scope_dependent={op.scope_dependent}")
    return op


COVERAGE_NAMES = frozenset(
    ["entries", "entry", "camera", "cameras", "groups", "sequences"]
)
"""Every spelling a coverage has worn in an op model."""


SCOPE_SHAPED_NAMES = COVERAGE_NAMES | {"overwrite"}
"""The coverage spellings plus the recompute decision.

Everything an attempt decides and a recipe does not. No op params model
declares any of them, and the two tests below pin the halves separately so a
regression names which one came back.
"""


class TestNoScopeReachesTheHashedPayload:
    """What a run covers and whether it recomputes never name the run.

    The direct leak, over **every** registered op rather than the sixteen the
    dataset gate below covers. Every op now takes both as arguments and declares no
    such field, and that is why this passes for all eighteen. It
    stays as the gate an op reintroducing one meets. The field would be
    ``HASH_EXCLUDE`` or the payload would name it, and a payload naming either
    gives one computation two names as soon as a caller narrows it.

    Read over the whole forbidden vocabulary rather than ``entries`` alone,
    because the spelling has differed per op and a new op is free to invent
    another one.
    """

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_no_scope_shaped_name_reaches_the_identity_payload(self, kind: str) -> None:
        params = OPS[kind].Params.model_validate(minimal_op_params(kind))
        leaked = SCOPE_SHAPED_NAMES & set(params.identity_dump())
        assert not leaked, f"{kind} hashes {sorted(leaked)}"


class TestADeclaredIndependenceHoldsAgainstTheDataset:
    """``scope_dependent = False`` names one run whatever the scope covers.

    The indirect leak, which the payload check cannot see. ``resample-tracks``
    has that shape. Its scope filters the tracks index and the surviving variant
    enters every identifier, while its params payload holds no scope at all.

    Partial today, and prospective rather than additive. Seventeen ops declare
    independence, sixteen of them are gated here, and this dataset lets three
    answer: ``extract-frames`` and ``trex``, both pure functions of their
    params, and ``transcode``, whose identity is its recipe and which reads
    the dataset only for refusals that answer the same way for every entry
    set. The other twelve defer, each deferral recorded as a skip stating the
    reason the op gave: a training op has no data.yaml here, an inference op
    no weights. It proves nothing the payload check does not, and it grows as
    the fixture gains those artifacts.

    Whole :class:`OpIdentity` values are compared. ``run_id`` alone leaves a
    ``tracks_variant`` free to move with the scope, which mints one variant
    directory for two coverages.
    """

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_the_identity_is_unchanged_across_entry_sets(
        self, kind: str, three_entry_dataset: Dataset
    ) -> None:
        op = _op_these_gates_cover(kind)
        identities: set[OpIdentity] = set()
        deferred: dict[str, str] = {}
        params = op.Params.model_validate(minimal_op_params(kind))
        entry_sets = _entry_sets_for(op.scope_takes)
        for entries in entry_sets:
            scope = three_entry_dataset.resolve_scope(_selector(entries))
            try:
                identities.add(op().plan_identity(three_entry_dataset, params, scope))
            except IdentityDeferred as exc:
                deferred[repr(entries)] = exc.because
        if len(deferred) == len(entry_sets):
            reasons = "; ".join(sorted(set(deferred.values())))
            pytest.skip(f"{kind} defers its identity: {reasons}")
        if deferred:
            scopes = "; ".join(
                f"{entries} ({because})"
                for entries, because in sorted(deferred.items())
            )
            message = (
                f"{kind} declares scope_dependent = False, defers its identity "
                f"for {scopes}, and answers for the other scopes. Deferring "
                f"under one coverage and answering under another is a "
                f"dependence on the scope."
            )
            pytest.fail(message)
        assert len(identities) == 1


class TestTheIndirectPathIsMeasured:
    """``resample-tracks`` reaches its identity from the scope off-payload.

    Its params hold no scope term the payload check can see. The scope filters
    the tracks index, and the one variant that survives enters both the run
    identifier and the tracks variant. Both are asserted here.

    The parametrized gate above cannot reach this shape, because a dataset with
    no tracks defers the identity before the scope is read. This measures it
    against a dataset holding two variants, and ties the declaration to the
    measurement.
    """

    def test_two_scopes_over_two_variants_name_two_runs(self, tmp_path: Path) -> None:
        dataset = make_dataset(tmp_path / "resample")
        add_tracks_variant(dataset, "trex.0.1-aaaaaaaaaa", "one")
        add_tracks_variant(dataset, "trex.0.1-bbbbbbbbbb", "two")
        op = OPS["resample-tracks"]

        def identity_over(sequence: str) -> OpIdentity:
            entries: list[Entry] = [("", sequence)]
            return op().plan_identity(
                dataset,
                op.Params.model_validate(minimal_op_params("resample-tracks")),
                dataset.resolve_scope(_selector(entries)),
            )

        first, second = identity_over("one"), identity_over("two")
        assert first.run_id != second.run_id
        assert first.tracks_variant != second.tracks_variant
        # The one line tying a measured behavior to a declared value. The pinned
        # sets above compare a declaration with a declaration.
        assert op.scope_dependent


class TestPublished:
    """Both declarations reach a client that reads an op without running it."""

    def test_list_ops_carries_both_declarations(self) -> None:
        rows = {row["kind"]: row for row in list_ops()}
        assert rows["resample-tracks"]["scope_takes"] == "any"
        assert rows["resample-tracks"]["scope_dependent"] is True
        assert rows["train-pose"]["scope_takes"] == "none"
        assert rows["train-pose"]["scope_dependent"] is False

    def test_describe_op_carries_both_declarations(self) -> None:
        assert describe_op("resample-tracks")["scope_dependent"] is True
        described = describe_op("export-store")
        assert described["scope_takes"] == "exactly-one"
        assert described["scope_dependent"] is False

    def test_neither_reaches_the_params_schema(self) -> None:
        """A client drawing controls from the schema must not draw a declaration.

        Read over the whole rendered document rather than its top-level
        properties, which covers a nested appearance too.
        """
        rendered = json.dumps(describe_op("transcode")["params_schema"], default=str)
        assert "scope_takes" not in rendered
        assert "scope_dependent" not in rendered
        assert "target" in rendered, "the settings are published"

    def test_no_op_declares_a_coverage_field(self) -> None:
        """A client draws its controls from these fields, and a coverage is not one.

        Which entries a run covers is an argument to the run. No op params
        model declares a coverage under any spelling the eighteen have used,
        and the published schema is generated from these fields.
        """
        declaring = {
            kind for kind in OPS if COVERAGE_NAMES & set(OPS[kind].Params.model_fields)
        }
        assert declaring == set()

    def test_no_op_params_model_declares_overwrite(self) -> None:
        """Op params are settings. Whether to recompute arrives beside them.

        Two attempts differing only in whether they redo the work are one
        recipe. Every op takes the decision as the ``overwrite`` argument
        :meth:`~mosaic.core.pipeline.ops.Op.run` receives, whose name and
        position :class:`TestOpInterface` pins for all eighteen.

        That a body reads it is measured per op, in both directions, wherever
        an op has a reuse gate: the five training ops and ``convert-points`` in
        ``tests/test_training_reuse.py``, ``tests/test_train_sleap.py`` and
        ``tests/test_train_litpose.py``, ``transcode`` and ``export-store`` in
        their own suites, and ``extract-frames`` in
        ``tests/test_frame_extraction.py``, where the argument answers a
        refusal instead of a recompute.
        """
        declaring = {
            kind for kind in OPS if "overwrite" in OPS[kind].Params.model_fields
        }
        assert declaring == set()

    def test_the_declaration_catalog_carries_them(self) -> None:
        """A canvas refuses a wire with no dataset and reads them here."""
        catalog = declaration_catalog()
        assert catalog.entries["transcode"].scope_takes == "at-least-one"
        assert catalog.entries["transcode"].scope_dependent is False
        assert catalog.entries["resample-tracks"].scope_dependent is True

    def test_a_feature_declaration_takes_neither(self) -> None:
        """No feature refuses a scope, and one legal value teaches nothing."""
        catalog = declaration_catalog()
        assert catalog.entries["speed-angvel"].scope_takes == ""

    def test_the_tracks_declaration_takes_neither(self) -> None:
        """The dataset's tracks are a producer rather than a step."""
        assert TRACKS_DECLARATION.scope_takes == ""
        assert TRACKS_DECLARATION.scope_dependent is False


class TestOpInterface:
    """Every op takes its coverage and its recompute decision as arguments.

    A recipe states settings. Which entries a run covers and whether it
    recomputes belong to the attempt, and the signature is the one place that
    states it for every registered op at once.
    """

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_run_takes_scope_and_overwrite(self, kind: str) -> None:
        parameters = list(inspect.signature(OPS[kind].run).parameters)
        assert parameters == ["self", "ds", "params", "scope", "overwrite", "ctx"]

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_plan_identity_takes_scope(self, kind: str) -> None:
        parameters = list(inspect.signature(OPS[kind].plan_identity).parameters)
        assert parameters[:4] == ["self", "ds", "params", "scope"]

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_plan_identity_keeps_require_data(self, kind: str) -> None:
        """An override that drops it raises TypeError where a run body passes it.

        Five run bodies call ``plan_identity(..., require_data=False)``.
        Only basedpyright catches an override that omits the parameter, and
        CI does not gate basedpyright.
        """
        parameter = inspect.signature(OPS[kind].plan_identity).parameters[
            "require_data"
        ]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is True

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_target_takes_scope(self, kind: str) -> None:
        parameters = list(inspect.signature(OPS[kind].target).parameters)
        assert parameters == ["self", "params", "scope"]
