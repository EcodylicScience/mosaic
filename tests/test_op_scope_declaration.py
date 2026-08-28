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
            "train-pose",
            "train-points",
            "train-localizer",
            "train-sleap",
            "train-litpose",
        }

    def test_the_scope_dependent_ops(self) -> None:
        dependent = {kind for kind, op in OPS.items() if op.scope_dependent}
        assert dependent == {"transcode", "export-store", "resample-tracks"}

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


def _selector(entries: list[Entry] | None) -> Scope:
    """The same coverage as a selector, matching the params field beside it.

    An op reads its entries from its params field, and these gates write
    them there. Passing the same entries through the scope argument keeps
    the two saying one thing.
    """
    return Scope(entries=entries)


GATED_OPS = frozenset(
    [
        "extract-frames",
        "infer-localizer",
        "infer-points",
        "infer-pose",
        "litpose",
        "sleap",
        "trex",
        "ultralytics",
    ]
)
"""The ops both gates below cover: they take a scope and declare independence."""


def _gated_kinds() -> frozenset[str]:
    """The op kinds whose declarations the gates below check.

    Selected from the two declarations rather than from a params field name.
    Reading the population off a field spelled ``entries`` is the inference
    these declarations replace, and it answers wrongly for an op spelling its
    scope otherwise -- ``export-store`` already spells one ``entry``.
    """
    return frozenset(
        kind
        for kind, op in OPS.items()
        if op.scope_takes != "none" and not op.scope_dependent
    )


def _op_these_gates_cover(kind: str) -> type[Op[Params]]:
    """Op *kind*, or a skip when its declarations put it outside these gates.

    An op declaring ``scope_dependent`` claims the dependence these gates refuse
    to find, and one declaring ``scope_takes = "none"`` takes no scope to leak.
    Both are skipped by name rather than passed, which keeps the count of ops
    actually exercised visible in the report.

    An op inside the population whose params hold no ``entries`` field fails
    instead. The gates read a scope off that field, and a skip there would
    report a widening they need as an op they cover.
    """
    op = OPS[kind]
    if kind not in _gated_kinds():
        pytest.skip(
            f"{kind} declares scope_takes={op.scope_takes!r} and "
            f"scope_dependent={op.scope_dependent}"
        )
    if "entries" not in op.Params.model_fields:
        pytest.fail(
            f"{kind} takes a scope, declares independence, and holds no "
            f"'entries' field. These gates read the scope off that field and "
            f"need widening for an op that spells it otherwise."
        )
    return op


class TestADeclaredIndependenceHoldsInThePayload:
    """``scope_dependent = False`` keeps the scope out of the hashed payload.

    The direct leak, and complete for every op it covers. A scope reaching
    ``identity_dump`` names one computation two things once a caller narrows it.
    """

    @pytest.mark.parametrize("kind", sorted(OPS))
    def test_entries_do_not_reach_the_identity_payload(self, kind: str) -> None:
        op = _op_these_gates_cover(kind)
        params = op.Params.model_validate(minimal_op_params(kind))
        assert "entries" not in params.identity_dump()


class TestADeclaredIndependenceHoldsAgainstTheDataset:
    """``scope_dependent = False`` names one run whatever the scope covers.

    The indirect leak, which the payload check cannot see. ``resample-tracks``
    has that shape. Its scope filters the tracks index and the surviving variant
    enters every identifier, while its params payload holds no scope at all.

    Partial today, and prospective rather than additive. The two ops it reaches
    are pure functions of their params. It currently proves nothing the payload
    check does not, and it grows with the fixtures. Six ops defer on this
    dataset, and each deferral is recorded as a skip carrying the reason the op
    gave.

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
        for entries in ENTRY_SETS:
            payload = minimal_op_params(kind)
            payload["entries"] = entries
            params = op.Params.model_validate(payload)
            scope = three_entry_dataset.resolve_scope(_selector(entries))
            try:
                identities.add(op().plan_identity(three_entry_dataset, params, scope))
            except IdentityDeferred as exc:
                deferred[repr(entries)] = exc.because
        if len(deferred) == len(ENTRY_SETS):
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
            payload = minimal_op_params("resample-tracks")
            payload["entries"] = entries
            return op().plan_identity(
                dataset,
                op.Params.model_validate(payload),
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
        assert rows["transcode"]["scope_takes"] == "at-least-one"
        assert rows["transcode"]["scope_dependent"] is True
        assert rows["train-pose"]["scope_takes"] == "none"
        assert rows["train-pose"]["scope_dependent"] is False

    def test_describe_op_carries_both_declarations(self) -> None:
        described = describe_op("export-store")
        assert described["scope_takes"] == "exactly-one"
        assert described["scope_dependent"] is True

    def test_neither_reaches_the_params_schema(self) -> None:
        """A client drawing controls from the schema must not draw a declaration.

        Read over the whole rendered document rather than its top-level
        properties, which covers a nested appearance too.
        """
        rendered = json.dumps(describe_op("transcode")["params_schema"], default=str)
        assert "scope_takes" not in rendered
        assert "scope_dependent" not in rendered
        assert "entries" in rendered

    def test_the_declaration_catalog_carries_them(self) -> None:
        """A canvas refuses a wire with no dataset and reads them here."""
        catalog = declaration_catalog()
        assert catalog.entries["transcode"].scope_takes == "at-least-one"
        assert catalog.entries["transcode"].scope_dependent is True

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
