"""What every registered feature must declare, asserted for all of them at once.

The tracker suite next door explains the shape and the reason: a declaration
nobody parametrized is a declaration the next copied-from-a-neighbour class ships
without, and the omission surfaces later as silence rather than as a failure.

The feature-side sweeps that existed before this file were unparametrized loops
inside ``test_hashing_rules.py`` that collected every violation into one
assertion message. That reports *that* something is undeclared; parametrizing
reports *which*, by name, which is what a person fixing it needs.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest
from pydantic import BaseModel

from mosaic.behavior.feature_library.registry import FEATURES

EMITS_LEVELS = frozenset({"individual", "pair", "unidentified", "as-input"})
"""Spelled out rather than derived from the type alias, so a value silently
added to the alias still has to be added here deliberately."""

FEATURE_CLASSES = sorted(FEATURES.values(), key=lambda cls: cls.__name__)
FEATURE_IDS = [cls.__name__ for cls in FEATURE_CLASSES]


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_every_feature_declares_accepts_overlap(cls: type) -> None:
    """Whether a feature may be handed several entries at once is not a default.

    With ``overlap_frames > 0`` the frame handed to ``apply`` spans the
    neighbouring sequences, so ``group`` and ``sequence`` are no longer constant
    down it. A feature that reads its identity off row 0 then stamps its
    neighbour's name onto its output, and one that opens media for that identity
    reads the wrong video -- neither of which raises. Only the feature knows, so
    an undeclared one is an error rather than an assumed ``False``.
    """
    assert hasattr(cls, "accepts_overlap"), (
        f"{cls.__name__} declares no 'accepts_overlap'. Declare it: True if "
        f"apply() reads nothing from the frame that is true of one entry alone, "
        f"False otherwise."
    )
    assert isinstance(cls.accepts_overlap, bool)


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_every_global_and_media_feature_declares_costly_fit(cls: type) -> None:
    """Whether a superseded run may be removed is not a default for a fit.

    A superseded-run cleanup keeps every run of a feature that declares
    ``costly_fit``: a model that took a GPU or hours to fit is not removed to
    reclaim disk. A global or media feature that declared nothing would be read
    as cheap, which is the wrong direction for one copied from a cheap neighbour.
    """
    category: object = getattr(cls, "category", "")
    if category not in ("global", "media"):
        pytest.skip("not a global or media feature")
    assert "costly_fit" in vars(cls), (
        f"{cls.__name__} is a {category} feature and declares no "
        f"'costly_fit'. Declare it: True when fitting it again costs a GPU or "
        f"hours, False otherwise."
    )
    assert isinstance(vars(cls)["costly_fit"], bool)


def _sets_on_instance(cls: type, attribute: str) -> bool:
    """Whether the class's own ``__init__`` assigns ``self.<attribute>``."""
    init = vars(cls).get("__init__")
    if init is None:
        return False
    tree = ast.parse(textwrap.dedent(inspect.getsource(init)))
    return any(
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ast.Store)
        for node in ast.walk(tree)
    )


def _learns_or_depends_on_scope(cls: type) -> bool:
    """Whether a feature is a global fitter or names its run by its scope.

    The class default or an instance assignment in ``__init__`` counts for
    ``scope_dependent``, because a feature may declare it only per instance.
    """
    return (
        getattr(cls, "category", "") == "global"
        or getattr(cls, "scope_dependent", False) is True
        or _sets_on_instance(cls, "scope_dependent")
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_every_global_or_scope_dependent_feature_declares_fits(cls: type) -> None:
    """Whether a feature learns from its inputs is not implied by its scope.

    ``scope_dependent`` is not "fits": ``xgboost`` and the ``global-*`` fitters
    learn and name their run without the entries in scope, and ``evaluate-labels``
    names its run by them and learns nothing. A guard that keeps a held-out group
    out of a fit reads ``fits``, so a feature copied from a neighbour that
    declared nothing would be read as learning nothing, the wrong direction for
    a fitter.
    """
    if not _learns_or_depends_on_scope(cls):
        pytest.skip("neither a global feature nor scope-dependent")
    assert hasattr(cls, "fits"), (
        f"{cls.__name__} is global or scope-dependent and declares no 'fits'. "
        f"Declare it: True when it learns something from its inputs, False when "
        f"it only scores or applies them."
    )
    declared: object = getattr(cls, "fits")  # noqa: B009
    assert isinstance(declared, bool)


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_a_feature_that_clears_scope_dependent_per_instance_clears_fits(
    cls: type,
) -> None:
    """A run given a model or fixed weights fits nothing, and says so on the instance.

    The class declares the fitting answer, which the catalog reads. The instance
    narrows ``scope_dependent`` when the params make the run apply instead of
    fit, and ``fits`` must narrow beside it or a planner reading the instance
    would take an apply step for a fit.
    """
    if not _sets_on_instance(cls, "scope_dependent"):
        pytest.skip("scope_dependent is not set per instance")
    assert getattr(cls, "fits", False) is True, (
        f"{cls.__name__} sets 'scope_dependent' per instance but declares "
        f"fits = True on the class no differently from a fit."
    )
    assert _sets_on_instance(cls, "fits"), (
        f"{cls.__name__} narrows 'scope_dependent' in __init__ and does not "
        f"narrow 'fits' beside it."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_a_fitting_feature_given_a_model_param_narrows_fits_per_instance(
    cls: type,
) -> None:
    """A run told which model to apply fits nothing, whatever its identity covers.

    The held-out guard reads ``fits`` off the instance, and an apply step left
    declaring ``True`` would be refused over the test group. A fitter that hashes
    its scope narrows ``scope_dependent`` beside it (``arhmm``, ``kpms``, ``vame``,
    the identity models), which the check above also covers. A scope-free fitter is
    not the guard's concern, except one
    that declares ``fit_set_pinned_by``: ``sequence-classifier`` and
    ``lightning-action`` hash no scope and still narrow ``fits`` for a given
    ``model``, which the planner and the catalog read.
    """
    params: object = getattr(cls, "Params", None)
    if not (isinstance(params, type) and issubclass(params, BaseModel)):
        pytest.skip("no Params model")
    fits: object = getattr(cls, "fits", False)
    named: object = getattr(cls, "scope_dependent", False)
    pinned = getattr(cls, "fit_set_pinned_by", None) is not None
    if "model" not in params.model_fields or fits is not True or not (named or pinned):
        pytest.skip("takes no model, does not fit, or a scope-free fitter")
    assert _sets_on_instance(cls, "fits"), (
        f"{cls.__name__} fits and takes a 'model' param to apply, and does not "
        f"narrow 'fits' in __init__ for a run given one."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_a_scope_free_fitter_declares_fits_and_narrows_it_per_instance(
    cls: type,
) -> None:
    """A fit set pinned outside the scope leaves the identity scope-free and ``fits`` set.

    The class says ``fits = True`` for the catalog, and the instance clears it for
    a run that applies a stored model or fixed weights. Nothing sets
    ``scope_dependent`` on the instance: the identity does not depend on the params.
    """
    if getattr(cls, "fit_set_pinned_by", None) is None:
        pytest.skip("does not pin its fit set outside the scope")
    assert getattr(cls, "scope_dependent", None) is False
    assert not _sets_on_instance(cls, "scope_dependent"), (
        f"{cls.__name__} pins its fit set and sets 'scope_dependent' per instance."
    )
    assert getattr(cls, "fits", False) is True
    assert _sets_on_instance(cls, "fits"), (
        f"{cls.__name__} pins its fit set and does not narrow 'fits' in __init__ "
        "for a run that fits nothing."
    )


def _reads_row_zero_of(node: ast.AST, frame_name: str) -> list[str]:
    """Identity columns read as ``<frame_name>[...].iloc[0]`` under *node*.

    Precise about *which* frame is indexed, because the distinction is the whole
    point. Reading row 0 of the frame ``apply`` was handed is the defect: with
    overlap that row belongs to the previous sequence. Reading row 0 of a
    per-frame group is correct and is how a feature that accepts overlap has to
    get its identity, since identity is constant within a frame.
    """
    found: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        # <expr>.iloc[0]
        value = child.value
        if not (isinstance(value, ast.Attribute) and value.attr == "iloc"):
            continue
        if not (isinstance(child.slice, ast.Constant) and child.slice.value == 0):
            continue
        inner = value.value
        if not isinstance(inner, ast.Subscript):
            continue
        base = inner.value
        if not (isinstance(base, ast.Name) and base.id == frame_name):
            continue
        column = ast.unparse(inner.slice)
        if "seq_col" in column or "group_col" in column:
            found.append(column)
    return found


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_a_feature_accepting_overlap_never_reads_row_zero_identity(cls: type) -> None:
    """A feature that accepts overlap must not read its identity from row 0.

    Under overlap the frame handed to ``apply`` spans the neighbouring
    sequences, so its row 0 belongs to the *previous* one. A feature that reads
    ``group`` or ``sequence`` there labels its whole output with the neighbour's
    name, and one that resolves a per-entry dependency with it -- a body-scale
    run, a video -- silently uses the neighbour's. Neither raises.

    Checked against ``apply``'s own parameter only. Reading row 0 of a per-frame
    group is the correct way to do this and must stay available: a frame belongs
    to exactly one sequence, so identity is constant within it.
    """
    if not getattr(cls, "accepts_overlap", False):
        pytest.skip("declares accepts_overlap = False")
    apply_fn = getattr(cls, "apply", None)
    if apply_fn is None:  # pragma: no cover - every feature has apply
        pytest.skip("no apply")
    try:
        source = textwrap.dedent(inspect.getsource(apply_fn))
    except OSError:  # pragma: no cover - source always available in-tree
        pytest.skip("source unavailable")
    tree = ast.parse(source)
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    frame_param = function.args.args[1].arg
    offenders = _reads_row_zero_of(function, frame_param)
    assert not offenders, (
        f"{cls.__name__} declares accepts_overlap = True but reads "
        f"{offenders} from row 0 of the frame apply() was given, which under "
        f"overlap belongs to the previous sequence. Take identity per frame "
        f"instead."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_every_feature_declares_what_it_emits(cls: type) -> None:
    """At what entity level a feature's output is keyed is not a default.

    It is what lets a chain be refused before it runs. ``alignment_verdict``
    reads the level off the identity columns of a produced parquet; before
    anything has run there is no parquet, so the declaration is the only source.
    And the mistake it exists to catch is the expensive one: joining an
    individual-level output to a pair-level one shares no identity column, so
    the merge pairs every row of one with every row of the other and nothing
    raises.

    An undeclared feature would have to be assumed passthrough, which is wrong
    in the dangerous direction -- a pair-producing feature read as passthrough
    has its cartesian join permitted rather than refused.
    """
    assert hasattr(cls, "emits"), (
        f"{cls.__name__} declares no 'emits'. Declare it: 'individual' for one "
        f"row per (frame, id), 'pair' for one row per pair, 'unidentified' for "
        f"an aggregate carrying no per-animal identity, 'as-input' when the "
        f"level follows whatever was handed in."
    )
    declared: object = getattr(cls, "emits")
    assert declared in EMITS_LEVELS, (
        f"{cls.__name__} declares emits = {declared!r}, which is not one of "
        f"{sorted(EMITS_LEVELS)}."
    )


BANNED_IDENTITY_NAMES = frozenset(
    {"id_a", "id_b", "id_A", "id_B", "focal_id", "target_id"}
)
"""Pair spellings that used to coexist with ``id1`` / ``id2``.

Four of them, across six features, and the cost was not untidiness.
``entity_level_of`` reads identity by name, so a ``focal_id`` frame read as
carrying *no* identity and was permitted a frame-only join against an
individual-level input. ``feature_columns`` excludes only the canonical names, so
every one of these was handed to the scaler and the embedding as a measurement.
And two producers spelled the same pair differently, so a merge between them bound
each row to the wrong partner.

One spelling is what makes it a rule. This is the assertion that keeps it one.
"""


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_no_feature_writes_a_second_pair_spelling(cls: type) -> None:
    """No feature assigns a column under a retired identity name.

    Read out of the source rather than by running the feature, because most
    features need a shaped input to produce a frame at all, and the point is to
    catch the name at the moment somebody writes it.

    String literals only: a *local variable* named ``id_a`` is fine and common --
    six features enumerate their pairs that way -- so what is looked for is the
    name used as a column key.
    """
    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):  # pragma: no cover - defensive
        pytest.skip(f"{cls.__name__} has no readable source")

    offenders = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in BANNED_IDENTITY_NAMES
    }

    assert not offenders, (
        f"{cls.__name__} names {sorted(offenders)}. A pair row is keyed by "
        f"(frame, id1, id2, perspective) -- id1 the focal, id2 the other."
    )


@pytest.mark.parametrize("cls", FEATURE_CLASSES, ids=FEATURE_IDS)
def test_a_pair_feature_writes_perspective(cls: type) -> None:
    """A feature declaring ``emits = "pair"`` names ``perspective`` somewhere.

    A pair feature emits one row per *ordered* pair, so without ``perspective``
    its two rows per frame are the same row twice as far as any join can tell.
    Presence of the name is a weak check -- the values are asserted in
    ``tests/behavior/feature_library/test_pair_identity_convention.py``, which
    runs the producers -- but it is the one that covers every registered feature,
    including the ones that need a video or a fitted model to run.
    """
    if getattr(cls, "emits", None) != "pair":
        pytest.skip("does not declare emits = 'pair'")

    try:
        source = textwrap.dedent(inspect.getsource(cls))
    except (OSError, TypeError):  # pragma: no cover - defensive
        pytest.skip(f"{cls.__name__} has no readable source")

    assert "perspective" in source, (
        f"{cls.__name__} declares emits = 'pair' but never names 'perspective'. "
        f"Two rows of one frame that differ in nothing cannot be joined apart."
    )
