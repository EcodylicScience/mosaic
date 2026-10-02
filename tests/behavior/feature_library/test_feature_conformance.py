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
    ``tests/test_pair_identity_convention.py``, which runs the producers -- but it
    is the one that covers every registered feature, including the ones that need
    a video or a fitted model to run.
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
