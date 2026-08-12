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
