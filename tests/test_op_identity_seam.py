"""Every op says what it will be called, and says it in exactly one place.

A graph is planned by a topological walk: a step's identity is what its
downstream steps hash, so the whole chain resolves before anything runs or it
does not resolve at all. That is only possible if each op can be asked its
identity without performing it -- which every op used to make impossible by
minting its identifier inside ``run``.

Two properties, and the second is the one that rots quietly. Every op must
implement ``plan_identity``; and ``run`` must *call* it rather than compute the
same value a second way, because two answers to what a run is called is the
shape of mistake that reports a cache hit over another run's output, and the
copy that gets forgotten is always the one a planner reads.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from mosaic.core.pipeline.ops import OPS, IdentityDeferred, Op, OpIdentity
from mosaic.tracking import register_ops

register_ops()

OP_KINDS = sorted(OPS)


@pytest.mark.parametrize("kind", OP_KINDS)
def test_every_op_implements_plan_identity(kind: str) -> None:
    """An op that does not is a step a graph cannot plan around."""
    op_cls = OPS[kind]
    assert "plan_identity" in vars(op_cls) or any(
        "plan_identity" in vars(base) for base in op_cls.__mro__ if base is not Op
    ), (
        f"op {kind!r} ({op_cls.__name__}) does not implement plan_identity, so a "
        f"graph cannot say what it will produce or whether it has already run."
    )


@pytest.mark.parametrize("kind", OP_KINDS)
def test_plan_identity_returns_the_declared_shape(kind: str) -> None:
    """The return annotation is what a planner reads; it must be the record."""
    op_cls = OPS[kind]
    signature = inspect.signature(op_cls.plan_identity)
    assert signature.return_annotation in {OpIdentity, "OpIdentity"}, (
        f"{op_cls.__name__}.plan_identity must return OpIdentity, not "
        f"{signature.return_annotation!r}."
    )


def _calls_in(source: str) -> set[str]:
    """Every function or method name called anywhere in *source*."""
    called: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            called.add(target.id)
        elif isinstance(target, ast.Attribute):
            called.add(target.attr)
    return called


# The functions that actually mint an identifier. Reaching one of these from a
# ``run`` is fine when it happens *through* the op's own plan_identity, and is
# the defect when it happens beside it.
_RAW_MINTERS = frozenset(
    {
        "op_run_id",
        "tracks_run_id",
        "train_run_id",
        "infer_run_id",
        "convert_points_run_id",
        "transcode_run_id",
        "export_run_id",
        "frames_run_id",
    }
)

# The one place a family's identifiers are made, which its members reach through
# rather than around. These are minting *functions*, not blessed op names: a
# tracker reaches its identity through mint_tracker_run, which calls
# tracker_identity, which is the only caller of op_run_id on that path.
_FAMILY_MINTERS = frozenset({"tracker_identity", "mint_tracker_run", "infer_identity"})


def _delegated_bodies(source: str, owner: type) -> list[str]:
    """The sources of the module-level functions *source* delegates to.

    A tracker op's ``run`` is three lines: import the runner, map params to its
    arguments, return. The identity is then the runner's business, so checking
    the op's own body alone would pass any op that delegates -- which is all four
    trackers. Following one level makes the check about the chain rather than
    about a list of blessed names.

    Only the modules the body itself imports are followed, and only by name, so
    this cannot wander: it resolves exactly what the code in front of it names.
    """
    tree = ast.parse(source)
    bodies: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        try:
            module = __import__(node.module, fromlist=[a.name for a in node.names])
        except Exception:  # pragma: no cover - an optional extra is absent
            continue
        for alias in node.names:
            target = getattr(module, alias.name, None)
            if target is None or not callable(target):
                continue
            try:
                bodies.append(textwrap.dedent(inspect.getsource(target)))
            except (OSError, TypeError):  # pragma: no cover
                continue
    # The op's own module globals too, for a body that calls a sibling directly.
    for name in _calls_in(source):
        target = getattr(inspect.getmodule(owner), name, None)
        if callable(target):
            try:
                bodies.append(textwrap.dedent(inspect.getsource(target)))
            except (OSError, TypeError):  # pragma: no cover
                continue
    return bodies


@pytest.mark.parametrize("kind", OP_KINDS)
def test_run_does_not_mint_a_second_identifier(kind: str) -> None:
    """``run`` must take its identity from plan_identity, not compute one."""
    op_cls = OPS[kind]
    try:
        source = textwrap.dedent(inspect.getsource(op_cls.run))
    except OSError:  # pragma: no cover - source is available in-tree
        pytest.skip("source unavailable")
    minted_here = _calls_in(source) & _RAW_MINTERS
    assert not minted_here, (
        f"{op_cls.__name__}.run calls {sorted(minted_here)} directly, which is a "
        f"second answer to what this run is called. Take it from plan_identity."
    )
    reached = set(_calls_in(source))
    for body in _delegated_bodies(source, op_cls):
        reached |= _calls_in(body)
    assert reached & ({"plan_identity"} | _FAMILY_MINTERS), (
        f"{op_cls.__name__}.run reaches no identity at all -- not its own "
        f"plan_identity, and not a family minter. Its identifier is being made "
        f"somewhere this cannot see, which is where a second copy hides."
    )


def test_identity_deferred_names_the_op_and_the_reason() -> None:
    """A deferral a user cannot act on is no better than a wrong identifier."""
    deferred = IdentityDeferred("train-pose", "its data has not been written yet")
    assert deferred.kind == "train-pose"
    assert "has not been written" in deferred.because
    assert "train-pose" in str(deferred)


def test_an_op_identity_carries_its_variant_apart_from_its_run() -> None:
    """They answer different questions even where they are byte-identical."""
    identity = OpIdentity(run_id="trex.0.1-abc", tracks_variant="trex.0.1-abc")
    assert identity.run_id == identity.tracks_variant
    assert identity.model_run_id == ""
