"""Every op says what it will be called, and says it in exactly one place.

A graph is planned by a topological walk: a step's identity is what its
downstream steps hash, so the whole chain resolves before anything runs or it
does not resolve at all. That is only possible if each op can be asked its
identity without performing it -- which every op used to make impossible by
minting its identifier inside ``run``.

Two properties, and the second is the one that rots quietly. Every op must
implement ``plan_identity``; and ``run`` must take its identity down the same
path rather than compute the same value a second way, because two answers to
what a run is called is the shape of mistake that reports a cache hit over
another run's output, and the copy that gets forgotten is always the one a
planner reads.

The same path is ``plan_identity`` itself, a family minter, or an identity
helper that ``plan_identity`` calls: one that reaches a minter or
``plan_identity`` one level further down. Sharing that helper lets ``run`` keep
the inputs that the identity was read from instead of reading them a second
time. A helper that both call but that names nothing, such as a loader, is not
an identity path. A raw minter called from ``run``, or reached from a helper of
``run`` that ``plan_identity`` does not call, in that helper's body or one level
further down, is a second path.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from collections.abc import Callable
from types import ModuleType

import pytest

from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import OPS, IdentityDeferred, Op, OpIdentity
from mosaic.tracking import register_ops

from tests.helpers import imports_in

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
# ``run`` is fine when it happens through the op's own plan_identity, a family
# minter, or an identity helper that plan_identity also calls, and is the defect
# when it happens beside them.
_RAW_MINTERS = frozenset(
    {
        "op_run_id",
        "tracks_run_id",
        "train_run_id",
        "infer_run_id",
        "convert_points_run_id",
        "prepare_training_data_run_id",
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


def _self_calls(source: str) -> set[str]:
    """Every method name that *source* calls as ``self.<name>(...)``."""
    return {
        node.func.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }


def _followed(
    source: str, owner: type, home: ModuleType | None
) -> dict[str, Callable[..., object]]:
    """The callables that *source* calls, by the name that it calls them by.

    A tracker op's ``run`` is three lines: import the runner, map params to its
    arguments, return. The identity is then the runner's business, so checking
    the op's own body alone would pass any op that delegates -- which is all four
    trackers. Following one level makes the check about the chain rather than
    about a list of blessed names.

    Three kinds of name are resolved: a name that the body itself imports, a
    global of *home*, the module that defines the body, and a method of *owner*
    called as ``self.<name>(...)``. Each is resolved by name only, so this
    cannot wander: it resolves exactly what the code in front of it names.
    """
    found: dict[str, Callable[..., object]] = {}
    for imported in imports_in(ast.parse(source), scope="anywhere", relative=False):
        if imported.name is None:
            continue
        try:
            module = importlib.import_module(imported.module)
        except Exception:  # pragma: no cover - an optional extra is absent
            continue
        target: object = getattr(module, imported.name, None)
        if callable(target):
            found[imported.asname or imported.name] = target
    for name in _calls_in(source):
        target = getattr(home, name, None)
        if callable(target):
            _ = found.setdefault(name, target)
    for name in _self_calls(source):
        target = getattr(owner, name, None)
        if inspect.isfunction(target):
            found[name] = target
    return found


def _bodies(callables: dict[str, Callable[..., object]]) -> dict[str, str]:
    """The readable sources of *callables*, by the name that they are called by."""
    bodies: dict[str, str] = {}
    for name, target in callables.items():
        try:
            bodies[name] = textwrap.dedent(inspect.getsource(target))
        except (OSError, TypeError):  # pragma: no cover
            continue
    return bodies


def _method_target(owner: type, name: str) -> Callable[..., object]:
    """The method *name* of *owner*.

    Raises:
        AttributeError: *owner* has no callable *name*.
    """
    target: object = getattr(owner, name, None)
    if not callable(target):
        raise AttributeError(f"{owner.__name__}.{name} is not a method")
    return target


def _method_source(owner: type, name: str) -> str:
    """The source of the method *name* of *owner*.

    Raises:
        AttributeError: *owner* has no callable *name*.
        OSError: The source cannot be read.
    """
    return textwrap.dedent(inspect.getsource(_method_target(owner, name)))


# The names whose bodies a search for minters does not enter: reaching one of
# them is taking the identity down the sanctioned path, whatever it calls inside.
_SANCTIONED = _FAMILY_MINTERS | {"plan_identity"}


def _reached(
    target: Callable[..., object], owner: type, passed: frozenset[int]
) -> set[str]:
    """The names that *target* calls, in its own body or one level further down.

    The second level is the bodies of the functions that *target*'s body calls,
    except a sanctioned name and a function whose ``id`` is in *passed*, whose
    calls are another path's business. A class that the body constructs is not
    entered: its source is every method it has, not what the call runs.
    """
    try:
        source = textwrap.dedent(inspect.getsource(target))
    except (OSError, TypeError):  # pragma: no cover
        return set()
    callees = _followed(source, owner, inspect.getmodule(target))
    entered = {
        name: callee
        for name, callee in callees.items()
        if inspect.isfunction(callee)
        and name not in _SANCTIONED
        and id(callee) not in passed
    }
    return _calls_in(source).union(*map(_calls_in, _bodies(entered).values()))


def _reaches_an_identity(target: Callable[..., object], owner: type) -> bool:
    """Whether *target* names a run, in its own body or one level further down.

    Its body, or the body of a callable that its body calls, must call a raw
    minter, a family minter or ``plan_identity``.
    """
    return bool(_reached(target, owner, frozenset()) & (_RAW_MINTERS | _SANCTIONED))


def _second_identity(op_cls: type) -> str | None:
    """Why *op_cls*'s ``run`` names itself apart from plan_identity, or ``None``.

    ``run`` passes when it reaches ``plan_identity``, a family minter, or an
    identity helper that ``plan_identity`` calls, and reaches no raw minter
    except through one of those or through another helper of ``plan_identity``.
    An identity helper is a function or method that reaches a minter or
    ``plan_identity`` itself, so a loader that both call does not count. The
    calls of ``run`` and of the callables that it calls, one level down, are
    what it reaches. A helper of ``run`` that ``plan_identity`` does not share
    is searched for raw minters as deep as an identity helper is: its body and
    the bodies of what it calls, short of a sanctioned name or a shared helper.
    Helpers are matched by the callable that a name resolves to, not by the
    name.

    Raises:
        AttributeError: *op_cls* has no callable ``run`` or ``plan_identity``.
        OSError: The source of ``run`` or ``plan_identity`` cannot be read.
    """
    name = op_cls.__name__
    home = inspect.getmodule(op_cls)
    run_source = _method_source(op_cls, "run")
    plan_source = _method_source(op_cls, "plan_identity")
    plan_helpers = [
        target
        for target in _followed(plan_source, op_cls, home).values()
        if inspect.isfunction(target)
    ]
    plan_helpers.append(_method_target(op_cls, "plan_identity"))
    shared = frozenset(id(target) for target in plan_helpers)
    identity_helpers = {
        id(target) for target in plan_helpers if _reaches_an_identity(target, op_cls)
    }
    minted_here = _calls_in(run_source) & _RAW_MINTERS
    if minted_here:
        return (
            f"{name}.run calls {sorted(minted_here)} directly, which is a second "
            f"answer to what this run is called. Take it from plan_identity."
        )
    run_helpers = _followed(run_source, op_cls, home)
    bodies = _bodies(run_helpers)
    for helper in sorted(bodies):
        if id(run_helpers[helper]) in shared:
            continue
        minted_beside = _reached(run_helpers[helper], op_cls, shared) & _RAW_MINTERS
        if minted_beside:
            return (
                f"{name}.run reaches {sorted(minted_beside)} through {helper}, "
                f"which plan_identity does not call, so the run is named apart "
                f"from its plan. Name it through a helper that both share."
            )
    reached = _calls_in(run_source).union(*map(_calls_in, bodies.values()))
    shares_an_identity = any(
        id(target) in identity_helpers for target in run_helpers.values()
    )
    if not (reached & ({"plan_identity"} | _FAMILY_MINTERS) or shares_an_identity):
        return (
            f"{name}.run reaches no identity at all -- not its own "
            f"plan_identity, not a family minter, and not an identity helper "
            f"that plan_identity calls. Its identifier is being made somewhere "
            f"this cannot see, which is where a second copy hides."
        )
    return None


@pytest.mark.parametrize("kind", OP_KINDS)
def test_run_does_not_mint_a_second_identifier(kind: str) -> None:
    """``run`` must take its identity down the path that plan_identity takes."""
    try:
        problem = _second_identity(OPS[kind])
    except OSError:  # pragma: no cover - source is available in-tree
        pytest.skip("source unavailable")
    assert problem is None, problem


class _SharesTheHelper:
    """An op whose ``run`` keeps the inputs that its identity was read from."""

    def plan_identity(self) -> OpIdentity:
        """Name the run through the helper that ``run`` calls too."""
        _inputs, run_id = self._identified()
        return OpIdentity(run_id=run_id)

    def _identified(self) -> tuple[str, str]:
        """Read the inputs once, and name the run from that read."""
        inputs = "inputs"
        return inputs, op_run_id("shares-the-helper", "0.1", {"inputs": inputs})

    def run(self) -> str:
        """Run on the inputs that the identity was read from."""
        _inputs, run_id = self._identified()
        return run_id


class _MintsThroughItsOwnHelper:
    """An op whose ``run`` names itself through a helper of its own."""

    def plan_identity(self) -> OpIdentity:
        """Name the run through one helper."""
        return OpIdentity(run_id=self._planned())

    def _planned(self) -> str:
        """The identifier that a planner reads."""
        return op_run_id("own-helper", "0.1", {"inputs": "planned"})

    def _minted(self) -> str:
        """The identifier that the run stores its output under."""
        return op_run_id("own-helper", "0.1", {"inputs": "minted"})

    def run(self) -> str:
        """Run under an identifier that the plan never sees."""
        return self._minted()


class _MintsBesideItsPlan:
    """An op whose ``run`` calls plan_identity and also a helper that mints."""

    def plan_identity(self) -> OpIdentity:
        """Name the run from the plan."""
        return OpIdentity(run_id=op_run_id("beside", "0.1", {"inputs": "planned"}))

    def _minted(self) -> str:
        """The identifier that the run stores its output under."""
        return op_run_id("beside", "0.1", {"inputs": "minted"})

    def run(self) -> str:
        """Plan, then run under another identifier."""
        _ = self.plan_identity()
        return self._minted()


class _MintsDirectly:
    """An op whose ``run`` calls a raw minter itself."""

    def plan_identity(self) -> OpIdentity:
        """Name the run from the plan."""
        return OpIdentity(run_id=op_run_id("direct", "0.1", {"inputs": "planned"}))

    def run(self) -> str:
        """Plan, then mint again."""
        _ = self.plan_identity()
        return op_run_id("direct", "0.1", {"inputs": "minted"})


def _read_the_inputs() -> str:
    """Load the inputs, naming nothing."""
    return "inputs"


def _minted_two_levels_down() -> str:
    """The identifier that the run stores its output under."""
    return op_run_id("loader", "0.1", {"inputs": "minted"})


def _named_through_a_wrapper() -> str:
    """Pass the identifier up from one level further down."""
    return _minted_two_levels_down()


class _SharesOnlyALoader:
    """An op whose ``run`` shares only a loader with its plan."""

    def plan_identity(self) -> OpIdentity:
        """Name the run from what the loader read."""
        inputs = _read_the_inputs()
        return OpIdentity(run_id=op_run_id("loader", "0.1", {"inputs": inputs}))

    def run(self) -> str:
        """Load, and name the run nowhere that this check can see."""
        return _read_the_inputs()


class _MintsBesideItsPlanThroughAWrapper:
    """An op whose ``run`` calls plan_identity and a wrapper around a minter."""

    def plan_identity(self) -> OpIdentity:
        """Name the run from the plan."""
        return OpIdentity(run_id=op_run_id("wrapper", "0.1", {"inputs": "planned"}))

    def run(self) -> str:
        """Plan, then run under an identifier minted two levels down."""
        _ = self.plan_identity()
        return _named_through_a_wrapper()


class _SharesTheHelperAndWrapsAMinter(_SharesTheHelper):
    """An op that shares its identity helper and also mints through a wrapper."""

    def _other(self) -> str:
        """Pass up an identifier that the plan never sees."""
        return _minted_two_levels_down()

    def run(self) -> str:
        """Read through the shared helper, then run under another identifier."""
        _inputs, _run_id = self._identified()
        return self._other()


class _HasNoRun:
    """An op class that names its plan and cannot run."""

    def plan_identity(self) -> OpIdentity:
        """Name the run from the plan."""
        return OpIdentity(run_id=op_run_id("no-run", "0.1", {}))


def test_a_run_that_shares_a_helper_with_plan_identity_passes() -> None:
    """One read through a helper that both call is one identity path."""
    assert _second_identity(_SharesTheHelper) is None


@pytest.mark.parametrize(
    ("op_cls", "reason"),
    [
        (_MintsThroughItsOwnHelper, "through _minted"),
        (_MintsBesideItsPlan, "through _minted"),
        (_MintsDirectly, "directly"),
        (_SharesOnlyALoader, "reaches no identity"),
        (_MintsBesideItsPlanThroughAWrapper, "through _named_through_a_wrapper"),
        (_SharesTheHelperAndWrapsAMinter, "through _other"),
    ],
)
def test_a_run_that_mints_apart_from_plan_identity_fails(
    op_cls: type, reason: str
) -> None:
    """A minter that plan_identity does not reach is a second identity."""
    problem = _second_identity(op_cls)
    assert problem is not None
    assert reason in problem


def test_an_op_without_a_run_is_an_error_and_not_a_skip() -> None:
    """A missing method is a defect of the op, not unreadable source."""
    with pytest.raises(AttributeError, match="_HasNoRun.run"):
        _ = _second_identity(_HasNoRun)


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
