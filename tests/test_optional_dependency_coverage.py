"""Every guarded optional dependency is accounted for in exactly one place.

A guarded test proves nothing unless some environment is obliged to install what
it guards. ``tests/conftest.py`` states that rule in prose -- "a new optional
dependency joins the install line and this tuple in the same change" -- and prose
does not fail. This does.

The suite guards two ways, and both are audited here: ``pytest.importorskip``,
and a module-level ``importlib.util.find_spec`` probe fed to ``skipif``. The
second exists for a guard that has to work in both directions -- ``feral``'s
tests assert the ImportError raised in its *absence* as well as the behavior
present in its presence -- which ``importorskip`` cannot express. Auditing only
the first would leave that whole file's dependency outside every check below.

The guarded set is read from the suite's **own AST** rather than by grepping the
source text, because several docstrings in ``conftest.py`` discuss
``pytest.importorskip("torch")`` while explaining the rule; a text search reports
those as guards and produces a covered set that is quietly wrong.

Three properties, and each has already caught something real:

- a guarded module that no CI job installs (found ``timm`` and ``tables``, and
  now the answer for ``ultralytics``, which belongs in an environment of its own
  rather than in mosaic's),
- a guarded module that is a *core* dependency, so the guard can never fire and
  would hide a broken install rather than a missing extra (found ``yaml``),
- a required module that nothing in the suite reaches, i.e. a tuple entry kept
  alive by nobody.
"""

from __future__ import annotations

import ast
import importlib.metadata
import re
import tomllib
from collections import defaultdict
from pathlib import Path

# From `conftest` rather than from `tests.helpers`: these are pytest's own
# configuration, read by `pytest_configure`, not builders a test composes with.
from tests.conftest import (
    CI_FERAL_MODULES,
    CI_IDENTITY_MODULES,
    CI_REQUIRED_MODULES,
)

TESTS = Path(__file__).resolve().parent
PYPROJECT = TESTS.parent / "pyproject.toml"


def _canonical(distribution: str) -> str:
    """A distribution name in the one spelling both sides can be compared in.

    ``opencv-python`` and ``opencv_python`` name the same thing, and metadata and
    pyproject do not agree on which to write.
    """
    return distribution.lower().replace("-", "_")


def _probe_targets(attribute: str, *, literal_required: bool) -> dict[str, set[str]]:
    """``{module: {test filename}}`` for every ``<obj>.<attribute>("name")`` call."""
    found: dict[str, set[str]] = defaultdict(set)
    for path in sorted(TESTS.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == attribute):
                continue
            if not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                assert not literal_required, (
                    f"{path.name}:{node.lineno} calls {attribute} with a non-literal "
                    "module name, which cannot be audited by this file"
                )
                continue
            found[first.value].add(path.name)
    return dict(found)


def importorskip_targets() -> dict[str, set[str]]:
    """``{module: {test filename}}`` for every literal ``importorskip`` in tests.

    A non-literal argument is deliberately not tolerated: it cannot be audited,
    so it would silently leave a guard outside every check here.
    """
    return _probe_targets("importorskip", literal_required=True)


def find_spec_targets() -> dict[str, set[str]]:
    """``{module: {test filename}}`` for every literal ``find_spec`` probe in tests.

    The second way the suite guards on an optional dependency, and it exists
    because ``importorskip`` can only express *skip when absent*. ``feral`` needs
    both directions -- two tests assert the ImportError that fires when the
    package is missing -- so it is probed once into a module-level flag and used
    with ``skipif`` either way round.

    Unlike ``importorskip`` above, a non-literal argument is passed over rather
    than refused. ``find_spec`` has a second caller that is not a guard at all:
    ``conftest.pytest_configure`` loops it over the CI tuples to turn a missing
    module into an error, and its argument is the loop variable.
    """
    return _probe_targets("find_spec", literal_required=False)


def guarded_targets() -> dict[str, set[str]]:
    """Every module the suite guards on, by either mechanism."""
    merged: dict[str, set[str]] = defaultdict(set)
    for targets in (importorskip_targets(), find_spec_targets()):
        for name, files in targets.items():
            merged[name] |= files
    return dict(merged)


def _ci_modules() -> set[str]:
    """Every module some CI job is obliged to install into mosaic's environment.

    There is deliberately no tracking entry, and its absence is load-bearing
    rather than an oversight. What the tracking job installs is an *Ultralytics
    environment*, which is not a module and is not importable from here:
    Ultralytics is AGPL-3.0 and mosaic never imports it. A tuple naming it would
    assert the opposite, and would then excuse an ``importorskip("ultralytics")``
    that guards mosaic's environment for a package that does not belong in it.
    Without one, such a guard fails the audit below, which is the right answer.
    ``tests/conftest.py`` holds that job to its real requirement directly.
    """
    return set(CI_REQUIRED_MODULES) | set(CI_IDENTITY_MODULES) | set(CI_FERAL_MODULES)


def _core_import_names() -> set[str]:
    """Import names provided by the distributions in ``[project] dependencies``.

    Resolved through installed metadata rather than a hand-written table, so the
    ``pyyaml`` -> ``yaml`` and ``opencv-python`` -> ``cv2`` renames come from the
    environment instead of from a list that would drift on its own.
    """
    with PYPROJECT.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    assert isinstance(project, dict)
    # The distribution name is the leading run of name characters; everything
    # after it is an extras bracket, a version specifier or an environment
    # marker. Parsed here rather than with `packaging`, which this project does
    # not declare and which would make the test depend on a transitive.
    declared = {
        _canonical(match.group(0))
        for spec in project["dependencies"]
        if (match := re.match(r"[A-Za-z0-9._-]+", str(spec)))
    }

    names: set[str] = set()
    for module, dists in importlib.metadata.packages_distributions().items():
        if any(_canonical(dist) in declared for dist in dists):
            names.add(module)
    return names


def test_every_guarded_module_is_required_by_some_ci_job() -> None:
    covered = _ci_modules()
    uncovered = {
        name: sorted(files)
        for name, files in guarded_targets().items()
        if name not in covered
    }
    assert not uncovered, (
        f"guarded but required by no CI job: {uncovered}. "
        "Their tests skip green in every job, so they are not evidence. Either "
        "add the distribution to a CI install line and its import name to the "
        "matching tuple in tests/conftest.py, or delete the guard. A dependency "
        "that belongs in an external tool's own environment rather than in "
        "mosaic's -- ultralytics is the one -- has no tuple to join: guard the "
        "environment, as tests/test_ultralytics_preflight.py does."
    )


def test_no_core_dependency_is_guarded() -> None:
    """A core dependency behind ``importorskip`` is a guard that cannot fire.

    Worse than useless: if the core install really were broken, the guard would
    turn the failure into a skip and the suite would stay green over it.
    """
    guarded_core = sorted(set(guarded_targets()) & _core_import_names())
    assert not guarded_core, (
        f"{guarded_core} are core dependencies but are guarded anyway. "
        "Remove the guard -- a core dependency is not optional, and skipping "
        "instead of failing hides a broken install."
    )


def test_every_required_module_is_reachable_from_the_suite() -> None:
    """The other direction: a tuple entry nothing reaches is dead weight.

    Every entry is guarded today. ``pywt`` used to be the exception -- reached
    without a guard, because the wavelet features imported it directly -- and it
    was carried here by name for that. It is a base dependency now, so it left
    the tuple and the exception left with it.
    """
    reachable = set(guarded_targets())
    unreached = sorted(_ci_modules() - reachable)
    assert not unreached, (
        f"{unreached} are demanded of a CI job but nothing in the suite reaches "
        "them. Either a test needs them and should say so, or the tuple entry in "
        "tests/conftest.py is stale."
    )
