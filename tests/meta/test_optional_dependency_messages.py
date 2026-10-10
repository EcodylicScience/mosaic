"""Every extra a message names, and every extra a bundle references, must exist.

Two failure modes, both silent, both previously real.

**A message naming an extra that does not exist.** ``movement/convert.py`` told
users to run ``pip install 'mosaic[movement]'`` -- the wrong distribution name,
and an extra that has never been declared. Nothing connected that string to
``pyproject.toml``, so it was wrong from the day it was written until someone
tried it. Renaming an extra has the same shape: the declaration moves, ten
messages do not, and each one sends a user to install something that will not
fix their problem.

**A bundle whose self-reference dangles.** Bundles are declared by
self-reference (``all = ["mosaic-behavior[deep-learning,faiss]"]``) so they
cannot drift from their parts. The failure mode that buys is narrow but nasty:
rename ``deep-learning`` without updating ``all`` and pip resolves the bundle to
the project with an unknown extra, warns, and installs the base. The user gets a
working mosaic with no torch in it and no error to say so -- which is exactly
what the deprecated aliases exist to prevent, so they are checked too.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers import (
    REPO_ROOT,
    SELF_REFERENCE,
    SOURCE_ROOT,
    inside_a_virtualenv,
    project_requirements,
    requirements_reaching,
    self_referenced_extras,
)

_HINT_WITNESS = "cli/run.py"
_REQUIRE_WITNESSES = {
    "require": "behavior/feature_library/global_tsne.py",
    "OptionalModule": "behavior/feature_library/movement/convert.py",
}
"""Files each walk below must have read, as paths under ``src/mosaic``.

Named rather than counted: a threshold on how many files were read passes on a
walk that read the wrong ones, and reading installed third-party source is
exactly how a walk comes to read thousands of the wrong ones.
"""


def _declared_extras() -> dict[str, list[str]]:
    return project_requirements(REPO_ROOT / "pyproject.toml").extras


def _python_sources() -> list[Path]:
    """Mosaic's own Python files, and only those.

    An installed virtualenv under the package tree is skipped and nothing else
    is. The two external-environment trees stay *in*: the programs there are
    mosaic's own code, shipped from this repository, and a bad ``require()`` call
    or a hint naming an extra that does not exist would be exactly as wrong there
    as anywhere -- so the exemption a rule about mosaic's *wiring* earns does not
    apply to a rule about the strings mosaic writes.

    Without the virtualenv skip this walk read 8742 files of installed
    third-party source and failed on numpy's own test suite, which calls a
    ``require`` of its own.
    """
    return [
        source
        for source in sorted(SOURCE_ROOT.rglob("*.py"))
        if not inside_a_virtualenv(source, SOURCE_ROOT)
    ]


def test_every_extra_named_in_a_message_is_declared() -> None:
    """A ``pip install "mosaic-behavior[x]"`` hint must name a real extra.

    A hint is found in any of the quoting styles that the source uses.
    """
    declared = set(_declared_extras())
    bad: list[str] = []
    hinting: set[str] = set()
    for path in _python_sources():
        if path.name == "optional_dependency.py":
            # Its module docstring quotes the historical bug verbatim.
            continue
        for match in SELF_REFERENCE.finditer(path.read_text()):
            hinting.add(str(path.relative_to(SOURCE_ROOT)))
            bad.extend(
                f"{path.relative_to(REPO_ROOT)} names [{name}]"
                for name in self_referenced_extras(match.group(0))
                if name not in declared
            )
    # The walk proves it happened before its verdict is believed: an exclusion
    # one predicate too broad would find nothing and report a green invariant it
    # never checked.
    assert _HINT_WITNESS in hinting, (
        f"the walk read no install hint in {_HINT_WITNESS}, so it is not reading "
        f"mosaic's own source; it found hints in {sorted(hinting)}"
    )
    assert not bad, (
        "these install hints name an extra pyproject.toml does not declare, so "
        "following them cannot fix anything: " + "; ".join(sorted(set(bad)))
    )


def test_every_require_call_names_a_declared_extra() -> None:
    """The ``extra`` argument of ``require`` and ``OptionalModule`` must be real.

    ``OptionalModule`` declares the arguments of ``require`` once, for a feature's
    prerequisites, and its sentence names the extra as ``require``'s error does.
    Read out of the AST rather than by text search, so a call spread over several
    lines counts and a mention in a comment does not.
    """
    declared = set(_declared_extras())
    bad: list[str] = []
    calling: dict[str, set[str]] = {name: set() for name in _REQUIRE_WITNESSES}
    for path in _python_sources():
        if path.name == "optional_dependency.py":
            # It defines both, and ``OptionalModule.load`` forwards its fields.
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", "")
            )
            if name not in calling or len(node.args) < 2:
                continue
            extra = node.args[1]
            if not isinstance(extra, ast.Constant) or not isinstance(extra.value, str):
                bad.append(f"{path.relative_to(REPO_ROOT)} passes a non-literal extra")
                continue
            calling[name].add(str(path.relative_to(SOURCE_ROOT)))
            if extra.value not in declared:
                bad.append(f"{path.relative_to(REPO_ROOT)} requires [{extra.value}]")
    for name, witness in _REQUIRE_WITNESSES.items():
        assert witness in calling[name], (
            f"the walk found no {name}() call in {witness}, a file of mosaic's "
            f"source. It found calls in {sorted(calling[name])}"
        )
    assert not bad, (
        "these require() and OptionalModule() calls name an extra pyproject.toml "
        "does not declare: " + "; ".join(sorted(set(bad)))
    )


def test_every_self_reference_resolves() -> None:
    """A bundle or alias must reference an extra that exists.

    A dangling reference does not fail the install -- pip warns about the unknown
    extra and carries on -- so nothing but this notices.
    """
    extras = _declared_extras()
    dangling: list[str] = []
    for name, specs in extras.items():
        for spec in specs:
            dangling.extend(
                f"[{name}] -> [{referenced}]"
                for referenced in self_referenced_extras(spec)
                if referenced not in extras
            )
    assert not dangling, (
        "these extras reference an extra that does not exist; pip would warn and "
        "install the base instead: " + "; ".join(sorted(dangling))
    )


def test_the_documented_bundle_lands_torch_and_the_retired_aliases_are_gone() -> None:
    """``all`` is the documented install, and must keep landing torch.

    The six aliases were removed in 0.14.0. A saved ``pip install .[recommended]``
    now makes pip warn and install the base, which the changelog states. A
    declared alias would instead resolve and hide that the name was retired.
    """
    extras = _declared_extras()

    resolved = requirements_reaching(extras, "all")
    assert any(spec.startswith("torch") for spec in resolved), (
        "[all] must land torch; it is what the documented install points at"
    )

    retired = {"recommended", "identity", "localizer", "gpu", "pose", "polo"}
    assert not retired & set(extras), (
        f"{sorted(retired & set(extras))} were removed in 0.14.0 and are declared again"
    )
