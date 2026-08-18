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
self-reference (``all = ["mosaic-behavior[pose,faiss]"]``) so they cannot drift
from their parts. The failure mode that buys is narrow but nasty: rename ``pose``
without updating ``all`` and pip resolves ``mosaic-behavior[pose]`` to the
project with an unknown extra, warns, and installs the base. The user gets a
working mosaic with no torch in it and no error to say so -- which is exactly
what the deprecated aliases exist to prevent, so they are checked too.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import cast

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "mosaic"

# `pip install "mosaic-behavior[x,y]"`, in any of the quoting styles the source
# uses. The distribution name is part of the pattern deliberately: a message
# naming `mosaic[...]` is a bug rather than something to check the extras of.
_INSTALL_HINT = re.compile(r"mosaic-behavior\[([a-z0-9,\-]+)\]")
_SELF_REFERENCE = re.compile(r"^mosaic-behavior\[([a-z0-9,\-]+)\]$")


def _declared_extras() -> dict[str, list[str]]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    return cast(dict[str, list[str]], pyproject["project"]["optional-dependencies"])


def _python_sources() -> list[Path]:
    return sorted(_SRC.rglob("*.py"))


def test_every_extra_named_in_a_message_is_declared() -> None:
    """A ``pip install "mosaic-behavior[x]"`` hint must name a real extra."""
    declared = set(_declared_extras())
    bad: list[str] = []
    for path in _python_sources():
        if path.name == "optional_dependency.py":
            # Its module docstring quotes the historical bug verbatim.
            continue
        for match in _INSTALL_HINT.finditer(path.read_text()):
            for name in match.group(1).split(","):
                if name.strip() not in declared:
                    rel = path.relative_to(_REPO_ROOT)
                    bad.append(f"{rel} names [{name.strip()}]")
    assert not bad, (
        "these install hints name an extra pyproject.toml does not declare, so "
        "following them cannot fix anything: " + "; ".join(sorted(set(bad)))
    )


def test_every_require_call_names_a_declared_extra() -> None:
    """The ``extra`` argument of ``optional_dependency.require`` must be real.

    Read out of the AST rather than by text search, so a call spread over several
    lines counts and a mention in a comment does not.
    """
    declared = set(_declared_extras())
    bad: list[str] = []
    for path in _python_sources():
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
            if name != "require" or len(node.args) < 2:
                continue
            extra = node.args[1]
            if not isinstance(extra, ast.Constant) or not isinstance(extra.value, str):
                bad.append(f"{path.relative_to(_REPO_ROOT)} passes a non-literal extra")
                continue
            if extra.value not in declared:
                bad.append(f"{path.relative_to(_REPO_ROOT)} requires [{extra.value}]")
    assert not bad, (
        "these require() calls name an extra pyproject.toml does not declare: "
        + "; ".join(sorted(set(bad)))
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
            match = _SELF_REFERENCE.match(spec.strip())
            if match is None:
                continue
            for referenced in match.group(1).split(","):
                if referenced.strip() not in extras:
                    dangling.append(f"[{name}] -> [{referenced.strip()}]")
    assert not dangling, (
        "these extras reference an extra that does not exist; pip would warn and "
        "install the base instead: " + "; ".join(sorted(dangling))
    )


def test_bundles_and_deprecated_aliases_are_not_empty() -> None:
    """Resolving a bundle or alias must reach a real requirement.

    ``recommended`` in particular has to keep landing torch: it is what every
    saved lab command and README still says, and an alias that quietly resolved
    to nothing would produce a torch-less environment with no error.
    """
    extras = _declared_extras()

    def resolve(name: str, seen: set[str] | None = None) -> list[str]:
        seen = seen if seen is not None else set()
        if name in seen:
            return []
        seen.add(name)
        out: list[str] = []
        for spec in extras.get(name, []):
            match = _SELF_REFERENCE.match(spec.strip())
            if match is None:
                out.append(spec)
                continue
            for referenced in match.group(1).split(","):
                out.extend(resolve(referenced.strip(), seen))
        return out

    for name in ("all", "recommended", "identity", "localizer", "gpu"):
        assert name in extras, f"[{name}] is expected to exist until 0.13"
        assert resolve(name), f"[{name}] resolves to no requirements at all"

    for name in ("all", "recommended"):
        resolved = resolve(name)
        assert any(spec.startswith("torch") for spec in resolved), (
            f"[{name}] must still land torch; it is what the documented install "
            "and every saved command point at"
        )
