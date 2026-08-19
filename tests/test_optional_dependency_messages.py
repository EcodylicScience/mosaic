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
import re
import tomllib
from pathlib import Path
from typing import cast

from tests.helpers import inside_a_virtualenv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src" / "mosaic"

# `pip install "mosaic-behavior[x,y]"`, in any of the quoting styles the source
# uses. The distribution name is part of the pattern deliberately: a message
# naming `mosaic[...]` is a bug rather than something to check the extras of.
_INSTALL_HINT = re.compile(r"mosaic-behavior\[([a-z0-9,\-]+)\]")
_SELF_REFERENCE = re.compile(r"^mosaic-behavior\[([a-z0-9,\-]+)\]$")

_HINT_WITNESS = "cli/run.py"
_REQUIRE_WITNESS = "behavior/feature_library/global_tsne.py"
"""Files each walk below must have read, as paths under ``src/mosaic``.

Named rather than counted: a threshold on how many files were read passes on a
walk that read the wrong ones, and reading installed third-party source is
exactly how a walk comes to read thousands of the wrong ones.
"""


def _declared_extras() -> dict[str, list[str]]:
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    return cast(dict[str, list[str]], pyproject["project"]["optional-dependencies"])


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
        for source in sorted(_SRC.rglob("*.py"))
        if not inside_a_virtualenv(source, _SRC)
    ]


def test_every_extra_named_in_a_message_is_declared() -> None:
    """A ``pip install "mosaic-behavior[x]"`` hint must name a real extra."""
    declared = set(_declared_extras())
    bad: list[str] = []
    hinting: set[str] = set()
    for path in _python_sources():
        if path.name == "optional_dependency.py":
            # Its module docstring quotes the historical bug verbatim.
            continue
        for match in _INSTALL_HINT.finditer(path.read_text()):
            hinting.add(str(path.relative_to(_SRC)))
            for name in match.group(1).split(","):
                if name.strip() not in declared:
                    rel = path.relative_to(_REPO_ROOT)
                    bad.append(f"{rel} names [{name.strip()}]")
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
    """The ``extra`` argument of ``optional_dependency.require`` must be real.

    Read out of the AST rather than by text search, so a call spread over several
    lines counts and a mention in a comment does not.
    """
    declared = set(_declared_extras())
    bad: list[str] = []
    calling: set[str] = set()
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
            calling.add(str(path.relative_to(_SRC)))
            if extra.value not in declared:
                bad.append(f"{path.relative_to(_REPO_ROOT)} requires [{extra.value}]")
    assert _REQUIRE_WITNESS in calling, (
        f"the walk read no require() call in {_REQUIRE_WITNESS}, so it is not "
        f"reading mosaic's own source; it found calls in {sorted(calling)}"
    )
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

    # One release behind the four above, because they were retired later: when
    # pose and point training moved into environments mosaic does not install,
    # and there was no longer anything for either name to install here.
    for name in ("pose", "polo"):
        assert name in extras, f"[{name}] is expected to exist until 0.14"
        assert resolve(name), f"[{name}] resolves to no requirements at all"

    for name in ("all", "recommended"):
        resolved = resolve(name)
        assert any(spec.startswith("torch") for spec in resolved), (
            f"[{name}] must still land torch; it is what the documented install "
            "and every saved command point at"
        )
