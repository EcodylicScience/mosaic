"""Regenerate the third-party license inventory from the resolved uv lock files.

One question per distribution: what license does it carry, and does that license
put an obligation on a commercial user. The answer is derived, never
hand-maintained -- the extras, the transitive closure, and the counts all come
out of the lock, so the inventory moves when the lock moves and a stale number
cannot survive in it.

Closure, not the direct list. An extra's obligation is whatever its whole
subtree carries: `pose` names `ultralytics`, and `ultralytics` is what drags in
`ultralytics-thop`, which is AGPL too and is named in no pyproject.toml. Each
group is walked from the root package's entry for that group through every
package's `dependencies`, following the `extra` key on a dependency entry into
that package's own optional-dependencies -- which is how `mosaic-media` pulls
`[io,cli]`. Platform markers are ignored on purpose: an obligation that attaches
only on Linux is still an obligation.

Licenses come from PyPI, in the order PEP 639 makes authoritative:
`license_expression`, then the `License ::` trove classifiers, then the
free-text `license` field. The free-text field is last because it is not a
license identifier -- some projects put their entire license text in it -- so it
is reported truncated to its first line and marked as needing a human.

A package resolved from a git or a path source is never looked up on PyPI. It
has no PyPI metadata for the revision in the lock, and worse, the name there can
be taken by different code: `ultralytics` in this lock is the POLO fork at
version 8.4.7, and PyPI serves a real 8.4.7 for upstream Ultralytics. A
name-keyed lookup would attribute the wrong project and look right doing it.
Those sources are answered from OVERRIDES, keyed by source URL rather than by
name, and a non-registry source with no override is reported unknown rather than
guessed.

Two locks are read by default. The main one resolves mosaic-behavior itself. The
second, under feature_library/external, resolves the isolated environment that
runs keypoint-MoSeq -- the only non-commercial component in the tree, and
invisible to any tool that reads the main lock alone.

Counts are per distribution, not per lock entry, and the difference is real: a
package resolved for more than one platform gets one `[[package]]` entry each,
and those entries are the same project under the same license. Counting them
twice would overstate the inventory, so they collapse by name. The root package
is excluded as well, being the thing the inventory is *for*. Reconciling against
a raw `[[package]]` count means allowing for both.

Writes nothing. The report goes to stdout; redirect it or paste it.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeGuard

SourceKind = Literal["registry", "git", "directory", "editable", "virtual", "unknown"]
LicenseOrigin = Literal[
    "override", "expression", "classifier", "free-text", "missing", "not-queried"
]
Obligation = Literal[
    "permissive",
    "weak-copyleft",
    "strong-copyleft",
    "non-commercial",
    "proprietary",
    "unknown",
]
ReportFormat = Literal["text", "markdown", "json"]

REPOSITORY_ROOT: Final = Path(__file__).resolve().parent.parent
DEFAULT_LOCKS: Final = (
    REPOSITORY_ROOT / "uv.lock",
    REPOSITORY_ROOT / "src/mosaic/behavior/feature_library/external/uv.lock",
)
PYPI_ENDPOINT: Final = "https://pypi.org/pypi"
USER_AGENT: Final = "mosaic-third-party-inventory"
FREE_TEXT_LIMIT: Final = 72

_SOURCE_KEYS: Final[tuple[SourceKind, ...]] = (
    "registry",
    "git",
    "directory",
    "editable",
    "virtual",
)

# Checked in order, first match wins. Two orderings here are load-bearing.
#
# Weak copyleft precedes strong because "Lesser General Public License" contains
# "General Public License"; testing in the other order would call every LGPL
# package GPL.
#
# "Proprietary" is its own verdict and is deliberately NOT folded into
# non-commercial. The trove string "Other/Proprietary License" says only that
# the license is not an OSI one -- it says nothing about commercial use, and the
# CUDA redistributables that carry it are in fact commercially redistributable
# under NVIDIA's terms. Collapsing the two would raise fourteen false alarms
# here and bury the one component that really is non-commercial.
_OBLIGATION_PATTERNS: Final[tuple[tuple[Obligation, str], ...]] = (
    ("non-commercial", r"non-?commercial|cc-by-nc|\bnc-sa\b"),
    ("proprietary", r"proprietary|all rights reserved"),
    (
        "weak-copyleft",
        r"\blgpl|lesser general public|\bmpl\b|mozilla public|\bepl\b"
        r"|eclipse public|\bcddl\b",
    ),
    (
        "strong-copyleft",
        r"\bagpl|affero|\bgpl\b|gplv[23]|gpl-[23]|general public license",
    ),
    (
        "permissive",
        r"\bmit\b|\bbsd\b|apache|\bisc\b|\bzlib\b|unlicense|\bcc0\b"
        r"|python software foundation|\bpsf\b|historical permission",
    ),
)


@dataclass(frozen=True)
class Override:
    """A hand-verified answer for a source PyPI cannot be asked about."""

    expression: str
    obligation: Obligation
    note: str


_POLO_NOTE: Final = (
    "Third-party fork of Ultralytics YOLO; ships the AGPL-3.0 text. The "
    "Ultralytics Enterprise License covers Ultralytics' own distribution and "
    "is not available for this fork. PyPI serves different code under the "
    "same name and version, so it is never queried for this entry."
)
_FERAL_NOTE: Final = (
    "Read from the project's own LICENSE, because its PyPI metadata carries no "
    "license identifier -- no expression and no classifier, only the MIT text in "
    "the free-text field, which this script reports truncated and flags for a "
    "human. Also downloads backbone weights from the HuggingFace hub at first "
    "use; those carry their own terms -- see docs/licensing.md."
)
_KPMS_NOTE: Final = (
    "Read from the project's own LICENSE.md, which PyPI does not carry: "
    "licensed by the President and Fellows of Harvard College for "
    "non-commercial research and academic use. Commercial use is expressly "
    "prohibited and no paid exception cures it; a separate agreement with "
    "Harvard's Office of Technology Development is required. Mosaic never "
    "installs or imports it -- see docs/licensing.md."
)

# Keyed by source URL with uv's revision query and fragment stripped, never by
# distribution name -- see the module docstring for why the name is unsafe.
OVERRIDES: Final[dict[str, Override]] = {
    "https://github.com/mooch443/POLO.git": Override(
        expression="AGPL-3.0-only",
        obligation="strong-copyleft",
        note=_POLO_NOTE,
    ),
    # FERAL released to PyPI at 1.0.0 and the extra names it from there, so new
    # locks resolve it from the registry and answer from REGISTRY_OVERRIDES
    # below. This entry is for a lock that still carries the git reference, which
    # includes the one in the tree: it cannot be regenerated while `pose`'s
    # `ultralytics>=8.4.63` floor and `polo`'s fork of the same distribution name
    # are both resolvable in one universal lock. Remove it with the relock.
    "https://github.com/Skovorp/feral.git": Override(
        expression="MIT",
        obligation="permissive",
        note=_FERAL_NOTE,
    ),
    "https://github.com/microsoft/python-type-stubs": Override(
        expression="MIT",
        obligation="permissive",
        note="Development-only type stubs; not distributed with mosaic.",
    ),
}

# Keyed by distribution name, and consulted only for registry sources, where the
# name is the identity. This is for the case PyPI cannot answer: a project that
# publishes no license metadata at all, whose real terms are in a license
# document in its repository. Reporting "unknown" for keypoint-moseq would be
# accurate about PyPI and useless about the obligation, which is the sharpest
# one in the whole tree.
#
# `feral` is the milder version of the same problem. It publishes its license
# only as free text, which is the tier this script reports truncated and marks
# as needing a human -- an answer that is accurate about the metadata and says
# nothing about the obligation. It moved here from OVERRIDES when FERAL began
# releasing to PyPI, since a registry source never reaches that table.
REGISTRY_OVERRIDES: Final[dict[str, Override]] = {
    "keypoint-moseq": Override(
        expression="Harvard OTD Non-Commercial Research and Academic Use",
        obligation="non-commercial",
        note=_KPMS_NOTE,
    ),
    "feral": Override(
        expression="MIT",
        obligation="permissive",
        note=_FERAL_NOTE,
    ),
}


@dataclass(frozen=True)
class Requirement:
    """A resolved edge in the lock graph: a name, plus any extras requested."""

    name: str
    extras: tuple[str, ...]


@dataclass(frozen=True)
class LockedPackage:
    """One ``[[package]]`` entry, reduced to what licensing needs."""

    name: str
    version: str
    source_kind: SourceKind
    source_url: str
    requires: tuple[Requirement, ...]
    optional_requires: dict[str, tuple[Requirement, ...]]
    development_requires: dict[str, tuple[Requirement, ...]]


@dataclass(frozen=True)
class LicenseFact:
    """What a distribution is licensed under, and how we came to know it."""

    expression: str
    origin: LicenseOrigin
    obligation: Obligation
    note: str


@dataclass(frozen=True)
class InventoryRow:
    """One distribution, with the groups that reach it."""

    name: str
    version: str
    source_kind: SourceKind
    source_url: str
    license: LicenseFact
    groups: tuple[str, ...]


def is_mapping(value: object) -> TypeGuard[dict[object, object]]:
    """Narrow a parsed TOML or JSON value to a mapping of checked contents.

    ``isinstance(value, dict)`` alone narrows to ``dict[Unknown, Unknown]``,
    which strict mode rejects. Stating the element types here is what lets every
    caller stay free of ``Any`` and of a cast.
    """
    return isinstance(value, dict)


def is_sequence(value: object) -> TypeGuard[list[object]]:
    """Narrow a parsed TOML or JSON value to a list of checked contents."""
    return isinstance(value, list)


def read_table(value: object) -> dict[str, object]:
    """Read a TOML or JSON mapping, or an empty one when it is not a mapping."""
    if not is_mapping(value):
        return {}
    return {str(key): item for key, item in value.items()}


def read_array(value: object) -> list[object]:
    """Read a TOML or JSON sequence, or an empty one when it is not one."""
    if not is_sequence(value):
        return []
    return value


def read_text(value: object) -> str:
    """Read a string, or the empty string when the value is not one."""
    return value if isinstance(value, str) else ""


def base_url(url: str) -> str:
    """A source URL without the revision query and commit fragment uv appends."""
    for separator in ("?", "#"):
        url = url.split(separator, 1)[0]
    return url


def read_requirements(value: object) -> tuple[Requirement, ...]:
    """The dependency entries in one array, keeping the ``extra`` key on each."""
    found: list[Requirement] = []
    for item in read_array(value):
        entry = read_table(item)
        name = read_text(entry.get("name"))
        if not name:
            continue
        extras = tuple(
            extra
            for extra in (read_text(value) for value in read_array(entry.get("extra")))
            if extra
        )
        found.append(Requirement(name=name, extras=extras))
    return tuple(found)


def read_requirement_groups(value: object) -> dict[str, tuple[Requirement, ...]]:
    """A table of named dependency arrays, as optional-dependencies is shaped."""
    return {name: read_requirements(item) for name, item in read_table(value).items()}


def read_package(entry: dict[str, object]) -> LockedPackage:
    """One ``[[package]]`` entry reduced to a LockedPackage."""
    source = read_table(entry.get("source"))
    kind: SourceKind = "unknown"
    url = ""
    for key in _SOURCE_KEYS:
        candidate = source.get(key)
        if isinstance(candidate, str):
            kind = key
            url = candidate
            break
    return LockedPackage(
        name=read_text(entry.get("name")),
        version=read_text(entry.get("version")),
        source_kind=kind,
        source_url=url,
        requires=read_requirements(entry.get("dependencies")),
        optional_requires=read_requirement_groups(entry.get("optional-dependencies")),
        development_requires=read_requirement_groups(entry.get("dev-dependencies")),
    )


def read_lock(path: Path) -> dict[str, LockedPackage]:
    """Every package in one lock file, keyed by name."""
    with path.open("rb") as handle:
        document: object = tomllib.load(handle)
    packages: dict[str, LockedPackage] = {}
    for entry in read_array(read_table(document).get("package")):
        package = read_package(read_table(entry))
        if package.name:
            packages[package.name] = package
    return packages


def find_root(packages: Mapping[str, LockedPackage]) -> LockedPackage | None:
    """The package the lock was resolved for: the one whose source is ``.``."""
    for package in packages.values():
        if package.source_url == "." and package.source_kind in (
            "editable",
            "virtual",
            "directory",
        ):
            return package
    return None


def group_requirements(root: LockedPackage) -> dict[str, tuple[Requirement, ...]]:
    """Every named group of the root: core, each extra, each dev group."""
    named: dict[str, tuple[Requirement, ...]] = {"(core)": root.requires}
    for extra, requirements in sorted(root.optional_requires.items()):
        named[f"[{extra}]"] = requirements
    for group, requirements in sorted(root.development_requires.items()):
        named[f"dev:{group}"] = requirements
    return named


def closure(
    packages: Mapping[str, LockedPackage],
    roots: Iterable[Requirement],
    *,
    root_name: str | None = None,
) -> set[str]:
    """Every distribution reachable from these requirements.

    An extra is expanded wherever it is requested, not only the first time the
    package is reached -- a package can arrive plain from one edge and with an
    extra from another, and the second edge's subtree is part of the obligation.
    The (name, extra) pairs already expanded are tracked so a cycle through an
    extra terminates.

    ``root_name`` names the project itself, whose extras are declared by
    self-reference (``all = ["mosaic-behavior[pose,faiss]"]``). Such an edge means
    "and also that extra", never "and also every core dependency": following the
    core requirements would put the whole base tree inside each extra's licence
    closure, and this report exists to say what a *particular* extra obliges you
    to. So the root's extras are expanded and its core requirements are not.
    """
    reached: set[str] = set()
    expanded: set[tuple[str, str]] = set()
    pending: list[Requirement] = list(roots)
    while pending:
        requirement = pending.pop()
        package = packages.get(requirement.name)
        if package is None:
            continue
        if requirement.name != root_name and requirement.name not in reached:
            reached.add(requirement.name)
            pending.extend(package.requires)
        for extra in requirement.extras:
            marker = (requirement.name, extra)
            if marker in expanded:
                continue
            expanded.add(marker)
            pending.extend(package.optional_requires.get(extra, ()))
    return reached


def classify(text: str) -> Obligation:
    """The obligation a license string implies, or ``unknown``."""
    lowered = text.lower()
    for obligation, pattern in _OBLIGATION_PATTERNS:
        if re.search(pattern, lowered):
            return obligation
    return "unknown"


def fetch_json(url: str, timeout: float) -> object:
    """One GET returning parsed JSON, or None on any failure. Never raises."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: bytes = response.read()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    try:
        document: object = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return document


def license_from_pypi_info(info: dict[str, object]) -> LicenseFact:
    """PEP 639 order: expression, then trove classifiers, then free text."""
    expression = read_text(info.get("license_expression")).strip()
    if expression:
        return LicenseFact(expression, "expression", classify(expression), "")

    trove = [
        entry
        for entry in (read_text(item) for item in read_array(info.get("classifiers")))
        if entry.startswith("License :: ")
    ]
    if trove:
        shortened = " OR ".join(
            entry.removeprefix("License :: OSI Approved :: ").removeprefix(
                "License :: "
            )
            for entry in trove
        )
        return LicenseFact(shortened, "classifier", classify(shortened), "")

    free_text = read_text(info.get("license")).strip()
    if free_text:
        first = free_text.splitlines()[0].strip()
        shown = (
            first if len(first) <= FREE_TEXT_LIMIT else f"{first[:FREE_TEXT_LIMIT]}..."
        )
        return LicenseFact(
            shown,
            "free-text",
            classify(shown),
            "free-text license field, not an identifier -- verify by hand",
        )

    return LicenseFact(
        "",
        "missing",
        "unknown",
        "no license metadata published on PyPI -- verify by hand",
    )


def resolve_license(
    package: LockedPackage, *, offline: bool, timeout: float
) -> LicenseFact:
    """This package's license, from an override or from PyPI, never a guess."""
    if package.source_kind != "registry":
        override = OVERRIDES.get(base_url(package.source_url))
        if override is not None:
            return LicenseFact(
                override.expression, "override", override.obligation, override.note
            )
        missing = (
            f"{package.source_kind} source with no override; PyPI is not queried "
            "for it because the name there may be different code. Add an entry "
            "to OVERRIDES."
        )
        return LicenseFact("", "missing", "unknown", missing)

    known = REGISTRY_OVERRIDES.get(package.name)
    if known is not None:
        return LicenseFact(known.expression, "override", known.obligation, known.note)

    if offline:
        return LicenseFact(
            "", "not-queried", "unknown", "offline: PyPI was not queried"
        )

    document = fetch_json(
        f"{PYPI_ENDPOINT}/{package.name}/{package.version}/json", timeout
    )
    if document is None:
        document = fetch_json(f"{PYPI_ENDPOINT}/{package.name}/json", timeout)
    if document is None:
        return LicenseFact("", "missing", "unknown", "PyPI lookup failed")
    return license_from_pypi_info(read_table(read_table(document).get("info")))


def build_inventory(
    packages: Mapping[str, LockedPackage],
    root: LockedPackage,
    *,
    offline: bool,
    timeout: float,
) -> list[InventoryRow]:
    """Every distribution any group reaches, with the groups that reach it."""
    reached_by: dict[str, list[str]] = {}
    for group, requirements in group_requirements(root).items():
        for name in sorted(closure(packages, requirements, root_name=root.name)):
            if name == root.name:
                continue
            reached_by.setdefault(name, []).append(group)

    rows: list[InventoryRow] = []
    for name in sorted(reached_by):
        package = packages[name]
        rows.append(
            InventoryRow(
                name=package.name,
                version=package.version,
                source_kind=package.source_kind,
                source_url=base_url(package.source_url),
                license=resolve_license(package, offline=offline, timeout=timeout),
                groups=tuple(reached_by[name]),
            )
        )
    return rows


def is_flagged(row: InventoryRow) -> bool:
    """True when a commercial user needs to look at this row."""
    return row.license.obligation != "permissive"


def render_text(
    lock: Path, root: LockedPackage, rows: list[InventoryRow], show_all: bool
) -> str:
    """The default human report."""
    lines: list[str] = []
    lines.append(f"lock:  {lock}")
    lines.append(f"root:  {root.name} {root.version}")
    lines.append(f"third-party distributions reached: {len(rows)}")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.license.obligation] = counts.get(row.license.obligation, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    lines.append(f"by obligation: {summary}")
    lines.append("")

    lines.append("Groups")
    lines.append("------")
    per_group: dict[str, list[InventoryRow]] = {}
    for row in rows:
        for group in row.groups:
            per_group.setdefault(group, []).append(row)
    for group in sorted(per_group):
        members = per_group[group]
        flagged = [row for row in members if is_flagged(row)]
        counted = f"{len(members):>4} distributions, {len(flagged):>3} flagged"
        lines.append(f"  {group:<22} {counted}")
    lines.append("")

    shown = rows if show_all else [row for row in rows if is_flagged(row)]
    heading = "All distributions" if show_all else "Flagged: not verified permissive"
    lines.append(heading)
    lines.append("-" * len(heading))
    if not shown:
        lines.append("  (none)")
    for row in shown:
        origin = row.source_kind
        if row.source_kind != "registry":
            origin = f"{row.source_kind} {row.source_url}"
        lines.append(f"  {row.name} {row.version}  [{origin}]")
        expression = row.license.expression or "(none published)"
        lines.append(f"      license:    {expression}  ({row.license.origin})")
        lines.append(f"      obligation: {row.license.obligation}")
        lines.append(f"      reached by: {', '.join(row.groups)}")
        if row.license.note:
            lines.append(f"      note:       {row.license.note}")
    return "\n".join(lines)


def render_markdown(
    lock: Path, root: LockedPackage, rows: list[InventoryRow], show_all: bool
) -> str:
    """A table that can be pasted into docs/licensing.md."""
    shown = rows if show_all else [row for row in rows if is_flagged(row)]
    header = (
        "| Distribution | Version | Source | License | Origin | Obligation "
        "| Reached by |"
    )
    lines: list[str] = []
    lines.append(f"<!-- generated from {lock.name} by gen_third_party_inventory.py -->")
    lines.append("")
    lines.append(f"{len(rows)} reached from `{root.name}`, {len(shown)} shown.")
    lines.append("")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|")
    for row in shown:
        expression = row.license.expression or "(none published)"
        groups = ", ".join(f"`{group}`" for group in row.groups)
        lines.append(
            f"| `{row.name}` | {row.version} | {row.source_kind} | {expression} "
            f"| {row.license.origin} | {row.license.obligation} | {groups} |"
        )
    return "\n".join(lines)


def render_json(lock: Path, root: LockedPackage, rows: list[InventoryRow]) -> str:
    """The whole inventory, for a downstream consumer."""
    payload = {
        "lock": str(lock),
        "root": {"name": root.name, "version": root.version},
        "distribution_count": len(rows),
        "distributions": [
            {
                "name": row.name,
                "version": row.version,
                "source_kind": row.source_kind,
                "source_url": row.source_url,
                "license": row.license.expression,
                "license_origin": row.license.origin,
                "obligation": row.license.obligation,
                "note": row.license.note,
                "groups": list(row.groups),
            }
            for row in rows
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--lock",
        type=Path,
        action="append",
        help="a uv.lock to inventory; repeatable (default: the project locks)",
    )
    _ = parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="report format (default: text)",
    )
    _ = parser.add_argument(
        "--all",
        action="store_true",
        help="list every distribution, not only the flagged ones",
    )
    _ = parser.add_argument(
        "--offline",
        action="store_true",
        help="skip PyPI; report every non-overridden license as unknown",
    )
    _ = parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="seconds per PyPI request (default: 15)",
    )
    arguments = parser.parse_args()

    requested: list[Path] | None = arguments.lock
    locks = list(requested) if requested else list(DEFAULT_LOCKS)
    report_format: ReportFormat = arguments.format
    show_all: bool = arguments.all
    offline: bool = arguments.offline
    timeout: float = arguments.timeout

    for index, lock in enumerate(locks):
        if index:
            print()
        if not lock.exists():
            print(f"missing lock: {lock}")
            continue
        packages = read_lock(lock)
        root = find_root(packages)
        if root is None:
            print(f"no root package in {lock}; nothing to walk from")
            continue
        rows = build_inventory(packages, root, offline=offline, timeout=timeout)
        if report_format == "json":
            print(render_json(lock, root, rows))
        elif report_format == "markdown":
            print(render_markdown(lock, root, rows, show_all))
        else:
            print(render_text(lock, root, rows, show_all))


if __name__ == "__main__":
    main()
