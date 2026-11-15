"""The names, the on-disk layout and the errors of feature definitions.

A dataset keeps each definition under ``<root>/<name>/``, where ``<root>`` is the
dataset's ``feature_definitions`` root. A version is a pair of files named by its
stem, ``<YYYY-MM-DD>_<sha256>``: the source (``.py``) and its sidecar (``.json``).
The text file ``current`` holds the stem of the version that a bare ``def:<name>``
resolves to. Files are never edited, so a digest names one source under one name.

The module imports the standard library, pydantic and the error base only. The
agent's tool contract imports :class:`DefinitionRecord` and the status reads import
the grammar, and neither may load pandas or the feature library.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.strict_model import StrictModel

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue

__all__ = [
    "DEFINITIONS_ROOT",
    "DIGEST_PREFIX_MIN",
    "NAME_MAX_LENGTH",
    "NAME_PATTERN",
    "SLUG_PATTERN",
    "DefinitionCheck",
    "DefinitionCorrupt",
    "DefinitionDigestAmbiguous",
    "DefinitionFinding",
    "DefinitionInvalid",
    "DefinitionMismatch",
    "DefinitionNameInvalid",
    "DefinitionNameTaken",
    "DefinitionNotFound",
    "DefinitionNotLoaded",
    "DefinitionRecord",
    "DefinitionToolkitModelMissing",
    "FeatureDefinitionsNotAllowed",
    "VersionStem",
    "check_definition_name",
    "definition_module_name",
    "is_under_definitions_root",
    "list_definition_names",
    "list_versions",
    "parse_definition_slug",
    "read_current",
    "resolve_digest",
]

DEFINITIONS_ROOT: Final = "feature_definitions"
"""The name of the dataset root that holds the definitions, and its default path."""

NAME_PATTERN: Final = "[a-z][a-z0-9]*(-[a-z0-9]+)*"
"""The grammar of a definition name: kebab-case, lowercase only.

Lowercase keeps two names from colliding on a case-insensitive filesystem. The
grammar excludes ``_``, ``.``, ``:`` and ``@``, which the module name, the storage
name and the slug use as separators.
"""

NAME_MAX_LENGTH: Final = 64
"""The most characters a definition name has."""

DIGEST_PREFIX_MIN: Final = 12
"""The fewest hex characters of a digest that a slug may give."""

SLUG_PATTERN: Final = (
    f"def:<name>[@<{DIGEST_PREFIX_MIN} to 64 lowercase hex characters>] where "
    f"<name> matches {NAME_PATTERN}, at most {NAME_MAX_LENGTH} characters"
)
"""The grammar of a definition slug, as the error for a malformed one prints it."""

_DIGEST_LENGTH: Final = 64
_NAME = re.compile(NAME_PATTERN)
_DIGEST = re.compile(rf"[0-9a-f]{{{DIGEST_PREFIX_MIN},{_DIGEST_LENGTH}}}")
_STEM = re.compile(r"(?P<day>[0-9]{4}-[0-9]{2}-[0-9]{2})_(?P<digest>[0-9a-f]{64})")
_VERSION_FILE = re.compile(r"(?P<stem>[^.]+)\.(?P<suffix>py|json)")
_CURRENT: Final = "current"
_SLUG_HEAD: Final = "def:"
_MODULE_PREFIX: Final = "mosaic_definitions."


class _Coded:
    """The mixin that makes ``error_reason`` return the class's ``code``."""

    code: ClassVar[str]

    def error_reason(self) -> str | None:
        """Return the code that names this failure."""
        return self.code


class DefinitionNameInvalid(_Coded, ClassifiedError):
    """A name or slug that does not match its grammar."""

    error_kind: ClassVar[ErrorKind] = "invalid"
    code: ClassVar[str] = "definition_name_invalid"

    def __init__(self, name: str, *, pattern: str = NAME_PATTERN) -> None:
        limit = (
            f", at most {NAME_MAX_LENGTH} characters" if pattern == NAME_PATTERN else ""
        )
        super().__init__(f"{name!r} does not match {pattern}{limit}")
        self.name: str = name
        self.pattern: str = pattern

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the text refused and the grammar it should match."""
        return {"name": self.name, "pattern": self.pattern}


class DefinitionNameTaken(_Coded, ClassifiedError):
    """A definition name that is the name of a built-in feature."""

    error_kind: ClassVar[ErrorKind] = "conflict"
    code: ClassVar[str] = "definition_name_taken"

    def __init__(self, name: str) -> None:
        super().__init__(f"{name!r} is the name of a built-in feature")
        self.name: str = name

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name that is taken."""
        return {"name": self.name}


type DefinitionCheck = Literal[
    "class",
    "attribute",
    "flag_in_init",
    "temporal_reach",
    "inputs",
    "params_base",
    "params_default",
    "params_type",
    "params_metadata",
    "params_alias",
    "params_config",
    "params_validator",
    "params_serializer",
    "params_post_init",
    "params_reference",
    "model_marker",
    "model_rebuild",
    "construction",
]
"""The kinds of finding that make a source an invalid definition."""


@dataclass(frozen=True)
class DefinitionFinding:
    """One reason a source is not a valid definition.

    Attributes:
        check: Which check failed.
        field: The field of the class that the check concerns, when it has one.
        message: The sentence a person reads.
    """

    check: DefinitionCheck
    field: str | None
    message: str

    def as_json(self) -> dict[str, JsonValue]:
        """Return the finding as the JSON object an error detail carries."""
        return {"check": self.check, "field": self.field, "message": self.message}


class DefinitionInvalid(_Coded, ClassifiedError):
    """A source that fails the checks a definition must pass."""

    error_kind: ClassVar[ErrorKind] = "invalid"
    code: ClassVar[str] = "definition_invalid"

    def __init__(self, name: str, findings: Sequence[DefinitionFinding]) -> None:
        reasons = "; ".join(finding.message for finding in findings)
        super().__init__(f"the definition {name!r} is invalid: {reasons}")
        self.name: str = name
        self.findings: tuple[DefinitionFinding, ...] = tuple(findings)

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name and every finding."""
        findings: list[JsonValue] = [finding.as_json() for finding in self.findings]
        return {"name": self.name, "findings": findings}


class DefinitionCorrupt(_Coded, ClassifiedError):
    """A stored file that does not hold what its name says."""

    error_kind: ClassVar[ErrorKind] = "conflict"
    code: ClassVar[str] = "definition_corrupt"

    def __init__(self, name: str, digest: str | None, path: Path) -> None:
        super().__init__(f"{path} of the definition {name!r} is corrupt")
        self.name: str = name
        self.digest: str | None = digest
        self.path: Path = path

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the name, the digest when the file names one, and the path."""
        return {"name": self.name, "digest": self.digest, "path": str(self.path)}


class DefinitionNotFound(_Coded, ClassifiedError):
    """A slug that names no version."""

    error_kind: ClassVar[ErrorKind] = "not_found"
    code: ClassVar[str] = "definition_not_found"

    def __init__(self, slug: str) -> None:
        super().__init__(f"no feature definition matches {slug!r}")
        self.slug: str = slug

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the slug that matched nothing."""
        return {"slug": self.slug}


class DefinitionDigestAmbiguous(_Coded, ClassifiedError):
    """A digest prefix that starts more than one version."""

    error_kind: ClassVar[ErrorKind] = "conflict"
    code: ClassVar[str] = "definition_digest_ambiguous"

    def __init__(self, slug: str, matches: Sequence[str]) -> None:
        super().__init__(f"{slug!r} matches {', '.join(matches)}")
        self.slug: str = slug
        self.matches: tuple[str, ...] = tuple(matches)

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the slug and the full digests it matches."""
        matches: list[JsonValue] = list(self.matches)
        return {"slug": self.slug, "matches": matches}


class DefinitionMismatch(_Coded, ClassifiedError):
    """A loaded class that differs from the sidecar its plan was made from."""

    error_kind: ClassVar[ErrorKind] = "conflict"
    code: ClassVar[str] = "definition_mismatch"

    def __init__(self, name: str, digest: str, fields: Sequence[str]) -> None:
        super().__init__(
            f"the definition {name!r} ({digest[:DIGEST_PREFIX_MIN]}) differs from "
            f"its sidecar in {', '.join(fields)}"
        )
        self.name: str = name
        self.digest: str = digest
        self.fields: tuple[str, ...] = tuple(fields)

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the definition and each field that differs."""
        fields: list[JsonValue] = list(self.fields)
        return {"name": self.name, "digest": self.digest, "fields": fields}


class DefinitionToolkitModelMissing(_Coded, ClassifiedError):
    """A definition that names a toolkit model this installation does not have."""

    error_kind: ClassVar[ErrorKind] = "unavailable"
    code: ClassVar[str] = "definition_toolkit_model_missing"

    def __init__(self, name: str, ref: str) -> None:
        super().__init__(
            f"the definition {name!r} uses {ref}, which this installation lacks"
        )
        self.name: str = name
        self.ref: str = ref

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the definition and the missing model's reference."""
        return {"name": self.name, "ref": self.ref}


class DefinitionNotLoaded(_Coded, ClassifiedError):
    """A method called on a declared class whose source was not imported."""

    error_kind: ClassVar[ErrorKind] = "invalid"
    code: ClassVar[str] = "definition_not_loaded"

    def __init__(self, name: str, method: str) -> None:
        super().__init__(
            f"{method} of the definition {name!r} needs its source loaded, and "
            "this read does not import it"
        )
        self.name: str = name
        self.method: str = method

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the definition and the method that was called."""
        return {"name": self.name, "method": self.method}


class FeatureDefinitionsNotAllowed(_Coded, ClassifiedError):
    """A ``def:`` slug used where definitions are switched off."""

    error_kind: ClassVar[ErrorKind] = "unavailable"
    code: ClassVar[str] = "feature_definitions_not_allowed"

    def __init__(self, slug: str) -> None:
        super().__init__(
            f"feature definitions are not allowed here, so {slug!r} is refused"
        )
        self.slug: str = slug

    def error_detail(self) -> dict[str, JsonValue]:
        """Return the slug that was refused."""
        return {"slug": self.slug}


class DefinitionRecord(StrictModel):
    """What defining a source did.

    Attributes:
        name: The definition's name.
        digest: The full SHA-256 of the source.
        defined_on: The UTC date of the version's first definition.
        reused: Whether the source already had a version, which was reused.
        previous_digest: The digest that ``current`` named before, when it moved.
    """

    name: str
    digest: str
    defined_on: date
    reused: bool
    previous_digest: str | None = None


@dataclass(frozen=True, order=True)
class VersionStem:
    """The stem ``<YYYY-MM-DD>_<sha256>`` that names a version.

    Instances sort by date, then digest, which is the order of their stems.

    Attributes:
        date: The UTC date of the source's first definition under its name.
        digest: The 64-character lowercase hex SHA-256 of the source.
    """

    date: date
    digest: str

    @property
    def stem(self) -> str:
        """The file stem of the version's two files."""
        return f"{self.date.isoformat()}_{self.digest}"

    @staticmethod
    def parse(text: str) -> VersionStem | None:
        """Return the stem *text* names, or ``None`` when it is not one."""
        matched = _STEM.fullmatch(text)
        if matched is None:
            return None
        try:
            day = date.fromisoformat(matched["day"])
        except ValueError:
            return None
        return VersionStem(day, matched["digest"])


def check_definition_name(name: str) -> None:
    """Raise :class:`DefinitionNameInvalid` unless *name* matches the grammar."""
    if len(name) > NAME_MAX_LENGTH or _NAME.fullmatch(name) is None:
        raise DefinitionNameInvalid(name)


def parse_definition_slug(slug: str) -> tuple[str, str | None]:
    """Split ``def:<name>[@<digest>]`` into the name and the digest or prefix.

    Raises:
        DefinitionNameInvalid: *slug* is not a definition slug, its name is not
            valid, or its digest is not 12 to 64 lowercase hex characters.
    """
    if not slug.startswith(_SLUG_HEAD):
        raise DefinitionNameInvalid(slug, pattern=SLUG_PATTERN)
    name, separator, digest = slug.removeprefix(_SLUG_HEAD).partition("@")
    try:
        check_definition_name(name)
    except DefinitionNameInvalid:
        raise DefinitionNameInvalid(slug, pattern=SLUG_PATTERN) from None
    if not separator:
        return name, None
    if _DIGEST.fullmatch(digest) is None:
        raise DefinitionNameInvalid(slug, pattern=SLUG_PATTERN)
    return name, digest


def definition_module_name(name: str, digest: str) -> str:
    """Return the module name a version is imported under.

    ``mosaic_definitions.<name with - as _>__<sha256>``. The grammar excludes ``_``,
    so the mapping from a name is one to one, and the full digest keeps two versions
    apart in one process.

    Raises:
        DefinitionNameInvalid: *name* does not match the grammar.
    """
    check_definition_name(name)
    return f"{_MODULE_PREFIX}{name.replace('-', '_')}__{digest}"


def is_under_definitions_root(parts: tuple[str, ...]) -> bool:
    """Does a path with these components pass through ``feature_definitions``?

    Component-wise, never by prefix string, so a sibling named
    ``feature_definitions_backup`` is not under it. A scan is handed arbitrary
    search directories and matches the root's default path, as
    ``is_under_tracking_root`` matches ``_tracking``, because ``exclude_patterns``
    match basenames only.
    """
    return DEFINITIONS_ROOT in parts


def list_versions(directory: Path) -> list[VersionStem]:
    """Return the complete versions in *directory*, sorted by stem.

    A version is complete when its ``.py`` and its ``.json`` both exist. The listing
    reads only names of the form ``<YYYY-MM-DD>_<64 hex>.py`` or ``.json``, so it
    skips ``current``, ``current.lock``, the hidden ``.tmp`` files an interrupted
    write leaves, and any other file. An absent directory has no versions.
    """
    if not directory.is_dir():
        return []
    suffixes: dict[VersionStem, set[str]] = {}
    for path in directory.iterdir():
        matched = _VERSION_FILE.fullmatch(path.name)
        if matched is None:
            continue
        stem = VersionStem.parse(matched["stem"])
        if stem is not None:
            suffixes.setdefault(stem, set()).add(matched["suffix"])
    return sorted(stem for stem, found in suffixes.items() if found == {"py", "json"})


def read_current(directory: Path) -> VersionStem | None:
    """Return the version that ``current`` of *directory* names.

    An absent or empty ``current`` means the name has no current version.

    Raises:
        DefinitionCorrupt: ``current`` holds text that is not a version stem.
    """
    path = directory / _CURRENT
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    stem = VersionStem.parse(text)
    if stem is None:
        raise DefinitionCorrupt(directory.name, None, path)
    return stem


def list_definition_names(root: Path) -> list[str]:
    """Return the sorted names under *root* whose ``current`` names a complete version.

    A directory that is not a valid name, has no ``current``, or whose ``current``
    is empty or holds text that is not a version stem is skipped.
    """
    if not root.is_dir():
        return []
    names: list[str] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        try:
            check_definition_name(directory.name)
        except DefinitionNameInvalid:
            continue
        try:
            current = read_current(directory)
        except DefinitionCorrupt:
            continue
        if current is not None and current in list_versions(directory):
            names.append(directory.name)
    return sorted(names)


def resolve_digest(directory: Path, digest_or_prefix: str) -> str:
    """Return the full digest of the one version that *digest_or_prefix* names.

    Args:
        directory: The definition's directory, which is named for the definition.
        digest_or_prefix: A full digest, or a prefix of at least
            :data:`DIGEST_PREFIX_MIN` hex characters.

    Raises:
        DefinitionNameInvalid: The text is not 12 to 64 lowercase hex characters.
        DefinitionNotFound: No complete version starts with it.
        DefinitionDigestAmbiguous: More than one does.
    """
    slug = f"{_SLUG_HEAD}{directory.name}@{digest_or_prefix}"
    if _DIGEST.fullmatch(digest_or_prefix) is None:
        raise DefinitionNameInvalid(slug, pattern=SLUG_PATTERN)
    matches = sorted(
        {
            version.digest
            for version in list_versions(directory)
            if version.digest.startswith(digest_or_prefix)
        }
    )
    if not matches:
        raise DefinitionNotFound(slug)
    if len(matches) > 1:
        raise DefinitionDigestAmbiguous(slug, matches)
    return matches[0]
