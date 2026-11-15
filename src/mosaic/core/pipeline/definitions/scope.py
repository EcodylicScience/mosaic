"""The definition scope: the definitions a ``def:`` slug may name, and its class.

A scope is the definitions of one dataset and whether its caller may load their
code. It lives in a ``ContextVar`` and belongs to one thread or asyncio task. A
thread starts with none. Under no scope a ``def:`` slug names nothing. The slug
lookups read the ambient scope through :func:`definition_class_for_slug`, which
returns the declared class of a sidecar under a ``load=False`` scope and the loaded
class under a ``load=True`` one.

:func:`definition_scope` nests. An entry of the dataset that the outer scope holds,
or under an outer scope without a dataset, reuses that scope: it fills in its
dataset, raises ``load`` to its own, and adds only the pins and sidecars that the
outer scope lacks. An entry of another dataset replaces the outer scope until it
exits. A disabled scope is kept as it is by every entry under it.

At import the module loads the standard library, pydantic, the store and the
sidecar only. Building a declared class imports the column contract and the
registries' lookups, and loading a definition imports the loader, both inside the
function that needs them.
"""

from __future__ import annotations

import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import BaseModel

from mosaic.core.pipeline.definitions.sidecar import (
    DefinitionRef,
    DefinitionSidecar,
    default_inputs,
    read_sidecar,
    rebuild_inputs_model,
    rebuild_params_model,
)
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionCorrupt,
    DefinitionDigestAmbiguous,
    DefinitionNameInvalid,
    DefinitionNotFound,
    DefinitionNotLoaded,
    FeatureDefinitionsNotAllowed,
    list_definition_names,
    list_versions,
    parse_definition_slug,
    resolve_digest,
    resolve_stored_slug,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.json_value import JsonValue
    from mosaic.core.params import Params

__all__ = [
    "DefinitionScope",
    "clear_definition_caches",
    "current_scope",
    "declared_class",
    "declared_class_of",
    "definition_class_for_slug",
    "definition_scope",
    "definition_sidecar_for_slug",
    "resolve_definition_slug",
    "scope_definition_names",
    "scope_version_slugs",
]


@dataclass(frozen=True)
class DefinitionScope:
    """The definitions of one dataset, and whether this caller may load their code.

    Attributes:
        base_dir: The dataset's directory, ``None`` for a scope of no dataset.
        load: Whether a ``def:`` slug gives the loaded class rather than the
            class declared from its sidecar.
        pins: Definition name to full digest, as a request records them.
        sidecars: Full digest to sidecar, for definitions not written to the
            dataset. Each takes precedence over the dataset's file of its digest.
        allowed: Whether definitions may be used at all. Under a disabled scope a
            ``def:`` slug is refused.
        definitions_dir: The dataset's ``feature_definitions`` root, ``None`` for
            a scope of no dataset or of a dataset that declares no
            ``feature_definitions`` root.
    """

    base_dir: Path | None
    load: bool
    pins: Mapping[str, str] = field(default_factory=dict[str, str])
    sidecars: Mapping[str, DefinitionSidecar] = field(
        default_factory=dict[str, DefinitionSidecar]
    )
    allowed: bool = True
    definitions_dir: Path | None = None


_SCOPE: Final[ContextVar[DefinitionScope | None]] = ContextVar(
    "mosaic_definition_scope", default=None
)
_CACHE_LOCK: Final = threading.Lock()
_SIDECARS: Final[dict[Path, DefinitionSidecar]] = {}
"""Each sidecar read, by path: a sidecar file never changes."""
_DECLARED: Final[dict[str, tuple[DefinitionSidecar, type]]] = {}
"""Each declared class built, by digest, beside the sidecar it was built from."""


def current_scope() -> DefinitionScope | None:
    """The scope of this thread or task, ``None`` when none was entered."""
    return _SCOPE.get()


@contextmanager
def definition_scope(
    ds_or_base: Dataset | Path | None,
    *,
    load: bool,
    pins: Mapping[str, str] | None = None,
    sidecars: Mapping[str, DefinitionSidecar] | None = None,
    allowed: bool = True,
) -> Generator[DefinitionScope]:
    """Enter the scope of a dataset for the block, nested in the scope found.

    Args:
        ds_or_base: The dataset, its directory, whose definitions are under the
            default ``feature_definitions`` path, or ``None`` for no dataset.
        load: Whether the block may load definitions' code.
        pins: Definition name to full digest, for the names the scope found does
            not pin.
        sidecars: Full digest to sidecar, for the digests the scope found does not
            hold.
        allowed: ``False`` disables definitions for the block and every entry
            under it.

    Yields:
        The scope in force for the block.
    """
    base_dir, definitions_dir = _located(ds_or_base)
    entered = DefinitionScope(
        base_dir=base_dir,
        load=load,
        pins=dict(pins or {}),
        sidecars=dict(sidecars or {}),
        allowed=allowed,
        definitions_dir=definitions_dir,
    )
    active = _nested(_SCOPE.get(), entered)
    token = _SCOPE.set(active)
    try:
        yield active
    finally:
        _SCOPE.reset(token)


def _located(ds_or_base: Dataset | Path | None) -> tuple[Path | None, Path | None]:
    """The dataset directory and its ``feature_definitions`` root.

    A dataset that declares its roots without ``feature_definitions`` holds no
    definitions. Its root is ``None``.
    """
    if ds_or_base is None:
        return None, None
    if isinstance(ds_or_base, Path):
        return ds_or_base, ds_or_base / DEFINITIONS_ROOT
    if not ds_or_base.has_root(DEFINITIONS_ROOT):
        return ds_or_base.base_dir, None
    return ds_or_base.base_dir, ds_or_base.get_root(DEFINITIONS_ROOT)


def _nested(outer: DefinitionScope | None, entered: DefinitionScope) -> DefinitionScope:
    """The scope in force when *entered* is entered under *outer*."""
    if outer is None:
        return entered
    if not outer.allowed:
        return outer
    if not (
        outer.base_dir is None
        or entered.base_dir is None
        or outer.base_dir.resolve() == entered.base_dir.resolve()
    ):
        return entered
    widened = replace(
        outer,
        base_dir=outer.base_dir or entered.base_dir,
        definitions_dir=outer.definitions_dir or entered.definitions_dir,
        load=outer.load or entered.load,
        pins={**entered.pins, **outer.pins},
        sidecars={**entered.sidecars, **outer.sidecars},
        allowed=entered.allowed,
    )
    return outer if widened == outer else widened


def resolve_definition_slug(slug: str) -> tuple[str, str]:
    """Return the name and the full digest of the version *slug* names in this scope.

    A digest or a prefix names the one version, in the dataset or among the
    scope's sidecars, that starts with it. A bare ``def:<name>`` names the scope's
    pin for the name, else the dataset's ``current``.

    Raises:
        FeatureDefinitionsNotAllowed: The scope is disabled.
        DefinitionNameInvalid: *slug* is not a definition slug.
        DefinitionNotFound: No scope is entered, or no version matches.
        DefinitionDigestAmbiguous: A prefix matches more than one version.
        DefinitionCorrupt: The dataset's ``current`` is not a version stem.
    """
    scope = _SCOPE.get()
    if scope is not None and not scope.allowed:
        raise FeatureDefinitionsNotAllowed(slug)
    name, digest = parse_definition_slug(slug)
    if scope is None:
        raise DefinitionNotFound(slug)
    digest = digest or scope.pins.get(name)
    if digest is None:
        if scope.definitions_dir is None:
            raise DefinitionNotFound(slug)
        return resolve_stored_slug(scope.definitions_dir, slug)
    matches = {
        held
        for held, sidecar in scope.sidecars.items()
        if sidecar.name == name and held.startswith(digest)
    }
    if scope.definitions_dir is not None:
        try:
            matches.add(resolve_digest(scope.definitions_dir / name, digest))
        except DefinitionNotFound:
            if not matches:
                raise
    if not matches:
        raise DefinitionNotFound(slug)
    if len(matches) > 1:
        raise DefinitionDigestAmbiguous(slug, sorted(matches))
    return name, matches.pop()


def definition_class_for_slug(slug: str) -> type | None:
    """The class a ``def:`` slug names in this scope, or ``None`` when it names none.

    Under no scope, and for a slug that is malformed or matches no version, it is
    ``None``. Under a ``load=False`` scope it is the class declared from the
    version's sidecar, and under a ``load=True`` scope the class of the loaded
    source.

    Raises:
        FeatureDefinitionsNotAllowed: The scope is disabled.
        DefinitionDigestAmbiguous: A prefix matches more than one version.
        DefinitionCorrupt: A stored file is not what its name says.
        DefinitionInvalid: A loaded source fails a check.
        DefinitionMismatch: A loaded class differs from its sidecar.
        DefinitionToolkitModelMissing: The definition names a toolkit model this
            installation lacks.
    """
    resolved = _resolved(slug)
    if resolved is None:
        return None
    scope, name, digest = resolved
    if not scope.load:
        return declared_class(_scope_sidecar(scope, name, digest))
    if scope.definitions_dir is None:
        return None
    from mosaic.core.pipeline.definitions.loader import load_definition

    return load_definition(scope.definitions_dir, name, digest)


def _resolved(slug: str) -> tuple[DefinitionScope, str, str] | None:
    """The scope in force, and the name and digest *slug* resolves to in it.

    ``None`` under no scope, and for a slug that is malformed or matches no version.

    Raises:
        FeatureDefinitionsNotAllowed: The scope is disabled.
        DefinitionDigestAmbiguous: A prefix matches more than one version.
        DefinitionCorrupt: The dataset's ``current`` is not a version stem.
    """
    scope = _SCOPE.get()
    if scope is None:
        return None
    try:
        name, digest = resolve_definition_slug(slug)
    except (DefinitionNameInvalid, DefinitionNotFound):
        return None
    return scope, name, digest


def definition_sidecar_for_slug(slug: str) -> DefinitionSidecar | None:
    """The sidecar of the version a ``def:`` slug names in this scope, or ``None``.

    ``None`` under no scope, and for a slug that is malformed or matches no version.
    The sidecar is read once per process and imports no source, whatever the scope
    loads.

    Raises:
        FeatureDefinitionsNotAllowed: The scope is disabled.
        DefinitionDigestAmbiguous: A prefix matches more than one version.
        DefinitionCorrupt: A stored file is not what its name says.
    """
    resolved = _resolved(slug)
    if resolved is None:
        return None
    scope, name, digest = resolved
    return _scope_sidecar(scope, name, digest)


def scope_definition_names() -> list[str]:
    """The names of the definitions this scope lists, sorted.

    The names the dataset holds a current version of and the names the scope pins.
    Empty under no scope and under a disabled one. Each name's bare slug resolves
    to the scope's pin for it, else the dataset's ``current``.
    """
    scope = _SCOPE.get()
    if scope is None or not scope.allowed:
        return []
    names = set(scope.pins)
    if scope.definitions_dir is not None:
        names.update(list_definition_names(scope.definitions_dir))
    return sorted(names)


def scope_version_slugs(slug: str) -> list[str]:
    """The slug of each version of the definition *slug* names that this scope holds.

    Every version of the name in the dataset and among the scope's sidecars, as
    ``def:<name>@<digest>`` sorted by digest, whatever digest *slug* gives. Empty
    under no scope and under a disabled one, and for a slug that is not a
    definition's.
    """
    scope = _SCOPE.get()
    if scope is None or not scope.allowed:
        return []
    try:
        name, _ = parse_definition_slug(slug)
    except DefinitionNameInvalid:
        return []
    digests = {held for held, sidecar in scope.sidecars.items() if sidecar.name == name}
    if scope.definitions_dir is not None:
        digests.update(
            version.digest for version in list_versions(scope.definitions_dir / name)
        )
    return [f"def:{name}@{digest}" for digest in sorted(digests)]


def declared_class_of(loaded: type) -> type | None:
    """The declared class of the version *loaded* is, under a ``load=True`` scope.

    ``None`` under any other scope, and for a class that is not a loaded
    definition.
    """
    from mosaic.core.pipeline.definitions.loader import DEFINITION_MARK

    scope = _SCOPE.get()
    ref: object = vars(loaded).get(DEFINITION_MARK)
    if scope is None or not scope.load or not isinstance(ref, DefinitionRef):
        return None
    return declared_class(_scope_sidecar(scope, ref.name, ref.digest))


def _scope_sidecar(scope: DefinitionScope, name: str, digest: str) -> DefinitionSidecar:
    """The sidecar of a version: the scope's own first, then the dataset's file.

    Raises:
        DefinitionNotFound: Neither holds the version.
        DefinitionCorrupt: The dataset's file is unreadable or describes another
            version.
    """
    held = scope.sidecars.get(digest)
    if held is not None and held.name == name:
        return held
    if scope.definitions_dir is not None:
        directory = scope.definitions_dir / name
        for version in list_versions(directory):
            if version.digest == digest:
                return _read_sidecar(directory / f"{version.stem}.json", name, digest)
    raise DefinitionNotFound(f"def:{name}@{digest}")


def _read_sidecar(path: Path, name: str, digest: str) -> DefinitionSidecar:
    """The sidecar at *path*, read once per process."""
    with _CACHE_LOCK:
        cached = _SIDECARS.get(path)
    if cached is not None:
        return cached
    sidecar = read_sidecar(path)
    if (sidecar.name, sidecar.digest) != (name, digest):
        raise DefinitionCorrupt(name, digest, path, reason="describes another version")
    with _CACHE_LOCK:
        return _SIDECARS.setdefault(path, sidecar)


def clear_definition_caches() -> None:
    """Forget every sidecar read and every declared class built in this process.

    A test that changes a toolkit model in-process calls it. The next lookup then
    rebuilds the declared class against the changed model.
    """
    with _CACHE_LOCK:
        _SIDECARS.clear()
        _DECLARED.clear()


class _DeclaredFeature:
    """The base of a declared class, whose source is not imported.

    It builds its inputs and params as a definition does, and each method that
    would run the source raises :class:`DefinitionNotLoaded`.
    """

    Inputs: ClassVar[type[BaseModel]]
    Params: ClassVar[type[Params]]
    definition_inputs: ClassVar[list[JsonValue]]
    definition_name: ClassVar[str]

    def __init__(
        self, inputs: BaseModel | None = None, params: dict[str, object] | None = None
    ) -> None:
        cls = type(self)
        self.inputs: BaseModel = (
            cls.Inputs.model_validate(cls.definition_inputs)
            if inputs is None
            else inputs
        )
        self.params: Params = cls.Params.from_overrides(params)

    def _not_loaded(self, method: str) -> DefinitionNotLoaded:
        return DefinitionNotLoaded(type(self).definition_name, method)

    def load_state(self, *_: object) -> bool:
        """Refuse: the state is the source's to read."""
        raise self._not_loaded("load_state")

    def fit(self, *_: object) -> None:
        """Refuse: fitting runs the source."""
        raise self._not_loaded("fit")

    def save_state(self, *_: object) -> None:
        """Refuse: the state is the source's to write."""
        raise self._not_loaded("save_state")

    def apply(self, *_: object) -> object:
        """Refuse: applying runs the source."""
        raise self._not_loaded("apply")


def declared_class(sidecar: DefinitionSidecar) -> type:
    """The class declared by *sidecar*, built without importing the source.

    It carries the sidecar's flags and ``version``, ``name = "def:<name>"``, the
    rebuilt ``Inputs`` and ``Params``, ``__mosaic_definition__``, the
    prerequisites as ``OptionalModule``s and every attribute the sidecar records,
    ``requires`` and ``provides`` as the column contract's records. Classes are
    cached by digest.

    Raises:
        DefinitionInvalid: A toolkit model the sidecar names is not one.
        DefinitionToolkitModelMissing: This installation lacks a toolkit model the
            sidecar names.
    """
    with _CACHE_LOCK:
        cached = _DECLARED.get(sidecar.digest)
    if cached is not None and cached[0] == sidecar:
        return cached[1]
    built = _build_declared_class(sidecar)
    with _CACHE_LOCK:
        _DECLARED[sidecar.digest] = (sidecar, built)
    return built


def _build_declared_class(sidecar: DefinitionSidecar) -> type:
    from functools import partial

    from pydantic import TypeAdapter

    from mosaic.core.pipeline.column_contract import Provides, Requirement
    from mosaic.core.pipeline.definitions.loader import DEFINITION_MARK
    from mosaic.core.pipeline.graph.compatibility import GpuWhen
    from mosaic.core.pipeline.graph.resolve import toolkit_model_for_ref
    from mosaic.optional_dependency import OptionalModule

    namespace: dict[str, object] = {
        "__module__": __name__,
        "__qualname__": sidecar.class_name,
        "__doc__": sidecar.summary,
        "name": f"def:{sidecar.name}",
        "version": sidecar.version,
        "Inputs": rebuild_inputs_model(sidecar),
        "Params": rebuild_params_model(
            sidecar, partial(toolkit_model_for_ref, definition=sidecar.name)
        ),
        "definition_inputs": default_inputs(sidecar.inputs),
        "definition_name": sidecar.name,
        DEFINITION_MARK: DefinitionRef(sidecar.name, sidecar.digest),
        "prerequisites": tuple(
            OptionalModule(each.module, each.extra, each.purpose)
            for each in sidecar.prerequisites
        ),
    }
    flags = sidecar.flags
    for flag, info in type(flags).model_fields.items():
        if info.default is not None or flag in flags.model_fields_set:
            namespace[flag] = getattr(flags, flag)
    attributes = sidecar.attributes
    for attribute in sorted(attributes.model_fields_set):
        namespace[attribute] = getattr(attributes, attribute)
    if "track_spaces" in attributes.model_fields_set:
        namespace["track_spaces"] = frozenset(attributes.track_spaces)
    if attributes.gpu_when is not None:
        namespace["gpu_when"] = GpuWhen(
            attributes.gpu_when.field, attributes.gpu_when.values
        )
    if attributes.requires is not None:
        namespace["requires"] = TypeAdapter(tuple[Requirement, ...]).validate_python(
            attributes.requires
        )
    if attributes.provides is not None:
        namespace["provides"] = TypeAdapter(Provides).validate_python(
            attributes.provides
        )
    return type(sidecar.class_name, (_DeclaredFeature,), namespace)
