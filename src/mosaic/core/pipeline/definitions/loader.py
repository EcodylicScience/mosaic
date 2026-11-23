"""Importing a stored definition under its module name, with its checks.

One :class:`DefinitionFinder` per process, at the front of ``sys.meta_path``,
resolves ``mosaic_definitions`` as a namespace package and
``mosaic_definitions.<name with - as _>__<sha256>`` from the
``feature_definitions`` roots registered with it. Any other name under the package
is refused. Every import of a definition's module, a worker's by name included,
goes through :meth:`_DefinitionLoader.exec_module`. The class it yields therefore
has ``name = "def:<name>"`` and ``__mosaic_definition__``.

At import the module loads the standard library and the store and sidecar modules
only. Executing a definition imports the checks, the registries and the feature
library inside :meth:`_DefinitionLoader.exec_module`.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
import types
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypeIs

from pydantic import BaseModel

from mosaic.core.pipeline.definitions.sidecar import (
    DefinitionRef,
    DefinitionSidecar,
    read_sidecar,
)
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_PACKAGE,
    MODEL_FIELD,
    DefinitionCorrupt,
    DefinitionInvalid,
    DefinitionMismatch,
    DefinitionNameInvalid,
    definition_module_name,
    list_versions,
)

if TYPE_CHECKING:
    from mosaic.core.json_value import JsonValue
    from mosaic.core.pipeline.types.feature import Feature
    from mosaic.core.pipeline.types.results import DefinitionModel

__all__ = [
    "DEFINITION_MARK",
    "DefinitionFinder",
    "applied_definition",
    "definition_ref_of",
    "load_definition",
    "narrow_to_apply",
    "register_definitions_root",
]

DEFINITION_MARK: Final = "__mosaic_definition__"
"""The class attribute holding a loaded definition's :class:`DefinitionRef`."""

_COMPARED: Final = (
    "class_name",
    "version",
    "summary",
    "flags",
    "inputs",
    "params",
    "attributes",
    "prerequisites",
)
"""The sidecar fields a definition owns, which its loaded class must reproduce."""

_METHODS: Final = ("load_state", "fit", "save_state", "apply")
_DIGEST_SEPARATOR: Final = "__"
_INSTALL_LOCK: Final = threading.Lock()


class _PackageLoader(importlib.abc.Loader):
    """Creates the empty ``mosaic_definitions`` package."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        """Use the default module."""
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        """Run nothing: the package holds only the definitions' modules."""


class _DefinitionLoader(importlib.abc.Loader):
    """Executes one stored version of a definition and checks its class.

    Args:
        name: The definition's name.
        digest: The full SHA-256 that names the version.
        path: The version's ``.py``, beside its ``.json`` sidecar.
    """

    def __init__(self, name: str, digest: str, path: Path) -> None:
        self.name: str = name
        self.digest: str = digest
        self.path: Path = path

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        """Use the default module."""
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        """Execute the source into *module* and mark its class.

        The source's bytes must hash to the digest. Every registry is filled with
        the built-ins first, and neither the execution nor the checks of the class
        may change one. The class is checked again in this process, and the facts
        the definition owns must equal its sidecar's. The class then gets
        ``name = "def:<name>"`` and ``__mosaic_definition__``.

        Raises:
            DefinitionCorrupt: The source does not hash to its digest, or the
                sidecar is unreadable or describes another version.
            DefinitionInvalid: The source changed a registry, or its class fails
                a check of a definition.
            DefinitionMismatch: The class differs from its sidecar, naming each
                differing field.
            DefinitionToolkitModelMissing: The class names a toolkit model this
                installation lacks.
        """
        from functools import partial

        from mosaic.core.pipeline.definitions.declare import (
            declare_class,
            definition_class,
            registry_findings,
        )
        from mosaic.core.pipeline.graph.resolve import (
            fill_builtin_registries,
            registry_snapshot,
            toolkit_model_for_ref,
        )

        source = self.path.read_bytes()
        if hashlib.sha256(source).hexdigest() != self.digest:
            raise DefinitionCorrupt.of_digest(self.name, self.digest, self.path)
        sidecar_path = self.path.with_suffix(".json")
        stored = read_sidecar(sidecar_path)
        if (stored.name, stored.digest) != (self.name, self.digest):
            raise DefinitionCorrupt(
                self.name, self.digest, sidecar_path, reason="describes another version"
            )
        fill_builtin_registries()
        before = registry_snapshot()
        exec(compile(source, str(self.path), "exec", dont_inherit=True), vars(module))
        changed = registry_findings(before, registry_snapshot())
        if changed:
            raise DefinitionInvalid(self.name, changed)
        loaded = declare_class(
            module,
            self.name,
            self.digest,
            stored.defined_on,
            partial(toolkit_model_for_ref, definition=self.name),
        )
        changed = registry_findings(before, registry_snapshot())
        if changed:
            raise DefinitionInvalid(self.name, changed)
        differing = _differing_fields(loaded, stored)
        if differing:
            raise DefinitionMismatch(self.name, self.digest, differing)
        cls = definition_class(module, self.name)
        setattr(cls, "name", f"def:{self.name}")
        setattr(cls, DEFINITION_MARK, DefinitionRef(self.name, self.digest))


class DefinitionFinder(importlib.abc.MetaPathFinder):
    """Finds the definitions' modules in the ``feature_definitions`` roots it holds.

    The root registered last is searched first. A load therefore reads the version
    from the dataset it names when another dataset holds the same digest. The first
    root that holds a complete version of the digest supplies the file.
    """

    def __init__(self) -> None:
        self._roots: list[Path] = []
        self._lock: threading.Lock = threading.Lock()

    def add_root(self, root: Path) -> None:
        """Search *root* for definitions before every root added earlier."""
        resolved = root.resolve()
        with self._lock:
            if resolved in self._roots:
                self._roots.remove(resolved)
            self._roots.insert(0, resolved)

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Return the spec of the package or of a stored definition's module.

        Returns ``None`` for a name outside ``mosaic_definitions``, which leaves it
        to the other finders.

        Raises:
            ModuleNotFoundError: *fullname* is under ``mosaic_definitions`` and no
                registered root holds the version it names.
        """
        if fullname == DEFINITIONS_PACKAGE:
            return importlib.machinery.ModuleSpec(
                fullname, _PackageLoader(), is_package=True
            )
        if not fullname.startswith(f"{DEFINITIONS_PACKAGE}."):
            return None
        located = self._locate(fullname)
        if located is None:
            raise ModuleNotFoundError(
                f"no feature definition in a registered directory is imported as "
                f"{fullname!r}",
                name=fullname,
            )
        spec = importlib.machinery.ModuleSpec(
            fullname, _DefinitionLoader(*located), origin=str(located[2])
        )
        spec.has_location = True
        return spec

    def _locate(self, fullname: str) -> tuple[str, str, Path] | None:
        """The name, the digest and the ``.py`` of the version *fullname* names."""
        head, separator, digest = fullname.rpartition(_DIGEST_SEPARATOR)
        name = head.removeprefix(f"{DEFINITIONS_PACKAGE}.").replace("_", "-")
        try:
            if not separator or definition_module_name(name, digest) != fullname:
                return None
        except DefinitionNameInvalid:
            return None
        with self._lock:
            roots = list(self._roots)
        for root in roots:
            for version in list_versions(root / name):
                if version.digest == digest:
                    return name, digest, root / name / f"{version.stem}.py"
        return None


def _installed_finder() -> DefinitionFinder:
    """The finder of this process, put first in ``sys.meta_path`` on first use."""
    with _INSTALL_LOCK:
        for finder in sys.meta_path:
            if isinstance(finder, DefinitionFinder):
                return finder
        finder = DefinitionFinder()
        sys.meta_path.insert(0, finder)
        return finder


def register_definitions_root(path: Path) -> None:
    """Let the definitions under the ``feature_definitions`` root *path* be imported."""
    _installed_finder().add_root(path)


def load_definition(definitions_dir: Path, name: str, digest: str) -> type[Feature]:
    """Import the version *digest* of the definition *name* and return its class.

    The module stays in ``sys.modules``. A later load or an import by its name
    returns the same class.

    Args:
        definitions_dir: The dataset's ``feature_definitions`` root.
        name: The definition's name.
        digest: The full SHA-256 of the version's source.

    Raises:
        DefinitionNameInvalid: *name* does not match the grammar.
        ModuleNotFoundError: No registered root holds the version.
        DefinitionCorrupt: The stored source or sidecar is not what its name says.
        DefinitionInvalid: The source changed a registry or fails a check.
        DefinitionMismatch: The loaded class differs from its sidecar.
        DefinitionToolkitModelMissing: The class names a toolkit model this
            installation lacks.
    """
    register_definitions_root(definitions_dir)
    module = importlib.import_module(definition_module_name(name, digest))
    ref = DefinitionRef(name, digest)
    for value in vars(module).values():
        if not (isinstance(value, type) and DEFINITION_MARK in vars(value)):
            continue
        mark: object = getattr(value, DEFINITION_MARK)
        if mark == ref and _is_feature_class(value):
            return value
    raise ImportError(
        f"{module.__name__} holds no loaded class of the definition {name!r}",
        name=module.__name__,
    )


def definition_ref_of(cls: type) -> DefinitionRef | None:
    """The version of a definition that *cls* is, ``None`` for any other class.

    Both a loaded class and a class declared from a sidecar carry the mark.
    """
    ref: object = getattr(cls, DEFINITION_MARK, None)
    return ref if isinstance(ref, DefinitionRef) else None


def applied_definition(
    feature: Feature,
) -> tuple[DefinitionModel, DefinitionRef] | None:
    """The fit that *feature* applies and the definition version that applies it.

    The fit is the instance's ``model`` param when its class is a definition that
    declares ``fits = True`` and the value is a
    :class:`~mosaic.core.pipeline.types.results.DefinitionModel`. A built-in, a
    definition that fits nothing and an instance given no model answer
    ``None``. The class decides, so the answer holds after
    :func:`narrow_to_apply` has cleared the instance's own ``fits``.
    """
    from mosaic.core.pipeline.types.results import DefinitionModel

    cls = type(feature)
    ref = definition_ref_of(cls)
    if ref is None or getattr(cls, "fits", False) is not True:
        return None
    model: object = getattr(getattr(feature, "params", None), MODEL_FIELD, None)
    return (model, ref) if isinstance(model, DefinitionModel) else None


def narrow_to_apply(feature: Feature) -> None:
    """Clear ``fits`` and ``scope_dependent`` on an instance applying a fit.

    An instance that applies a fit learns nothing, and its identity covers no
    entries. Any other instance is left as it is, and a second call changes
    nothing.
    """
    if applied_definition(feature) is None:
        return
    setattr(feature, "fits", False)
    setattr(feature, "scope_dependent", False)


def _is_feature_class(cls: type) -> TypeIs[type[Feature]]:
    """Whether *cls* has the four methods of a feature.

    The loader's checks have already held the class to the whole protocol.
    """
    for method in _METHODS:
        value: object = getattr(cls, method, None)
        if not callable(value):
            return False
    return True


def _differing_fields(
    loaded: DefinitionSidecar, stored: DefinitionSidecar
) -> list[str]:
    """Each definition-owned field, dotted into records, where the two sidecars differ.

    Values are compared through their canonical JSON, a field left unset reading as its
    default. The result is the same in every process, and a sidecar that omits a
    defaulted fact still matches. A field of ``flags`` or ``attributes`` whose default
    is ``None`` differs also when only one side declares it, because a declared ``None``
    is not an undeclared attribute.
    """
    from mosaic.core.json_object import JSON_OBJECT

    fields: set[str] = set(_COMPARED)
    loaded_facts = JSON_OBJECT.validate_json(loaded.model_dump_json(include=fields))
    stored_facts = JSON_OBJECT.validate_json(stored.model_dump_json(include=fields))
    differing = [
        difference
        for field in _COMPARED
        for difference in _differences(
            field, loaded_facts.get(field), stored_facts.get(field)
        )
    ]
    for record, loaded_record, stored_record in (
        ("flags", loaded.flags, stored.flags),
        ("attributes", loaded.attributes, stored.attributes),
    ):
        declared = _declared_none_defaults(loaded_record) ^ _declared_none_defaults(
            stored_record
        )
        differing.extend(
            path
            for path in (f"{record}.{field}" for field in sorted(declared))
            if path not in differing
        )
    return differing


def _declared_none_defaults(record: BaseModel) -> set[str]:
    """The fields of *record* whose default is ``None`` and that are set on it."""
    return {
        name
        for name, info in type(record).model_fields.items()
        if info.default is None and name in record.model_fields_set
    }


def _differences(path: str, loaded: JsonValue, stored: JsonValue) -> list[str]:
    """*path*, or the dotted path of each differing key when both values are objects."""
    from mosaic.core.pipeline._utils import canonical_json

    if isinstance(loaded, dict) and isinstance(stored, dict):
        return [
            difference
            for key in sorted(loaded.keys() | stored.keys())
            for difference in _differences(
                f"{path}.{key}", loaded.get(key), stored.get(key)
            )
        ]
    return [] if canonical_json(loaded) == canonical_json(stored) else [path]
