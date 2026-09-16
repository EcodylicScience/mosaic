"""Every parameter field declares itself, and every declared fact reaches the schema.

Two states are separable at the schema and each has its own ratchet. A field
with no ``Declared`` publishes no description key and sits in
``ALLOWED_MISSING_DESCRIPTION``, bounded by ``ALLOWLIST_CEILING``. A field
declared ``Declared(NEEDS_DESCRIPTION)`` publishes an empty one. Declaring a
field and describing it are separate pieces of work, so
``UNDOCUMENTED_CEILING`` bounds the two states together: moving a field between
them leaves it flat and only writing prose brings it down. Both ceilings only
ever come down, which is what stops either state from being reached by adding a
line. A client drawing a form from the schema has no other source for a
control's label.

``ALLOWED_FIELD_DESCRIPTION`` is the second ratchet, over the forbidden
spelling.
``Field(description=...)`` overrides a ``Declared`` silently, in either
declaration order, and the resulting property still carries a description -- so
the prose guard passes while the ``Declared`` is dead. ``FieldInfo.description
is None`` is the signal that separates the two. ``FIELD_DESCRIPTION_CEILING``
bounds that allowlist as ``ALLOWLIST_CEILING`` bounds the first. Unbounded, the
one-line repair in front of whoever writes the spelling is to list the field.
Its prose then survives and its unit, its hash-exclude flag and its unwired
record do not, and the unit guard stops seeing it.

The prose guard reads ``model_json_schema()`` rather than ``FieldInfo.metadata``.
A constraint reached through a reusable ``Annotated`` alias lives inside that
alias's own annotation, so the outer field reports an empty ``metadata`` while
its ``minimum`` and ``maximum`` do appear in the schema.
``test_hash_exclude_agrees_between_metadata_and_schema`` reads both on purpose:
their disagreement is what would let ``identity_dump()`` and the published schema
describe different sets of fields.

**Coverage is the import list.** ``Params.__subclasses__()`` sees a class only
once its module has been imported, so the models enumerated here are the two
registries plus ``UNREGISTERED_PARAMS_MODULES``.
:func:`test_the_import_list_reaches_every_declared_params_model` reads
``src/mosaic`` as source and names any module that list stops reaching, which
is the one gap no other guard here can report: an unimported model is invisible
to
every walk over ``__subclasses__()``. The two modules a name match suggests it
misses, ``feature_library.external.kpms_server`` and
``pose_training.localizer_model``, declare no ``Params`` subclass at all.

**Reached by subclass, not by annotation.** ``DECLARED_MODELS`` holds every
``DeclaredModel`` in the tree, and every guard reads it or the
``NON_PARAMS_DECLARED_MODELS`` half of it -- the prose home, the repeated
``Declared``, the unit rule, the ``HASH_EXCLUDE`` agreement, the description.
Following field annotations out from a ``Params`` model reaches fewer: not a
model no annotation names, which is how ``ResultColumn`` and
``FeatureLabelsSource`` each shipped a defect through a sweep that had just
repaired their siblings.

**Two tiers, two bars.** The description guard is the one place they differ.
``UNDOCUMENTED_CEILING`` counts over ``PARAMS_MODELS``, so a
``Declared(NEEDS_DESCRIPTION)`` placeholder there is a tracked debt; on a
``NON_PARAMS_DECLARED_MODELS`` entry nothing would count it, so a placeholder
is refused outright. The ``HASH_EXCLUDE`` guard splits the same way for its own
reason: a top-level marker must agree with the schema, and a nested one is
refused, because ``identity_dump()`` pops top-level names only.

"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Annotated

import mosaic
import pytest
from pydantic import Field

from mosaic.core.params import (
    HASH_EXCLUDE,
    NEEDS_DESCRIPTION,
    Declared,
    DeclaredModel,
    HashExclude,
    Params,
    own_declaration,
)
from mosaic.core.strict_model import StrictModel
from mosaic.tracking import register_ops
from tests.helpers import (
    inside_a_virtualenv,
    runs_in_an_external_environment,
    source_tree,
)

UNREGISTERED_PARAMS_MODULES: tuple[str, ...] = (
    "mosaic.behavior.feature_library.feature_template__global",
    "mosaic.behavior.feature_library.feature_template__per_sequence",
    "mosaic.behavior.label_library",
    "mosaic.behavior.label_library.custom_label_template",
    "mosaic.behavior.label_library.label_converter_template",
    "mosaic.core.annotations.bbox",
    "mosaic.core.track_library.track_converter_template",
)
"""Modules holding a ``Params`` subclass that neither registry reaches.

The label and track converters reach users through the converter CLIs and hash
through the same ``identity_dump()``. The ``*_template`` modules are the classes
a person copies to add a converter or a feature, which is where an undescribed
field propagates from.
"""

# The op registry fills on an explicit call rather than on importing the
# package, so the op params models are unreachable without it.
register_ops()
for module_name in UNREGISTERED_PARAMS_MODULES:
    _ = importlib.import_module(module_name)


def params_models() -> dict[str, type[Params]]:
    """Return every shipped ``Params`` subclass, keyed by qualified name.

    Walks ``__subclasses__()`` transitively rather than reading the op and
    feature registries, which hold 62 between them and reach neither the
    converters nor the ``*_template`` modules.

    Two kinds of subclass are skipped. A model declared by a test module is a
    fixture rather than a shipped parameter model, so the walk keeps only
    ``mosaic.*``. Pydantic mints a concrete class for each parametrization of a
    generic model, named ``GlobalModelParams[KMeansModelArtifact]`` and carrying
    the generic's own fields, so the walk descends through those and records
    them under the generic's name instead.
    """
    models: dict[str, type[Params]] = {}
    seen: set[type[Params]] = set()
    pending: list[type[Params]] = list(Params.__subclasses__())
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        pending.extend(model.__subclasses__())
        if not model.__module__.startswith("mosaic."):
            continue
        if "[" in model.__qualname__:
            continue
        held = models.get(model.__qualname__, model)
        clash = (
            f"{model.__module__}.{model.__qualname__} and "
            f"{held.__module__}.{held.__qualname__} share one key; one of them "
            f"would go unchecked"
        )
        assert held is model, clash
        models[model.__qualname__] = model
    return models


PARAMS_MODELS = params_models()
MODEL_NAMES = sorted(PARAMS_MODELS)


def subclasses_in_the_tree(base: type[StrictModel]) -> dict[str, type[StrictModel]]:
    """Every shipped subclass of *base*, transitively, *base* included.

    ``__subclasses__()`` rather than a registry or a field walk, because the
    guards reading this ask about models those would not have reached: one that
    picked the wrong base, or one reachable through no field annotation at all.
    ``ResultColumn`` and ``FeatureLabelsSource`` were each the second of those.

    Keyed by module and qualified name together. The qualified name alone is
    unique -- it is what tells 48 models named ``Params`` apart, each named for
    the feature nesting it -- but a failure reports its key, and a bare
    ``KpmsFeature.Params`` does not say which file to open. Test fixtures are
    left out by module.
    """
    found: dict[str, type[StrictModel]] = {}
    seen: set[type[StrictModel]] = set()
    pending: list[type[StrictModel]] = [base]
    while pending:
        model = pending.pop()
        if model in seen:
            continue
        seen.add(model)
        pending.extend(model.__subclasses__())
        if model.__module__.startswith("mosaic."):
            found[f"{model.__module__}.{model.__qualname__}"] = model
    return found


DECLARED_MODELS = subclasses_in_the_tree(DeclaredModel)
DECLARED_MODEL_NAMES = sorted(DECLARED_MODELS)
"""Every model that takes ``DeclaredModel``'s hooks, however it is reached.

By subclass, because no narrower rule reaches them all. Following annotations
out from a ``Params`` field misses a model nothing annotates, and that is not a
corner: ``ResultColumn`` published a hand-written ``Attributes:`` block long
after the sweep that removed the others, and ``FeatureLabelsSource`` shipped a
blank control beside the labelled one its sibling declares.
"""


NON_PARAMS_DECLARED_MODELS = {
    name: model
    for name, model in DECLARED_MODELS.items()
    if not issubclass(model, Params)
}
NON_PARAMS_DECLARED_NAMES = sorted(NON_PARAMS_DECLARED_MODELS)
"""The declaring models that are not parameter models.

The two tiers meet a description bar of different strictness, and the split is
not style: :data:`UNDOCUMENTED_CEILING` counts over ``PARAMS_MODELS`` alone, so
a
``Declared(NEEDS_DESCRIPTION)`` placeholder is a tracked debt on a ``Params``
model and an untracked one anywhere else. These are refused a placeholder
outright.
"""


ALLOWLISTED_MODELS: dict[str, type[StrictModel]] = {
    **PARAMS_MODELS,
    **DECLARED_MODELS,
}
"""Every key space an allowlist entry can be written in, for resolving one.

The two allowlists are keyed by whatever their guard calls a model, and those
guards read different collections: ``ALLOWED_MISSING_DESCRIPTION`` by qualified
name out of ``PARAMS_MODELS``, ``ALLOWED_FIELD_DESCRIPTION`` by module and
qualified name out of ``DECLARED_MODELS``. The two shapes cannot collide, since
only the second carries a module prefix.
"""

PACKAGE_ROOT = Path(mosaic.__file__ or "").resolve().parent

PARAMS_MODULE_EXEMPTIONS: tuple[tuple[str, str], ...] = ()
"""Modules declaring a ``Params`` subclass that ``PARAMS_MODELS`` may skip,
each with the reason its fields reach no client. Empty: every module under
``src/mosaic`` that declares one is imported by a registry or by
``UNREGISTERED_PARAMS_MODULES``."""

ALLOWLIST_CEILING = 0
"""Zero, and only ever lowered. Every parameter field mosaic ships declares a
description, so an entry here would be a field reaching a client unlabelled."""

# Both ceilings reached zero together. This one was raised exactly once, from
# 488, when the label strict_schema and two template verbose fields were found
# carrying descriptions no source supported: an honest placeholder counts here
# where a false description did not. All three are settled -- strict_schema
# describes what it would mean and declares that nothing reads it, and the
# templates describe theirs -- so nothing is owed.
UNDOCUMENTED_CEILING = 0
"""Only ever lowered. Counts every field a client cannot draw a label for.

The sum of both states rather than the placeholder count alone, which is what
keeps it monotone through the bulk declaration: declaring an undeclared field
with :data:`NEEDS_DESCRIPTION` moves it between the two and leaves the sum flat,
and only writing prose brings it down.
"""

_PENDING_PROSE_NAMED = 10
"""How many pending fields a failure names, so hundreds cannot bury the fix."""

FIELD_DESCRIPTION_CEILING = 0
"""Zero, and only ever lowered. Every field mosaic ships states its prose in a
``Declared``, where the unit, the hash-exclude flag and the unwired record are
published beside it. Listing a field here keeps the prose and drops the rest."""

REF_SIBLING_REPAIR = (
    "{fields} render as a bare $ref, so check the declaration before assuming "
    "there is none: pydantic's handle_ref_overrides deletes a sibling key "
    "beside a $ref whose value equals the referenced schema's, so a Declared "
    "description character-identical to the referenced model's object "
    "description is dropped from the property outright. Reword one of the two "
    "-- a field description that restates its type's summary was saying "
    "nothing about the field anyway."
)
"""What to check first when a field that visibly declares prose fails a guard.

Read by :func:`missing_description_repair`, and so by both guards that report
an undescribed field, for the fields whose property is a ``$ref``. Nothing else
distinguishes the two causes at the schema: a field carrying no ``Declared``
and a field whose ``Declared`` collided with its type's docstring both publish
a property with no description key.

It replaces the "add a ``Declared``" advice for those fields rather than
following it, because for a ``$ref`` the likely repair is the opposite one --
the declaration is already written and one of the two strings has to move.
"""


def missing_description_repair(
    properties: dict[str, dict[str, object]], fields: list[str]
) -> str:
    """What to do about *fields*, which publish no description on that model.

    Splits on how the property renders, because the two shapes fail for
    different reasons and the wrong advice sends a reader to write a
    declaration that is already there.
    """
    by_ref = {field for field in fields if "$ref" in properties[field]}
    refs = sorted(by_ref)
    plain = [field for field in fields if field not in by_ref]
    parts: list[str] = []
    if plain:
        parts.append(f"Add a Declared(...) to the Annotated of {plain}.")
    if refs:
        parts.append(REF_SIBLING_REPAIR.format(fields=refs))
    return " ".join(parts)


ALLOWED_FIELD_DESCRIPTION: frozenset[tuple[str, str]] = frozenset()

ALLOWED_MISSING_DESCRIPTION: frozenset[tuple[str, str]] = frozenset()


def field_schemas(model: type[StrictModel]) -> dict[str, dict[str, object]]:
    """Return the property schema of each of *model*'s fields."""
    properties: dict[str, dict[str, object]] = model.model_json_schema()["properties"]
    return properties


def metadata_of(model: type[StrictModel], field: str) -> list[object]:
    """Return the ``Annotated`` metadata pydantic retained for *field*."""
    metadata: list[object] = model.model_fields[field].metadata
    return metadata


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_every_field_declares_a_description(model_name: str) -> None:
    """A field carries a ``Declared``, or it is on the shrinking allowlist.

    Presence of the description key is the question, not whether the prose is
    written: ``Declared(NEEDS_DESCRIPTION)`` publishes an empty one, which
    declares the field and hands the debt to
    :func:`test_the_undocumented_count_only_shrinks`. A field with no
    ``Declared`` at all publishes no key, and nothing else writes one --
    ``use_attribute_docstrings`` is off and
    :func:`test_no_field_states_its_prose_on_field` forbids the ``Field``
    spelling.

    The failure names its own repair through
    :func:`missing_description_repair`, which splits on how the property
    renders: one way to fail this while visibly carrying a ``Declared`` reaches
    only the fields typed as a model. See :data:`REF_SIBLING_REPAIR`.
    """
    model = PARAMS_MODELS[model_name]
    properties = field_schemas(model)
    undescribed = {
        field for field, spec in properties.items() if "description" not in spec
    }
    unlisted = sorted(
        field
        for field in undescribed
        if (model_name, field) not in ALLOWED_MISSING_DESCRIPTION
    )
    undeclared = (
        f"{model_name} declares no description for {unlisted}. "
        + missing_description_repair(properties, unlisted)
    )
    assert not unlisted, undeclared

    listed = {
        field
        for allowed_model, field in ALLOWED_MISSING_DESCRIPTION
        if allowed_model == model_name
    }
    stale = sorted(listed & set(field_schemas(model)) - undescribed)
    described = (
        f"{model_name} now describes {stale}. Remove those entries from "
        f"ALLOWED_MISSING_DESCRIPTION and lower ALLOWLIST_CEILING to match."
    )
    assert not stale, described


@pytest.mark.parametrize("model_name", DECLARED_MODEL_NAMES)
def test_no_field_states_its_prose_on_field(model_name: str) -> None:
    """``Declared`` is the one home for a description, at either depth.

    A ``description=`` on ``Field`` wins over a ``Declared`` in the same
    ``Annotated``, whichever order the two are written in, and the property still
    carries a description -- so the prose guard above cannot see it. A non-None
    ``FieldInfo.description`` is the precise signal.

    A nested config's fields render as controls the same way a top-level
    field's do. A ``Field`` description there loses the same three records.
    """
    model = DECLARED_MODELS[model_name]
    on_field = sorted(
        field
        for field, info in model.model_fields.items()
        if info.description is not None
        and (model_name, field) not in ALLOWED_FIELD_DESCRIPTION
    )
    misplaced = (
        f"{model_name} states prose on Field for {on_field}. Move it into a "
        f"Declared(...) and delete the description= argument."
    )
    assert not on_field, misplaced


@pytest.mark.parametrize("model_name", DECLARED_MODEL_NAMES)
def test_no_field_declares_two_descriptions(model_name: str) -> None:
    """Two ``Declared`` in one ``Annotated``: the last one wins, silently."""
    model = DECLARED_MODELS[model_name]
    doubled = sorted(
        field
        for field in model.model_fields
        if sum(isinstance(m, Declared) for m in metadata_of(model, field)) > 1
    )
    repeated = (
        f"{model_name} declares more than one Declared on {doubled}. Only the "
        f"last takes effect; state the prose once."
    )
    assert not doubled, repeated


@pytest.mark.parametrize("model_name", DECLARED_MODEL_NAMES)
def test_hash_exclude_agrees_between_metadata_and_schema(model_name: str) -> None:
    """The set ``identity_dump()`` strips is the set the schema publishes.

    ``identity_dump()`` reads ``FieldInfo.metadata`` and a client reads
    ``x-mosaic-hash-exclude``. A marker whose hook stopped being called would
    leave the two describing different fields, and only a client would notice.

    A nested config field carries no marker at all, a stronger bar than
    agreement. The marker's hook fires wherever the ``Annotated`` sits, and
    ``x-mosaic-hash-exclude`` does appear under ``$defs``, while
    ``identity_dump()`` pops top-level names and never descends into a nested
    model. The schema then states the field sits outside run identity while the
    hash covers it. A throughput knob belongs on the ``Params`` model, where the
    strip reaches it.
    """
    model = DECLARED_MODELS[model_name]
    from_metadata = {
        field
        for field in model.model_fields
        if any(isinstance(m, HashExclude) for m in metadata_of(model, field))
    }
    from_schema = {
        field
        for field, spec in field_schemas(model).items()
        if spec.get("x-mosaic-hash-exclude") is True
    }
    disagreement = (
        f"{model_name} marks {sorted(from_metadata)} in metadata but publishes "
        f"{sorted(from_schema)} in the schema."
    )
    assert from_metadata == from_schema, disagreement

    if issubclass(model, Params):
        return
    nested = (
        f"{model_name} is not a Params model and marks {sorted(from_metadata)} "
        f"HASH_EXCLUDE. identity_dump() strips top-level names only. The hash "
        f"still covers those fields while the schema says otherwise. Move the "
        f"knob onto the Params model."
    )
    assert not from_metadata, nested


def test_a_placeholder_declares_the_field_and_leaves_the_prose_empty() -> None:
    """The two states the guards above tell apart, pinned on a throwaway model.

    Everything here rests on a field with no ``Declared`` publishing no
    description key at all, while a placeholder publishes an empty one. Were
    pydantic to start emitting ``description`` for an undeclared field, or to
    drop an empty one, the declaration guard would silently stop guarding and
    the prose ceiling would silently stop counting.
    """

    class Sampled(Params):
        declared: Annotated[int, Declared("How many samples to take.")] = 1
        pending: Annotated[int, Declared(NEEDS_DESCRIPTION)] = 2
        bare: int = 3

    schemas = field_schemas(Sampled)
    assert schemas["declared"]["description"] == "How many samples to take."
    assert schemas["pending"]["description"] == NEEDS_DESCRIPTION
    assert "description" not in schemas["bare"], (
        "an undeclared field must publish no key, or the declaration guard "
        "cannot tell it from a placeholder"
    )


def test_the_attributes_section_states_only_what_was_declared() -> None:
    """Three ways the rendered section could state something false.

    Appending to ``cls.__doc__`` rather than ``cls.__dict__``'s entry would copy
    a parent's prose onto a child that declared none. Rendering a field left at
    ``NEEDS_DESCRIPTION`` would publish an attribute entry that says nothing.
    Synthesizing a summary for a class whose author wrote none would put words in
    their mouth, which is the reason ``gen_docs_reference.summary_line`` renders
    an empty cell instead.
    """

    class Documented(Params):
        """A model whose author wrote a summary."""

        stated: Annotated[int, Declared("How many.")] = 1
        owed: Annotated[int, Declared(NEEDS_DESCRIPTION)] = 2

    class Undocumented(Params):
        stated: Annotated[int, Declared("How many.")] = 1

    documented = Documented.__doc__ or ""
    assert documented.startswith("A model whose author wrote a summary.")
    assert "stated: How many." in documented
    assert "owed" not in documented, "a placeholder states nothing, so it is left out"

    undocumented = Undocumented.__dict__.get("__doc__") or ""
    assert undocumented.startswith("Attributes:"), (
        "a class that declared no docstring gets the section and no invented summary"
    )
    inherited = (Params.__doc__ or "").splitlines()[0]
    assert inherited, "read from Params so this cannot go stale against its docstring"
    assert inherited not in undocumented, (
        "the parent's prose must not be copied onto a child that declared none"
    )


def test_the_attributes_section_carries_a_declared_unit() -> None:
    """``x-mosaic-unit`` reaches a schema client; this is the same fact for a reader."""

    class Timed(Params):
        idle_timeout: Annotated[float, Declared("How long to wait.", unit="s")] = 900

    assert "idle_timeout: How long to wait. [s]" in (Timed.__doc__ or "")


def test_the_undocumented_count_only_shrinks() -> None:
    """A placeholder is a debt with a bound, never a resting state.

    ``Declared(NEEDS_DESCRIPTION)`` passes the declaration guard, so without this
    the cheapest way to satisfy that guard would be to declare every field and
    describe none -- a schema full of controls a client can label with nothing.

    Undeclared and placeholder fields are counted together because the bulk
    declaration moves fields from the first state to the second. Counting the
    second alone would make that ceiling climb, which is no ratchet at all.
    """
    undeclared: list[str] = []
    pending: list[str] = []
    for model_name, model in PARAMS_MODELS.items():
        for field, spec in field_schemas(model).items():
            if "description" not in spec:
                undeclared.append(f"{model_name}.{field}")
            elif spec["description"] == NEEDS_DESCRIPTION:
                pending.append(f"{model_name}.{field}")

    total = len(undeclared) + len(pending)
    named = ", ".join(sorted(pending)[:_PENDING_PROSE_NAMED])
    rest = len(pending) - _PENDING_PROSE_NAMED
    grown = (
        f"{len(undeclared)} fields declare no description and {len(pending)} "
        f"declare NEEDS_DESCRIPTION, {total} against a ceiling of "
        f"{UNDOCUMENTED_CEILING}. Placeholders: {named}"
        f"{f' and {rest} more' if rest > 0 else ''}. Write prose and lower "
        f"UNDOCUMENTED_CEILING to match."
    )
    assert total <= UNDOCUMENTED_CEILING, grown


def test_the_allowlist_only_shrinks() -> None:
    """Nothing may join either allowlist, and nothing may rot on either.

    A ceiling stops one added line from admitting what the guards above refuse:
    an undescribed field, or a field whose prose sits on ``Field``. Resolving
    each entry stops a renamed or deregistered field from sitting there forever,
    which the per-model staleness check cannot see: it intersects with the fields
    a model still has.
    """
    grown = (
        f"ALLOWED_MISSING_DESCRIPTION holds {len(ALLOWED_MISSING_DESCRIPTION)} "
        f"entries against a ceiling of {ALLOWLIST_CEILING}."
    )
    assert len(ALLOWED_MISSING_DESCRIPTION) <= ALLOWLIST_CEILING, grown

    spelled = (
        f"ALLOWED_FIELD_DESCRIPTION holds {len(ALLOWED_FIELD_DESCRIPTION)} "
        f"entries against a ceiling of {FIELD_DESCRIPTION_CEILING}. Move the "
        f"prose into a Declared(...) beside the Field instead of listing the "
        f"field here."
    )
    assert len(ALLOWED_FIELD_DESCRIPTION) <= FIELD_DESCRIPTION_CEILING, spelled

    allowed = ALLOWED_MISSING_DESCRIPTION | ALLOWED_FIELD_DESCRIPTION
    unknown_models = sorted({model for model, _ in allowed} - set(ALLOWLISTED_MODELS))
    vanished = (
        f"allowlisted models {unknown_models} resolve to no declaring model. "
        f"Delete their entries."
    )
    assert not unknown_models, vanished

    unknown_fields = sorted(
        f"{model}.{field}"
        for model, field in allowed
        if field not in ALLOWLISTED_MODELS[model].model_fields
    )
    renamed = (
        f"allowlisted fields {unknown_fields} no longer exist. Delete their "
        f"entries and lower the ceiling that bounds them to match."
    )
    assert not unknown_fields, renamed


def base_class_name(node: ast.expr) -> str:
    """Return the name a base-class expression ends in, or empty for anything else.

    A subscript is unwrapped so that ``GlobalModelParams[KMeansModelArtifact]``
    reports the generic the class inherits from.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return base_class_name(node.value)
    return ""


def class_declarations() -> list[tuple[str, str, tuple[str, ...]]]:
    """Every class under ``src/mosaic``, as ``(module, class, base names)``.

    Two trees are skipped, matching the five sibling structural walks. A
    virtualenv under the package root holds installed third-party code, where
    ``class Foo(RequestParams)`` is common and a non-UTF-8 source file raises
    instead of failing by name. The external-environment trees hold programs run
    under interpreters mosaic builds nothing of, whose parameter payloads are
    never published as a mosaic schema.
    """
    found: list[tuple[str, str, tuple[str, ...]]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if inside_a_virtualenv(path, PACKAGE_ROOT):
            continue
        if runs_in_an_external_environment(path, PACKAGE_ROOT):
            continue
        relative = path.relative_to(PACKAGE_ROOT.parent).with_suffix("")
        module = ".".join(relative.parts)
        for node in ast.walk(source_tree(path)):
            if isinstance(node, ast.ClassDef):
                bases = tuple(base_class_name(base) for base in node.bases)
                found.append((module, node.name, bases))
    return found


def modules_declaring_params() -> dict[str, tuple[str, ...]]:
    """Every ``src/mosaic`` module declaring a ``Params`` subclass.

    Read out of the source rather than off ``__subclasses__()``, which is the
    thing under test: a class becomes a subclass only once its module has been
    imported.

    The base chain is resolved to a fixed point rather than one level deep. The
    ``*Params`` suffix seeds the set. Every class it matches then counts as a
    base itself, which reaches a model inheriting through an intermediate whose
    name breaks the convention. Nine shipped models are that shape already --
    three ``*InferParams`` deriving from ``_InferParamsBase``, six tracker
    backends from ``_BackendConfig``. Both intermediates would carry their
    subclasses out of this walk the day either moves to a module of its own.

    A base is matched by name, which is what a walk over source can read. The
    error that leaves is a module listed because it reuses a name. A listed
    module has to be reached or exempted rather than dropped.
    """
    records = class_declarations()
    matched: set[tuple[str, str]] = set()
    known: set[str] = set()
    while True:
        found = {
            (module, name)
            for module, name, bases in records
            if any(base.endswith("Params") or base in known for base in bases)
        }
        if found == matched:
            break
        matched = found
        known = {name for _, name in matched}

    declared: dict[str, tuple[str, ...]] = {}
    for module, name, _ in records:
        if (module, name) in matched:
            declared[module] = declared.get(module, ()) + (name,)
    return declared


def test_the_import_list_reaches_every_declared_params_model() -> None:
    """A model in an unimported module is guarded by nothing, and reported by nothing.

    Every guard in this file is parametrized over ``PARAMS_MODELS``, which walks
    ``Params.__subclasses__()`` and therefore sees only what the registries and
    ``UNREGISTERED_PARAMS_MODULES`` have imported. A new model in a module
    neither reaches passes all of them by being absent from all of them. This
    reads the source instead and names the module the day it is added.
    """
    declared = modules_declaring_params()
    canary = (
        f"the source walk under {PACKAGE_ROOT} reached no module declaring a "
        f"Params subclass. It is agreeing with itself over an empty set."
    )
    assert "mosaic.behavior.feature_library.speed_angvel" in declared, canary

    exempt = {module for module, _ in PARAMS_MODULE_EXEMPTIONS}
    stale = sorted(exempt - set(declared))
    gone = (
        f"exempted modules {stale} declare no Params subclass any more. Delete "
        f"their entries from PARAMS_MODULE_EXEMPTIONS."
    )
    assert not stale, gone

    reached = {model.__module__ for model in PARAMS_MODELS.values()}
    unreached = sorted(set(declared) - reached - exempt)
    named = {module: declared[module] for module in unreached}
    missing = (
        f"{named} declare a Params subclass PARAMS_MODELS never sees. Register "
        f"the model, add its module to UNREGISTERED_PARAMS_MODULES, or name it "
        f"in PARAMS_MODULE_EXEMPTIONS with the reason its fields reach no "
        f"client."
    )
    assert not unreached, missing


class DeclarationSample(Params):
    """A field of each declared kind, for reading the emitted schema back."""

    speed: Annotated[
        float | None,
        Field(gt=0.0),
        Declared("the maximum plausible speed", unit="cm/s"),
    ] = None
    workers: Annotated[int, HASH_EXCLUDE, Declared("how many workers run")] = 1
    inert: Annotated[
        str, Declared("which accelerator runs it", unwired="nothing reads it")
    ] = "auto"


def test_every_declared_fact_appears_in_the_schema() -> None:
    """Read the four keys a client reads, straight out of the schema.

    This is what fails when pydantic stops calling a marker's
    ``__get_pydantic_json_schema__``.
    """
    properties = field_schemas(DeclarationSample)

    assert properties["speed"]["description"] == "the maximum plausible speed"
    assert properties["speed"]["x-mosaic-unit"] == "cm/s"
    assert "x-mosaic-hash-exclude" not in properties["speed"]
    assert "x-mosaic-unwired" not in properties["speed"]

    assert properties["workers"]["description"] == "how many workers run"
    assert properties["workers"]["x-mosaic-hash-exclude"] is True
    assert "x-mosaic-unit" not in properties["workers"]

    assert properties["inert"]["description"] == "which accelerator runs it"
    assert properties["inert"]["x-mosaic-unwired"] == "nothing reads it"


def test_every_model_carrying_a_declared_subclasses_declared_model() -> None:
    """``DeclaredModel`` is what reading a ``Declared`` is opted into by.

    Nothing reads a ``Declared`` off a plain ``StrictModel``: the rendered
    ``Attributes:`` section and the inherited-description pass both live on
    ``DeclaredModel``'s hooks. A model that declares its fields and then picks
    the wrong base publishes them on its properties -- ``Declared`` writes
    those itself -- and silently gets no section at all. That is how
    ``FeralTrainingConfig`` came to declare twenty fields and print no section
    for any of them.

    The walk is over ``StrictModel``'s subclasses rather than over
    ``DECLARED_MODELS``, because a model that picked the wrong base is by
    definition not in the latter.
    """
    offenders: list[str] = []
    for name, model in sorted(subclasses_in_the_tree(StrictModel).items()):
        if issubclass(model, DeclaredModel):
            continue
        declared = sorted(
            field
            for field, info in model.model_fields.items()
            if own_declaration(info) is not None
        )
        if declared:
            offenders.append(f"{name} declares {declared}")
    assert not offenders, (
        "; ".join(offenders)
        + ". Each subclasses StrictModel, so its declared prose reaches no "
        "Attributes: section and no inherited-description pass. Parent it on "
        "DeclaredModel."
    )


@pytest.mark.parametrize("model_name", NON_PARAMS_DECLARED_NAMES)
def test_every_non_params_field_declares_a_description(model_name: str) -> None:
    """The same bar the parameter fields meet, with no allowlist behind it.

    These were declared after the top-level sweep reached zero, so this starts
    with nothing owed and there is no ceiling to lower. A new undescribed field
    fails here the day it is added, and a placeholder is refused rather than
    counted -- :data:`UNDOCUMENTED_CEILING` does not reach this tier.

    Reached by subclass rather than by following annotations out from a
    ``Params`` field, which never arrives at a model nothing annotates.
    ``FeatureLabelsSource`` is one: it is exported beside
    ``GroundTruthLabelsSource`` and shipped a blank ``source`` control next to
    that model's labelled one.
    """
    model = NON_PARAMS_DECLARED_MODELS[model_name]
    properties = field_schemas(model)
    undescribed = sorted(
        field for field, spec in properties.items() if not spec.get("description")
    )
    assert not undescribed, (
        f"{model_name} declares no description for {undescribed}. "
        "A client drawing this nested form renders those controls unlabelled. "
        + missing_description_repair(properties, undescribed)
    )


@pytest.mark.parametrize("model_name", DECLARED_MODEL_NAMES)
def test_no_model_publishes_a_field_list_in_its_description(model_name: str) -> None:
    """The generated section reaches ``help()`` and stops there.

    A client renders a model as a group of controls, with the model's own
    description above them and each field's description on its control.
    Publishing an ``Attributes:`` section as that group description prints
    every child twice: once in the header and once on the control.

    Parametrized over every ``DeclaredModel`` subclass rather than over what a
    ``Params`` field annotates, because the one model that reached this state,
    ``ResultColumn``, is annotated by nothing. ``DeclaredModel`` trims a
    section it generated; a block written by hand sits inside the prose it
    preserves, and only deleting that block helps.
    """
    model = DECLARED_MODELS[model_name]
    description = str(model.model_json_schema().get("description", ""))
    assert "Attributes:" not in description, (
        f"{model_name} publishes its field list inside its schema description. "
        f"Delete the hand-written Attributes: block; DeclaredModel renders one "
        f"for help() from the fields' Declared entries."
    )


@pytest.mark.parametrize("model_name", DECLARED_MODEL_NAMES)
def test_no_description_states_the_unit_it_already_declares(model_name: str) -> None:
    """``Declared``'s two halves must not say the same thing twice.

    ``core/params.py`` contracts the description as stating the quantity
    *without* its unit, and both readers append the unit themselves --
    ``scripts/gen_docs_reference.py`` renders ``description [unit]`` and the
    generated ``Attributes:`` section does the same. A description naming its
    own unit therefore ships it twice: "The number of epochs to train for.
    [epochs]".

    Written as a test rather than left to convention because eight of these
    reached a commit while the rule was a brief clause with a worked example
    beside it. A unit is a short token, so the match is on a word boundary:
    "framerate" and "seconds" do not trip "frames" or "s".

    A nested config renders the same ``description [unit]`` cell and meets the
    same bar under the same match.
    """
    model = DECLARED_MODELS[model_name]
    offenders: list[str] = []
    for field, spec in field_schemas(model).items():
        unit = str(spec.get("x-mosaic-unit", "") or "").strip()
        description = str(spec.get("description", "") or "")
        if not unit or not description:
            continue
        for token in re.split(r"[^a-z]+", unit.lower()):
            if len(token) < 2:
                continue
            # A unit named inside a formula is informative, not a restatement:
            # "derives dt as frame_diff / fps" earns the word.
            for match in re.finditer(rf"\b{re.escape(token)}\b", description.lower()):
                before = description.lower()[: match.start()].rstrip()
                after = description.lower()[match.end() :]
                # A token naming a column is not the unit restated.
                if before.endswith("/") or before.endswith("per"):
                    continue
                if after.startswith(" column"):
                    continue
                offenders.append(f"{field} says {token!r} and declares unit={unit!r}")
                break
    assert not offenders, (
        f"{model_name}: "
        + "; ".join(offenders)
        + ". State the quantity and let unit= supply the unit."
    )


def test_the_schema_description_is_the_prose_without_the_field_list() -> None:
    """The two halves of the split, pinned against one model at once.

    pydantic copies ``__doc__`` into the schema whole, and the ``Attributes:``
    section is appended to ``__doc__``. Published as-is, every description
    appears twice -- once under ``properties``, where a client reads it, and
    once inside one long newline-laden string a client has to parse back apart.
    ``mosaic tracking describe`` printed both.

    The negative half is held over the whole tree by
    :func:`test_no_model_publishes_a_field_list_in_its_description`. What is
    here and nowhere else is the pair. The exact description says the prose is
    published whole and *alone*, which is what rules out publishing nothing at
    all -- the reading that a rendered docstring carrying the section would
    otherwise be perfectly happy with.
    """
    schema = DeclarationSample.model_json_schema()

    assert (
        schema["description"]
        == "A field of each declared kind, for reading the emitted schema back."
    )

    rendered = DeclarationSample.__doc__ or ""
    assert "Attributes:" in rendered
    assert "the maximum plausible speed" in rendered


def test_a_model_with_no_prose_publishes_no_description() -> None:
    """A model that declares no docstring does not inherit its parent's.

    The same rule ``__doc__`` follows: prose is never invented for a class that
    wrote none, and publishing the base's would describe the wrong model.
    """

    class Undocumented(Params):
        count: Annotated[int, Declared("how many")] = 1

    schema = Undocumented.model_json_schema()

    assert "description" not in schema
    assert schema["properties"]["count"]["description"] == "how many"


def test_a_re_annotated_field_keeps_its_parents_declaration() -> None:
    """Narrowing a field or pinning its default does not strip its prose.

    Re-annotating a field replaces its whole ``Annotated``, so the ``Declared``
    goes with it -- which is how twenty-two of the twenty-three
    ``ArtifactSpec`` subclasses came to publish a ``load`` and a ``pattern``
    with no description while ``ArtifactSpec`` declared both. Over the whole
    ``Result`` tree, the wider denominator, twenty-six models lost at least one
    declaration this way.

    :func:`mosaic.core.params._declared_for` reads the nearest class in the
    method resolution order that still holds one, and both consumers go through
    it: the property a client reads, and the section ``help()`` prints.
    """

    class Parent(DeclaredModel):
        """Parent prose."""

        kept: Annotated[
            str, Declared("what kept means", unit="px", unwired="nothing reads it")
        ] = "a"

    class Child(Parent):
        """Child prose."""

        kept: str = "b"

    field = Child.model_json_schema()["properties"]["kept"]

    assert field["description"] == "what kept means"
    # Every key ``Declared`` publishes is inherited, not the description alone.
    # No shipped field inherits a declaration carrying a unit or an unwired
    # record, so dropping either from the pass is green without these two.
    assert field["x-mosaic-unit"] == "px"
    assert field["x-mosaic-unwired"] == "nothing reads it"
    assert "kept: what kept means [px]" in (Child.__doc__ or "")


def test_a_re_annotated_field_may_restate_its_own_prose() -> None:
    """A subclass whose meaning changed declares again, and its own wins.

    The lookup stops at the first class holding a ``Declared``, so a subclass
    that narrows what a field means is not stuck with the description of the
    wider one it replaced.
    """

    class Parent(DeclaredModel):
        """Parent prose."""

        kept: Annotated[str, Declared("the wide meaning")] = "a"

    class Child(Parent):
        """Child prose."""

        kept: Annotated[str, Declared("the narrow meaning")] = "b"

    field = Child.model_json_schema()["properties"]["kept"]
    rendered = Child.__doc__ or ""

    assert field["description"] == "the narrow meaning"
    # The property is written by ``Declared``'s own hook, which the lookup
    # never reaches, so the rendered section is the only consumer that reads
    # the search order. Without this line a lookup walking the bases outward is
    # green.
    assert "kept: the narrow meaning" in rendered
    assert "the wide meaning" not in rendered


def test_an_unwired_field_still_reaches_the_run_identity() -> None:
    """``unwired`` records what a field fails to reach, and changes nothing else.

    A field nothing reads is still a field a caller can set, so it keeps its
    place in ``model_dump`` and in the identity hash unless it also carries
    ``HASH_EXCLUDE``. Publishing the record must not quietly move an identifier
    and invalidate every cached run that named one.
    """
    assert "inert" in DeclarationSample().identity_dump()


def test_the_sample_model_stays_out_of_the_walk() -> None:
    """A ``Params`` subclass declared by a test is a fixture, not a shipped model.

    Test modules throughout the suite declare one. Were the walk to pick them
    up, this file's own sample would be the first to break the guard it
    supports.
    """
    assert DeclarationSample.__qualname__ not in PARAMS_MODELS


def test_the_constraint_stays_inside_the_optional_branch() -> None:
    """``Declared`` writes at the field level; a constraint stays in its branch.

    A client unwrapping the non-null branch of an optional field finds the
    constraint there and must read the description from the property itself.
    """
    speed = field_schemas(DeclarationSample)["speed"]
    branches = speed["anyOf"]
    assert branches == [{"exclusiveMinimum": 0.0, "type": "number"}, {"type": "null"}]
