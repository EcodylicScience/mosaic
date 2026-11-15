"""The definition scope: which definitions a slug names, and the class it gives."""

from __future__ import annotations

import hashlib
import sys
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest
from pydantic import BaseModel

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.definitions.loader import DEFINITION_MARK
from mosaic.core.pipeline.definitions.scope import (
    DefinitionScope,
    clear_definition_caches,
    current_scope,
    definition_scope,
    resolve_definition_slug,
)
from mosaic.core.pipeline.definitions.sidecar import DefinitionSidecar, read_sidecar
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionMismatch,
    DefinitionNotLoaded,
    FeatureDefinitionsNotAllowed,
    list_versions,
)
from mosaic.core.pipeline.graph.compatibility import TRACKS_DECLARATION
from mosaic.core.pipeline.graph.reach import feature_reach
from mosaic.core.pipeline.graph.resolve import (
    build_feature,
    declared_fit,
    declared_version,
    feature_class_for_slug,
    feature_parallelizable,
    run_reference_fields,
)
from tests.helpers import (
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    defined_dataset,
    run_fresh,
)

SCALED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
)
TRIPLED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 3.0"
)

TRAITS_NAME: Final = "trait-distance"
TRAITS_SOURCE: Final = (
    MINIMAL_SOURCE.replace(f'name = "{MINIMAL_NAME}"', f'name = "{TRAITS_NAME}"')
    .replace(
        "    consumed_roots = ()\n", "    consumed_roots = ()\n    temporal_reach = 3\n"
    )
    .replace("scale: float = 1.0", 'scale: float = 1.0\n        label: str = "a"')
)

NARROWING_NAME: Final = "narrow-distance"
NARROWING_SOURCE: Final = (
    MINIMAL_SOURCE.replace(f'name = "{MINIMAL_NAME}"', f'name = "{NARROWING_NAME}"')
    .replace("scale: float = 1.0", "scale: float = 1.0\n        wide: bool = False")
    .replace(
        "        self.params = self.Params.from_overrides(params)\n",
        "        self.params = self.Params.from_overrides(params)\n"
        "        if self.params.wide:\n"
        "            self.scope_dependent = True\n",
    )
)

SLUG: Final = f"def:{MINIMAL_NAME}"


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A dataset whose ``origin-distance`` has two versions, the scaled one current."""
    defined = defined_dataset(
        tmp_path_factory.mktemp("scope"),
        {
            MINIMAL_NAME: MINIMAL_SOURCE,
            GLOBAL_NAME: GLOBAL_SOURCE,
            TRAITS_NAME: TRAITS_SOURCE,
            NARROWING_NAME: NARROWING_SOURCE,
        },
    )
    _ = define_feature(defined, MINIMAL_NAME, SCALED_SOURCE)
    return defined


@pytest.fixture(scope="module")
def other_dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """Another dataset, whose ``origin-distance`` is a third source."""
    return defined_dataset(
        tmp_path_factory.mktemp("other"), {MINIMAL_NAME: TRIPLED_SOURCE}
    )


@pytest.fixture(autouse=True)
def forget_loaded_definitions() -> Iterator[None]:
    """Remove the definition modules a test imported, so each test loads afresh."""
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        if name.startswith("mosaic_definitions."):
            _ = sys.modules.pop(name)


def _sidecar(dataset: Dataset, name: str, digest: str) -> DefinitionSidecar:
    directory = dataset.get_root(DEFINITIONS_ROOT) / name
    (version,) = [each for each in list_versions(directory) if each.digest == digest]
    return read_sidecar(directory / f"{version.stem}.json")


def _scale_default(cls: type | None) -> object:
    params: object = getattr(cls, "Params", None)
    assert isinstance(params, type) and issubclass(params, BaseModel)
    return params.model_fields["scale"].default


# --- the class a slug names --------------------------------------------------


def test_no_scope_names_no_definition(dataset: Dataset) -> None:
    assert current_scope() is None
    assert feature_class_for_slug(SLUG) is None
    assert feature_class_for_slug("def:no-such-name") is None


def test_a_read_scope_gives_the_declared_class(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        cls = feature_class_for_slug(SLUG)
        feature = build_feature(SLUG, ["tracks"], None)
    assert cls is not None
    assert getattr(cls, "name") == SLUG
    assert _scale_default(cls) == 2.0
    with pytest.raises(DefinitionNotLoaded) as raised:
        feature.load_state(Path(), {}, {})
    assert raised.value.method == "load_state"
    assert not [name for name in sys.modules if name.startswith("mosaic_definitions.")]


def test_a_read_scope_imports_no_definition_and_no_declaring_module(
    dataset: Dataset,
) -> None:
    code = (
        "import sys\n"
        "from mosaic.core.dataset import open_dataset\n"
        "from mosaic.core.pipeline.definitions.scope import definition_scope\n"
        "from mosaic.core.pipeline.graph.resolve import feature_class_for_slug\n"
        "dataset = open_dataset(sys.argv[1], ensure_roots=False)\n"
        "with definition_scope(dataset, load=False):\n"
        "    assert feature_class_for_slug(sys.argv[2]) is not None\n"
        "print(sorted(name for name in sys.modules if name.startswith(\n"
        "    ('mosaic_definitions', 'mosaic.core.pipeline.definitions.declare'))))\n"
    )
    assert run_fresh(code, args=[str(dataset.manifest_path), SLUG]) == "[]"


def test_a_load_scope_gives_the_loaded_class(dataset: Dataset) -> None:
    with definition_scope(dataset, load=True):
        cls = feature_class_for_slug(SLUG)
    assert cls is load_feature_definition(dataset, SLUG)
    assert DEFINITION_MARK in vars(cls)


def test_the_readers_answer_for_a_declared_class_from_its_sidecar(
    dataset: Dataset,
) -> None:
    fitting = replace(TRACKS_DECLARATION, fits=True)
    with definition_scope(dataset, load=False):
        assert declared_version("feature", SLUG) == "1"
        assert feature_parallelizable(SLUG)
        assert not feature_parallelizable(f"def:{GLOBAL_NAME}")
        assert declared_fit(f"def:{GLOBAL_NAME}", {}, fitting) == (True, True)
        assert declared_fit(SLUG, {}, fitting) == (True, False)
        assert run_reference_fields(f"def:{TRAITS_NAME}") == frozenset({"label"})
        assert feature_reach(build_feature(f"def:{TRAITS_NAME}", None, None)) == 3


# --- resolving a slug -----------------------------------------------------------


def test_a_bare_slug_resolves_to_current_or_to_its_pin(dataset: Dataset) -> None:
    first, scaled = _digest(MINIMAL_SOURCE), _digest(SCALED_SOURCE)
    with definition_scope(dataset, load=False):
        assert resolve_definition_slug(SLUG) == (MINIMAL_NAME, scaled)
        assert resolve_definition_slug(f"{SLUG}@{first[:12]}") == (MINIMAL_NAME, first)
    with definition_scope(dataset, load=False, pins={MINIMAL_NAME: first}):
        assert resolve_definition_slug(SLUG) == (MINIMAL_NAME, first)
        assert _scale_default(feature_class_for_slug(SLUG)) == 1.0
        assert resolve_definition_slug(f"{SLUG}@{scaled}") == (MINIMAL_NAME, scaled)


# --- nesting ------------------------------------------------------------------


def test_an_inner_scope_of_the_same_dataset_adds_only_new_pins(
    dataset: Dataset,
) -> None:
    first, scaled = _digest(MINIMAL_SOURCE), _digest(SCALED_SOURCE)
    global_digest = _digest(GLOBAL_SOURCE)
    with definition_scope(dataset, load=False, pins={MINIMAL_NAME: first}):
        outer = current_scope()
        with definition_scope(
            dataset.base_dir,
            load=False,
            pins={MINIMAL_NAME: scaled, GLOBAL_NAME: global_digest},
        ):
            inner = current_scope()
            assert inner is not None
            assert dict(inner.pins) == {
                MINIMAL_NAME: first,
                GLOBAL_NAME: global_digest,
            }
        with definition_scope(dataset, load=False):
            assert current_scope() is outer
        assert current_scope() is outer


def test_a_scope_without_a_dataset_is_filled_and_its_sidecars_come_first(
    dataset: Dataset,
) -> None:
    scaled = _digest(SCALED_SOURCE)
    renamed = _sidecar(dataset, MINIMAL_NAME, scaled).model_copy(
        update={"version": "9"}
    )
    with definition_scope(None, load=False, sidecars={scaled: renamed}):
        assert feature_class_for_slug(f"def:{GLOBAL_NAME}") is None
        with definition_scope(dataset, load=False):
            filled = current_scope()
            assert filled is not None
            assert filled.base_dir == dataset.base_dir
            assert declared_version("feature", SLUG) == "9"
            assert declared_version("feature", f"def:{GLOBAL_NAME}") == "1"
        restored = current_scope()
        assert restored is not None
        assert restored.base_dir is None


def test_a_scope_of_another_dataset_is_replaced_and_restored(
    dataset: Dataset, other_dataset: Dataset
) -> None:
    with definition_scope(dataset, load=False):
        outer = current_scope()
        with definition_scope(other_dataset, load=False):
            assert _scale_default(feature_class_for_slug(SLUG)) == 3.0
            assert feature_class_for_slug(f"def:{GLOBAL_NAME}") is None
        assert current_scope() is outer
        assert _scale_default(feature_class_for_slug(SLUG)) == 2.0


def test_a_load_entry_under_a_read_scope_loads(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        with definition_scope(dataset, load=True):
            loaded = feature_class_for_slug(SLUG)
            assert loaded is not None
            assert DEFINITION_MARK in vars(loaded)
        scope = current_scope()
        assert scope is not None
        assert not scope.load


# --- threads ------------------------------------------------------------------


def test_threads_keep_their_own_scopes(
    dataset: Dataset, other_dataset: Dataset
) -> None:
    seen: dict[str, object] = {}
    barrier = threading.Barrier(2)

    def read(key: str, source: Dataset) -> None:
        with definition_scope(source, load=False):
            _ = barrier.wait(timeout=30)
            seen[key] = _scale_default(feature_class_for_slug(SLUG))
            seen[f"{key}-global"] = feature_class_for_slug(f"def:{GLOBAL_NAME}")

    threads = [
        threading.Thread(target=read, args=("first", dataset)),
        threading.Thread(target=read, args=("other", other_dataset)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert seen["first"] == 2.0
    assert seen["other"] == 3.0
    assert seen["first-global"] is not None
    assert seen["other-global"] is None


def test_a_thread_started_under_a_scope_starts_with_none(dataset: Dataset) -> None:
    seen: list[DefinitionScope | None] = []
    with definition_scope(dataset, load=False):
        thread = threading.Thread(target=lambda: seen.append(current_scope()))
        thread.start()
        thread.join(timeout=60)
    assert seen == [None]


# --- the identity check ---------------------------------------------------------


def test_a_flag_narrowed_by_a_param_is_a_mismatch_when_loaded(
    dataset: Dataset,
) -> None:
    slug = f"def:{NARROWING_NAME}"
    with definition_scope(dataset, load=False):
        declared = build_feature(slug, ["tracks"], {"wide": True})
    assert not getattr(declared, "scope_dependent")
    with definition_scope(dataset, load=True):
        _ = build_feature(slug, ["tracks"], None)
        with pytest.raises(DefinitionMismatch) as raised:
            _ = build_feature(slug, ["tracks"], {"wide": True})
    assert raised.value.fields == ("scope_dependent",)
    assert raised.value.error_reason() == "definition_mismatch"


# --- caches ---------------------------------------------------------------------


def test_clearing_the_caches_rebuilds_the_declared_class(dataset: Dataset) -> None:
    with definition_scope(dataset, load=False):
        first = feature_class_for_slug(SLUG)
        assert feature_class_for_slug(SLUG) is first
        clear_definition_caches()
        rebuilt = feature_class_for_slug(SLUG)
    assert rebuilt is not None
    assert rebuilt is not first


# --- a disabled scope -----------------------------------------------------------


def test_a_disabled_scope_refuses_a_definition_and_is_never_widened(
    dataset: Dataset, other_dataset: Dataset
) -> None:
    with definition_scope(None, load=False, allowed=False):
        disabled = current_scope()
        with pytest.raises(FeatureDefinitionsNotAllowed):
            _ = feature_class_for_slug(SLUG)
        for entry in (dataset, other_dataset, dataset.base_dir):
            with definition_scope(
                entry, load=True, pins={MINIMAL_NAME: _digest(MINIMAL_SOURCE)}
            ):
                assert current_scope() is disabled
                with pytest.raises(FeatureDefinitionsNotAllowed):
                    _ = resolve_definition_slug(SLUG)
        assert current_scope() is disabled
    assert current_scope() is None


def test_an_inner_disabled_entry_disables_the_scope_it_enters(
    dataset: Dataset, other_dataset: Dataset
) -> None:
    with definition_scope(dataset, load=False):
        for entry in (dataset, other_dataset):
            with definition_scope(entry, load=False, allowed=False):
                with pytest.raises(FeatureDefinitionsNotAllowed):
                    _ = resolve_definition_slug(SLUG)
        assert resolve_definition_slug(SLUG)[0] == MINIMAL_NAME
