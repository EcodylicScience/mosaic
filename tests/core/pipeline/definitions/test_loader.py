"""Loading a stored definition: the module name, the class's marks and its checks."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.define import (
    define_feature,
    load_feature_definition,
)
from mosaic.core.pipeline.definitions.loader import (
    applied_definition_model,
    load_definition,
    narrow_to_apply,
    register_definitions_root,
)
from mosaic.core.pipeline.definitions.sidecar import DefinitionRef
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionCorrupt,
    DefinitionInvalid,
    DefinitionMismatch,
    definition_module_name,
    list_versions,
)
from mosaic.core.pipeline.types import DefinitionModel, Feature
from tests.helpers import (
    APPLYING_NAME,
    APPLYING_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    PlainFeature,
    built_by_class,
    defined_dataset,
)

SCALED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
)
TRIPLED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 3.0"
)
VERSION_VARIABLE: Final = "MOSAIC_TEST_DEFINITION_VERSION"
OS_SOURCE: Final = MINIMAL_SOURCE.replace(
    "from pathlib import Path\n", "import os\nfrom pathlib import Path\n", 1
)
ENVIRONMENT_SOURCE: Final = OS_SOURCE.replace(
    'version = "1"', f'version = os.environ.get("{VERSION_VARIABLE}", "1")'
)

REGISTER_VARIABLE: Final = "MOSAIC_TEST_DEFINITION_REGISTER"
STRAY_FEATURE: Final = "StrayDistance"
REGISTERING_SOURCE: Final = (
    OS_SOURCE
    + f"""

if os.environ.get("{REGISTER_VARIABLE}"):
    from mosaic.behavior.feature_library.registry import FEATURES

    FEATURES.setdefault("{STRAY_FEATURE}", OriginDistance)
"""
)
REACH_SOURCE: Final = MINIMAL_SOURCE.replace(
    "    consumed_roots = ()\n",
    "    consumed_roots = ()\n    temporal_reach = None\n",
    1,
)


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _root(dataset: Dataset) -> Path:
    return dataset.get_root(DEFINITIONS_ROOT)


def _version_file(dataset: Dataset, name: str, source: str, suffix: str) -> Path:
    directory = _root(dataset) / name
    (version,) = [
        version
        for version in list_versions(directory)
        if version.digest == _digest(source)
    ]
    return directory / f"{version.stem}.{suffix}"


def _edit_sidecar(dataset: Dataset, source: str, record: str, key: str) -> Path:
    """Remove *key* from the *record* of the stored sidecar of *source*."""
    path = _version_file(dataset, MINIMAL_NAME, source, "json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored[record][key]
    _ = path.write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")
    return path


def _scale_default(cls: type) -> object:
    params: object = getattr(cls, "Params")
    assert isinstance(params, type)
    return getattr(params(), "scale")


def test_a_defined_source_loads_as_a_marked_class(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    digest = _digest(MINIMAL_SOURCE)

    cls = load_definition(_root(dataset), MINIMAL_NAME, digest)

    assert getattr(cls, "name") == f"def:{MINIMAL_NAME}"
    assert getattr(cls, "__mosaic_definition__") == DefinitionRef(MINIMAL_NAME, digest)
    assert load_definition(_root(dataset), MINIMAL_NAME, digest) is cls
    module = importlib.import_module(definition_module_name(MINIMAL_NAME, digest))
    assert getattr(module, "OriginDistance") is cls


def test_versions_and_datasets_load_side_by_side(tmp_path: Path) -> None:
    first = defined_dataset(tmp_path / "first", {MINIMAL_NAME: MINIMAL_SOURCE})
    _ = define_feature(first, MINIMAL_NAME, SCALED_SOURCE)
    second = defined_dataset(tmp_path / "second", {MINIMAL_NAME: TRIPLED_SOURCE})

    loaded = [
        load_definition(_root(dataset), MINIMAL_NAME, _digest(source))
        for dataset, source in (
            (first, MINIMAL_SOURCE),
            (first, SCALED_SOURCE),
            (second, TRIPLED_SOURCE),
        )
    ]

    assert [_scale_default(cls) for cls in loaded] == [1.0, 2.0, 3.0]
    assert len({id(cls) for cls in loaded}) == 3


def test_a_sidecar_altered_on_disk_is_a_mismatch(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    path = _version_file(dataset, MINIMAL_NAME, MINIMAL_SOURCE, "json")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    recorded["flags"]["scope_dependent"] = True
    _ = path.write_text(json.dumps(recorded, sort_keys=True), encoding="utf-8")

    with pytest.raises(DefinitionMismatch) as caught:
        _ = load_definition(_root(dataset), MINIMAL_NAME, _digest(MINIMAL_SOURCE))

    assert caught.value.fields == ("flags.scope_dependent",)
    assert caught.value.error_reason() == "definition_mismatch"


def test_a_class_read_from_the_environment_is_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(VERSION_VARIABLE, "1")
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: ENVIRONMENT_SOURCE})
    monkeypatch.setenv(VERSION_VARIABLE, "2")

    with pytest.raises(DefinitionMismatch) as caught:
        _ = load_definition(_root(dataset), MINIMAL_NAME, _digest(ENVIRONMENT_SOURCE))

    assert caught.value.fields == ("version",)


def test_a_source_edited_on_disk_is_corrupt(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    path = _version_file(dataset, MINIMAL_NAME, MINIMAL_SOURCE, "py")
    _ = path.write_text(MINIMAL_SOURCE + "# edited\n", encoding="utf-8")

    with pytest.raises(DefinitionCorrupt) as caught:
        _ = load_definition(_root(dataset), MINIMAL_NAME, _digest(MINIMAL_SOURCE))

    assert caught.value.path == path
    assert caught.value.error_reason() == "definition_corrupt"


@pytest.mark.parametrize(
    "module",
    [
        f"mosaic_definitions.origin_distance__{'0' * 64}",
        "mosaic_definitions.os",
        f"mosaic_definitions.origin_distance__{'0' * 64}.inner",
    ],
)
def test_a_module_no_registered_directory_holds_is_refused(
    tmp_path: Path, module: str
) -> None:
    register_definitions_root(tmp_path)

    with pytest.raises(ModuleNotFoundError, match="no feature definition"):
        _ = importlib.import_module(module)


def test_a_slug_loads_its_version(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    _ = define_feature(dataset, MINIMAL_NAME, SCALED_SOURCE)
    first = _digest(MINIMAL_SOURCE)

    pinned = load_feature_definition(dataset, f"def:{MINIMAL_NAME}@{first[:12]}")
    current = load_feature_definition(dataset, f"def:{MINIMAL_NAME}")

    assert getattr(pinned, "__mosaic_definition__") == DefinitionRef(
        MINIMAL_NAME, first
    )
    assert _scale_default(current) == 2.0


def test_a_load_reads_the_dataset_it_names(tmp_path: Path) -> None:
    edited = defined_dataset(tmp_path / "edited", {MINIMAL_NAME: MINIMAL_SOURCE})
    intact = defined_dataset(tmp_path / "intact", {MINIMAL_NAME: MINIMAL_SOURCE})
    path = _version_file(edited, MINIMAL_NAME, MINIMAL_SOURCE, "py")
    _ = path.write_text(MINIMAL_SOURCE + "# edited\n", encoding="utf-8")
    digest = _digest(MINIMAL_SOURCE)

    with pytest.raises(DefinitionCorrupt):
        _ = load_definition(_root(edited), MINIMAL_NAME, digest)
    cls = load_definition(_root(intact), MINIMAL_NAME, digest)

    assert getattr(cls, "__mosaic_definition__") == DefinitionRef(MINIMAL_NAME, digest)


def test_a_sidecar_without_a_defaulted_fact_loads(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    _ = _edit_sidecar(dataset, MINIMAL_SOURCE, "flags", "consumed_roots")

    cls = load_definition(_root(dataset), MINIMAL_NAME, _digest(MINIMAL_SOURCE))

    assert getattr(cls, "name") == f"def:{MINIMAL_NAME}"


def test_a_reach_declared_none_differs_from_an_undeclared_one(tmp_path: Path) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: REACH_SOURCE})
    _ = _edit_sidecar(dataset, REACH_SOURCE, "attributes", "temporal_reach")

    with pytest.raises(DefinitionMismatch) as caught:
        _ = load_definition(_root(dataset), MINIMAL_NAME, _digest(REACH_SOURCE))

    assert caught.value.fields == ("attributes.temporal_reach",)


@pytest.mark.parametrize("key", ["digest", "name"])
def test_a_sidecar_of_another_version_is_corrupt(tmp_path: Path, key: str) -> None:
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: MINIMAL_SOURCE})
    path = _version_file(dataset, MINIMAL_NAME, MINIMAL_SOURCE, "json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored[key] = {"digest": "0" * 64, "name": "elsewhere"}[key]
    _ = path.write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")

    with pytest.raises(DefinitionCorrupt) as caught:
        _ = load_definition(_root(dataset), MINIMAL_NAME, _digest(MINIMAL_SOURCE))

    assert caught.value.path == path


def test_a_registration_made_at_load_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mosaic.behavior.feature_library.registry import FEATURES

    monkeypatch.delenv(REGISTER_VARIABLE, raising=False)
    dataset = defined_dataset(tmp_path, {MINIMAL_NAME: REGISTERING_SOURCE})
    monkeypatch.setenv(REGISTER_VARIABLE, "1")

    try:
        with pytest.raises(DefinitionInvalid) as caught:
            _ = load_definition(
                _root(dataset), MINIMAL_NAME, _digest(REGISTERING_SOURCE)
            )
    finally:
        _ = FEATURES.pop(STRAY_FEATURE, None)

    assert [(each.check, each.field) for each in caught.value.findings] == [
        ("registry", "FEATURES")
    ]


# --- an instance given a model fits nothing ----------------------------------


FIT_STORAGE: Final = f"def.{APPLYING_NAME}__from__tracks"
FIT_RUN_ID: Final = "1-abc"
MODEL: Final = {"feature": FIT_STORAGE, "run_id": FIT_RUN_ID}


class _BuiltIn(PlainFeature):
    """A built-in feature of the full protocol."""

    accepts_overlap = False


def _flags(feature: object) -> tuple[object, object]:
    """The ``fits`` and ``scope_dependent`` values of *feature*, ``None`` if unset."""
    return getattr(feature, "fits", None), getattr(feature, "scope_dependent", None)


def _applying(tmp_path: Path, params: dict[str, object] | None) -> Feature:
    """An instance of the applying definition's loaded class, built by its class."""
    dataset = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    cls = load_definition(_root(dataset), APPLYING_NAME, _digest(APPLYING_SOURCE))
    return built_by_class(cls, params)


def test_an_instance_given_a_model_is_narrowed_to_an_apply(tmp_path: Path) -> None:
    feature = _applying(tmp_path, {"model": MODEL})
    assert _flags(feature) == (True, True)
    expected = DefinitionModel(feature=FIT_STORAGE, run_id=FIT_RUN_ID)

    assert applied_definition_model(feature) == expected
    narrow_to_apply(feature)

    assert _flags(feature) == (False, False)
    assert applied_definition_model(feature) == expected


def test_narrowing_is_idempotent(tmp_path: Path) -> None:
    feature = _applying(tmp_path, {"model": MODEL})

    narrow_to_apply(feature)
    narrow_to_apply(feature)

    assert _flags(feature) == (False, False)


def test_an_instance_without_a_model_keeps_its_flags(tmp_path: Path) -> None:
    feature = _applying(tmp_path, None)

    narrow_to_apply(feature)

    assert applied_definition_model(feature) is None
    assert _flags(feature) == (True, True)


def test_a_built_in_is_neither_narrowed_nor_reported_as_applying() -> None:
    feature = _BuiltIn()
    flags = _flags(feature)

    narrow_to_apply(feature)

    assert applied_definition_model(feature) is None
    assert _flags(feature) == flags
