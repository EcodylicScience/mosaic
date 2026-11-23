"""The sidecar format, its reader and the models rebuilt from it."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.json_value import JsonValue
from mosaic.core.params import DeclaredModel, Params
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.loader import load_definition
from mosaic.core.pipeline.definitions.sidecar import (
    SIDECAR_FORMAT,
    SIDECAR_FORMAT_VERSION,
    Attributes,
    DefinitionRef,
    ToolkitModelLookup,
    read_sidecar,
    rebuild_inputs_model,
    rebuild_params_model,
    sidecar_json,
)
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionCorrupt,
    DefinitionInvalid,
    VersionStem,
)
from mosaic.core.pipeline.graph.resolve import toolkit_model_for_ref
from mosaic.core.pipeline.types.data_config import PoseConfig
from mosaic.core.pipeline.types.results import DefinitionModel, Result
from tests.helpers import APPLYING_NAME, APPLYING_SOURCE, defined_dataset

DIGEST = "a" * 64
POSE_REF = f"{PoseConfig.__module__}:{PoseConfig.__qualname__}"

INT: dict[str, JsonValue] = {"kind": "int"}
STR: dict[str, JsonValue] = {"kind": "str"}
POSE: dict[str, JsonValue] = {"kind": "toolkit_model", "ref": POSE_REF}


class Holder(DeclaredModel):
    """A toolkit-like model that holds another."""

    poses: list[PoseConfig] = Field(default_factory=list[PoseConfig])
    label: str = ""


HOLDER_REF = f"{__name__}:{Holder.__qualname__}"
TOOLKIT_MODELS: dict[str, type[DeclaredModel]] = {
    POSE_REF: PoseConfig,
    HOLDER_REF: Holder,
}


def resolve_toolkit_model(ref: str) -> type[DeclaredModel]:
    return TOOLKIT_MODELS[ref]


toolkit_model: ToolkitModelLookup = resolve_toolkit_model


def marker(**fields: JsonValue) -> dict[str, JsonValue]:
    return {"$model": POSE_REF, "set": fields}


def field(
    name: str, kind: dict[str, JsonValue], default: JsonValue, **extra: JsonValue
) -> dict[str, JsonValue]:
    return {"name": name, "type": kind, "default": default, **extra}


def sidecar_data(
    *params: dict[str, JsonValue],
    inputs: dict[str, JsonValue] | None = None,
    prerequisites: list[JsonValue] | None = None,
    format_version: int = SIDECAR_FORMAT_VERSION,
) -> dict[str, JsonValue]:
    return {
        "format": SIDECAR_FORMAT,
        "format_version": format_version,
        "name": "demo",
        "digest": DIGEST,
        "defined_on": "2026-11-13",
        "mosaic_version": "0.21.0",
        "class_name": "Demo",
        "version": "1",
        "summary": "A demo feature.",
        "flags": {
            "parallelizable": True,
            "scope_dependent": False,
            "accepts_overlap": False,
            "emits": "individual",
            "category": "per-frame",
            "consumed_roots": ["tracks"],
        },
        "inputs": inputs
        or {"items": ["tracks"], "require": "nonempty", "track_input": False},
        "params": list(params),
        "attributes": {},
        "prerequisites": prerequisites or [],
    }


def write(directory: Path, data: dict[str, JsonValue], *, sort_keys: bool) -> Path:
    path = directory / "demo" / f"2026-11-13_{DIGEST}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(data, indent=2, sort_keys=sort_keys) + "\n")
    return path


def rebuilt(tmp_path: Path, *params: dict[str, JsonValue]) -> type[Params]:
    sidecar = read_sidecar(write(tmp_path, sidecar_data(*params), sort_keys=True))
    return rebuild_params_model(sidecar, toolkit_model)


def test_a_format_one_file_round_trips_byte_for_byte_with_sorted_keys(
    tmp_path: Path,
) -> None:
    data = sidecar_data(
        field(
            "window",
            {"kind": "tuple", "items": [INT, INT]},
            [1, 2],
            constraints={"min_length": 2},
            description="The window.",
            unit="frames",
        ),
        field("pose", POSE, marker(pose_n=5)),
        field("workers", INT, 1, hash_excluded=True),
        prerequisites=[{"module": "movement", "extra": "movement", "purpose": "it"}],
    )
    data["attributes"] = {
        "fits": True,
        "temporal_reach": None,
        "gpu_when": {"field": "device", "values": ["cuda"]},
        "requires": [{"kind": "column", "name": "X"}],
    }
    expected = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()

    for sort_keys in (True, False):
        sidecar = read_sidecar(write(tmp_path, data, sort_keys=sort_keys))
        assert sidecar_json(sidecar) == expected


def test_an_unset_attribute_stays_absent_and_a_declared_none_is_kept(
    tmp_path: Path,
) -> None:
    data = sidecar_data()
    data["attributes"] = {"temporal_reach": None}
    sidecar = read_sidecar(write(tmp_path, data, sort_keys=True))

    assert sidecar.attributes.model_fields_set == {"temporal_reach"}


def test_a_later_format_version_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path, sidecar_data(format_version=2), sort_keys=True)

    with pytest.raises(DefinitionCorrupt) as caught:
        _ = read_sidecar(path)
    assert caught.value.error_detail() == {
        "name": "demo",
        "digest": DIGEST,
        "path": str(path),
    }
    assert str(caught.value) == (
        f"{path} of the definition 'demo' was written in sidecar format version 2, "
        f"newer than the version {SIDECAR_FORMAT_VERSION} this toolkit reads"
    )


def test_an_emits_outside_its_vocabulary_is_corrupt(tmp_path: Path) -> None:
    data = sidecar_data()
    flags = data["flags"]
    assert isinstance(flags, dict)
    flags["emits"] = "herd"

    with pytest.raises(DefinitionCorrupt):
        _ = read_sidecar(write(tmp_path, data, sort_keys=True))


def test_a_file_that_is_not_a_sidecar_is_corrupt(tmp_path: Path) -> None:
    path = write(tmp_path, {"format": SIDECAR_FORMAT}, sort_keys=True)

    with pytest.raises(DefinitionCorrupt):
        _ = read_sidecar(path)


@pytest.mark.parametrize(
    ("kind", "default", "accepted", "refused"),
    [
        pytest.param(
            {"kind": "list", "item": {"kind": "union", "of": [INT, STR]}},
            [1, "a"],
            [[2, "b"], [2, "b"]],
            [None],
            id="list[int | str]",
        ),
        pytest.param(
            {"kind": "tuple", "items": [INT, INT]},
            [1, 2],
            [[3, 4], (3, 4)],
            [1, 2, 3],
            id="tuple[int, int]",
        ),
        pytest.param(
            {"kind": "tuple", "item": STR},
            [],
            [["a", "b"], ("a", "b")],
            [1],
            id="tuple[str, ...]",
        ),
        pytest.param(
            {
                "kind": "dict",
                "value": {"kind": "union", "of": [STR, {"kind": "list", "item": STR}]},
            },
            {},
            [{"a": "x", "b": ["y"]}, {"a": "x", "b": ["y"]}],
            {"a": 1},
            id="dict[str, str | list[str]]",
        ),
        pytest.param(
            {"kind": "literal", "values": ["a", "b"]},
            "a",
            ["b", "b"],
            "c",
            id="Literal['a', 'b']",
        ),
        pytest.param(
            {"kind": "union", "of": [INT, {"kind": "null"}]},
            None,
            [3, 3],
            "x",
            id="int | None",
        ),
    ],
)
def test_each_type_kind_rebuilds_and_validates(
    tmp_path: Path,
    kind: dict[str, JsonValue],
    default: JsonValue,
    accepted: list[object],
    refused: JsonValue,
) -> None:
    model = rebuilt(tmp_path, field("value", kind, default))
    given, held = accepted

    assert model.model_validate({"value": given}).model_dump()["value"] == held
    with pytest.raises(ValidationError):
        _ = model.model_validate({"value": refused})


@pytest.mark.parametrize(
    ("kind", "constraints", "default", "refused"),
    [
        pytest.param(INT, {"ge": 1}, 1, 0, id="ge"),
        pytest.param(STR, {"pattern": "^[a-z]+$"}, "a", "A1", id="pattern"),
        pytest.param(
            {"kind": "list", "item": INT},
            {"max_length": 2},
            [],
            [1, 2, 3],
            id="max_length",
        ),
    ],
)
def test_a_recorded_constraint_refuses_through_the_rebuilt_model(
    tmp_path: Path,
    kind: dict[str, JsonValue],
    constraints: dict[str, JsonValue],
    default: JsonValue,
    refused: JsonValue,
) -> None:
    model = rebuilt(tmp_path, field("value", kind, default, constraints=constraints))

    _ = model.model_validate({"value": default})
    with pytest.raises(ValidationError):
        _ = model.model_validate({"value": refused})


def test_a_hash_excluded_field_is_left_out_of_the_identity_dump(
    tmp_path: Path,
) -> None:
    model = rebuilt(
        tmp_path,
        field("window", INT, 3, description="The window.", unit="frames"),
        field("workers", INT, 1, hash_excluded=True),
    )
    params = model()

    assert params.model_dump() == {"window": 3, "workers": 1}
    assert params.identity_dump() == {"window": 3}
    schema = model.model_json_schema()["properties"]["window"]
    assert (schema["description"], schema["x-mosaic-unit"]) == ("The window.", "frames")


def test_the_keypoint_and_unwired_facts_reach_the_rebuilt_schema(
    tmp_path: Path,
) -> None:
    model = rebuilt(
        tmp_path,
        field(
            "keypoints",
            {"kind": "list", "item": STR},
            [],
            description="The keypoints.",
            keypoint="name",
            unwired="no code path reads it",
        ),
    )

    schema = model.model_json_schema()["properties"]["keypoints"]
    assert schema["x-mosaic-keypoint"] == "name"
    assert schema["x-mosaic-unwired"] == "no code path reads it"


def test_track_spaces_are_recorded_sorted_and_unique() -> None:
    attributes = Attributes(track_spaces=("world_m", "image_px", "world_m"))

    assert attributes.model_dump(mode="json")["track_spaces"] == [
        "image_px",
        "world_m",
    ]


def test_an_unknown_track_space_is_corrupt(tmp_path: Path) -> None:
    data = sidecar_data()
    data["attributes"] = {"track_spaces": ["image_px", "screen"]}

    with pytest.raises(DefinitionCorrupt):
        _ = read_sidecar(write(tmp_path, data, sort_keys=True))


@pytest.mark.parametrize(
    ("default", "expected"),
    [
        pytest.param(marker(), PoseConfig(), id="nothing set"),
        pytest.param(marker(pose_n=5), PoseConfig(pose_n=5), id="pose_n set"),
    ],
)
def test_a_toolkit_model_default_rebuilds_from_its_set_fields(
    tmp_path: Path, default: JsonValue, expected: PoseConfig
) -> None:
    model = rebuilt(tmp_path, field("pose", POSE, default))
    built: object = model().__dict__["pose"]

    assert isinstance(built, PoseConfig)
    assert built == expected
    assert built.model_fields_set == expected.model_fields_set


class FactoryPose(Params):
    pose: PoseConfig = Field(default_factory=lambda: PoseConfig(pose_n=5))


class PlainPose(Params):
    pose: PoseConfig = PoseConfig(pose_n=5)


@pytest.mark.parametrize(
    ("source", "factory", "pose_n"),
    [
        pytest.param(FactoryPose, True, 5, id="default_factory merges"),
        pytest.param(PlainPose, False, 7, id="plain default validates as given"),
    ],
)
def test_a_partial_override_follows_the_recorded_default_form(
    tmp_path: Path, source: type[Params], factory: bool, pose_n: int
) -> None:
    model = rebuilt(
        tmp_path, field("pose", POSE, marker(pose_n=5), default_factory=factory)
    )
    overrides: dict[str, object] = {"pose": {"x_prefix": "q"}}

    expected = source.from_overrides(overrides).model_dump()
    assert expected["pose"]["pose_n"] == pose_n
    assert model.from_overrides(overrides).model_dump() == expected
    assert model().model_dump() == source().model_dump()


def test_a_list_default_rebuilds_each_toolkit_model(tmp_path: Path) -> None:
    model = rebuilt(
        tmp_path,
        field("poses", {"kind": "list", "item": POSE}, [marker(), marker(pose_n=5)]),
    )

    poses: object = model().__dict__["poses"]
    assert poses == [PoseConfig(), PoseConfig(pose_n=5)]


def test_an_own_file_model_rebuilds_the_toolkit_model_inside_it(
    tmp_path: Path,
) -> None:
    inner: dict[str, JsonValue] = {
        "kind": "model",
        "name": "Inner",
        "base": "NestedParams",
        "fields": [field("pose", POSE, marker()), field("k", INT, 1)],
    }
    outer: dict[str, JsonValue] = {
        "kind": "model",
        "name": "Outer",
        "base": "DeclaredModel",
        "fields": [
            field("inner", inner, {"pose": marker(), "k": 1}),
            field("label", STR, "x"),
        ],
    }
    model = rebuilt(
        tmp_path,
        field(
            "outer", outer, {"inner": {"pose": marker(pose_n=5), "k": 2}, "label": "y"}
        ),
    )

    built = model()
    held: object = built.__dict__["outer"]
    assert isinstance(held, BaseModel)
    assert held.model_dump() == {
        "inner": {"pose": PoseConfig(pose_n=5).model_dump(), "k": 2},
        "label": "y",
    }
    pose: object = held.__dict__["inner"].__dict__["pose"]
    assert isinstance(pose, PoseConfig)
    assert pose.model_fields_set == {"pose_n"}
    definitions = model.model_json_schema()["$defs"]
    assert (definitions["Outer"]["title"], definitions["Inner"]["title"]) == (
        "Outer",
        "Inner",
    )
    assert model.model_validate({"outer": {"inner": {"k": 4}}}).model_dump()[
        "outer"
    ] == {"inner": {"pose": PoseConfig().model_dump(), "k": 4}, "label": "x"}


def test_a_toolkit_model_inside_a_toolkit_model_rebuilds(tmp_path: Path) -> None:
    holder: dict[str, JsonValue] = {
        "$model": HOLDER_REF,
        "set": {"poses": [marker(pose_n=5)]},
    }
    model = rebuilt(
        tmp_path, field("holder", {"kind": "toolkit_model", "ref": HOLDER_REF}, holder)
    )

    built: object = model().__dict__["holder"]
    assert isinstance(built, Holder)
    assert built.model_fields_set == {"poses"}
    assert built.poses == [PoseConfig(pose_n=5)]
    assert built.poses[0].model_fields_set == {"pose_n"}


def test_a_union_with_a_toolkit_model_decodes_its_marker(tmp_path: Path) -> None:
    kind: dict[str, JsonValue] = {"kind": "union", "of": [POSE, {"kind": "null"}]}
    model = rebuilt(tmp_path, field("pose", kind, marker(pose_n=3)))

    assert model().__dict__["pose"] == PoseConfig(pose_n=3)
    assert model.model_validate({"pose": None}).model_dump() == {"pose": None}


def test_a_tuple_with_both_or_neither_form_is_corrupt(tmp_path: Path) -> None:
    kinds: list[dict[str, JsonValue]] = [
        {"kind": "tuple"},
        {"kind": "tuple", "item": INT, "items": [INT]},
    ]
    for kind in kinds:
        data = sidecar_data(field("value", kind, []))
        with pytest.raises(DefinitionCorrupt):
            _ = read_sidecar(write(tmp_path, data, sort_keys=True))


def test_a_model_marker_where_no_toolkit_model_fits_is_plain_data(
    tmp_path: Path,
) -> None:
    kind: dict[str, JsonValue] = {
        "kind": "dict",
        "value": {"kind": "dict", "value": STR},
    }
    model = rebuilt(tmp_path, field("table", kind, {"k": {"$model": "x"}}))

    assert model().model_dump() == {"table": {"k": {"$model": "x"}}}


@pytest.mark.parametrize("module", ["os.path", "mosaic_definitions_x", "a-b"])
def test_a_prerequisite_that_is_not_a_plain_top_level_name_is_refused(
    tmp_path: Path, module: str
) -> None:
    prerequisite: dict[str, JsonValue] = {
        "module": module,
        "extra": "x",
        "purpose": "y",
    }
    path = write(tmp_path, sidecar_data(prerequisites=[prerequisite]), sort_keys=True)
    before = set(sys.modules)

    with pytest.raises(DefinitionInvalid) as caught:
        _ = read_sidecar(path)

    assert set(sys.modules) == before
    finding = caught.value.findings[0]
    assert (finding.check, finding.field) == ("attribute", "prerequisites")
    assert module in finding.message


def test_a_stored_prerequisite_naming_an_undeclared_extra_is_read(
    tmp_path: Path,
) -> None:
    """A sidecar recorded before an extra was renamed still reads: only its hint is stale."""
    prerequisite: dict[str, JsonValue] = {
        "module": "movement",
        "extra": "no-such-extra",
        "purpose": "y",
    }
    path = write(tmp_path, sidecar_data(prerequisites=[prerequisite]), sort_keys=True)

    assert [each.extra for each in read_sidecar(path).prerequisites] == [
        "no-such-extra"
    ]


def test_the_rebuilt_inputs_follow_the_recorded_require(tmp_path: Path) -> None:
    tracks = read_sidecar(write(tmp_path, sidecar_data(), sort_keys=True))
    track_inputs = rebuild_inputs_model(tracks)

    with pytest.raises(ValidationError):
        _ = track_inputs.model_validate([])
    assert track_inputs.model_validate(["tracks"]).model_dump() == ("tracks",)
    assert getattr(track_inputs, "_track_input", None) is False

    any_inputs = sidecar_data(
        inputs={"items": ["result"], "require": "any", "track_input": True}
    )
    results = rebuild_inputs_model(
        read_sidecar(write(tmp_path, any_inputs, sort_keys=True))
    )
    result = Result(feature="speed-angvel__from__tracks")
    assert results.model_validate([result]).model_dump() == (result.model_dump(),)
    assert results.model_validate([]).model_dump() == ()
    assert getattr(results, "_track_input", None) is True


def test_a_definition_ref_is_a_frozen_value() -> None:
    ref = DefinitionRef("demo", DIGEST)

    assert ref == DefinitionRef("demo", DIGEST)
    assert hash(ref) == hash(DefinitionRef("demo", DIGEST))


def test_the_rebuilt_params_of_a_definition_that_applies_a_model_dump_alike(
    tmp_path: Path,
) -> None:
    dataset: Dataset = defined_dataset(tmp_path, {})
    record = define_feature(dataset, APPLYING_NAME, APPLYING_SOURCE)
    root = dataset.get_root(DEFINITIONS_ROOT)
    stem = VersionStem(record.defined_on, record.digest).stem
    sidecar = read_sidecar(root / APPLYING_NAME / f"{stem}.json")
    rebuilt_params = rebuild_params_model(
        sidecar, partial(toolkit_model_for_ref, definition=APPLYING_NAME)
    )
    own_params: object = getattr(
        load_definition(root, APPLYING_NAME, record.digest), "Params"
    )
    assert isinstance(own_params, type) and issubclass(own_params, Params)
    fit: dict[str, object] = {
        "feature": "def.centered-distance__from__tracks",
        "run_id": "x",
    }

    cases: tuple[dict[str, object], ...] = ({}, {"model": fit})
    for overrides in cases:
        assert (
            rebuilt_params.from_overrides(overrides).identity_dump()
            == own_params.from_overrides(overrides).identity_dump()
        )
    assert rebuilt_params.from_overrides({}).identity_dump() == {"model": None}
    assert rebuilt_params.from_overrides({"model": fit}).identity_dump() == {
        "model": fit
    }
    annotation = rebuilt_params.model_fields["model"].annotation
    assert DefinitionModel.__name__ in repr(annotation)
