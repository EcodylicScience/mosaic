"""Declaring a definition's class: the checks it must pass and the record it gives."""

from __future__ import annotations

import types
from datetime import date
from functools import partial
from pathlib import Path
from typing import Annotated, Final, Literal, Self

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from mosaic.core.json_value import JsonValue
from mosaic.core.params import Declared, DeclaredModel, NestedParams
from mosaic.core.params import Params as ParamsBase
from mosaic.core.pipeline.definitions.declare import declare_class
from mosaic.core.pipeline.definitions.sidecar import (
    DefinitionSidecar,
    ListType,
    LiteralType,
    ParamRecord,
    rebuild_inputs_model,
    rebuild_params_model,
)
from mosaic.core.pipeline.definitions.store import (
    DefinitionCheck,
    DefinitionInvalid,
    DefinitionToolkitModelMissing,
)
from mosaic.core.pipeline.graph.compatibility import GpuWhen
from mosaic.core.pipeline.graph.resolve import toolkit_model_for_ref
from mosaic.core.pipeline.types.data_config import PoseConfig
from mosaic.core.pipeline.types.inputs import Inputs as InputsBase
from mosaic.core.pipeline.types.inputs import TrackInput
from mosaic.core.pipeline.types.results import Result

NAME: Final = "demo"
DIGEST: Final = "b" * 64
DEFINED_ON: Final = date(2026, 11, 13)
POSE_REF: Final = f"{PoseConfig.__module__}:{PoseConfig.__qualname__}"

toolkit_model = partial(toolkit_model_for_ref, definition=NAME)


class _Absent:
    """Leaves an attribute off the class :func:`feature` builds."""


ABSENT: Final = _Absent()


class TracksInputs(InputsBase[TrackInput]):
    pass


class EmptyParams(ParamsBase):
    pass


class Methods:
    """A feature that computes nothing."""

    def __init__(self, inputs: BaseModel, params: dict[str, object] | None) -> None:
        params_model: object = getattr(type(self), "Params")
        assert isinstance(params_model, type) and issubclass(params_model, ParamsBase)
        self.inputs: BaseModel = inputs
        self.params: ParamsBase = params_model.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return False

    def fit(self, inputs: object) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: object) -> object:
        return df


FLAGS: Final[dict[str, object]] = {
    "name": NAME,
    "version": "1",
    "parallelizable": True,
    "scope_dependent": False,
    "accepts_overlap": False,
    "emits": "individual",
    "consumed_roots": (),
    "Inputs": TracksInputs,
    "Params": EmptyParams,
}


def feature(**attributes: object) -> type:
    """A class with the flags of :data:`FLAGS`, changed by *attributes*."""
    namespace = {
        key: value
        for key, value in {**FLAGS, **attributes}.items()
        if value is not ABSENT
    }
    namespace["__doc__"] = "A demo feature.\n\nIt computes nothing."
    return type("Demo", (Methods,), namespace)


def declare(*classes: type) -> DefinitionSidecar:
    module = types.ModuleType("mosaic_definitions.demo")
    for index, cls in enumerate(classes):
        setattr(module, f"Class{index}", cls)
    return declare_class(module, NAME, DIGEST, DEFINED_ON, toolkit_model)


def findings_of(*classes: type) -> list[tuple[DefinitionCheck, str | None]]:
    with pytest.raises(DefinitionInvalid) as caught:
        _ = declare(*classes)
    return [(finding.check, finding.field) for finding in caught.value.findings]


def record_of(sidecar: DefinitionSidecar, name: str) -> ParamRecord:
    return next(record for record in sidecar.params if record.name == name)


type Aggregation = Literal["mean", "max"]


class Window(DeclaredModel):
    """A model of the definition's own file."""

    size: Annotated[int, Field(ge=1), Declared("The window length.", unit="frames")] = 3


class AllowedParams(ParamsBase):
    pose: PoseConfig = Field(default_factory=PoseConfig)
    keypoints: list[int | str] = [0, "nose"]
    window: tuple[int, int] = (0, 10)
    labels: dict[str, str | list[str]] = {"a": "x", "b": ["y"]}
    nested: Window = Window()
    aggregations: list[Aggregation] = ["mean"]
    poses: list[PoseConfig] = [PoseConfig(), PoseConfig(pose_n=5)]


def test_a_minimal_class_records_its_flags_and_inputs() -> None:
    sidecar = declare(feature())

    assert (sidecar.name, sidecar.digest, sidecar.defined_on) == (
        NAME,
        DIGEST,
        DEFINED_ON,
    )
    assert (sidecar.class_name, sidecar.version, sidecar.summary) == (
        "Demo",
        "1",
        "A demo feature.",
    )
    assert sidecar.flags.emits == "individual"
    assert sidecar.flags.scope_dependent is False
    assert sidecar.inputs.items == ("tracks",)
    assert sidecar.inputs.require == "nonempty"
    assert sidecar.params == ()


def test_the_allowed_types_are_accepted_and_recorded() -> None:
    sidecar = declare(feature(Params=AllowedParams))

    assert [record.name for record in sidecar.params] == list(
        AllowedParams.model_fields
    )
    assert record_of(sidecar, "pose").default == {"$model": POSE_REF, "set": {}}
    assert record_of(sidecar, "poses").default == [
        {"$model": POSE_REF, "set": {}},
        {"$model": POSE_REF, "set": {"pose_n": 5}},
    ]
    assert record_of(sidecar, "nested").default == {"size": 3}


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"pose": {"pose_n": 3}}, id="pose"),
        pytest.param({"keypoints": [1, "tail"]}, id="keypoints"),
        pytest.param({"window": [2, 5]}, id="window"),
        pytest.param({"labels": {"c": ["z"]}}, id="labels"),
        pytest.param({"nested": {"size": 4}}, id="nested"),
        pytest.param({"aggregations": ["max"]}, id="aggregations"),
    ],
)
def test_a_steps_params_dump_alike_through_the_rebuilt_model(
    overrides: dict[str, object],
) -> None:
    rebuilt = rebuild_params_model(
        declare(feature(Params=AllowedParams)), toolkit_model
    )

    assert (
        rebuilt.from_overrides(overrides).identity_dump()
        == AllowedParams.from_overrides(overrides).identity_dump()
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"pose": {"pose_n": "x"}}, id="pose"),
        pytest.param({"keypoints": [1.5]}, id="keypoints"),
        pytest.param({"window": [1]}, id="window"),
        pytest.param({"labels": {"a": 1}}, id="labels"),
        pytest.param({"nested": {"size": 0}}, id="nested"),
        pytest.param({"aggregations": ["sum"]}, id="aggregations"),
    ],
)
def test_a_steps_params_are_refused_alike_through_the_rebuilt_model(
    overrides: dict[str, object],
) -> None:
    rebuilt = rebuild_params_model(
        declare(feature(Params=AllowedParams)), toolkit_model
    )

    with pytest.raises(ValidationError):
        _ = AllowedParams.from_overrides(overrides)
    with pytest.raises(ValidationError):
        _ = rebuilt.from_overrides(overrides)


def test_a_type_alias_is_recorded_as_the_type_it_names() -> None:
    sidecar = declare(feature(Params=AllowedParams))

    assert record_of(sidecar, "aggregations").type == ListType(
        kind="list", item=LiteralType(kind="literal", values=("mean", "max"))
    )


def test_the_rebuilt_inputs_accept_what_the_class_accepts() -> None:
    sidecar = declare(feature())
    rebuilt = rebuild_inputs_model(sidecar)

    assert (
        rebuilt.model_validate(["tracks"]).model_dump()
        == TracksInputs.model_validate(["tracks"]).model_dump()
    )
    with pytest.raises(ValidationError):
        _ = rebuilt.model_validate([])


class NoDefaultParams(ParamsBase):
    window: int


class ValidatedParams(ParamsBase):
    window: int = 3

    @field_validator("window")
    @classmethod
    def _positive(cls, value: int) -> int:
        return abs(value)


class ResultParams(ParamsBase):
    source: Result | None = None


class FieldInputs(InputsBase[TrackInput]):
    _window: int = 3


class ValidatedInputs(InputsBase[TrackInput]):
    @model_validator(mode="after")
    def _single(self) -> Self:
        return self


class MetadataParams(ParamsBase):
    rate: float = Field(default=1.0, allow_inf_nan=False)


class DescribedParams(ParamsBase):
    rate: float = Field(default=1.0, description="The rate.")


class AliasParams(ParamsBase):
    rate: float = Field(default=1.0, alias="speed")


class NestedAnnotatedParams(ParamsBase):
    values: list[Annotated[int, Field(ge=0)]] = []


class OptionalDeclaredParams(ParamsBase):
    names: Annotated[list[str], Declared("The names.")] | None = None


class StrippedParams(ParamsBase):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: str = ""


class SerializedParams(ParamsBase):
    label: str = ""

    @field_serializer("label")
    def _upper(self, value: str) -> str:
        return value.upper()


class ComputedParams(ParamsBase):
    window: int = 3

    @computed_field
    @property
    def doubled(self) -> int:
        return 2 * self.window


class PostInitParams(ParamsBase):
    window: int = 3

    def model_post_init(self, context: object, /) -> None:
        self.window = 4


class InitParams(ParamsBase):
    window: int = 3

    def __init__(self, **data: object) -> None:
        super().__init__(**data)


class AliasedInner(NestedParams):
    k: int = Field(default=1, alias="key")


class NestedAliasParams(ParamsBase):
    inner: AliasedInner = AliasedInner()


class IdentityInner(NestedParams):
    k: int = 1

    def identity_dump(self) -> dict[str, object]:
        return {}


class NestedIdentityParams(ParamsBase):
    inner: IdentityInner = IdentityInner()


class ConstructedPoseParams(ParamsBase):
    pose: PoseConfig = PoseConfig.model_construct(pose_n="x")


class MarkerParams(ParamsBase):
    table: dict[str, PoseConfig | dict[str, str]] = {"k": {"$model": "x"}}


class PlainMarkerParams(ParamsBase):
    table: dict[str, dict[str, str]] = {"k": {"$model": "x"}}


class ModelParams(ParamsBase):
    model: str | None = None


class NarrowingInit(Methods):
    """Narrows two flags when given a model."""

    name = NAME
    version = "1"
    parallelizable = True
    scope_dependent = True
    accepts_overlap = False
    emits = "individual"
    consumed_roots = ()
    fits = True
    Inputs = TracksInputs
    Params = ModelParams

    def __init__(self, inputs: BaseModel, params: dict[str, object] | None) -> None:
        super().__init__(inputs, params)
        model: object = getattr(self.params, "model")
        self.scope_dependent: bool = model is None
        self.fits: bool = model is None


class RaisingInit(Methods):
    """Cannot be built."""

    name = NAME
    version = "1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    emits = "individual"
    consumed_roots = ()
    Inputs = TracksInputs
    Params = EmptyParams

    def __init__(self, inputs: BaseModel, params: dict[str, object] | None) -> None:
        raise RuntimeError("no")


def gpu_when_outside_json() -> GpuWhen:
    """A ``GpuWhen`` holding a value that is not JSON, which its type forbids."""
    gpu_when = GpuWhen(field="device", values=("cuda",))
    object.__setattr__(gpu_when, "values", (object(),))
    return gpu_when


def _reach(params: object) -> int:
    return 3


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        pytest.param(
            (feature(scope_dependent=ABSENT),),
            [("attribute", "scope_dependent")],
            id="missing scope_dependent",
        ),
        pytest.param(
            (feature(version=""),), [("attribute", "version")], id="empty version"
        ),
        pytest.param(
            (feature(emits="crowd"),), [("attribute", "emits")], id="unknown emits"
        ),
        pytest.param(
            (feature(__mosaic_definition__=True),),
            [("attribute", "__mosaic_definition__")],
            id="mosaic definition marker",
        ),
        pytest.param(
            (feature(gpu_when=gpu_when_outside_json()),),
            [("attribute", "gpu_when")],
            id="gpu_when outside JSON",
        ),
        pytest.param((), [("class", None)], id="no class"),
        pytest.param((feature(), feature()), [("class", None)], id="two classes"),
        pytest.param(
            (feature(temporal_reach=_reach),),
            [("temporal_reach", "temporal_reach")],
            id="callable temporal_reach",
        ),
        pytest.param(
            (feature(temporal_reach_by_option=_reach),),
            [("temporal_reach", "temporal_reach_by_option")],
            id="temporal_reach_by_option",
        ),
        pytest.param(
            (feature(Inputs=FieldInputs),), [("inputs", "_window")], id="inputs field"
        ),
        pytest.param(
            (feature(Inputs=ValidatedInputs),),
            [("inputs", None)],
            id="inputs validator",
        ),
        pytest.param(
            (feature(Params=BaseModel),), [("params_base", None)], id="params base"
        ),
        pytest.param(
            (feature(Params=NoDefaultParams),),
            [("params_default", "window")],
            id="no default",
        ),
        pytest.param(
            (feature(Params=ValidatedParams),),
            [("params_validator", "window")],
            id="field validator",
        ),
        pytest.param(
            (feature(Params=ResultParams),),
            [("params_reference", "source")],
            id="Result field",
        ),
        pytest.param(
            (feature(Params=MetadataParams),),
            [("params_metadata", "rate")],
            id="Field metadata",
        ),
        pytest.param(
            (feature(Params=AliasParams),), [("params_alias", "rate")], id="alias"
        ),
        pytest.param(
            (feature(Params=NestedAnnotatedParams),),
            [("params_type", "values")],
            id="nested Annotated",
        ),
        pytest.param(
            (feature(Params=OptionalDeclaredParams),),
            [("params_type", "names")],
            id="Declared inside a union",
        ),
        pytest.param(
            (feature(Params=StrippedParams),),
            [("params_config", None)],
            id="model_config",
        ),
        pytest.param(
            (feature(Params=SerializedParams),),
            [("params_serializer", "label")],
            id="field_serializer",
        ),
        pytest.param(
            (feature(Params=ComputedParams),),
            [("params_serializer", "doubled")],
            id="computed_field",
        ),
        pytest.param(
            (feature(Params=PostInitParams),),
            [("params_post_init", None)],
            id="model_post_init",
        ),
        pytest.param(
            (feature(Params=InitParams),),
            [("params_post_init", None)],
            id="__init__ on Params",
        ),
        pytest.param(
            (feature(Params=NestedAliasParams),),
            [("params_alias", "inner.k")],
            id="nested alias",
        ),
        pytest.param(
            (feature(Params=NestedIdentityParams),),
            [("params_serializer", "inner")],
            id="nested identity_dump",
        ),
        pytest.param(
            (feature(Params=ConstructedPoseParams),),
            [("model_rebuild", "pose")],
            id="model_construct default",
        ),
        pytest.param(
            (feature(Params=MarkerParams),),
            [("model_marker", "table")],
            id="model marker",
        ),
        pytest.param(
            (NarrowingInit,),
            [("flag_in_init", "scope_dependent"), ("flag_in_init", "fits")],
            id="flag narrowed in __init__",
        ),
        pytest.param((RaisingInit,), [("construction", None)], id="construction"),
    ],
)
def test_each_check_refuses_with_its_finding(
    classes: tuple[type, ...], expected: list[tuple[DefinitionCheck, str | None]]
) -> None:
    assert findings_of(*classes) == expected


def test_a_model_marker_where_no_toolkit_model_fits_is_plain_data() -> None:
    sidecar = declare(feature(Params=PlainMarkerParams))
    rebuilt = rebuild_params_model(sidecar, toolkit_model)

    expected: JsonValue = {"k": {"$model": "x"}}
    assert record_of(sidecar, "table").default == expected
    assert rebuilt().model_dump() == {"table": expected}


def test_a_toolkit_model_is_imported_by_its_ref() -> None:
    assert toolkit_model_for_ref(POSE_REF, definition=NAME) is PoseConfig


@pytest.mark.parametrize(
    "ref",
    [
        pytest.param("mosaic_definitions.demo__x:Thing", id="a definition module"),
        pytest.param("os.path:Thing", id="outside mosaic"),
        pytest.param(
            "mosaic.core.pipeline.definitions.store:VersionStem",
            id="not a DeclaredModel",
        ),
        pytest.param("mosaic.core.pipeline.types.data_config", id="no class"),
    ],
)
def test_a_ref_outside_the_toolkit_models_is_invalid(ref: str) -> None:
    with pytest.raises(DefinitionInvalid) as caught:
        _ = toolkit_model_for_ref(ref, definition=NAME)

    assert caught.value.name == NAME


@pytest.mark.parametrize(
    "ref",
    [
        pytest.param(
            "mosaic.core.pipeline.types.data_config:RemovedConfig", id="removed class"
        ),
        pytest.param("mosaic.core.removed_module:PoseConfig", id="removed module"),
    ],
)
def test_a_ref_this_installation_lacks_is_missing(ref: str) -> None:
    with pytest.raises(DefinitionToolkitModelMissing) as caught:
        _ = toolkit_model_for_ref(ref, definition=NAME)

    assert (caught.value.name, caught.value.ref) == (NAME, ref)


def test_the_record_is_the_same_on_each_declaration() -> None:
    first = declare(feature(Params=AllowedParams))
    second = declare(feature(Params=AllowedParams))

    assert first == second


def test_a_field_description_is_refused_naming_declared() -> None:
    with pytest.raises(DefinitionInvalid) as caught:
        _ = declare(feature(Params=DescribedParams))

    (finding,) = caught.value.findings
    assert (finding.check, finding.field) == ("params_metadata", "rate")
    assert "Declared(...)" in finding.message
