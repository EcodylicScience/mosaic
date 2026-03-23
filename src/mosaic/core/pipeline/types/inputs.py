from __future__ import annotations

from typing import ClassVar, Generic, Literal, Protocol, Self

from pydantic import RootModel, model_validator
from typing_extensions import TypeVar

from mosaic.core.pipeline.types.results import Result

InputRequire = Literal["nonempty", "empty", "any"]

TrackInput = Literal["tracks"]

InputItem = TypeVar("InputItem", bound=TrackInput | Result, default=TrackInput | Result)


class InputsLike(Protocol):
    """Read-only interface satisfied by any Inputs[InputItem]."""

    @property
    def root(self) -> tuple[TrackInput | Result, ...]: ...
    @property
    def has_tracks(self) -> bool: ...
    @property
    def is_single_tracks(self) -> bool: ...
    @property
    def is_single_feature(self) -> bool: ...
    @property
    def is_multi(self) -> bool: ...
    @property
    def is_empty(self) -> bool: ...
    @property
    def feature_inputs(self) -> tuple[Result, ...]: ...
    def storage_suffix(self) -> str | None: ...
    def model_dump(self) -> dict[str, object]: ...


class Inputs(RootModel[tuple[InputItem, ...]], Generic[InputItem]):
    """Base class for feature input collections. Mirrors Params.

    Each Feature subclasses to narrow allowed input types,
    paralleling class Params(Params):.

    Examples:
        Inputs(("tracks",))
        Inputs((Result(feature="speed-angvel"),))
        Inputs(("tracks", Result(feature="nn", run_id="0.1-abc")))

    Per-feature narrowing:
        class Inputs(Inputs[TrackInput]):
            pass

    Features that take no pipeline inputs:
        class Inputs(Inputs[Result]):
            _require: ClassVar[InputRequire] = "empty"

    Self-loading features that optionally accept inputs (e.g. fit + assign):
        class Inputs(Inputs[Result]):
            _require: ClassVar[InputRequire] = "any"
    """

    # "nonempty" (default): at least one input required
    # "empty": must be empty (no pipeline inputs)
    # "any": both empty and non-empty are valid
    _require: ClassVar[InputRequire] = "nonempty"

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self._require == "empty":
            if self.root:
                raise ValueError("This feature takes no pipeline inputs")
            return self
        if self._require == "nonempty" and not self.root:
            raise ValueError("Inputs must have at least one item")
        keys = [i if isinstance(i, str) else (i.feature, i.run_id) for i in self.root]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate inputs: {keys}")
        return self

    @property
    def has_tracks(self) -> bool:
        return any(i == "tracks" for i in self.root)

    @property
    def feature_inputs(self) -> tuple[Result, ...]:  # type: ignore[type-arg]
        return tuple(i for i in self.root if isinstance(i, Result))

    @property
    def is_single_tracks(self) -> bool:
        return len(self.root) == 1 and self.root[0] == "tracks"

    @property
    def is_single_feature(self) -> bool:
        return len(self.root) == 1 and isinstance(self.root[0], Result)

    @property
    def is_multi(self) -> bool:
        return len(self.root) > 1

    @property
    def is_empty(self) -> bool:
        return len(self.root) == 0

    def storage_suffix(self) -> str | None:
        if not self.root:
            return None
        parts: list[str] = []
        for item in self.root:
            if isinstance(item, str):
                parts.append(item)
            else:
                parts.append(item.feature)
        return "+".join(parts)
