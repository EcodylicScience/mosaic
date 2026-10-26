"""Maps a model's keypoints onto a pose, by name, by override or by index.

A model orders its keypoints independently of a set's pose, and weights
trained elsewhere may name them ``kp0``, ``kp1`` and so on. A
:class:`KeypointMap` records which source position fills each target keypoint,
so predictions read in the model's layout are compared with, or accepted into,
a set's pose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, StrictInt

from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.json_value import JsonValue
from mosaic.core.keypoint_refs import resolve_keypoint_refs
from mosaic.core.params import Declared, DeclaredModel

__all__ = ["KeypointMap", "KeypointMapParams", "resolve_keypoint_map"]


@dataclass(frozen=True, slots=True)
class KeypointMap:
    """Records which source keypoint fills each keypoint of a target pose.

    Construction refuses a map that does not align to the target, names a
    position outside the source, maps one source position to two target
    keypoints, or leaves every target keypoint unmapped.

    Attributes:
        target: The pose the keypoints are mapped onto.
        source_names: The model's keypoint names, in its order.
        sources: One per target keypoint, in target order: the source position
            that fills it, or ``None`` when it is unmapped.
    """

    target: KeypointSchema
    source_names: tuple[str, ...]
    sources: tuple[int | None, ...]

    def __post_init__(self) -> None:
        """Refuse a map unreadable by :meth:`apply`, and an empty map."""
        if len(self.sources) != self.target.num_keypoints:
            msg = (
                f"a keypoint map needs one source per target keypoint, and "
                f"{len(self.sources)} were given for {list(self.target.names)}"
            )
            raise ValueError(msg)
        claimed: dict[int, str] = {}
        for name, source in zip(self.target.names, self.sources, strict=True):
            if source is None:
                continue
            if not 0 <= source < len(self.source_names):
                msg = (
                    f"target keypoint {name!r} maps source position {source}, "
                    f"which is outside the {len(self.source_names)} source "
                    f"keypoints {list(self.source_names)}"
                )
                raise ValueError(msg)
            if source in claimed:
                msg = (
                    f"source keypoint {self.source_names[source]!r} is mapped to "
                    f"two target keypoints, {claimed[source]!r} and {name!r}; a "
                    "source keypoint fills one target"
                )
                raise ValueError(msg)
            claimed[source] = name
        if not claimed:
            msg = (
                f"the keypoint map maps none of the target keypoints "
                f"{list(self.target.names)} from the source keypoints "
                f"{list(self.source_names)}; map one in keypoints, by source name "
                "or index"
            )
            raise ValueError(msg)

    def apply(self, keypoints: Sequence[Keypoint]) -> tuple[Keypoint, ...]:
        """Return *keypoints*, given in source order, aligned to the target.

        Each mapped keypoint is returned as it was given, score included, and
        an unmapped target keypoint is :meth:`Keypoint.absent`.

        Raises:
            ValueError: When the count of *keypoints* differs from the count
                of source names.
        """
        if len(keypoints) != len(self.source_names):
            msg = (
                f"{len(keypoints)} keypoints were given to a map whose source "
                f"has {len(self.source_names)}: {list(self.source_names)}"
            )
            raise ValueError(msg)
        return tuple(
            Keypoint.absent() if source is None else keypoints[source]
            for source in self.sources
        )

    @property
    def is_identity(self) -> bool:
        """Whether the source names equal the target names, each at its position."""
        return self.source_names == self.target.names and all(
            source == position for position, source in enumerate(self.sources)
        )

    def identity_payload(self) -> dict[str, JsonValue]:
        """Return the resolved map, for a run identity that reads through it.

        The payload records positions and no overrides. Two spellings that
        resolve to the same pairs therefore give the same payload.
        """
        target: list[JsonValue] = [*self.target.names]
        source: list[JsonValue] = [*self.source_names]
        sources: list[JsonValue] = [*self.sources]
        return {"target": target, "source": source, "sources": sources}

    @property
    def coverage(self) -> tuple[str, ...]:
        """The mapped target keypoint names, in target order."""
        return tuple(
            name
            for name, source in zip(self.target.names, self.sources, strict=True)
            if source is not None
        )

    @property
    def unmapped_sources(self) -> tuple[str, ...]:
        """The source keypoint names left unmapped, in source order."""
        mapped = set(self.sources)
        return tuple(
            name for index, name in enumerate(self.source_names) if index not in mapped
        )


def resolve_keypoint_map(
    target: KeypointSchema,
    source_names: Sequence[str],
    *,
    overrides: Mapping[str, str | int] | None = None,
    by_name: bool = True,
) -> KeypointMap:
    """Resolve a map from *source_names* onto *target*.

    An override maps a target keypoint to the source keypoint it names, or to
    the source position it gives, and wins over a name match. With *by_name*,
    every other target name that is also a source name maps to it. Names
    match only when equal, with no case folding or fuzzy match.

    Args:
        target: The pose the keypoints are mapped onto.
        source_names: The model's keypoint names, in its order.
        overrides: Target keypoint name to source keypoint name or index.
        by_name: Whether a target name that is also a source name maps to it.

    Returns:
        The resolved map.

    Raises:
        ValueError: On an override key that is not a target name, a source
            name that does not exist, an index out of range, one source
            position mapped to two target keypoints, or a map that leaves every
            target keypoint unmapped.
    """
    names = tuple(source_names)
    table = dict(overrides or {})
    unknown = sorted(set(table) - set(target.names))
    if unknown:
        msg = (
            f"the keypoint map names {unknown}, which the target pose does not "
            f"have; its keypoints are {list(target.names)}"
        )
        raise ValueError(msg)
    by_source_name = {name: index for index, name in enumerate(names)}
    sources: list[int | None] = []
    for name in target.names:
        if name in table:
            (index,) = resolve_keypoint_refs(
                [table[name]],
                names,
                len(names),
                who=f"the keypoint map entry {name!r}",
            )
            sources.append(index)
        elif by_name:
            sources.append(by_source_name.get(name))
        else:
            sources.append(None)
    return KeypointMap(target=target, source_names=names, sources=tuple(sources))


class KeypointMapParams(DeclaredModel):
    """Declares how a model's keypoints map onto the pose that reads them."""

    keypoints: Annotated[
        dict[str, str | StrictInt],
        Declared("Target keypoint name to source keypoint name or index."),
    ] = Field(default_factory=dict)
    by_name: Annotated[
        bool, Declared("Map every target name that is also a source name to it.")
    ] = True
