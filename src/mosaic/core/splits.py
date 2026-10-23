"""Reading the roles a dataset gives its groups in a held-out protocol.

The manifest's ``splits`` mapping names each group ``train``, ``test`` or
``sealed`` (:attr:`~mosaic.core.manifest.DatasetManifest.splits`). The functions
here answer the two questions a planner asks of it -- what role does this group
have, and which groups have this role -- without reaching into the manifest.

They take anything that carries a ``splits`` mapping: a
:class:`~mosaic.core.manifest.DatasetManifest`, a
:class:`~mosaic.core.dataset.Dataset`, or the plain mapping itself. All three
are pure reads. A group with no entry has no role, which is not the same as
``train``: nothing here guesses one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Protocol

if TYPE_CHECKING:
    from mosaic.core.manifest import SplitRole

__all__ = [
    "HELD_OUT_ROLES",
    "HasSplits",
    "SplitSource",
    "groups_with_role",
    "held_out_groups",
    "role_of",
]

HELD_OUT_ROLES: Final[tuple["SplitRole", ...]] = ("test", "sealed")
"""The roles no fit may include: a group with one is held out."""


class HasSplits(Protocol):
    """An object that carries a ``splits`` mapping: a manifest or a dataset."""

    @property
    def splits(self) -> Mapping[str, SplitRole]:
        """The role of each named group."""
        ...


SplitSource = HasSplits | Mapping[str, "SplitRole"]
"""What the readers accept: a manifest, a dataset, or the mapping itself."""


def _roles(source: SplitSource) -> Mapping[str, SplitRole]:
    return source if isinstance(source, Mapping) else source.splits


def role_of(source: SplitSource, group: str) -> SplitRole | None:
    """The role of *group*, or ``None`` when the manifest gives it none.

    Args:
        source: A manifest, a dataset, or a ``splits`` mapping.
        group: The group name. The empty group never has a role.

    Returns:
        ``"train"``, ``"test"``, ``"sealed"``, or ``None``.
    """
    return _roles(source).get(group)


def groups_with_role(source: SplitSource, *roles: SplitRole) -> tuple[str, ...]:
    """The groups whose role is one of *roles*, sorted by name.

    Args:
        source: A manifest, a dataset, or a ``splits`` mapping.
        *roles: The roles to select. With none, every group that has a role.

    Returns:
        The group names in ascending order, so the answer does not depend on the
        order the manifest was written in.
    """
    held = _roles(source)
    wanted = set(roles) if roles else None
    return tuple(
        sorted(
            group for group, role in held.items() if wanted is None or role in wanted
        )
    )


def held_out_groups(source: SplitSource) -> tuple[str, ...]:
    """The groups with role ``test`` or ``sealed``, sorted by name.

    These are the groups no fit may include. See :data:`HELD_OUT_ROLES`.
    """
    return groups_with_role(source, *HELD_OUT_ROLES)
