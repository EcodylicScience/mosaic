"""Keypoint references, by index or by name, resolved to indices.

A leaf module by design: the standard library only, so a parameter model can
resolve references without importing numpy or pandas.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["resolve_keypoint_refs"]


def resolve_keypoint_refs(
    refs: Sequence[int | str],
    names: Sequence[str] | None,
    n_keypoints: int,
    *,
    who: str,
) -> list[int]:
    """Resolve keypoint references, by index or by name, to indices.

    A parameter naming keypoints accepts either form. A recipe can say
    ``"left_wrist"`` where a name list is given and ``9`` where none is. Both
    forms are checked against the table's keypoint count, and an out-of-range
    index raises as an unknown name does.

    Args:
        refs: Keypoint indices or names, in the order wanted.
        names: The keypoint names, in index order, or ``None`` when none were
            given.
        n_keypoints: How many keypoints the table contains.
        who: The caller, for the message.

    Returns:
        The keypoint indices, in the order of *refs*.

    Raises:
        ValueError: On an unknown name, a name with no name list to resolve it
            in, or an index outside ``0..n_keypoints-1``.
    """
    lookup = {name: index for index, name in enumerate(names or [])}
    resolved: list[int] = []
    for ref in refs:
        if isinstance(ref, str):
            if not names:
                raise ValueError(
                    f"{who} names keypoint {ref!r} but was not given keypoint names; "
                    "pass keypoint_names or refer to keypoints by index"
                )
            if ref not in lookup:
                raise ValueError(
                    f"{who}: no keypoint named {ref!r}; the names are {list(names)}"
                )
            index = lookup[ref]
        else:
            index = int(ref)
        if not 0 <= index < n_keypoints:
            raise ValueError(
                f"{who}: keypoint {ref!r} is outside the table's {n_keypoints} "
                f"keypoints (0..{n_keypoints - 1})"
            )
        resolved.append(index)
    return resolved
