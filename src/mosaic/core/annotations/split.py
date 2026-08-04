"""Assigning labelled frames to train, validation and test.

Moved out of the CVAT converter, which is only where it was written first. Five
other modules imported it from there, three of them reaching past the underscore
on its companions -- a shared primitive living inside one caller's namespace,
which is the arrangement where the next caller writes a sixth copy instead.

It is not converter machinery. Which subset a frame belongs to is a property of
the labelled set, decided once and read by every emitter, so it belongs beside
the representation rather than beside one of its writers.

**The group key is load-bearing.** Splitting per image puts frames of one video
on both sides of the train/validation line, and frames of one video are not
independent samples: a model that has seen frame 100 has largely seen frame 101,
so the score it earns across that line is not a score on unseen data.
``split_by="group"`` keeps a video whole, and the default key reads the
``{video}__frame_{n}`` naming that frame extraction produces.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

__all__ = [
    "assign_splits",
    "default_group_key",
    "print_split_summary",
    "split_filenames",
]


def print_split_summary(
    assignment: dict[str, str],
    n_train: int,
    n_valid: int,
    n_total: int,
    split_by: str,
    key_fn: Callable[[str], str],
) -> None:
    """Print split counts, with group counts when ``split_by="group"``."""
    n_test = n_total - n_train - n_valid
    if split_by == "group":
        groups_per_split: dict[str, set[str]] = {
            "train": set(),
            "valid": set(),
            "test": set(),
        }
        for fname, subset in assignment.items():
            groups_per_split[subset].add(key_fn(fname))
        gt = len(groups_per_split["train"])
        gv = len(groups_per_split["valid"])
        gte = len(groups_per_split["test"])
        print(
            f"  Splits (by group): train={n_train} ({gt} groups), "
            f"valid={n_valid} ({gv} groups), test={n_test} ({gte} groups)"
        )
    else:
        print(f"  Splits: train={n_train}, valid={n_valid}, test={n_test}")


def default_group_key(filename: str) -> str:
    """Extract video stem from ``{stem}__frame_{N}.ext`` naming convention.

    Falls back to the full stem (each image is its own group) if no
    ``__frame`` delimiter is found.
    """
    base = Path(filename).stem
    idx = base.find("__frame")
    return base[:idx] if idx >= 0 else base


def assign_splits(
    usable: Sequence[tuple[Mapping[str, object], Path]],
    split: tuple[float, float, float],
    seed: int,
    *,
    split_by: str = "image",
    group_key: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], int, int]:
    """Shuffle and assign images to train/valid/test splits.

    Thin wrapper around :func:`split_filenames` for converters that hold their
    images as ``(record, path)`` tuples, where the record carries a ``name``.

    Parameters
    ----------
    split_by : ``"image"`` or ``"group"``
        When ``"group"``, all images sharing the same *group_key* are kept
        together in the same split (e.g. all frames from one video).
    group_key : callable, optional
        Function ``filename -> group_name``.  Defaults to
        :func:`default_group_key` which splits on ``__frame``.
    """
    filenames = [str(record["name"]) for record, _ in usable]
    return split_filenames(
        filenames,
        split,
        seed,
        split_by=split_by,
        group_key=group_key,
    )


def split_filenames(
    filenames: list[str],
    split: tuple[float, float, float],
    seed: int,
    *,
    split_by: str = "image",
    group_key: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], int, int]:
    """Assign filenames to train/valid/test splits.

    This is the low-level helper used by all converters.  It works on
    plain filename strings so it is agnostic to the converter's data
    structures.

    Parameters
    ----------
    filenames : list[str]
        Image filenames (or any string identifiers).
    split : (train, valid, test) floats
        Target fractions per split.
    seed : int
        Random seed.
    split_by : ``"image"`` or ``"group"``
        When ``"group"``, all filenames sharing the same *group_key* are
        kept together in the same split.
    group_key : callable, optional
        Function ``filename -> group_name``.  Defaults to
        :func:`default_group_key` (splits on ``__frame``).

    Returns
    -------
    assignment : dict[str, str]
        Mapping ``filename -> "train" | "valid" | "test"``.
    n_train, n_valid : int
        Number of items assigned to train / valid.
    """
    rng = random.Random(seed)

    if split_by == "group":
        key_fn = group_key or default_group_key
        groups: dict[str, list[str]] = {}
        for fn in filenames:
            groups.setdefault(key_fn(fn), []).append(fn)

        group_names = list(groups.keys())
        rng.shuffle(group_names)
        n_groups = len(group_names)
        n_train_g = int(n_groups * split[0])
        n_valid_g = int(n_groups * split[1])

        assignment: dict[str, str] = {}
        for gname in group_names[:n_train_g]:
            for fn in groups[gname]:
                assignment[fn] = "train"
        for gname in group_names[n_train_g : n_train_g + n_valid_g]:
            for fn in groups[gname]:
                assignment[fn] = "valid"
        for gname in group_names[n_train_g + n_valid_g :]:
            for fn in groups[gname]:
                assignment[fn] = "test"
    else:
        shuffled = list(filenames)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_items_train = int(n * split[0])
        n_items_valid = int(n * split[1])

        assignment = {}
        for fn in shuffled[:n_items_train]:
            assignment[fn] = "train"
        for fn in shuffled[n_items_train : n_items_train + n_items_valid]:
            assignment[fn] = "valid"
        for fn in shuffled[n_items_train + n_items_valid :]:
            assignment[fn] = "test"

    n_train = sum(1 for v in assignment.values() if v == "train")
    n_valid = sum(1 for v in assignment.values() if v == "valid")
    return assignment, n_train, n_valid
