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

``VideoFrame`` and the key functions (``sequence_key``, ``recording_unit_key``,
``frame_key``, ``unit_key``) name the recorded frames and units that splits are
drawn from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from mosaic.core.entry import to_safe_name
from mosaic.core.recording import RecordingKey
from mosaic.core.seeded_digest import seeded_digest

__all__ = [
    "SplitBy",
    "Subset",
    "VideoFrame",
    "assign_splits",
    "default_group_key",
    "frame_key",
    "print_split_summary",
    "recording_unit_key",
    "sequence_key",
    "split_filenames",
    "unit_key",
]

Subset = Literal["train", "valid", "test"]
"""The partitions a name is assigned to."""

SplitBy = Literal["recording", "sequence", "group", "frame"]
"""What a split keeps together: a recording, a sequence, a ``sequence_groups`` label or
one frame."""


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One recorded frame, named by where it was recorded.

    Attributes:
        origin_uuid: The manifest uuid of the dataset that owns the image. An
            image under no frames-index entry takes the origin of the revision
            that names it.
        group: The entry's group, or ``""`` for an image under no frames-index
            entry.
        sequence: The entry's sequence.
        camera: The camera, or ``""`` for an image under no frames-index entry.
        frame_index: The index in the recording, or -1 when the source did not
            record it.
        image_stem: The image file stem, read only when ``frame_index`` is -1.
    """

    origin_uuid: str
    group: str
    sequence: str
    camera: str
    frame_index: int
    image_stem: str = ""


def sequence_key(frame: VideoFrame) -> str:
    """Return ``"{origin}:{group}:{sequence}"``, each part percent-encoded."""
    return recording_unit_key(
        frame.origin_uuid, RecordingKey(frame.group, frame.sequence)
    )


def recording_unit_key(origin_uuid: str, key: RecordingKey) -> str:
    """Return ``"{origin}:{group}:{sequence}"`` for the recording *key* of *origin_uuid*.

    Each part is percent-encoded. A divided recording ``(group, "")`` ends in an
    empty part, which no sequence key can equal, because a sequence name is never
    empty. An undivided recording's key is its sequence key.
    """
    parts = (origin_uuid, key.group, key.sequence)
    return ":".join(to_safe_name(part) for part in parts)


def _index_part(frame: VideoFrame) -> str:
    if frame.frame_index == -1:
        return "@" + to_safe_name(frame.image_stem)
    return str(frame.frame_index)


def frame_key(frame: VideoFrame) -> str:
    """Return ``"{origin}:{group}:{sequence}:{camera}:{index}"`` for one image.

    ``index`` is the decimal frame index, or ``@`` followed by the encoded image
    stem when the index is -1. Every part is percent-encoded, so a ``:`` inside a
    part cannot split the key.
    """
    return f"{sequence_key(frame)}:{to_safe_name(frame.camera)}:{_index_part(frame)}"


def unit_key(
    frame: VideoFrame,
    split_by: SplitBy,
    sequence_groups: Mapping[str, str],
    recording_units: Mapping[str, str],
) -> str:
    """Return the key of the unit a split keeps together.

    The camera is in no unit key, so the cameras of one recording split together.

    Args:
        frame: The recorded frame.
        split_by: ``"frame"`` gives the frame key without the camera, so the
            cameras' images of one instant are one unit. ``"recording"`` gives
            the unit that ``recording_units`` holds for the sequence key, else the
            sequence key. ``"sequence"`` gives the sequence key. ``"group"`` gives
            the label that ``sequence_groups`` holds for the sequence key, else for
            the bare sequence name, else the sequence key.
        sequence_groups: Maps a sequence key or a bare sequence name to a label.
        recording_units: Maps the sequence key of a member of a divided recording
            to that recording's :func:`recording_unit_key`.

    Returns:
        The unit key.
    """
    sequence = sequence_key(frame)
    if split_by == "frame":
        return f"{sequence}:{_index_part(frame)}"
    if split_by == "recording":
        return recording_units.get(sequence, sequence)
    if split_by == "group":
        label = sequence_groups.get(sequence)
        if label is None:
            label = sequence_groups.get(frame.sequence)
        return sequence if label is None else label
    return sequence


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
    """Assign images to train/valid/test splits.

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
    """Assign each name to ``"train"``, ``"valid"`` or ``"test"`` by a seeded hash.

    The unit is the name under ``split_by="image"`` and its group key under
    ``"group"``. The names of one group share one subset. Each unit draws
    ``u = int.from_bytes(blake2b(f"{seed}\\0{unit}", digest_size=8), "big") / 2**64``
    and is ``train`` when ``u < split[0]``, ``valid`` when ``u < split[0] + split[1]``
    and ``test`` otherwise. A unit's subset depends only on the seed, its key and
    the fractions. The draw leaves every existing unit in its subset when units are
    added, a changed fraction moves only the units in the band that moved, and a
    changed seed redraws every unit. The fractions are expected shares.

    A small-set guard then runs over the ``n`` units, ordered by ``u`` with ties
    broken by unit key:

    1. When ``split[0] > 0``, ``n >= 2`` and no unit drew ``train``, the unit of
       smallest ``u`` moves to ``train``.
    2. When ``split[1] > 0``, ``n >= 2`` and ``valid`` is empty, the ``test``
       unit of smallest ``u`` moves to ``valid``. Without a ``test`` unit, the
       ``train`` unit of largest ``u`` moves instead.

    From two units on, both partitions with a positive share are non-empty. The
    guard leaves the draw unchanged when the draw fills both. A larger set is
    therefore assigned by the threshold alone.

    With units only added, at a fixed seed and fractions, and from two units before
    the addition, rule 2 keeps every existing ``train`` unit out of ``valid``. A
    guarded ``valid`` unit is either a ``test`` unit, which no trainer trains on, or
    the ``train`` unit of largest ``u``, which returns to ``train`` once a later unit
    draws ``valid``.

    That stability starts at two units. The guard skips a lone unit drawn ``train``,
    and adding a second unit of smaller ``u`` makes the first the ``train`` unit of
    largest ``u``, which rule 2 moves. Rule 1 fires only when every unit drew outside
    ``train``, with probability ``(1 - split[0]) ** n``, and its unit can later
    return to ``valid`` or ``test``. The stability covers the units that rule 1 did
    not promote.

    Args:
        filenames: Image file names, or any string identifiers.
        split: The ``(train, valid, test)`` fractions.
        seed: The seed of every unit's draw.
        split_by: ``"group"`` keeps every name of one group key in one subset. Any
            other value draws each name as a unit.
        group_key: Maps a name to its group under ``split_by="group"``. Defaults to
            :func:`default_group_key`, which reads the ``__frame`` delimiter.

    Returns:
        The subset of every name, and the numbers of names assigned to ``train``
        and to ``valid``.
    """
    if split_by == "group":
        key_fn = group_key or default_group_key
        unit_of = {name: key_fn(name) for name in filenames}
    else:
        unit_of = {name: name for name in filenames}
    subset_of_unit = _assign_units(set(unit_of.values()), split, seed)
    assignment: dict[str, str] = {
        name: subset_of_unit[unit] for name, unit in unit_of.items()
    }
    n_train = sum(1 for subset in assignment.values() if subset == "train")
    n_valid = sum(1 for subset in assignment.values() if subset == "valid")
    return assignment, n_train, n_valid


def _draw(seed: int, unit: str) -> float:
    """Return the unit's position in ``[0, 1)``, drawn from the seed and its key alone."""
    return seeded_digest(seed, unit) / 2**64


def _assign_units(
    units: set[str], split: tuple[float, float, float], seed: int
) -> dict[str, Subset]:
    """Assign each unit by the threshold of its draw, then apply the small-set guard."""
    f_train, f_valid = split[0], split[1]
    draws = {unit: _draw(seed, unit) for unit in units}
    ordered = sorted(units, key=lambda unit: (draws[unit], unit))
    subset: dict[str, Subset] = {}
    for unit in ordered:
        if draws[unit] < f_train:
            subset[unit] = "train"
        elif draws[unit] < f_train + f_valid:
            subset[unit] = "valid"
        else:
            subset[unit] = "test"
    if len(ordered) < 2:
        return subset
    if f_train > 0 and "train" not in subset.values():
        subset[ordered[0]] = "train"
    if f_valid > 0 and "valid" not in subset.values():
        tests = [unit for unit in ordered if subset[unit] == "test"]
        trains = [unit for unit in ordered if subset[unit] == "train"]
        subset[tests[0] if tests else trains[-1]] = "valid"
    return subset
