"""The train/valid/test assignment, pinned by value and by its properties.

``split_filenames`` decides the output layout of every converter and of
prepare-training-data. A change to the draw, to the guard or to the group key
changes all of them at once, and the literal pins catch any such change. The draw
is a seeded blake2b digest of each unit's key. A literal assignment is therefore
reproducible across machines, Python versions and ``PYTHONHASHSEED``.

The property tests check the guarantees of the hash draw. An addition leaves every
existing unit in its subset. A changed fraction moves only the units in the band
that moved. The small-set guard fills both positive-share partitions from two units
and keeps a drawn training unit out of validation.
"""

from __future__ import annotations

import hashlib

import pytest

from mosaic.core.annotations.split import Subset
from mosaic.tracking.pose_training.converters import split_filenames


def _draw(seed: int, unit: str) -> float:
    """Compute the draw that ``split_filenames`` documents, apart from its code."""
    digest = hashlib.blake2b(f"{seed}\0{unit}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def _threshold(draw: float, split: tuple[float, float, float]) -> Subset:
    """Return the subset of a draw when the guard does not fire."""
    if draw < split[0]:
        return "train"
    if draw < split[0] + split[1]:
        return "valid"
    return "test"


def _units_per_subset(
    assignment: dict[str, str], unit_of: dict[str, str]
) -> dict[str, int]:
    """Count the distinct units of each subset."""
    units: dict[str, set[str]] = {"train": set(), "valid": set(), "test": set()}
    for name, subset in assignment.items():
        units[subset].add(unit_of[name])
    return {subset: len(members) for subset, members in units.items()}


# Four videos of three frames each: enough that group mode and image mode
# disagree, and small enough to read the whole assignment at a glance.
_FILENAMES: list[str] = [
    f"vid{video}__frame_{index:04d}.png"
    for video in ("A", "B", "C", "D")
    for index in range(3)
]

_SPLIT: tuple[float, float, float] = (0.5, 0.25, 0.25)
_SEED: int = 20260731


def test_the_image_split_is_pinned() -> None:
    """Per-image assignment, exact."""
    assignment, n_train, n_valid = split_filenames(
        _FILENAMES, _SPLIT, _SEED, split_by="image"
    )
    assert assignment == {
        "vidA__frame_0001.png": "train",
        "vidA__frame_0002.png": "train",
        "vidB__frame_0002.png": "train",
        "vidC__frame_0002.png": "train",
        "vidA__frame_0000.png": "valid",
        "vidC__frame_0000.png": "valid",
        "vidC__frame_0001.png": "valid",
        "vidB__frame_0000.png": "test",
        "vidB__frame_0001.png": "test",
        "vidD__frame_0000.png": "test",
        "vidD__frame_0001.png": "test",
        "vidD__frame_0002.png": "test",
    }
    assert (n_train, n_valid) == (4, 3)


def test_the_group_split_is_pinned() -> None:
    """Per-group assignment, exact. A different partition of the same inputs."""
    assignment, n_train, n_valid = split_filenames(
        _FILENAMES, _SPLIT, _SEED, split_by="group"
    )
    assert assignment == {
        "vidA__frame_0000.png": "train",
        "vidA__frame_0001.png": "train",
        "vidA__frame_0002.png": "train",
        "vidB__frame_0000.png": "valid",
        "vidB__frame_0001.png": "valid",
        "vidB__frame_0002.png": "valid",
        "vidC__frame_0000.png": "valid",
        "vidC__frame_0001.png": "valid",
        "vidC__frame_0002.png": "valid",
        "vidD__frame_0000.png": "test",
        "vidD__frame_0001.png": "test",
        "vidD__frame_0002.png": "test",
    }
    assert (n_train, n_valid) == (3, 6)


def _subsets_per_group(assignment: dict[str, str]) -> dict[str, set[str]]:
    """Group each assigned filename by its leading ``vid*`` token."""
    grouped: dict[str, set[str]] = {}
    for filename, subset in assignment.items():
        grouped.setdefault(filename.split("__", 1)[0], set()).add(subset)
    return grouped


def test_a_group_is_never_split_across_subsets() -> None:
    """The load-bearing property, asserted as itself and not only by the literal.

    ``split_by="group"`` exists so frames of one video cannot leak between train
    and validation. The pinned assignment above happens to satisfy this; stating
    it separately means a re-pin cannot quietly give it up.
    """
    assignment, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED, split_by="group")
    assert all(len(s) == 1 for s in _subsets_per_group(assignment).values())


def test_image_mode_does_split_a_group() -> None:
    """The contrast that gives the test above its meaning.

    Without this, "groups stayed together" is equally true of a splitter that
    ignores ``split_by`` entirely.
    """
    assignment, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED, split_by="image")
    assert any(len(s) > 1 for s in _subsets_per_group(assignment).values())


def test_the_same_seed_assigns_the_same_way() -> None:
    """Determinism, the reason the seed is a parameter at all."""
    first, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED)
    second, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED)
    assert first == second


def test_a_different_seed_assigns_differently() -> None:
    """Otherwise the seed is decorative and the pins above prove nothing."""
    first, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED)
    second, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED + 1)
    assert first != second


def test_every_filename_is_assigned_exactly_once() -> None:
    """No item is dropped by the rounding, in either mode."""
    for split_by in ("image", "group"):
        assignment, _, _ = split_filenames(_FILENAMES, _SPLIT, _SEED, split_by=split_by)
        assert set(assignment) == set(_FILENAMES)


def test_a_custom_group_key_overrides_the_default() -> None:
    """Converters pass ``group_key=`` when their filenames are not ``__frame``-shaped."""
    assignment, _, _ = split_filenames(
        _FILENAMES, _SPLIT, _SEED, split_by="group", group_key=lambda _: "one"
    )
    assert len(set(assignment.values())) == 1, "a single group cannot be partitioned"


def _co_grouped(filenames: list[str]) -> list[set[str]]:
    """Which filenames the default group key keeps together, read off the split.

    Observed through the public splitter rather than by calling the private
    ``_default_group_key``: two filenames are in one group exactly when they
    cannot be separated, and a half-and-half split over distinct groups forces
    them apart when they are separable.
    """
    assignment, _, _ = split_filenames(
        filenames, (0.5, 0.5, 0.0), _SEED, split_by="group"
    )
    by_subset: dict[str, set[str]] = {}
    for filename, subset in assignment.items():
        by_subset.setdefault(subset, set()).add(filename)
    return sorted(by_subset.values(), key=lambda names: sorted(names))


def test_the_default_group_key_reads_the_frame_delimiter() -> None:
    """``{stem}__frame_{N}.ext`` groups by ``{stem}``."""
    assert _co_grouped(
        [
            "clipA__frame_0001.png",
            "clipA__frame_0002.png",
            "clipB__frame_0001.png",
            "clipB__frame_0002.png",
        ]
    ) == [
        {"clipA__frame_0001.png", "clipA__frame_0002.png"},
        {"clipB__frame_0001.png", "clipB__frame_0002.png"},
    ]


def test_a_name_without_the_delimiter_is_its_own_group() -> None:
    """No ``__frame``: the whole stem is the key, so each image can land anywhere."""
    assert _co_grouped(["one.png", "two.png"]) == [{"one.png"}, {"two.png"}]


def test_the_delimiter_match_is_the_bare_token() -> None:
    """``__frame`` and not ``__frame_<digits>`` -- ``a__frameXX`` still keys on ``a``."""
    assert _co_grouped(["a__frameXX.jpg", "a__frame_1.png", "b__frame_1.png"]) == [
        {"a__frameXX.jpg", "a__frame_1.png"},
        {"b__frame_1.png"},
    ]


def test_directory_components_are_dropped_before_the_delimiter() -> None:
    """Two paths under different directories share a group when their stems do.

    Worth pinning because it is a collision, not a convenience: distinct source
    directories holding the same filename become one group and can never be
    separated.
    """
    assert _co_grouped(["d/e/f__frame_1.png", "z/f__frame_2.png"]) == [
        {"d/e/f__frame_1.png", "z/f__frame_2.png"}
    ]


# --- the hash draw ----------------------------------------------------------------

# Sixty units. Every partition of the fractions below contains several of them,
# and the guard fires in none of the assignments the tests compare.
_UNITS: list[str] = [f"clip{index:02d}" for index in range(60)]


def _videos(names: list[str], frames: int = 3) -> list[str]:
    """Return *frames* extract-shaped file names for each video in *names*."""
    return [
        f"{name}__frame_{index:04d}.png" for name in names for index in range(frames)
    ]


def _video_of(filename: str) -> str:
    return filename.split("__", 1)[0]


@pytest.mark.parametrize("split_by", ["image", "group"])
def test_each_unit_is_assigned_by_its_seeded_draw(split_by: str) -> None:
    """Without the guard, a unit's subset is the band its draw falls in."""
    split = (0.6, 0.25, 0.15)
    filenames = _videos(_UNITS) if split_by == "group" else list(_UNITS)
    unit_of = {
        name: _video_of(name) if split_by == "group" else name for name in filenames
    }

    assignment, n_train, n_valid = split_filenames(
        filenames, split, _SEED, split_by=split_by
    )

    assert min(_units_per_subset(assignment, unit_of).values()) >= 2, "no guard"
    assert assignment == {
        name: _threshold(_draw(_SEED, unit_of[name]), split) for name in filenames
    }
    assert n_train == sum(subset == "train" for subset in assignment.values())
    assert n_valid == sum(subset == "valid" for subset in assignment.values())


@pytest.mark.parametrize("split_by", ["image", "group"])
def test_adding_units_moves_no_existing_unit(split_by: str) -> None:
    """Every unit keeps its subset when ten more are drawn beside it.

    Each subset contains at least two units before and after. The guard leaves one
    unit in a subset it fills. Both assignments are therefore pure threshold draws.
    """
    split = (0.6, 0.25, 0.15)
    if split_by == "group":
        before = _videos(_UNITS[:50])
        # Ten new videos, and a new frame of a video already drawn.
        after = _videos(_UNITS) + ["clip00__frame_0003.png"]
        unit_of = {name: _video_of(name) for name in after}
    else:
        before, after = _UNITS[:50], list(_UNITS)
        unit_of = {name: name for name in after}

    first, _, _ = split_filenames(before, split, _SEED, split_by=split_by)
    second, _, _ = split_filenames(after, split, _SEED, split_by=split_by)

    assert min(_units_per_subset(first, unit_of).values()) >= 2
    assert min(_units_per_subset(second, unit_of).values()) >= 2
    assert {name: second[name] for name in before} == first
    if split_by == "group":
        assert second["clip00__frame_0003.png"] == first["clip00__frame_0000.png"]


def test_the_order_of_the_names_does_not_reach_the_assignment() -> None:
    forward, _, _ = split_filenames(list(_UNITS), _SPLIT, _SEED)
    backward, _, _ = split_filenames(list(reversed(_UNITS)), _SPLIT, _SEED)

    assert forward == backward


def test_changing_a_fraction_moves_only_the_units_in_the_band_that_moved() -> None:
    """Training grows from 0.6 to 0.7 and test shrinks: two bands move up one subset."""
    before_split, after_split = (0.6, 0.2, 0.2), (0.7, 0.2, 0.1)
    unit_of = {name: name for name in _UNITS}

    before, _, _ = split_filenames(list(_UNITS), before_split, _SEED)
    after, _, _ = split_filenames(list(_UNITS), after_split, _SEED)

    assert min(_units_per_subset(before, unit_of).values()) >= 2
    assert min(_units_per_subset(after, unit_of).values()) >= 2
    moved = {
        name: (before[name], after[name])
        for name in _UNITS
        if before[name] != after[name]
    }
    in_band = {
        name
        for name in _UNITS
        if 0.6 <= _draw(_SEED, name) < 0.7 or 0.8 <= _draw(_SEED, name) < 0.9
    }
    assert set(moved) == in_band and moved, "the bands contain units at this seed"
    assert set(moved.values()) <= {("valid", "train"), ("test", "valid")}


# --- the small-set guard -----------------------------------------------------------

# Five units and their draws at _SEED, which the guard cases below choose their
# fractions against: a 0.206, b 0.659, c 0.536, d 0.855, e 0.695.
_FIVE: list[str] = ["a", "b", "c", "d", "e"]


def test_the_five_draws_are_the_ones_the_guard_cases_assume() -> None:
    assert [round(_draw(_SEED, unit), 3) for unit in _FIVE] == [
        0.206,
        0.659,
        0.536,
        0.855,
        0.695,
    ]


@pytest.mark.parametrize(
    ("split", "expected"),
    [
        pytest.param(
            (0.1, 0.45, 0.45),
            {"a": "train", "b": "test", "c": "valid", "d": "test", "e": "test"},
            id="no-training-draw-promotes-the-smallest-draw",
        ),
        pytest.param(
            (0.55, 0.05, 0.4),
            {"a": "train", "b": "valid", "c": "train", "d": "test", "e": "test"},
            id="no-validation-draw-promotes-the-smallest-test-draw",
        ),
        pytest.param(
            (0.9, 0.1, 0.0),
            {"a": "train", "b": "train", "c": "train", "d": "valid", "e": "train"},
            id="no-test-unit-moves-the-largest-training-draw",
        ),
        pytest.param(
            (0.1, 0.05, 0.85),
            {"a": "train", "b": "test", "c": "valid", "d": "test", "e": "test"},
            id="both-rules-fill-both-partitions",
        ),
    ],
)
def test_the_guard_fills_both_positive_share_partitions(
    split: tuple[float, float, float], expected: dict[str, str]
) -> None:
    assignment, n_train, n_valid = split_filenames(_FIVE, split, _SEED)

    assert assignment == expected
    assert (n_train, n_valid) == (
        sum(subset == "train" for subset in expected.values()),
        sum(subset == "valid" for subset in expected.values()),
    )


@pytest.mark.parametrize("split_by", ["image", "group"])
def test_the_guard_counts_units_and_moves_a_group_whole(split_by: str) -> None:
    """Under ``group``, two videos fill training and validation one video each."""
    filenames = _videos(["clipA", "clipB"])

    assignment, _, _ = split_filenames(
        filenames, (0.9, 0.1, 0.0), _SEED, split_by=split_by
    )

    if split_by == "group":
        assert all(len(s) == 1 for s in _subsets_per_group(assignment).values())
        assert list(assignment.values()).count("valid") == 3, (
            "one video of three frames"
        )
    else:
        assert "valid" in assignment.values() and "train" in assignment.values()


def test_a_lone_unit_keeps_its_draw() -> None:
    """One unit cannot fill two partitions. The guard leaves it where it drew."""
    assert split_filenames(["d"], (0.1, 0.05, 0.85), _SEED)[0] == {"d": "test"}
    assert split_filenames(["d"], (0.9, 0.1, 0.0), _SEED)[0] == {"d": "train"}


@pytest.mark.parametrize(
    ("split", "absent"),
    [((0.0, 0.5, 0.5), "train"), ((0.5, 0.0, 0.5), "valid")],
)
def test_a_zero_share_partition_stays_empty(
    split: tuple[float, float, float], absent: str
) -> None:
    assignment, _, _ = split_filenames(list(_UNITS), split, _SEED)

    assert absent not in assignment.values()


def test_ties_in_the_draw_break_by_unit_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Equal draws order by key: the smallest key is promoted first, the largest last."""

    def tied(seed: int, unit: str) -> float:
        _ = (seed, unit)
        return 0.5

    monkeypatch.setattr("mosaic.core.annotations.split._draw", tied)

    assert split_filenames(["c", "a", "b"], (0.2, 0.2, 0.6), _SEED)[0] == {
        "a": "train",
        "b": "valid",
        "c": "test",
    }
    assert split_filenames(["c", "a", "b"], (0.9, 0.1, 0.0), _SEED)[0] == {
        "a": "train",
        "b": "train",
        "c": "valid",
    }


def test_an_addition_never_moves_a_drawn_training_unit_into_validation() -> None:
    """From two units on, a unit that drew training stays in training.

    Every seed, size and addition of the grid is checked. A unit the guard promoted
    into training is left out, because it drew outside training and may return
    there once a later unit draws training.
    """
    units = [f"u{index}" for index in range(8)]
    checked = guarded = 0
    for split in ((0.8, 0.2, 0.0), (0.7, 0.2, 0.1), (0.6, 0.1, 0.3)):
        for seed in range(40):
            for n in range(2, 6):
                before, _, _ = split_filenames(units[:n], split, seed)
                drawn_train = [
                    unit
                    for unit in units[:n]
                    if before[unit] == "train" and _draw(seed, unit) < split[0]
                ]
                guarded += any(
                    before[unit] == "valid" and _draw(seed, unit) < split[0]
                    for unit in units[:n]
                )
                for added in range(1, 4):
                    after, _, _ = split_filenames(units[: n + added], split, seed)
                    for unit in drawn_train:
                        checked += 1
                        assert after[unit] == "train", (split, seed, n, added, unit)
    assert checked and guarded, "the grid covers the guard's move out of training"


def test_from_one_unit_an_addition_can_move_a_training_unit() -> None:
    """The guard skips a lone unit. The property above starts at two units.

    ``b`` drew training alone. Adding ``a``, whose draw is smaller, makes ``b`` the
    largest training draw, which the guard moves to validation. ``d`` then draws
    validation itself, and ``b`` returns to training.
    """
    split = (0.8, 0.2, 0.0)

    assert split_filenames(["b"], split, _SEED)[0] == {"b": "train"}
    assert split_filenames(["a", "b"], split, _SEED)[0] == {"a": "train", "b": "valid"}
    assert split_filenames(["a", "b", "d"], split, _SEED)[0] == {
        "a": "train",
        "b": "train",
        "d": "valid",
    }
