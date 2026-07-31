"""The train/valid/test assignment, pinned by value before it changes module.

``split_filenames`` is the one genuinely shared piece of the converter package:
five modules reach into ``cvat_points``' underscore namespace for it and its two
companions, and every converter's output layout is downstream of what it decides.
It is also unpinned -- nothing asserted a single assignment, so a change to the
shuffle order, to how group counts round, or to the group key would move every
converter's output at once and no test would say so.

Pinned by value rather than by property. A property test ("the fractions are
roughly right", "groups stay together") stays green through a reordering of the
RNG calls, and a reordering is exactly what a move between modules invites.
``random.Random`` is a documented Mersenne Twister, so a literal assignment is
reproducible across machines and Python versions -- there is nothing to be
tolerant about.
"""

from __future__ import annotations

from mosaic.tracking.pose_training.converters import split_filenames

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
        "vidC__frame_0000.png": "train",
        "vidB__frame_0002.png": "train",
        "vidD__frame_0001.png": "train",
        "vidD__frame_0002.png": "train",
        "vidA__frame_0002.png": "train",
        "vidC__frame_0001.png": "train",
        "vidC__frame_0002.png": "valid",
        "vidA__frame_0001.png": "valid",
        "vidD__frame_0000.png": "valid",
        "vidB__frame_0000.png": "test",
        "vidB__frame_0001.png": "test",
        "vidA__frame_0000.png": "test",
    }
    assert (n_train, n_valid) == (6, 3)


def test_the_group_split_is_pinned() -> None:
    """Per-group assignment, exact -- a different partition of the same inputs."""
    assignment, n_train, n_valid = split_filenames(
        _FILENAMES, _SPLIT, _SEED, split_by="group"
    )
    assert assignment == {
        "vidC__frame_0000.png": "train",
        "vidC__frame_0001.png": "train",
        "vidC__frame_0002.png": "train",
        "vidD__frame_0000.png": "train",
        "vidD__frame_0001.png": "train",
        "vidD__frame_0002.png": "train",
        "vidB__frame_0000.png": "valid",
        "vidB__frame_0001.png": "valid",
        "vidB__frame_0002.png": "valid",
        "vidA__frame_0000.png": "test",
        "vidA__frame_0001.png": "test",
        "vidA__frame_0002.png": "test",
    }
    assert (n_train, n_valid) == (6, 3)


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
