"""The seeded digest that split draws and selection permutations read."""

from __future__ import annotations

import pytest

from mosaic.core.seeded_digest import seeded_digest


@pytest.mark.parametrize(
    ("seed", "key", "expected"),
    [
        (0, "g__s1", 7353594157802530171),
        (7, "g__s1", 8938394611376822188),
        (0, "g__s2", 5749115157940668750),
    ],
)
def test_the_digest_is_pinned_for_a_seed_and_key(
    seed: int, key: str, expected: int
) -> None:
    assert seeded_digest(seed, key) == expected


def test_the_digest_is_a_64_bit_unsigned_integer() -> None:
    assert 0 <= seeded_digest(-3, "") < 2**64
