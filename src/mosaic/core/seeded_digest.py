"""A seeded digest of one key, the draw behind every hash-seeded choice.

The split of a labeled unit and the permutation of a selection's candidates
both read it, so one seed and one key give one number on every platform and in
every process. The builtin ``hash()`` is salted per process and is never used.
"""

from __future__ import annotations

import hashlib

__all__ = ["seeded_digest"]


def seeded_digest(seed: int, key: str) -> int:
    """Return the 64-bit digest of *key* under *seed*.

    The value is
    ``int.from_bytes(blake2b(f"{seed}\\0{key}", digest_size=8), "big")``, an
    integer in ``[0, 2**64)`` that depends on the seed and the key alone.

    Args:
        seed: The seed.
        key: The key drawn for, such as a split unit or an entry key.

    Returns:
        The digest, read as a big-endian unsigned integer.
    """
    unit = f"{seed}\0{key}"
    digest = hashlib.blake2b(unit.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")
