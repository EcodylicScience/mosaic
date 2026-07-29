"""A content digest for a file that identity has to name but cannot recompute.

Model weights are the case this exists for. ``resolve_model`` accepts either a
training run identity or a bare filesystem path, and a path is a mutable key:
swap the file and every consumer reuses output produced by different weights,
reporting a cache hit. A digest is what makes those weights identifiable when
there is no run to name them by.

Deliberately not ``hash_params``. That digests a JSON payload, and a weights file
is bytes -- ``json.dumps`` cannot carry three hundred megabytes. Its fixed width
is also an argument about identifier namespaces, and importing that constraint
here would be borrowing a reason that does not apply.

Deliberately not ``Dataset._md5`` either. That one means *source-file
provenance*: it is the ``md5`` column on a raw-tracks row, copied into the label
index and every registered converter. Keeping a model digest distinguishable from
it is worth one function.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

__all__ = ["MODEL_DIGEST_HEX", "file_digest"]

MODEL_DIGEST_HEX: Final = 16
"""Hex characters kept from the digest -- 64 bits.

Wider than ``hash_params``'s ten, on purpose. That width is argued inside one
``features/<name>/`` directory, where the birthday bound at a thousand distinct
parameter sets is 4.5e-7. A weights digest has no such namespace: it is compared
across machines and across years of retraining, so the population it must stay
distinct within is much larger and has no boundary to bound it.
"""


def file_digest(path: Path, *, chunk: int = 1 << 20) -> str:
    """A blake2b digest of *path*'s bytes, as ``MODEL_DIGEST_HEX`` hex characters.

    blake2b to match ``mosaic-media``'s content hashing, so a reader who knows
    one knows the other. Streamed in 1 MiB chunks: a weights file does not fit
    comfortably in memory and there is no reason to ask it to.

    **Cost, stated because it is paid on every op invocation:** one full read of
    the file, roughly 0.3 s for a 300 MB ``best.pt`` -- on a job that then loads
    the same bytes onto a GPU and runs for minutes. Not cached, because the thing
    a cache would key on is the path, and a mutable path is the defect this
    function exists to close.

    **What it means, and does not.** "These exact weights", not "this training".
    Two identical trainings produce different bytes under nondeterministic CUDA,
    so this digest cannot say two models are the same model. For a *registered*
    model that never matters -- the training run identity names it and this is
    only recorded beside it. It surfaces only on the bare-path escape hatch,
    where there is nothing better to say.
    """
    digest = hashlib.blake2b(digest_size=MODEL_DIGEST_HEX // 2)
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()
