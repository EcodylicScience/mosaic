"""The ``mosaic run`` command line that a refusal or a plan prints for a user to run.

This module imports only the standard library. A reader that prints a command loads
no third-party package to build it.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from mosaic.core.entry import Entry

__all__ = ["mosaic_run_command"]

_MANIFEST_PLACEHOLDER: Final = "<manifest>"


def mosaic_run_command(
    kind: str,
    entries: Iterable[Entry],
    *,
    params: str = "",
    manifest: str = _MANIFEST_PLACEHOLDER,
) -> str:
    """Return the ``mosaic run`` line that runs op *kind* over *entries*, unindented.

    Each entry is one ``--entries group:sequence``, sorted and named once.
    :func:`shlex.quote` quotes the manifest, each entry and the params. The
    ``<manifest>`` placeholder is printed bare, as the slot for a manifest path.

    Args:
        kind: The op kind, for ``--kind``.
        entries: The ``(group, sequence)`` pairs that the run covers.
        params: The ``--params`` value, a JSON object or ``@file``. An empty value
            omits ``--params``.
        manifest: The manifest path, for ``-m``.
    """
    target = manifest if manifest == _MANIFEST_PLACEHOLDER else shlex.quote(manifest)
    words = [f"mosaic run -m {target} --kind {kind}"]
    words.extend(
        f"--entries {shlex.quote(f'{group}:{sequence}')}"
        for group, sequence in sorted(set(entries))
    )
    if params:
        words.append(f"--params {shlex.quote(params)}")
    return " ".join(words)
