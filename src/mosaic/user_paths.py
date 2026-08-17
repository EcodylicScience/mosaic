"""Expanding ``~`` in a path a person supplied.

A path typed by a human may begin with ``~``; one mosaic wrote never does. The
rule is that a ``~`` is expanded exactly once, where a path first enters the
process, and interior code receives a path that is already expanded. There are
four such boundaries and nothing else is one:

1. A CLI argument or option (``--manifest``, ``--notes-file``, ``--params @f``).
2. A public API parameter accepting ``str | Path`` (``Dataset(manifest_path=)``,
   ``new_dataset_manifest(base_dir=)``, ``resolve_model(ref=)``).
3. A value read back out of persisted text -- a manifest root, an index cell, an
   op's params blob.
4. An environment variable naming a path. This is the boundary that matters
   most: a shell expands ``~`` only in a word it is expanding itself, so
   ``MOSAIC_TREX_BIN=~/bin/trex`` reaches the process with the tilde intact.

A parameter annotated ``str | Path`` is a boundary and expands; one annotated
``Path`` is interior and does not, leaving its caller responsible.
``resolve_stored_path(stored: str | Path, anchor: Path)`` is the model -- it
expands *stored* and never touches *anchor*, and the signature says which is
which. Expansion is idempotent, so a second call is harmless rather than a bug,
but it does mean the boundary is not where the code claims it is.

Top level rather than under the core subpackage on purpose, the same shape as
``mosaic.runlog`` and ``mosaic.media_probe_config``: importing any ``core``
submodule runs ``core/__init__``, which pulls ``Dataset`` and the video readers,
hence pandas, cv2 and numpy. Every layer needs this rule -- ``cli``, ``core``,
``tracking.common.toolenv``, ``behavior.feature_library.kpms`` -- and a consumer
that needs one path rule should not pay for those to get it. ``core.stored_paths``
is the right shape but the wrong scope: its subject is the cells an index stores,
and it is a consumer of this rule rather than its home.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["user_path"]


def user_path(value: str | Path) -> Path:
    """Expand a leading ``~`` in a path a person supplied.

    Deliberately does not resolve. Expansion and resolution are separate
    questions, and bundling them would break the values that must keep their
    spelling: ``MOSAIC_TREX_BIN=trex`` is a bare name for ``$PATH`` to find, and
    resolving it would produce ``$CWD/trex``. A caller wanting an absolute path
    writes ``user_path(x).resolve()`` and says so.

    Surrounding whitespace is stripped, because a stored index cell may carry it.

    Args:
        value: The path as supplied, expanded or not.

    Returns:
        The path with a leading ``~`` or ``~user`` expanded, and otherwise
        unchanged -- still relative if it was relative.
    """
    path = Path(str(value).strip())
    try:
        return path.expanduser()
    except RuntimeError:
        # `Path.expanduser` raises when it cannot determine the home directory --
        # a `~unknownuser` prefix, or `~` with no `HOME` and no `USERPROFILE`, as
        # a service account has. `os.path.expanduser` returns the input unchanged
        # in exactly those cases, and that is the tolerant behavior every call
        # site here had before this function existed. Raising a message that
        # names no path, from any of forty boundaries, would be a worse answer
        # than leaving a pathological path alone for the caller to fail on.
        return path
