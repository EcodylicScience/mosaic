"""The parameters every tracker op shares: what to run over, and how to run it.

Three op modules declared the same scope block and the same execution block
before adding their tool's own knobs.

**Why the scope fields are not ``HASH_EXCLUDE``.** They arguably should be -- a
tracker's ``run_id`` comes from its settings dict, which is scope-free, so
whether ``groups`` reaches ``identity_dump()`` changes nothing about what any run
is called. But the three tracker ops declare them untagged today, unlike
``_InferParamsBase``, and six digests in ``tests/data/op_identity_golden.json``
are pinned to that. Moving the fields here preserves them exactly; tagging them
while moving them would move those six for no behavior change, and a moved digest
during a refactor should always mean a mistake. Tagging them is a separate
one-commit contract change.

**Why the move is digest-safe.** ``hash_params`` is
``sha1(json.dumps(identity_ready(d), sort_keys=True))[:10]`` and
``Params.identity_dump()`` iterates ``type(self).model_fields``, which includes
inherited fields. Field names, types, defaults and ``HASH_EXCLUDE`` tagging are
carried over verbatim, and ``sort_keys=True`` makes the change in declaration
order invisible.
"""

from __future__ import annotations

from typing import Annotated

from mosaic.core.helpers import parse_entry_tokens
from mosaic.core.pipeline.types import HASH_EXCLUDE, Params

__all__ = ["TrackerOpParams"]


class TrackerOpParams(Params):
    """Scope and execution knobs shared by every tracker op.

    A tracker's own parameters -- its model reference, its thresholds, its
    tracker flavor -- are declared by the subclass. Pydantic allows a required
    field after defaulted ones, so ``model_path`` and its kin stay required.

    The execution knobs are all ``HASH_EXCLUDE``: they change how a run happens,
    not what it produces, so folding them in would move an identifier without
    moving the output, which is a cache miss costing a recompute for nothing.
    Placement knobs (which conda environment, which binary, which display) are
    deliberately *not* here at all -- they are properties of a machine, so they
    are read from ``MOSAIC_<TOOL>_*`` rather than carried in params a queued job
    would ship to a different one.
    """

    # scope (empty -> all indexed media)
    groups: list[str] | None = None
    sequences: list[str] | None = None
    # "group:sequence" pairs; a bare token is a sequence in the empty group
    entries: list[str] | None = None
    convert_to_tracks: Annotated[bool, HASH_EXCLUDE] = True
    overwrite: Annotated[bool, HASH_EXCLUDE] = False
    # Inactivity (hang) watchdog: kill a phase after this many seconds with no
    # output from the tool. max_runtime is an optional absolute ceiling; None
    # leaves the ceiling to the queue.
    idle_timeout: Annotated[float, HASH_EXCLUDE] = 900
    max_runtime: Annotated[float | None, HASH_EXCLUDE] = None

    def entry_pairs(self) -> list[tuple[str, str]]:
        """The ``entries`` tokens as ``(group, sequence)`` pairs."""
        return parse_entry_tokens(self.entries)
