"""The declared identity of mosaic's TREx integration.

A leaf module with no imports, because both users need it at *module* scope and
one of them must stay import-light: ``TrexOp`` defers every TREx import into
``run()`` so registering the op does not drag in the subprocess machinery, and
reading a version constant must not undo that.
"""

from __future__ import annotations

from typing import Final

TREX_KIND: Final = "trex"

TREX_VERSION: Final = "0.2"
"""The declared compatibility version of the TREx integration.

**Declared, never detected.** TREx is updated continuously, so deriving this
from the installed binary's build string would invalidate every tracks variant
on every upstream commit, for bit-identical output. Bump it by hand when the
integration's *output semantics* change -- typically at an upstream major
release, or when the settings this integration builds stop meaning what they
meant. What the installed binary reports is provenance, recorded on the index
row, and never part of identity.

``TrexOp.version`` and the standalone ``run_trex`` both read this, so the two
entry points cannot drift into naming one run two ways.

**0.1 -> 0.2: a multi-clip entry is now converted from one joined video.** TRex
used to be handed the clip list and joined it itself, and its
``FFmpegVideoCapture`` under-counts every file it opens -- so each clip lost its
tail and the ``.pv`` frame index stopped addressing the media. mosaic joins the
clips instead (:mod:`mosaic.core.pipeline.joined_export`) and hands over one
file.

The bump is what makes the fix reachable, and nothing smaller would be. A
conversion slot is addressed by ``<convert run id>/<source uid>``, and neither
term moved: ``source_uid`` is the composition of the clips, which is the same
clips, and the convert run id is the settings, which are the same settings. So
without this the wrong ``.pv`` would be served as a cache hit for every session
already converted. The other three trackers needed no bump for the same change,
because they were truncated to clip 0 before it and their ``source_uid`` moves
from one clip's uuid to the composition of all of them by itself.
"""
