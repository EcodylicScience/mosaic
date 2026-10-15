"""The declared identity of mosaic's TREx integration.

A leaf module with no imports, because both users need it at *module* scope and
one of them must stay import-light: ``TrexOp`` defers every TREx import into
``run()`` so registering the op does not drag in the subprocess machinery, and
reading a version constant must not undo that.

**Two versions, one per identity.** :data:`TREX_TRACK_VERSION` names a tracking
run and the tracks variant it publishes. :data:`TREX_CONVERT_VERSION` names the
conversion cache, which holds each ``.pv`` that TREx's detector produced. A
change to tracking or to publication bumps the first alone, so the next run
re-tracks from the cached conversions without re-detecting. A change to the
conversion bumps both: a tracking run's working directory reuses the
conversion its markers record, so the tracking version must move too.

**Declared, never detected.** TREx is updated continuously, so deriving either
from the installed binary's build string would invalidate every tracks variant
on every upstream commit, for bit-identical output. Bump one by hand when the
*output semantics* of what it names change -- typically at an upstream major
release, or when the settings this integration builds stop meaning what they
meant. What the installed binary reports is provenance, recorded on the index
row, and never part of identity.
"""

from __future__ import annotations

from typing import Final

TREX_KIND: Final = "trex"

CLIP_LIST_VERSION: Final = "0.1"
"""The last tracking version that handed TREx a multi-clip entry's clips as a list.

Its run index rows prove, once a conversion's ``.pv`` is swept, that TREx read
several files and not one join of them. See the 0.1 -> 0.2 note on
:data:`TREX_CONVERT_VERSION`.
"""

TREX_TRACK_VERSION: Final = "0.3"
"""The declared version of a TREx tracking run, its run id and its tracks variant.

``TrexOp.version`` and the standalone ``run_trex`` both read this, so the two
entry points cannot drift into naming one run two ways.

0.2 -> 0.3: tables are timed at the measured frame rate of the media TREx read.
"""

TREX_CONVERT_VERSION: Final = "0.2"
"""The declared version of a TREx conversion, which names the conversion cache.

**0.1 -> 0.2: a multi-clip entry is now converted from one joined video.** TRex
used to be handed the clip list and joined it itself, and its
``FFmpegVideoCapture`` under-counts every file it opens -- so each clip lost its
tail and the ``.pv`` frame index stopped addressing the media. mosaic joins the
clips instead (:mod:`mosaic.core.pipeline.joined_export`) and hands over one
file. The tracking version moved with it, from :data:`CLIP_LIST_VERSION`.

The bump is what makes the fix reachable, and nothing smaller would be. A
conversion slot is addressed by ``<convert run id>/<source uid>``, and neither
term moved: ``source_uid`` is the composition of the clips, which is the same
clips, and the convert run id is the settings, which are the same settings. So
without this the wrong ``.pv`` would be served as a cache hit for every session
already converted. The other three trackers needed no bump. They read only clip 0
until mosaic joined the clips, and their markers record that clip's uuid, where the
reuse gate compares the composition of every clip, so each such entry is tracked
again on its next run.
"""
