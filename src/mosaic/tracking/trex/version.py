"""The declared identity of mosaic's TREx integration.

A leaf module with no imports, because both users need it at *module* scope and
one of them must stay import-light: ``TrexOp`` defers every TREx import into
``run()`` so registering the op does not drag in the subprocess machinery, and
reading a version constant must not undo that.
"""

from __future__ import annotations

from typing import Final

TREX_KIND: Final = "trex"

TREX_VERSION: Final = "0.1"
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
"""
