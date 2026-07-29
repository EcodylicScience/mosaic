"""The declared identity of mosaic's SLEAP integration.

A leaf module with no imports, because both users need it at *module* scope and
one of them must stay import-light: ``SleapOp`` defers every SLEAP import into
``run()`` so registering the op does not drag in the subprocess machinery, and
reading a version constant must not undo that.
"""

from __future__ import annotations

from typing import Final

SLEAP_KIND: Final = "sleap"

SLEAP_VERSION: Final = "1.6"
"""The declared compatibility version of the SLEAP integration.

Seeded at ``1.6`` to mark the SLEAP release line this integration targets, so
``tracks/sleap.1.6-<digest>/`` reads legibly against the SLEAP version a human
knows. It is still the *integration's* number, **declared, never detected**: it
is not read from the installed ``sleap-track``. Deriving it from the installed
tool would invalidate every tracks variant on every upstream patch release for
bit-identical output. Bump it by hand when the integration's *output semantics*
change -- typically at an upstream major/minor release, or when the settings
this integration builds stop meaning what they meant. What the installed SLEAP
reports is provenance, recorded on the index row / variant ``observed``, and
never part of identity.

``SleapOp.version`` and the standalone ``run_sleap`` both read this, so the two
entry points cannot drift into naming one run two ways.
"""
