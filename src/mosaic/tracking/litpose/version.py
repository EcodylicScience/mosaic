"""The declared identity of mosaic's Lightning Pose integration.

A leaf module with no imports, because both users need it at *module* scope and
one of them must stay import-light: ``LitposeOp`` defers every Lightning Pose
import into ``run()`` so registering the op does not drag in the subprocess
machinery, and reading a version constant must not undo that.
"""

from __future__ import annotations

from typing import Final

LITPOSE_KIND: Final = "litpose"

LITPOSE_VERSION: Final = "2.3"
"""The declared compatibility version of the Lightning Pose integration.

Seeded at ``2.3`` to mark the Lightning Pose release line this integration
targets, so ``tracks/litpose.2.3-<digest>/`` reads legibly against the Lightning
Pose version a human knows. It is still the *integration's* number, **declared,
never detected**: it is not read from the installed ``litpose``. Deriving it from
the installed tool would invalidate every tracks variant on every upstream patch
release for bit-identical output. Bump it by hand when the integration's *output
semantics* change -- typically at an upstream major/minor release, or when the
settings this integration builds stop meaning what they meant. What the installed
Lightning Pose reports is provenance, recorded on the index row / variant
``observed``, and never part of identity.

``LitposeOp.version`` and the standalone ``run_litpose`` both read this, so the
two entry points cannot drift into naming one run two ways.
"""
