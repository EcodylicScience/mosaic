"""The declared identity of mosaic's SLEAP integration.

A leaf module with no imports, because both users need it at *module* scope and
one of them must stay import-light: ``SleapOp`` defers every SLEAP import into
``run()`` so registering the op does not drag in the subprocess machinery, and
reading a version constant must not undo that.
"""

from __future__ import annotations

from typing import Final

SLEAP_KIND: Final = "sleap"

TRAIN_SLEAP_KIND: str = "train-sleap"
"""The op kind that *produces* the models this integration tracks with.

Here rather than beside ``TrainSleapOp`` because both halves of "train here,
track with it there" need it, and the tracker cannot reach the op module without
dragging the whole op-registration machinery into its import path.

A model reference resolves against ``models/<kind>/index.csv``, and the kind that
names that index is the one that *wrote* the row -- the training op's, never the
tracker's. ``MODEL_KINDS`` already declares ``train-sleap``, so passing it still
selects SLEAP's artifact shape; the two names answer different questions, which
is exactly what ``spec_for`` is written to keep apart.
"""

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
