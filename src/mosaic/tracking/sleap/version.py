"""The declared identity of mosaic's SLEAP integration.

A light module, because both users need it at *module* scope and one of them
must stay import-light: ``SleapOp`` defers every SLEAP import into ``run()`` so
registering the op does not drag in the subprocess machinery, and reading a
version constant must not undo that. Its only runtime import is the stdlib-only
:mod:`mosaic.tracking.common.release`, and it imports the probe's response type
for annotations alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mosaic.tracking.common.release import release_numbers

if TYPE_CHECKING:
    from mosaic.tracking.sleap.probe import SleapProbeResponse

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

PREPARE_EMBEDDING_DATA_KIND: Final = "prepare-embedding-data"
"""The op kind that cuts embedding training labels from a sleap tracker run.

train-sleap reads it off a labels run id to choose the embedding objective, so
it is here, beside the kinds that op already reads.
"""

SLEAP_VERSION: Final = "1.6"
"""The declared compatibility version of the SLEAP integration.

Seeded at ``1.6`` to mark the SLEAP release line this integration targets, so
``tracks/sleap.1.6-<digest>/`` reads legibly against the SLEAP version a human
knows. It is still the *integration's* number, **declared, never detected**: it
is not read from the installed ``sleap-nn``. Deriving it from the installed
tool would invalidate every tracks variant on every upstream patch release for
bit-identical output. Bump it by hand when the integration's *output semantics*
change -- typically at an upstream major/minor release, or when the settings
this integration builds stop meaning what they meant. What the installed SLEAP
reports is provenance, recorded on the index row / variant ``observed``, and
never part of identity.

``SleapOp.version`` and the standalone ``run_sleap`` both read this, so the two
entry points cannot drift into naming one run two ways.
"""

SLEAP_NN_EMBEDDING_MIN: Final = "0.3.5"
"""The oldest sleap-nn release that trains and predicts with an embedding model.

0.3.4 shipped the model type, and 0.3.5 is the release the embedding trial ran.
The installed release is provenance and enters no identifier.
"""


def require_sleap_nn_at_least(probe: SleapProbeResponse, minimum: str) -> None:
    """Refuse a sleap-nn release older than *minimum*.

    Compares the leading release numbers, so a pre-release of *minimum* such as
    ``0.3.5rc1`` is accepted.

    Args:
        probe: The probe of the SLEAP environment.
        minimum: The oldest acceptable release, such as
            :data:`SLEAP_NN_EMBEDDING_MIN`.

    Raises:
        ValueError: The installed release is older than *minimum*, or names no
            release. The message names both versions.
    """
    installed = probe.sleap_nn_version
    if release_numbers(installed) >= release_numbers(minimum):
        return
    message = (
        f"The SLEAP environment has sleap-nn {installed!r}, and this needs "
        f"{minimum} or later. Install a later release in the environment that "
        "MOSAIC_SLEAP_CONDA_ENV or MOSAIC_SLEAP_BIN names."
    )
    raise ValueError(message)
