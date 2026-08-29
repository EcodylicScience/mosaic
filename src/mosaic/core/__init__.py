"""Core data contracts, schemas, and dataset orchestration.

This file binds every public name on access, which keeps the modules beneath it
cheap. Python executes it before any ``mosaic.core.*`` submodule, so a name
bound at module scope is paid for by every consumer of every leaf below.
``Dataset`` reaches pandas through ``helpers`` and ``schema``, and
``track_library`` reached h5py through the SLEAP converter, so importing three
field names off ``mosaic.core.scope`` loaded a dataframe library and raised
``ModuleNotFoundError`` where h5py was absent. ``tests/test_core_lazy_exports.py``
holds that line.

Each name is declared under ``TYPE_CHECKING`` as well, so a type checker and an
editor still resolve it.

Track converters register as a side effect of importing
``mosaic.core.track_library``, which this file no longer does. Anything reading
``TRACK_CONVERTERS`` calls
:func:`~mosaic.core.track_converter.ensure_track_converters_registered` first;
:func:`~mosaic.core.track_converter.get_track_converter` already does.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mosaic.behavior.feature_library.registry import (
        register_feature as register_feature,
    )

    from . import track_library as track_library
    from .dataset import Dataset as Dataset, open_dataset as open_dataset
    from .helpers import from_safe_name as from_safe_name, to_safe_name as to_safe_name


def __getattr__(name: str) -> Any:
    """Import what defines *name* on first access.

    ``register_feature`` additionally has to be deferred rather than merely
    delayed: ``core`` is the layer ``behavior`` is built on, and importing it at
    module scope would make the two a cycle.
    """
    if name == "register_feature":
        from mosaic.behavior.feature_library.registry import register_feature

        return register_feature
    if name in ("Dataset", "open_dataset"):
        from . import dataset

        return getattr(dataset, name)
    if name in ("to_safe_name", "from_safe_name"):
        from . import helpers

        return getattr(helpers, name)
    if name == "track_library":
        # ``import_module`` rather than ``from . import track_library``. The
        # latter runs ``_handle_fromlist``, which probes the package with
        # ``hasattr`` -- re-entering this function, which runs the same import
        # again, until the stack ends. ``dataset`` and ``helpers`` escape it
        # only because no arm here claims their names, so the probe falls
        # through to the AttributeError below.
        return import_module(".track_library", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Dataset",
    "open_dataset",
    "register_feature",
    "to_safe_name",
    "from_safe_name",
    "track_library",
]
