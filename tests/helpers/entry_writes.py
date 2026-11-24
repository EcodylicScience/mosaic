"""Wrap the write that an entry task makes, to fault it or hold it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mosaic.core.pipeline import entry_task
from mosaic.core.pipeline._utils import FeatureMeta
from mosaic.core.pipeline.writers import FeatureOutput


class WriteOutput(Protocol):
    """The signature of ``entry_task.write_output``."""

    def __call__(
        self, meta: FeatureMeta, df_feat: FeatureOutput, *, create_parent: bool = True
    ) -> int: ...


def wrapped_write(before: Callable[[FeatureMeta], None]) -> WriteOutput:
    """Return the current ``entry_task.write_output``, calling *before* first.

    *before* receives the entry's meta and may raise to fail the write or sleep to
    slow it. The wrapped function is the one bound when this is called, so a test
    installs the result over ``entry_task.write_output`` itself.
    """
    real = entry_task.write_output

    def write_output(
        meta: FeatureMeta, df_feat: FeatureOutput, *, create_parent: bool = True
    ) -> int:
        before(meta)
        return real(meta, df_feat, create_parent=create_parent)

    return write_output
