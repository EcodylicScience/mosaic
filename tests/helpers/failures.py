"""Failure records written into a dataset, for the tests of quarantine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.graph import Recipe

__all__ = ["quarantine_entry"]


def quarantine_entry(
    dataset: Dataset, recipe: Recipe, step_id: str, sequence: str
) -> None:
    """Fail one entry of a step past the attempt bound, and record no exclusion."""
    from mosaic.core.pipeline.graph import plan_pipeline
    from mosaic.core.pipeline.graph.claims import FileFailureStore
    from mosaic.core.pipeline.graph.failures import QUARANTINE_AFTER

    planned = plan_pipeline(dataset, recipe).step(step_id)
    store = FileFailureStore(dataset.base_dir)
    for _ in range(QUARANTINE_AFTER):
        _ = store.note_entry_failure(
            (planned.storage_name, planned.run_id or "", "", sequence), error="boom"
        )
