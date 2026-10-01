"""Deleting a trained model's index row and run root.

A trained model is a row in ``models/<kind>/index.csv`` and the run root
``models/<kind>/<run_id>/`` that the row names. mosaic owns that layout, and a
caller that wants a model gone asks here instead of removing directories itself.

**The row goes first.** Every resolver finds a model through its row
(:func:`~mosaic.tracking.model_refs.resolve_model`), so once the row is dropped the
model stops resolving, from this dataset and from every dataset linking it as a
library, before a file is touched. A removal that then fails leaves unreferenced
files rather than a row naming a half-deleted model.

**Every refusal comes before anything is touched.** A kind that holds prepared
data, a run identifier of another kind, a run that no index registers, a run root
that is a link, and a run root whose directory resolves outside ``models`` are
each refused first. :func:`~mosaic.core.pipeline.models.holds_trained_models`
decides which kinds hold models, for the trained-model inventory too, so every
model that the inventory lists can be deleted.

**The run root is claimed before the row goes**, as an op claims it before
writing. Once the row is dropped, a resubmitted training finds no finished model
and claims the root to train into it. Holding the claim until the root is gone
refuses that training instead of deleting its files from under it. A root that
another execution already holds is refused.

The data that a model trained on is left in place. A prepared dataset is a run of
its own, and other trainings may read it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.markers import InflightMarker, clear_inflight
from mosaic.core.pipeline.models import (
    holds_trained_models,
    model_index_path,
    model_run_root,
)
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.runlog import new_execution_id
from mosaic.tracking.model_refs import ModelNotFoundError
from mosaic.tracking.ops._common import (
    RunRootHeld,
    claim_run_root,
    empty_claimed_run_root,
)
from mosaic.tracking.ops.train import trained_model_index

if TYPE_CHECKING:
    from pathlib import Path

    from mosaic.core.dataset import Dataset

__all__ = [
    "DeletedModel",
    "DeletionRefusalReason",
    "ModelDeletionRefusedError",
    "TrainedModelInUseError",
    "delete_trained_model",
]

type DeletionRefusalReason = Literal[
    "prepared_data",
    "not_a_run_of_kind",
    "run_root_is_a_link",
    "outside_models_root",
]
"""Why a deletion request was refused before anything changed."""


class ModelDeletionRefusedError(ValueError):
    """A deletion request refused before anything changed.

    Attributes:
        reason: Which refusal this is, for a caller that answers each one
            differently.
        kind: The kind as given.
        run_id: The run identifier as given.
    """

    def __init__(
        self, message: str, *, reason: DeletionRefusalReason, kind: str, run_id: str
    ) -> None:
        super().__init__(message)
        self.reason: DeletionRefusalReason = reason
        self.kind: str = kind
        self.run_id: str = run_id


class TrainedModelInUseError(RuntimeError):
    """Another execution holds the model's run root, so nothing was deleted.

    Attributes:
        kind: The training kind.
        run_id: The run identifier.
        execution_id: The execution holding the run root, or ``""`` when its
            claim could not be read.
        host: Where that execution runs, or ``""``.
        expires_at: When the claim lapses unless its holder renews it, as an
            ISO-8601 instant, or ``""``.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        run_id: str,
        execution_id: str,
        host: str,
        expires_at: str,
    ) -> None:
        super().__init__(message)
        self.kind: str = kind
        self.run_id: str = run_id
        self.execution_id: str = execution_id
        self.host: str = host
        self.expires_at: str = expires_at


@dataclass(frozen=True, slots=True)
class DeletedModel:
    """What :func:`delete_trained_model` removed.

    Attributes:
        row: The dropped index row, as text cells.
        directory_removed: Whether the run root was removed. ``False`` when there
            was none, or when removing it failed.
        removal_error: Why removing the run root failed; empty otherwise.
    """

    row: Mapping[str, str]
    directory_removed: bool
    removal_error: str = ""


def delete_trained_model(ds: Dataset, kind: str, run_id: str) -> DeletedModel:
    """Delete the trained model *run_id* of *kind* from *ds*.

    Claims the run root, drops the model's index row, then empties and removes
    the root. A model in a library dataset is deleted by passing that library as
    *ds*; datasets linking it stop resolving it with the row.

    Two deletes of one model racing each other are told apart by the claim: the
    second is refused as in use while the first holds the root, and finds no
    model once the first has dropped the row.

    Args:
        ds: The dataset whose ``models`` root holds the model.
        kind: The training op kind that wrote it, such as ``train-pose``.
        run_id: Its run identifier.

    Returns:
        The dropped row and whether the run root was removed. A removal that
        fails is reported here rather than raised, because by then the row is
        gone and the model no longer resolves: ``directory_removed`` is
        ``False`` and ``removal_error`` says why.

    Raises:
        ModelDeletionRefusedError: *kind* holds prepared data, *run_id* is not a
            run identifier of *kind*, the run root is a link, or the directory
            holding it resolves outside the ``models`` root.
        ModelNotFoundError: No row in *kind*'s index registers *run_id*,
            including when no index of *kind* exists or *ds* has no ``models``
            root.
        TrainedModelInUseError: Another live execution holds the run root.
        pandas.errors.ParserError: *kind*'s index cannot be parsed. It is a
            ``ValueError``, so a caller telling refusals apart catches
            :class:`ModelDeletionRefusedError`, not ``ValueError``.
        KeyError: *kind*'s index has no ``run_id`` column. It is a
            ``LookupError``, as :class:`ModelNotFoundError` is, so a caller
            telling a missing model apart catches that class by name.
    """
    _refuse_unless_a_model_run(kind, run_id)
    if not ds.has_root("models"):
        raise _not_found(kind, run_id, "this dataset has no models root")
    index_path = model_index_path(ds, kind)
    index = trained_model_index(index_path)
    try:
        _ = index.read(run_id=run_id)
    except FileNotFoundError as exc:
        raise _not_found(kind, run_id, f"{index_path} holds no row for it") from exc

    run_root = model_run_root(ds, kind, run_id)
    _refuse_unless_under_models(ds, kind, run_id, run_root)
    # Only a root that exists: claiming creates the directory it claims.
    claimed = run_root.is_dir()
    execution_id = new_execution_id()
    if claimed:
        try:
            _ = claim_run_root(ds, execution_id, run_root, kind, 0)
        except RunRootHeld as held:
            raise _in_use(kind, run_id, held.held) from None
    try:
        dropped = index_records(index.drop_runs([run_id]))
        if not dropped:
            raise _not_found(kind, run_id, "its row was dropped by another caller")
        if not claimed:
            return DeletedModel(row=dropped[0], directory_removed=False)
        try:
            empty_claimed_run_root(run_root)
            clear_inflight(run_root, execution_id=execution_id)
            run_root.rmdir()
        except OSError as exc:
            return DeletedModel(
                row=dropped[0], directory_removed=False, removal_error=str(exc)
            )
        return DeletedModel(row=dropped[0], directory_removed=True)
    finally:
        if claimed and run_root.is_dir():
            clear_inflight(run_root, execution_id=execution_id)


def _refuse_unless_a_model_run(kind: str, run_id: str) -> None:
    """Raise unless *kind* holds trained models and *run_id* is one of its runs.

    This also keeps both arguments one path component each. A parsed run
    identifier cannot contain a separator or ``..``, and *kind* must be the kind
    it parses to.
    """
    if not holds_trained_models(kind):
        message = (
            f"models/{kind} holds prepared training data, not trained models, so "
            f"it has no model to delete."
        )
        raise ModelDeletionRefusedError(
            message, reason="prepared_data", kind=kind, run_id=run_id
        )
    parsed = parse_op_run_id(run_id)
    if parsed is None or parsed.kind != kind:
        message = f"{run_id!r} is not a run identifier of {kind}."
        raise ModelDeletionRefusedError(
            message, reason="not_a_run_of_kind", kind=kind, run_id=run_id
        )


def _refuse_unless_under_models(
    ds: Dataset, kind: str, run_id: str, run_root: Path
) -> None:
    """Raise unless *run_root* is a directory of its own inside ``models``.

    A link is refused, because emptying the root through it would empty its
    target, and a claim written through it would be written there. Containment is
    judged on the directory containing the root, so a leaf link cannot pass on the
    strength of its target.
    """
    if run_root.is_symlink():
        message = (
            f"{run_root} is a link rather than a run root of its own; not deleting "
            "through it."
        )
        raise ModelDeletionRefusedError(
            message, reason="run_root_is_a_link", kind=kind, run_id=run_id
        )
    models_root = ds.get_root("models").resolve()
    holder = run_root.parent.resolve()
    if models_root not in holder.parents:
        message = (
            f"{run_root.parent} resolves to {holder}, outside the models root "
            f"{models_root}; not deleting from it."
        )
        raise ModelDeletionRefusedError(
            message, reason="outside_models_root", kind=kind, run_id=run_id
        )


def _in_use(
    kind: str, run_id: str, held: InflightMarker | None
) -> TrainedModelInUseError:
    if held is None:
        message = f"{kind} run {run_id} is held by another execution; not deleting it."
        return TrainedModelInUseError(
            message, kind=kind, run_id=run_id, execution_id="", host="", expires_at=""
        )
    message = (
        f"{kind} run {run_id} is held by execution {held.execution_id} on "
        f"{held.host}:{held.pid} until {held.expires_at or 'an unknown time'}; not "
        "deleting it while that runs."
    )
    return TrainedModelInUseError(
        message,
        kind=kind,
        run_id=run_id,
        execution_id=held.execution_id,
        host=held.host,
        expires_at=held.expires_at,
    )


def _not_found(kind: str, run_id: str, why: str) -> ModelNotFoundError:
    return ModelNotFoundError(
        f"No {kind} model {run_id} to delete: {why}.",
        reference=run_id,
        model_kind=kind,
    )
