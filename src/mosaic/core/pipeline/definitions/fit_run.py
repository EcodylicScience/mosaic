"""The fit runs that a definition's ``model`` may name, and the refusals of a reference.

A definition that declares ``fits = True`` and is given a ``model`` applies one
fit of the same version over the same inputs, named by its run id. Fit and apply
runs share the definition's storage. A reference without a run id is therefore
refused, and the refusal names the fits it could name.

Every read goes through the lookup of ``ds.lookup_order("features")``: a fit that
a linked dataset holds is a candidate, written ``<run_id> (link <link id>)``.
Without a candidate, an unreachable link that provides features raises
:class:`~mosaic.core.dataset.LinkUnreachableError`, because fitting again would
compute the fit that the offline link may hold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions.loader import (
    applied_definition,
)
from mosaic.core.pipeline.definitions.store import (
    MODEL_FIELD,
    DefinitionModelMissing,
    DefinitionModelRefused,
)
from mosaic.core.pipeline.fit_scope import FIT_SCOPE_NAME
from mosaic.core.pipeline.index import feature_index_path, feature_run_file
from mosaic.core.pipeline.inventory.params import read_linked_run_params
from mosaic.core.pipeline.storage_names import derive_storage_name

if TYPE_CHECKING:
    from mosaic.core.pipeline.types.feature import Feature
    from mosaic.core.pipeline.types.results import DefinitionModel

__all__ = [
    "FitCandidate",
    "FitRunReason",
    "fit_run_candidates",
    "is_apply_text",
    "other_feature_text",
    "other_version_text",
    "refuse_unpinned_model",
    "refuse_unusable_fit_run",
    "unpinned_text",
]

type FitRunReason = Literal[
    "unpinned", "missing", "unfitted", "other_feature", "is_apply", "other_version"
]
"""Why a run refuses the fit run that a ``model`` reference leads to."""

_FITTING_AGAIN: Final = "Fitting now would compute again a fit it may hold."
"""What a refusal names as the cost of answering while a link is unreachable."""


@dataclass(frozen=True, slots=True)
class FitCandidate:
    """A fit run that an unpinned ``model`` could name.

    Attributes:
        run_id: The fit's run id.
        link_id: The link of the dataset that holds the run, ``""`` for the
            reading dataset itself.
    """

    run_id: str
    link_id: str

    @property
    def label(self) -> str:
        """The run id, followed by its link when a linked dataset holds the run."""
        return (
            self.run_id if not self.link_id else f"{self.run_id} (link {self.link_id})"
        )


def fit_run_candidates(
    ds: Dataset, storage: str, digest: str
) -> tuple[FitCandidate, ...]:
    """The fits in *storage* made by the version *digest*, sorted by run id.

    A fit is a run root whose ``params.json`` records the version *digest* and no
    ``model``, and which holds ``fit_scope.json``, the file a fit writes after it
    saves its state. The run roots are listed from the storage's directory in
    each dataset of the lookup order, not from an index, since a fit that saved
    its state may have no index row yet. A run id is listed once, from the first
    dataset that holds it.

    Args:
        ds: The dataset reading.
        storage: The definition's storage directory.
        digest: The full digest of the applying version.

    Returns:
        The candidates, sorted by run id.
    """
    holders: dict[str, str] = {}
    for linked in ds.lookup_order("features"):
        directory = feature_index_path(linked.dataset, storage).parent
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_dir():
                _ = holders.setdefault(child.name, linked.link_id)
    return tuple(
        FitCandidate(run_id, holders[run_id])
        for run_id in sorted(holders)
        if _is_fit_of(ds, storage, run_id, digest)
    )


def _is_fit_of(ds: Dataset, storage: str, run_id: str, digest: str) -> bool:
    """Whether the run *run_id* is a saved fit of the version *digest*."""
    params = read_linked_run_params(ds, storage, run_id).params
    if params is None or params.definition is None:
        return False
    if params.definition.digest != digest or params.params.get(MODEL_FIELD) is not None:
        return False
    return feature_run_file(ds, storage, run_id, FIT_SCOPE_NAME) is not None


def refuse_unpinned_model(ds: Dataset, feature: Feature) -> None:
    """Refuse a ``model`` of *feature* that names no run.

    It acts only on a definition given a ``model`` whose ``run_id`` is ``None``,
    and returns without a read for a pinned reference, a built-in and a feature
    given no model.

    Args:
        ds: The dataset reading.
        feature: The feature about to be resolved.

    Raises:
        DefinitionModelRefused: The reference names another storage than
            *feature*'s own (``other_feature``).
        LinkUnreachableError: No dataset read holds a fit of the version, and a
            link that provides features is unreachable.
        DefinitionModelMissing: The reference names no run (``unpinned``), with
            the fits of the version as its candidates.
    """
    applied = applied_definition(feature)
    if applied is None or applied[0].run_id is not None:
        return
    model, definition = applied
    name = definition.name
    storage = derive_storage_name(feature.name, feature.inputs.storage_suffix())
    if model.feature != storage:
        raise _other_feature(name, model.feature, None, storage)
    offline = ds.fresh_unreachable_links("features")
    candidates = fit_run_candidates(ds, storage, definition.digest)
    if not candidates:
        ds.refuse_while_unreachable(
            "features",
            offline,
            f"whether def:{name} has a fit of this version",
            _FITTING_AGAIN,
        )
    message = unpinned_text(
        name, storage, [candidate.label for candidate in candidates]
    )
    raise DefinitionModelMissing(
        name,
        model.feature,
        None,
        "unpinned",
        message,
        candidates=[candidate.run_id for candidate in candidates],
    )


def refuse_unusable_fit_run(
    ds: Dataset, model: DefinitionModel, *, name: str, digest: str, storage: str
) -> None:
    """Refuse a pinned ``model`` that names no saved fit of the applying version.

    The run named is read through the lookup of ``ds.lookup_order("features")``:
    its ``params.json`` and the presence of its ``fit_scope.json``. A reference
    without a run id is :func:`refuse_unpinned_model`'s, and returns here.

    Args:
        ds: The dataset reading.
        model: The reference, which names a run.
        name: The applying definition's name.
        digest: The full digest of the applying version.
        storage: The applying step's storage directory.

    Raises:
        DefinitionModelRefused: The reference names another storage, or a run of
            another definition (``other_feature``), a run that applies a model
            itself (``is_apply``) or a fit of another version (``other_version``).
        LinkUnreachableError: No dataset read holds the run or its saved fit, and
            a link that provides features is unreachable.
        DefinitionModelMissing: No dataset read holds the run (``missing``) or its
            saved fit (``unfitted``).
    """
    run_id = model.run_id
    if run_id is None:
        return
    offline = ds.fresh_unreachable_links("features")
    if model.feature != storage:
        raise _other_feature(name, model.feature, run_id, storage)
    params = read_linked_run_params(ds, storage, run_id).params
    if params is None:
        ds.refuse_while_unreachable(
            "features",
            offline,
            f"whether run {run_id} of {storage} exists",
            _FITTING_AGAIN,
        )
        raise DefinitionModelMissing(
            name,
            storage,
            run_id,
            "missing",
            f"def:{name}'s 'model' names run {run_id} of {storage}, which does not "
            "exist. A step given a model applies it and fits nothing, so run the fit "
            "first",
        )
    fitted_by = params.definition
    if fitted_by is None or fitted_by.name != name:
        raise _other_feature(name, model.feature, run_id, storage)
    if params.params.get(MODEL_FIELD) is not None:
        raise DefinitionModelRefused(
            name,
            storage,
            run_id,
            "is_apply",
            is_apply_text(name, f"run {run_id}"),
        )
    if fitted_by.digest != digest:
        raise DefinitionModelRefused(
            name,
            storage,
            run_id,
            "other_version",
            other_version_text(name, f"run {run_id}", fitted_by.digest, digest),
        )
    if feature_run_file(ds, storage, run_id, FIT_SCOPE_NAME) is None:
        ds.refuse_while_unreachable(
            "features",
            offline,
            f"whether run {run_id} of {storage} has saved its fit",
            _FITTING_AGAIN,
        )
        raise DefinitionModelMissing(
            name,
            storage,
            run_id,
            "unfitted",
            f"run {run_id} of {storage} has not saved its fit. Let the fit finish, "
            "or run it again",
        )


def unpinned_text(name: str, feature: str, candidates: Sequence[str] | None) -> str:
    """The sentence for a ``model`` that names no run of *feature*.

    Args:
        name: The applying definition's name.
        feature: The storage the reference names.
        candidates: The fits the reference could name, as written to the user.
            ``None`` when no dataset was read, so the fits are unknown.
    """
    opening = (
        f"def:{name}'s 'model' names no run of {feature}. An apply reads one fit, "
        "named by its run id"
    )
    spelling = 'reference the fit step as {"step": ...}'
    if candidates is None:
        return f"{opening}: give run_id, or {spelling}"
    if not candidates:
        return f"{opening}, and no fit of this version exists: run the fit first"
    return f"{opening}: give run_id, one of {', '.join(candidates)}, or {spelling}"


def other_feature_text(name: str, feature: str, storage: str) -> str:
    """The sentence for a ``model`` of *feature* in a step of *storage*."""
    return (
        f"def:{name}'s 'model' names {feature}, and this step stores under "
        f"{storage}. A model is applied by the definition that fitted it, over the "
        f"same inputs: name a step of def:{name} that reads them"
    )


def is_apply_text(name: str, subject: str) -> str:
    """The sentence for a ``model`` naming *subject*, which applies a model.

    *subject* reads as ``run <run id>`` or ``step '<id>'``.
    """
    return (
        f"def:{name}'s 'model' names {subject}, which applies a model itself and "
        "holds no fit. Name the step that fits"
    )


def other_version_text(name: str, subject: str, fit_digest: str, digest: str) -> str:
    """The sentence for a ``model`` naming *subject*, fitted by a version.

    Args:
        name: The applying definition's name.
        subject: ``run <run id>`` or ``step '<id>'``.
        fit_digest: The digest of the version that made the fit.
        digest: The digest of the applying version.
    """
    fit_version = fit_digest[:12]
    return (
        f"def:{name}'s 'model' names {subject}, fitted by version {fit_version} of "
        f"def:{name}, and this step runs version {digest[:12]}. An apply reads a "
        "fit of its own version: fit again with this version, or run this step as "
        f"def:{name}@{fit_version}"
    )


def _other_feature(
    name: str, feature: str, run_id: str | None, storage: str
) -> DefinitionModelRefused:
    """The refusal of a reference to *feature*, a run the step does not store."""
    return DefinitionModelRefused(
        name,
        feature,
        run_id,
        "other_feature",
        other_feature_text(name, feature, storage),
    )
