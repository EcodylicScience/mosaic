"""Pin every unpinned upstream reference *before* identity is computed.

A ``Result`` -- or any of its subclasses, ``ArtifactSpec``, ``NNResult``,
``BodyScaleResult``, ``ResultColumn`` -- may be written with ``run_id=None``,
meaning "whichever run of that feature is latest". That is a convenience for the
author and a hole in the identity. :func:`~mosaic.core.pipeline.run.compute_run_id`
hashes ``feature.inputs.model_dump()``, so ``None`` is what reached the digest,
and the resolution to a concrete run happened *afterwards* -- in four separate
places, each into a local that was discarded:

- ``manifest._resolve_feature`` for a ``Result`` in ``feature.inputs``
- ``run._build_result_lookup`` for a ``Result`` in a params field
- ``run._resolve_dependencies`` for an ``ArtifactSpec``
- ``loading.build_nn_lookup`` for an ``NNResult`` pair filter

The consequence is not confined to the feature graph. ``ArtifactSpec`` extends
``Result``, and for a ``GlobalModelParams`` feature the ``templates`` reference
*is* the training set. Re-run ``extract-templates`` over a different scope, then
re-run ``xgboost`` unpinned: same identifier, same directory, a classifier
trained on the old templates, reported as cached. Six global features sit on that
path -- ``global-scaler``, ``global-tsne``, ``global-kmeans``, ``global-ward``,
``xgboost``, ``lightning-action``.

**Resolution reads the filesystem; hashing does not.** That split is why this is
a separate pass rather than a branch inside ``compute_run_id``: the hash function
stays pure and callable with no dataset, which is what lets the golden corpus pin
literal identifiers and what lets the control plane predict a ``run_id`` before
spawning work.

**It mutates the references in place.** ``Result`` and its subclasses are
``StrictModel`` -- ``extra="forbid"`` and nothing else, so no ``frozen`` and no
``validate_assignment`` -- and in-place mutation of a reference is already the
house idiom (``ArtifactSpec._derive_pattern`` writes ``self.pattern``;
``GlobalModelParams._exclusive_source`` writes ``self.templates``). Rebuilding
instead would mean assigning ``feature.inputs``, which the ``Feature`` protocol
declares read-only. Writing ``item.run_id`` on an element of the existing tuple
touches neither the tuple nor the attribute.

Two consequences of mutating rather than copying, both intended:

- The caller's reference objects are pinned as a side effect. Within one run that
  is the point -- ``_resolve_dependencies`` later reads the same objects, so it
  cannot pick a *different* "latest" than the one identity recorded, which a
  copy-based design would leave open as a race.
- The pinned value survives the process-worker round trip (``run.py`` dumps
  ``feature.inputs.model_dump()`` and the worker rebuilds it), because ``run_id``
  is a plain field of the dump.

**The ``"tracks"`` literal is deliberately untouched.** There is no tracks
identity to pin yet; implementation item 3.3 resolves it once ``tracks/`` is
variant-addressed. Pinning it to anything that varies with which sequences are on
disk would move a scope-free feature's identifier every time a sequence is added,
which is exactly what workflow H5 forbids.

**A reference to a feature that has not run yet stays unpinned**, and is reported
as ``run_id=None``. It must not raise: the chain runner previews a cold dataset
step by step, and ``build_manifest`` is documented to return an empty scope for a
not-yet-computed upstream rather than failing. The reference is then resolved --
or reported missing -- later, by whichever consumer needs the file.

Only top-level params fields are scanned, matching ``_resolve_dependencies`` and
``Params.identity_dump()``, neither of which recurses into a nested model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard

from .track_universe import current_run_id
from .types import Feature, Result

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["Resolution", "resolution_payload", "resolve_references"]


@dataclass(frozen=True, slots=True)
class Resolution:
    """One reference and the run it was pinned to.

    Attributes:
        where: Where the reference lives on the feature -- ``inputs[<i>]`` or
            ``params.<field>``. Names the site rather than the object so a
            reader of ``params.json`` can find it again.
        feature: Storage name of the upstream feature.
        run_id: The concrete run, or None when the upstream has not run yet.
            An honest None beats a confident wrong value.
    """

    where: str
    feature: str
    run_id: str | None


def resolve_references(ds: Dataset, feature: Feature) -> tuple[Resolution, ...]:
    """Pin every unpinned reference on *feature*, and report what was pinned.

    Mutates ``run_id`` on each ``Result``-shaped reference reachable from
    ``feature.inputs`` and from the top-level fields of ``feature.params``.
    Already-pinned references are left exactly as the caller wrote them --
    resolution never overrides an explicit choice, and re-running the pass is a
    no-op.

    Call it *before* ``compute_run_id``. Afterwards is the defect it fixes.

    Args:
        ds: Dataset whose feature indexes name the available runs.
        feature: The feature about to be run. Modified in place.

    Returns:
        One :class:`Resolution` per reference examined, in the order examined:
        inputs first, then params fields in declaration order. References with
        an empty feature name are skipped and absent from the report -- that is
        the unset ``GlobalModelParams.templates`` default, not a dependency.
    """
    resolutions: list[Resolution] = []

    for position, item in enumerate(feature.inputs.root):
        # The other member of the union is the "tracks" literal (item 3.3).
        if isinstance(item, Result):
            record = _pin(ds, item, f"inputs[{position}]")
            if record is not None:
                resolutions.append(record)

    params = feature.params
    for field_name in type(params).model_fields:
        value: object = getattr(params, field_name)
        if _is_reference(value):
            record = _pin(ds, value, f"params.{field_name}")
            if record is not None:
                resolutions.append(record)

    return tuple(resolutions)


def resolution_payload(
    resolutions: tuple[Resolution, ...],
) -> list[dict[str, str | None]]:
    """Render *resolutions* for the ``_resolved`` section of ``params.json``.

    Provenance, never identity: it records what the identifier already covers,
    so hashing it would be redundant, and it is written to the save payload
    rather than the hash payload. It is what a reverse-dependency walk (item
    6.1) reads to answer "which templates run did this classifier train on".
    """
    return [
        {"where": r.where, "feature": r.feature, "run_id": r.run_id}
        for r in resolutions
    ]


def _is_reference(value: object) -> TypeGuard[Result[str]]:
    """Is *value* a run reference?

    ``isinstance`` alone leaves the type argument unknown, because a bare
    ``object`` gives the checker nothing to infer it from. ``Result``'s TypeVar
    is bound to ``str`` and defaults to it, so ``Result[str]`` is the only
    argument it can have -- stated once here rather than at each call site.
    """
    return isinstance(value, Result)


def _pin(ds: Dataset, reference: Result[str], where: str) -> Resolution | None:
    """Fill in *reference*'s ``run_id`` if it is unset; report either way."""
    feature_name = reference.feature
    if not feature_name:
        return None
    if reference.run_id is None:
        reference.run_id = _latest_run_id(ds, feature_name)
    return Resolution(where=where, feature=feature_name, run_id=reference.run_id)


def _latest_run_id(ds: Dataset, feature_name: str) -> str | None:
    """The upstream's current run, or None when it has not run here yet.

    Uses the same rule as every consumer of an unpinned reference --
    ``track_universe.current_run_id`` -- so pinning cannot change *which* run a run
    would have read. That sentence used to be here and be false: this sorted on the
    recorded timestamps while the query path walked the chain, so the two disagreed
    on exactly the dataset the chain walk exists for.

    ``FileNotFoundError`` is a missing index (the feature has never run in this
    dataset); ``ValueError`` is an index with no rows (created by ``ensure()``,
    not yet written to). Both mean the same thing here.
    """
    try:
        return current_run_id(ds, feature_name)
    except (FileNotFoundError, ValueError):
        return None
