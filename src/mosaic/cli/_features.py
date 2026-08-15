"""Build a runnable ``Feature`` instance from a slug + JSON inputs/params.

``run_feature`` takes a *constructed* feature instance, and the library has no
"params-dict -> feature-instance" helper -- so the CLI owns that construction.
Every feature follows the uniform ``FeatureCls(inputs, params: dict | None)``
shape (``self.params = self.Params.from_overrides(params)``), which is exactly
what notebooks use. Passing ``--params`` straight through ``from_overrides``
reconstructs artifact-in-params dependencies (e.g. scaler/tsne ``templates``)
from plain dicts, giving full generality.

Two footguns worth surfacing to users (documented in ``describe`` / the README):
- Artifact refs default their glob to ``*.parquet``; a producer that emits more
  than one parquet (e.g. ``extract-templates``) needs the ref to pin ``pattern``
  (``{"feature": "extract-templates", "run_id": null, "pattern": "templates.parquet"}``)
  or the pipeline silently resolves the wrong file.
- ``GlobalModelParams`` requires exactly one of ``templates`` / ``model``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mosaic.cli._io import fail

if TYPE_CHECKING:
    from mosaic.core.pipeline.types import Feature


def available_slugs() -> list[str]:
    """Sorted list of every registered feature slug (for error messages/help)."""
    from mosaic.behavior.feature_library import FEATURES

    slugs: list[str] = []
    for cls in FEATURES.values():
        name_raw = getattr(cls, "name", None)
        slugs.append(name_raw if isinstance(name_raw, str) else cls.__name__)
    return sorted(slugs)


def feature_class_for_slug(slug: str) -> "type[Feature]":
    """Resolve a discovery slug (``cls.name``) to its feature class, or exit.

    The lookup itself lives in the graph package, which is where a recipe and a
    lane both need it too; this adds the CLI's own refusal, which lists what is
    available. Three copies of the scan existed -- here, in recipe resolution,
    and in mosaic-api's lane routing -- and a fourth would have been written the
    next time something had to turn a slug into a class.
    """
    from mosaic.core.pipeline.graph import feature_class_for_slug as lookup

    found = lookup(slug)
    if found is None:
        fail(f"Unknown feature '{slug}'. Available: {', '.join(available_slugs())}")
    return cast("type[Feature]", found)


def build_feature(
    slug: str,
    inputs_json: object | None,
    params_dict: object | None,
) -> "Feature":
    """Construct a runnable feature instance from a slug + JSON inputs/params.

    The construction itself lives in the graph package, which is where a recipe
    is validated and where a plan hashes one -- three constructions would be
    three answers to what a step means. What is here is the CLI's own refusal:
    the available slugs for an unknown one, and which flag to look at for the
    other two, phrased off the ``stage`` the error carries rather than by
    reading its text.
    """
    from mosaic.core.pipeline.graph import StepBuildError
    from mosaic.core.pipeline.graph import build_feature as construct

    try:
        return construct(slug, inputs_json, params_dict)
    except StepBuildError as exc:
        if exc.stage == "slug":
            fail(f"Unknown feature '{slug}'. Available: {', '.join(available_slugs())}")
        if exc.stage == "params":
            fail(f"Invalid --params for '{slug}': {exc.detail}")
        if inputs_json is None:
            fail(
                f"Feature '{slug}' does not read from tracks by default; pass "
                f'--inputs (e.g. --inputs \'[{{"feature":"<upstream-slug>"}}]\').'
            )
        fail(f"Invalid --inputs for '{slug}': {exc.detail}")
