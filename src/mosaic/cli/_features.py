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

from typing import TYPE_CHECKING

from mosaic.cli._io import fail

if TYPE_CHECKING:
    from pydantic import ValidationError

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
    """Resolve a discovery slug (``cls.name``) to its feature class.

    ``FEATURES`` is keyed by class name, so we scan values by ``.name``.
    """
    from mosaic.behavior.feature_library import FEATURES

    for cls in FEATURES.values():
        if getattr(cls, "name", None) == slug:
            return cls
    fail(f"Unknown feature '{slug}'. Available: {', '.join(available_slugs())}")


def build_feature(
    slug: str,
    inputs_json: object | None,
    params_dict: object | None,
) -> "Feature":
    """Construct a runnable feature instance from a slug + JSON inputs/params."""
    from pydantic import BaseModel, ValidationError

    cls = feature_class_for_slug(slug)

    inputs_cls = getattr(cls, "Inputs", None)
    if not (isinstance(inputs_cls, type) and issubclass(inputs_cls, BaseModel)):
        fail(f"Feature '{slug}' has no Inputs model.")

    # Build the Inputs via model_validate (works for both the default tracks
    # payload and an explicit --inputs list; RootModel validates the root value).
    inputs_payload: object = ["tracks"] if inputs_json is None else inputs_json
    if inputs_json is not None and not isinstance(inputs_json, list):
        fail(
            '--inputs must be a JSON array, e.g. ["tracks"] or [{"feature":"speed-angvel"}].'
        )
    try:
        inputs_obj = inputs_cls.model_validate(inputs_payload)
    except ValidationError as exc:
        if inputs_json is None:
            fail(
                f"Feature '{slug}' does not read from tracks by default; pass "
                f'--inputs (e.g. --inputs \'[{{"feature":"<upstream-slug>"}}]\').'
            )
        fail(f"Invalid --inputs for '{slug}': {_compact(exc)}")

    if params_dict is not None and not isinstance(params_dict, dict):
        fail("--params must be a JSON object.")

    try:
        # Uniform feature constructor: (inputs, params_dict). Feature is a
        # Protocol (no declared __init__), hence the targeted ignore.
        return cls(inputs_obj, params_dict)  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        fail(f"Invalid --params for '{slug}': {_compact(exc)}")


def _compact(exc: "ValidationError") -> str:
    """Render a pydantic ValidationError as ``field: message; ...``."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = str(err.get("msg", ""))
        parts.append(f"{loc}: {msg}" if loc else msg)
    return "; ".join(parts)
