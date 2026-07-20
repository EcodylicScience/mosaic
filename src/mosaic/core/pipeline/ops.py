"""Registry of ops under the Job Contract.

Every long-running operation that rides the Job Contract -- frame extraction,
pose/point/localizer training, pose/point/localizer inference, annotation
conversion, TREx, media transcode, and future domains -- is an ``Op``: a class
carrying a ``kind`` (its ``runs.kind``), a ``domain`` (which package owns it),
a Pydantic ``Params`` model, and a ``run(ds, params, ctx)`` body that computes
a content ``run_id``, does the work, writes an index row, and returns the
``run_id``. Ops self-register via ``@register_op`` -- so a new op plugs in by
adding a module, with no edit to the runner, the CLI, or the API.

One generic entry point, :func:`run_op`, wraps *every* op in the Job Contract
(`core/pipeline/job.py`), so attempt-recording, progress, heartbeat, and
cooperative cancellation are written once. Because each op declares a Pydantic
``Params``, discovery is schema-driven -- ``op.Params.model_json_schema()``
gives a CLI / mosaic-api / MCP a full param spec exactly the way features are
discovered today, with zero per-op schema code.

**Registration must stay import-light.** Op modules import only their ``Params``
and light deps at module top; heavy backends (``ultralytics`` / ``torch`` /
POLO) are imported *inside* ``run()`` so importing an op package never fails
when an optional extra is absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from mosaic.core.pipeline.job import CancelToken, JobContext, job_context
from mosaic.core.pipeline.types import Params

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.progress import ProgressCallback


# ---------------------------------------------------------------------------
# Op spec + registry
# ---------------------------------------------------------------------------

P = TypeVar("P", bound=Params)


class Op(Generic[P]):
    """Base class for a registered op.

    Generic over its ``Params`` type so a subclass can narrow ``run``/``target``
    without an LSP-incompatible override. Subclasses set the class attributes,
    implement :meth:`run`, and are stateless (the registry stores the class and
    instantiates it per call).
    """

    kind: ClassVar[str]
    category: ClassVar[str]  # "extract" | "train" | "infer" | "convert" | "transcode"
    domain: ClassVar[str]  # "tracking" | "media"
    version: ClassVar[str] = "0.1"
    # Compute-placement hint for schedulers / the execution router ("gpu" | "heavy" |
    # "cpu"). Empty ("") derives it from ``category`` (train/infer -> gpu, else cpu) via
    # :func:`op_resource_class`; an op overrides it when category is misleading (e.g. TREx
    # is category "convert" but needs the GPU for YOLO detection, so it declares "gpu").
    resource_class: ClassVar[str] = ""
    Params: ClassVar[type[Params]]

    def target(self, params: P) -> str:
        """A short human label for the ``runs.target`` column."""
        return self.kind

    def run(self, ds: "Dataset", params: P, ctx: JobContext) -> str:
        """Do the work using *ctx* (progress/cancel/run_id) and return the run_id."""
        raise NotImplementedError


OPS: dict[str, type[Op[Any]]] = {}


def register_op(cls: type[Op[Any]]) -> type[Op[Any]]:
    """Class decorator: register *cls* under its ``kind``."""
    if not getattr(cls, "kind", None):
        raise ValueError(f"{cls.__name__} must define a non-empty 'kind'")
    OPS[cls.kind] = cls
    return cls


# ---------------------------------------------------------------------------
# Generic runner (the single Job-Contract wrapper for all ops)
# ---------------------------------------------------------------------------


def run_op(
    ds: "Dataset",
    kind: str,
    params: Params | dict[str, Any],
    *,
    execution_id: str | None = None,
    owner: str = "",
    track: bool = True,
    progress_callback: "ProgressCallback | None" = None,
    cancel_token: CancelToken | None = None,
) -> str:
    """Run a registered op as a tracked Job-Contract attempt.

    *params* may be a validated ``Params`` instance or a plain dict (validated
    against the op's ``Params`` model). Returns the content ``run_id``.
    """
    op_cls = OPS.get(kind)
    if op_cls is None:
        raise KeyError(f"Unknown op '{kind}'. Registered: {sorted(OPS)}")
    op = op_cls()
    p = op.Params.model_validate(params) if isinstance(params, dict) else params
    with job_context(
        ds,
        kind=kind,
        target=op.target(p),
        execution_id=execution_id,
        owner=owner,
        track=track,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    ) as ctx:
        return op.run(ds, p, ctx)


# ---------------------------------------------------------------------------
# Discovery (schema-driven; consumed by the CLI / mosaic-api / MCP)
# ---------------------------------------------------------------------------


def list_ops(
    category: str | None = None, domain: str | None = None
) -> list[dict[str, object]]:
    """Enumerate registered ops as ``{kind, domain, category, version}`` dicts."""
    ops = sorted(OPS.values(), key=lambda c: c.kind)
    return [
        {
            "kind": c.kind,
            "domain": c.domain,
            "category": c.category,
            "version": c.version,
        }
        for c in ops
        if (category is None or c.category == category)
        and (domain is None or c.domain == domain)
    ]


_CATEGORY_RESOURCE_CLASS: dict[str, str] = {
    "train": "gpu",
    "infer": "gpu",
    "extract": "cpu",
    "convert": "cpu",
    "transcode": "cpu",
}


def op_resource_class(kind: str) -> str:
    """Return an op's compute-placement class (``"gpu"`` | ``"heavy"`` | ``"cpu"``).

    Prefers the op's explicit ``resource_class`` classvar; otherwise derives it from
    ``category`` (train/infer -> gpu, else cpu). Unknown kinds fall back to ``"cpu"``. Used by
    the execution router to send GPU work (training, inference, TREx) to a GPU lane / k8s
    without any per-op routing edits.
    """
    op_cls = OPS.get(kind)
    if op_cls is None:
        return "cpu"
    declared = getattr(op_cls, "resource_class", "")
    if declared:
        return declared
    return _CATEGORY_RESOURCE_CLASS.get(op_cls.category, "cpu")


def describe_op(kind: str) -> dict[str, object]:
    """Return ``{kind, domain, category, version, params_schema}`` for one op."""
    op_cls = OPS.get(kind)
    if op_cls is None:
        raise KeyError(f"Unknown op '{kind}'. Registered: {sorted(OPS)}")
    return {
        "kind": op_cls.kind,
        "domain": op_cls.domain,
        "category": op_cls.category,
        "version": op_cls.version,
        "params_schema": op_cls.Params.model_json_schema(),
    }
