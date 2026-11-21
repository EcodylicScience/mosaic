"""The op ``retire-recipes``: an approved retirement of saved recipes, run as a job.

A retirement removes the feature runs that retired recipes wrote and that no
rule keeps. mosaic-api plans it, a person approves the plan, and the queue runs
the approval as this op, the way it runs every op that is not a recipe step
(``mosaic run --kind retire-recipes --params ...``). The op plans and judges
again before it removes anything
(:func:`~mosaic.core.pipeline.graph.retire.run_retirement`), so a run that
stopped being removable since the approval is kept.

The attempt's run-log records each removal and each run kept, and a removal that
raised is a lost entry, so ``mosaic run`` reports ``partial`` for it. A recipe
cannot run this op (``EXCLUDED_KINDS``).

Params past the length of one command-line argument go to ``mosaic run`` as
``--params @<file>`` or ``--params @-``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.graph.model import RetirementSpec
from mosaic.core.pipeline.op_identity import op_run_id
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope
    from mosaic.core.pipeline.job import JobContext

__all__ = ["RetireRecipesOp", "RetireRecipesParams"]

_RETIREMENT_DESCRIPTION = (
    "The approved retirement: the retired recipes' digests, the pipelines that "
    "stay, the policy the plan judged under, the runs the plan found removable "
    "and the instant it planned at. The run plans and judges again, and removes "
    "only an approved run that is still removable."
)

_MANAGER_DESCRIPTION = (
    "The manager the retirement acts for. A dataset managed by that manager has "
    "its runs removed. Empty acts for no manager, so a managed dataset keeps "
    "every run."
)


class RetireRecipesParams(Params):
    """Parameters of one approved retirement."""

    retirement: Annotated[RetirementSpec, Declared(_RETIREMENT_DESCRIPTION)]
    manager: Annotated[str, Declared(_MANAGER_DESCRIPTION)] = ""


@register_op
class RetireRecipesOp(Op[RetireRecipesParams]):
    """Remove the runs an approved retirement of recipes found removable.

    The run id names the approval. ``overwrite`` is ignored: a rerun judges
    again, keeps what is no longer removable and removes nothing twice.
    """

    kind = "retire-recipes"
    domain = "pipeline"
    category = "cleanup"
    version = "0.1"
    scope_takes = "none"
    scope_dependent = False
    Params = RetireRecipesParams

    def plan_identity(
        self,
        ds: "Dataset",
        params: RetireRecipesParams,
        scope: "ResolvedScope",
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """What this retirement will be called, from its params alone."""
        return OpIdentity(
            run_id=op_run_id(self.kind, self.version, params.identity_dump())
        )

    def run(
        self,
        ds: "Dataset",
        params: RetireRecipesParams,
        scope: "ResolvedScope",
        overwrite: bool,
        ctx: "JobContext",
    ) -> str:
        from mosaic.core.pipeline.graph.retire import run_retirement

        run_id = self.plan_identity(ds, params, scope).run_id
        ctx.set_run_id(run_id)
        _ = run_retirement(ds, params.retirement, params.manager, ctx=ctx)
        return run_id
