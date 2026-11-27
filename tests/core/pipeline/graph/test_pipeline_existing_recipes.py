"""The recipes that exist today keep the problems and cautions they have.

A plan-time check that reads each join is new code over recipes written before
it. These two, the guppies paper's analysis graph and the worked recipe of the
plan golden, validate today with the findings recorded below, which were read
before the check existed. A finding that appears here later is a recipe that the
check refuses or warns about and a run accepts.
"""

from __future__ import annotations

from mosaic.core.pipeline.graph import validation_report
from tests.helpers import GUPPIES_RECIPE, WORKED_RECIPE, read_golden

_ANGLE_CAUTION = (
    "'ANGLE' may be missing from its merged input. An input may contain it, "
    "and none guarantees it"
)

GUPPIES_CAUTIONS = [
    (step, "inputs", _ANGLE_CAUTION)
    for step in (
        "delta_focal_n10",
        "delta_focal_bf_n10",
        "delta_model_n10",
        "delta_focal_n30",
        "delta_focal_bf_n30",
        "delta_model_n30",
    )
]
"""The guppies recipe's cautions: each nn-delta-response step reads ``ANGLE``."""


def test_existing_recipes_gain_no_problem() -> None:
    guppies = validation_report(read_golden(GUPPIES_RECIPE))
    worked = validation_report(WORKED_RECIPE)

    assert guppies.problems == ()
    assert [(c.step, c.where, c.message) for c in guppies.cautions] == GUPPIES_CAUTIONS
    assert guppies.unchecked == ()
    assert worked.problems == ()
    assert worked.cautions == ()
    assert worked.unchecked == ()
