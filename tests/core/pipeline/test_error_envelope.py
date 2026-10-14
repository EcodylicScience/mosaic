"""Each classified error renders as one envelope, with the kind its class declares.

A caller branches on ``kind`` and reads ``reason``, ``step`` and ``detail`` for
the specifics. Each case pins all five fields of one error type. Each error
keeps the base that its callers already catch it by.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from mosaic.core.manifest import ManifestVersionError
from mosaic.core.pipeline.error_envelope import ErrorEnvelope, error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError
from mosaic.core.pipeline.graph import RecipeInvalid, RecipeRefused, StepBuildError
from mosaic.core.pipeline.graph.preflight import CoverageShortfall
from mosaic.core.pipeline.graph.wire import Problem
from mosaic.core.pipeline.ops import ScopeRefused
from mosaic.core.pipeline.refusal import Refusal
from mosaic.optional_dependency import MissingOptionalDependency


def test_a_step_refusal_is_refused_with_its_reason_step_and_detail() -> None:
    error = CoverageShortfall("fit", covered=2, target=3, detail="one entry failed")

    envelope = error_envelope(error)

    assert envelope == ErrorEnvelope(
        kind="refused",
        reason="coverage_shortfall",
        step="fit",
        message=str(error),
        detail={"covered": 2, "target": 3, "detail": "one entry failed"},
    )


def test_a_refusal_raised_outside_a_step_names_no_step_and_no_detail() -> None:
    error = Refusal("clip0.mp4 is av1", reason="undecodable_codec")

    envelope = error_envelope(error)

    assert envelope == ErrorEnvelope(
        kind="refused", reason="undecodable_codec", message="clip0.mp4 is av1"
    )


def test_an_invalid_recipe_lists_each_problem_at_its_step() -> None:
    error = RecipeInvalid(
        (Problem("speed", "feature", "unknown feature"), Problem("", "steps", "empty"))
    )

    envelope = error_envelope(error)

    assert (envelope.kind, envelope.reason, envelope.step) == ("invalid", None, None)
    assert envelope.message == str(error)
    assert envelope.detail == {
        "problems": [
            {"step": "speed", "where": "feature", "message": "unknown feature"},
            {"step": "", "where": "steps", "message": "empty"},
        ]
    }


def test_a_recipe_one_dataset_refuses_is_invalid_like_its_base() -> None:
    assert RecipeRefused.error_kind == "invalid"


def test_a_step_that_cannot_be_built_names_its_stage_and_field_errors() -> None:
    error = StepBuildError(
        "params", "speed-angvel", "bad params", (("params.step", "must be > 0"),)
    )

    envelope = error_envelope(error)

    assert envelope == ErrorEnvelope(
        kind="invalid",
        reason="params",
        message="speed-angvel: bad params",
        detail={
            "name": "speed-angvel",
            "errors": [{"location": "params.step", "message": "must be > 0"}],
        },
    )


def test_a_refused_scope_is_invalid() -> None:
    envelope = error_envelope(ScopeRefused("train-pose covers one prepared directory"))

    assert envelope == ErrorEnvelope(
        kind="invalid", message="train-pose covers one prepared directory"
    )


def test_a_missing_optional_dependency_is_unavailable_and_names_the_extra() -> None:
    error = MissingOptionalDependency("torch", "deep-learning", "identity models")

    envelope = error_envelope(error)

    assert envelope.kind == "unavailable"
    assert envelope.message == str(error)
    assert envelope.detail == {"module": "torch", "extra": "deep-learning"}


def test_a_manifest_newer_than_this_version_is_unavailable() -> None:
    envelope = error_envelope(ManifestVersionError("manifest_version 9 is newer"))

    assert envelope == ErrorEnvelope(
        kind="unavailable", message="manifest_version 9 is newer"
    )


@pytest.mark.parametrize(
    ("error_type", "caught_as"),
    [
        (RecipeInvalid, ValueError),
        (StepBuildError, ValueError),
        (ScopeRefused, ValueError),
        (MissingOptionalDependency, ImportError),
        (ManifestVersionError, RuntimeError),
        (CoverageShortfall, RuntimeError),
    ],
)
def test_a_classified_error_keeps_the_base_its_callers_catch(
    error_type: type[ClassifiedError], caught_as: type[Exception]
) -> None:
    assert issubclass(error_type, ClassifiedError)
    assert issubclass(error_type, caught_as)


def test_a_subclass_that_declares_no_kind_is_refused_at_definition() -> None:
    with pytest.raises(TypeError, match="no error_kind"):
        _ = type("Unclassified", (ClassifiedError,), {})


def test_a_refused_subclass_that_names_no_reason_is_refused_at_definition() -> None:
    with pytest.raises(TypeError, match="names no reason"):
        _ = type("Unreasoned", (ClassifiedError,), {"error_kind": "refused"})


def test_an_envelope_survives_a_json_round_trip() -> None:
    envelope = error_envelope(
        CoverageShortfall("fit", covered=2, target=3, detail="one entry failed")
    )

    assert ErrorEnvelope.model_validate_json(envelope.model_dump_json()) == envelope


def test_the_optional_dependency_leaf_loads_no_pydantic() -> None:
    """``optional_dependency`` subclasses the base, and stays cheap to import."""
    code = (
        "import sys\n"
        "import mosaic.optional_dependency\n"
        "loaded = sorted(m for m in sys.modules if m.startswith(('mosaic', 'pydantic')))\n"
        "print(loaded)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == repr(
        [
            "mosaic",
            "mosaic.core",
            "mosaic.core.pipeline",
            "mosaic.core.pipeline.error_kind",
            "mosaic.optional_dependency",
        ]
    )
