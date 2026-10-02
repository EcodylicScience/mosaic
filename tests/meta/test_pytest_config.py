"""The pytest configuration itself, asserted rather than assumed.

Settings in ``[tool.pytest.ini_options]``, and the CI workflows' selection by its
markers, are load-bearing in ways that fail silently when they drift, so each
gets a test naming what it protects.
"""

from __future__ import annotations

import shlex
import tomllib

import pytest
import yaml
from pydantic import BaseModel

from tests.helpers import REPO_ROOT

PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

_UNCLAIMED_MARKERS = frozenset({"slow", "media"})
"""The markers that no CI job selects by name.

``slow`` divides each job's selection by cost rather than by environment, and
``media`` marks a test needing the ffmpeg toolchain, which every job installs.
Every other marker is a job marker: it names the job whose environment its tests
need.
"""


def _ini_options() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        config = tomllib.load(handle)
    tool = config["tool"]
    assert isinstance(tool, dict)
    options = tool["pytest"]["ini_options"]
    assert isinstance(options, dict)
    return options


def _declared_markers() -> list[str]:
    """The ``markers`` entries, as strings.

    ``tomllib`` types every value as ``object``, so the narrowing happens once
    here rather than at each use.
    """
    markers = _ini_options()["markers"]
    assert isinstance(markers, list)
    return [str(entry) for entry in markers]


class _Step(BaseModel):
    run: str = ""


class _Job(BaseModel):
    steps: list[_Step] = []


class _Workflow(BaseModel):
    jobs: dict[str, _Job]


def _marker_expressions(workflow: str) -> dict[str, str]:
    """Each job of *workflow* that runs pytest, and the ``-m`` expression it passes.

    A job passing no ``-m`` maps to the empty string.
    """
    text = (WORKFLOWS / workflow).read_text()
    document = _Workflow.model_validate(yaml.safe_load(text))
    expressions: dict[str, str] = {}
    for name, job in document.jobs.items():
        for step in job.steps:
            arguments = shlex.split(step.run, comments=True)
            if "pytest" not in arguments:
                continue
            selects = "-m" in arguments
            expressions[name] = arguments[arguments.index("-m") + 1] if selects else ""
    return expressions


def _terms(expression: str) -> set[str]:
    """The terms of an ``-m`` expression joined by ``and``."""
    return {term.strip() for term in expression.split(" and ")}


def _job_markers() -> set[str]:
    declared = {entry.split(":", 1)[0] for entry in _declared_markers()}
    return declared - _UNCLAIMED_MARKERS


def test_the_default_invocation_deselects_slow() -> None:
    """A bare ``pytest`` must stay the fast run, and it is one option away.

    pytest takes the *last* ``-m`` it is given rather than intersecting them, so
    the day someone runs ``pytest -m "not media"`` the ``slow`` deselection is
    gone and the keypoint-MoSeq integration suite silently joins the run. That is
    a four-minute suite becoming a much longer one with nothing on screen to say
    why, which is exactly the kind of drift a comment does not survive.
    """
    addopts = _ini_options()["addopts"]
    assert isinstance(addopts, list), (
        "addopts must be a list; the string form hides the quoting around "
        "`not slow` and reads as one argument containing a space"
    )
    assert ["-m", "not slow"] == addopts[:2], (
        f"addopts no longer begins with the slow deselection: {addopts}. "
        "A bare `pytest` is documented in CLAUDE.md as running everything except "
        "the slow tests."
    )


def test_every_marker_used_in_the_suite_is_declared() -> None:
    """``--strict-markers`` only helps if the declarations stay complete.

    It turns an undeclared marker into an error, which is the point -- a typo
    like ``@pytest.mark.slwo`` would otherwise attach a marker nobody selects on
    and leave the test running in every invocation it was meant to be excluded
    from. This asserts the other half: that the declared set is the one the
    suite's own markers need, so adding a marker without declaring it fails here
    with a readable message rather than at collection with a bare ``UsageError``.
    """
    declared = {entry.split(":", 1)[0] for entry in _declared_markers()}
    assert {"slow", "media", "tracker", "identity", "feral"} <= declared


@pytest.mark.parametrize("marker", ["slow", "media", "tracker", "identity", "feral"])
def test_each_declared_marker_carries_a_description(marker: str) -> None:
    """A bare name in ``markers`` tells ``pytest --markers`` nothing."""
    entry = next(m for m in _declared_markers() if m.split(":", 1)[0] == marker)
    _, _, description = entry.partition(":")
    assert description.strip(), f"marker {marker!r} is declared with no description"


def test_the_import_mode_is_pinned_to_prepend() -> None:
    """``from tests.helpers import`` must resolve to this checkout's ``tests``.

    Under ``prepend`` the repository root leads ``sys.path``. An installed
    distribution may ship a top-level ``tests`` package of its own, as an
    Ultralytics wheel does, and under ``importlib`` that package can answer the
    import instead.
    """
    addopts = _ini_options()["addopts"]
    assert isinstance(addopts, list)
    assert "--import-mode=prepend" in addopts


def test_each_job_marker_is_selected_by_exactly_one_ci_job() -> None:
    """A job marker that no job selects runs nowhere its environment exists.

    One that two jobs select reports each failure twice, which is how the
    Ultralytics runner's tests came to fail in both ``test`` and ``tracking``.
    """
    expressions = _marker_expressions("ci.yml")
    for marker in sorted(_job_markers()):
        selecting = sorted(job for job, expr in expressions.items() if expr == marker)
        assert len(selecting) == 1, f"ci.yml jobs selecting {marker!r}: {selecting}"


def test_the_test_job_runs_what_no_job_marker_claims() -> None:
    """``test`` deselects ``slow`` and every job marker, so each test runs once."""
    expected = {f"not {marker}" for marker in {"slow", *_job_markers()}}

    assert _terms(_marker_expressions("ci.yml")["test"]) == expected


def test_the_slow_workflow_runs_the_slow_tests_no_job_marker_claims() -> None:
    """Every slow test runs in CI: in its marker's job, or else in ``slow.yml``."""
    expected = {"slow", *(f"not {marker}" for marker in _job_markers())}

    assert _terms(_marker_expressions("slow.yml")["slow"]) == expected


def test_an_empty_parameter_set_fails_collection() -> None:
    """A parametrize over nothing would otherwise report one skip and check nothing.

    ``test_tools_import`` parametrizes over the scripts under ``tools/``, and
    pointed at a directory that is not there it would find none.
    """
    assert _ini_options()["empty_parameter_set_mark"] == "fail_at_collect"
