"""The pytest configuration itself, asserted rather than assumed.

Settings in ``[tool.pytest.ini_options]``, and the CI workflows' selection by its
markers, are load-bearing in ways that fail silently when they drift, so each
gets a test naming what it protects.
"""

from __future__ import annotations

import shlex
import tomllib
from dataclasses import dataclass

import pytest
import yaml
from pydantic import BaseModel, Field

from tests.helpers import REPO_ROOT, SEPARATED_TOOLS, CiJob

PYPROJECT = REPO_ROOT / "pyproject.toml"
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
_WORKFLOWS_RUNNING_PYTEST = ("ci.yml", "slow.yml")

_UNCLAIMED_MARKERS = frozenset({"slow", "media"})
"""The markers that no CI job selects by name.

``slow`` divides each job's selection by cost. It also selects ``slow.yml``,
the only workflow that builds VAME's environment. A test needing VAME's
environment is therefore marked ``slow``. ``media`` marks a test needing the
ffmpeg toolchain, which every job installs. Every other marker is a job marker:
it names the job whose environment its tests need.
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
    env: dict[str, object] = {}
    working_directory: str = Field(default="", alias="working-directory")

    @property
    def arguments(self) -> list[str]:
        """The words of :attr:`run`, without its comments."""
        return shlex.split(self.run, comments=True)


class _Job(BaseModel):
    steps: list[_Step] = []


class _Workflow(BaseModel):
    jobs: dict[str, _Job]


@dataclass(frozen=True, slots=True)
class _PytestJob:
    """A job that runs pytest.

    Attributes:
        job: The job.
        step: The step that runs pytest.
        expression: The ``-m`` expression that the step passes. Empty when it
            passes none.
    """

    job: _Job
    step: _Step
    expression: str


def _pytest_jobs(workflow: str) -> dict[str, _PytestJob]:
    """Each job of *workflow* that runs pytest, by name."""
    text = (WORKFLOWS / workflow).read_text()
    document = _Workflow.model_validate(yaml.safe_load(text))
    found: dict[str, _PytestJob] = {}
    for name, job in document.jobs.items():
        for step in job.steps:
            arguments = step.arguments
            if "pytest" not in arguments:
                continue
            selects = "-m" in arguments
            expression = arguments[arguments.index("-m") + 1] if selects else ""
            found[name] = _PytestJob(job, step, expression)
    return found


def _marker_expressions(workflow: str) -> dict[str, str]:
    """Each job of *workflow* that runs pytest, and the ``-m`` expression it passes.

    A job passing no ``-m`` maps to the empty string.
    """
    return {name: job.expression for name, job in _pytest_jobs(workflow).items()}


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


def _building_jobs() -> list[CiJob]:
    """The CI jobs that build an environment of a separated tool."""
    built = {
        environment.built_by
        for tool in SEPARATED_TOOLS
        for environment in tool.environments
        if environment.built_by is not None
    }
    return sorted(built, key=lambda job: job.variable)


def _job_id(job: CiJob) -> str:
    return job.variable


@pytest.mark.parametrize("ci_job", _building_jobs(), ids=_job_id)
def test_the_job_selected_by_a_building_marker_builds_and_requires_the_environment(
    ci_job: CiJob,
) -> None:
    """Checks the job that ``SEPARATED_TOOLS`` names against the workflows.

    The job whose ``-m`` expression selects the marker builds each environment
    that the registry records it building, and its pytest step sets the
    variable. ``tests/conftest.py`` reads that variable and refuses to start
    when an environment does not resolve. Without the variable, or without the
    build, the tests that need the environment skip and the job passes.
    """
    selecting = {
        f"{workflow}:{name}": job
        for workflow in _WORKFLOWS_RUNNING_PYTEST
        for name, job in _pytest_jobs(workflow).items()
        if ci_job.marker in _terms(job.expression)
    }
    assert len(selecting) == 1, (
        f"jobs selecting {ci_job.marker!r}: {sorted(selecting)}. The registry "
        f"records one job that builds environments under {ci_job.variable}."
    )
    [(name, selected)] = selecting.items()

    assert ci_job.variable in selected.step.env, (
        f"{name} does not set {ci_job.variable} in its pytest step. The tests "
        "that need the environment then skip, and the job passes."
    )
    builds = [
        step.working_directory
        for step in selected.job.steps
        if step.arguments[:2] == ["uv", "sync"]
    ]
    unbuilt = sorted(
        environment.directory
        for tool in SEPARATED_TOOLS
        for environment in tool.environments
        if environment.built_by == ci_job
        and not any(
            build.endswith(f"src/mosaic/{environment.directory}") for build in builds
        )
    )
    assert not unbuilt, (
        f"{name} does not build {unbuilt}. It runs `uv sync` in {builds}."
    )


def test_an_empty_parameter_set_fails_collection() -> None:
    """A parametrize over nothing would otherwise report one skip and check nothing.

    ``test_tools_import`` parametrizes over the scripts under ``tools/``, and
    pointed at a directory that is not there it would find none.
    """
    assert _ini_options()["empty_parameter_set_mark"] == "fail_at_collect"
