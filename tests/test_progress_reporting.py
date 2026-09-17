"""A denominator is a promise, and this is what holds every job to it.

``ctx.set_total`` writes the ``total`` event a client renders a progress bar
against. A job that declares one and then reports nothing leaves that bar at
``0/200`` for the length of the run -- which reads as *stuck*, not as *unknown*,
so it is worse than declaring no total at all. Two training ops were in exactly
that state for as long as they existed, and nothing anywhere would have said so:
each was correct in isolation, and what was missing was a call that is not
there to be read.

So the rule is checked across the source rather than per job. It is deliberately
structural -- what a function *contains*, not what it does at run time -- because
the alternative is driving every job to completion, and the ones that most need
the check are the ones that cost hours to run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import mosaic

SOURCE_ROOT = Path(mosaic.__file__).parent

REPORTING_CALLS = frozenset({"heartbeat", "on_epoch_end", "on_entry_end"})
"""Methods that move the reported position.

``heartbeat(done)`` for a job counting entries, ``on_epoch_end`` for a trainer
counting epochs, ``on_entry_end`` for one reporting an entry it finished. Each
lands as an event ``reduce_run_log`` folds into ``progress_done``.
"""

REPORTING_ARGUMENTS = frozenset(
    {"callback", "on_output", "on_activity", "progress_callback"}
)
"""Names under which a job hands its reporting away rather than doing it itself.

A tool running out of process reports through a callback over its output, and a
trainer running in process through a progress callback it is given. Both are
reporting; neither calls any of :data:`REPORTING_CALLS` at the site that
declared the total.
"""


def _receiver(node: ast.expr) -> str:
    """The trailing name of whatever a call was made on."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _calls(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    return [node for node in ast.walk(fn) if isinstance(node, ast.Call)]


def _declares_a_total(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Does *fn* ask a job context for a progress denominator?

    A ``set_total`` on a **run-log** is excluded: that is the mechanism writing
    the event down, one layer below the job that asked for it, and it reports
    nothing because reporting is not its job.
    """
    return any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "set_total"
        and _receiver(call.func.value) != "run_log"
        for call in _calls(fn)
    )


def _reports(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Does *fn* move the reported position, itself or through a callback?"""
    calls = _calls(fn)
    called = {call.func.attr for call in calls if isinstance(call.func, ast.Attribute)}
    passed = {kw.arg for call in calls for kw in call.keywords if kw.arg is not None}
    return bool(called & REPORTING_CALLS) or bool(passed & REPORTING_ARGUMENTS)


def _unreported_totals() -> list[str]:
    """Every function declaring a total that nothing then reports against."""
    offenders: list[str] = []
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # One hop is allowed, and only into this module's own functions: a job
        # may declare its total and hand the context to a helper that runs the
        # work, as the two Ultralytics training ops do with
        # `train_through_the_tool`. Resolving the name locally keeps that from
        # becoming "calls something, somewhere, that might report".
        helpers = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for fn in _functions(tree):
            if not _declares_a_total(fn):
                continue
            if _reports(fn):
                continue
            if any(
                isinstance(call.func, ast.Name)
                and call.func.id in helpers
                and _reports(helpers[call.func.id])
                for call in _calls(fn)
            ):
                continue
            relative = path.relative_to(SOURCE_ROOT.parent)
            offenders.append(f"{relative}:{fn.lineno} {fn.name}")
    return offenders


def test_no_job_declares_a_total_it_never_reports_against() -> None:
    """Every ``set_total`` has something moving the numerator underneath it."""
    offenders = _unreported_totals()
    assert not offenders, (
        "these declare a progress total and never report against it, so a "
        "client renders 0/total for the length of the run:\n  " + "\n  ".join(offenders)
    )


def test_the_rule_can_tell_a_reporting_job_from_a_silent_one() -> None:
    """The check itself, on two jobs written to differ in exactly one call.

    Without this the suite cannot tell a rule that holds from a rule that
    matches nothing -- and a structural check that has stopped matching passes
    silently forever.
    """
    module = ast.parse(
        "def silent(ctx):\n"
        "    ctx.set_total(10)\n"
        "    work()\n"
        "\n"
        "def reporting(ctx):\n"
        "    ctx.set_total(10)\n"
        "    for i in range(10):\n"
        "        ctx.heartbeat(i + 1)\n"
        "\n"
        "def delegating(ctx):\n"
        "    ctx.set_total(10)\n"
        "    reporting(ctx)\n"
    )
    silent, reporting, delegating = _functions(module)

    assert _declares_a_total(silent) and not _reports(silent)
    assert _declares_a_total(reporting) and _reports(reporting)
    assert _declares_a_total(delegating) and not _reports(delegating), (
        "delegating passes only because the hop resolves, which is the part "
        "worth keeping honest"
    )
