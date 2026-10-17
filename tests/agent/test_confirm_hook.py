"""The Claude Code Elicitation hook that answers the MCP server's confirmations."""

from __future__ import annotations

import pytest

from mosaic.agent.confirm_hook import HookFallback, answer_elicitation
from mosaic.agent.mcp.spec import confirmation_tag
from tests.helpers import modules_loaded_by

QUESTION = (
    "mosaic asks you to confirm submit_pipeline: step 'fit' runs on lane "
    "gpu-train; step 'fit' trains a model with train-pose. Proceed?\n\n"
    + confirmation_tag("submit_pipeline", ["gpu_lane", "training_op"])
)


def _hook_input(**changed: object) -> dict[str, object]:
    """The input Claude Code passes for the question above, with *changed* keys."""
    given: dict[str, object] = {
        "session_id": "s",
        "hook_event_name": "Elicitation",
        "mcp_server_name": "mosaic",
        "message": QUESTION,
        "mode": "form",
        "requested_schema": {
            "type": "object",
            "properties": {"confirm": {"type": "boolean", "title": "Proceed"}},
            "required": ["confirm"],
        },
    }
    given.update(changed)
    return given


def test_an_approved_condition_set_is_accepted_with_the_box_ticked() -> None:
    answer = answer_elicitation(
        _hook_input(), ["gpu_lane", "training_op", "bind"], otherwise="decline"
    )

    assert answer == {
        "hookSpecificOutput": {
            "hookEventName": "Elicitation",
            "action": "accept",
            "content": {"confirm": True},
        }
    }


@pytest.mark.parametrize("otherwise", ["ask", "cancel", "decline"])
def test_an_unapproved_condition_falls_back_as_told(otherwise: HookFallback) -> None:
    answer = answer_elicitation(_hook_input(), ["gpu_lane"], otherwise=otherwise)

    if otherwise == "ask":
        assert answer is None
    else:
        assert answer == {
            "hookSpecificOutput": {"hookEventName": "Elicitation", "action": otherwise}
        }


@pytest.mark.parametrize(
    "changed",
    [
        {"mcp_server_name": "other"},
        {"hook_event_name": "ElicitationResult"},
        {"mode": "url", "url": "https://example.org"},
        {"message": "mosaic asks you to confirm submit_pipeline. Proceed?"},
        {"message": f"{confirmation_tag('submit_pipeline', ['bind'])}\nProceed?"},
        {"requested_schema": {"type": "object", "properties": {"name": {}}}},
        {"requested_schema": None},
    ],
    ids=[
        "another-server",
        "another-event",
        "url",
        "untagged",
        "tag-not-last",
        "another-form",
        "no-form",
    ],
)
def test_another_server_a_url_or_an_untagged_form_is_left_to_the_dialog(
    changed: dict[str, object],
) -> None:
    assert (
        answer_elicitation(
            _hook_input(**changed),
            ["gpu_lane", "training_op", "bind"],
            otherwise="decline",
        )
        is None
    )


def test_a_form_without_a_mode_is_answered() -> None:
    given = _hook_input()
    del given["mode"]

    answer = answer_elicitation(given, ["gpu_lane", "training_op"])

    assert answer is not None


def test_a_server_listed_under_another_name_is_answered_when_named() -> None:
    given = _hook_input(mcp_server_name="mosaic-lab")

    assert answer_elicitation(given, ["gpu_lane", "training_op"]) is None
    assert (
        answer_elicitation(given, ["gpu_lane", "training_op"], server="mosaic-lab")
        is not None
    )


def test_importing_the_hook_loads_neither_an_sdk_nor_the_analysis_stack() -> None:
    loaded = modules_loaded_by(
        "import mosaic.agent.confirm_hook",
        ("mcp", "mcp_types", "pandas", "cv2", "av", "mosaic.behavior"),
    )

    assert loaded == []
