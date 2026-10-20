"""Which features need more than the base install, and what each one needs.

A feature declares its optional packages and the setups of its run in
``prerequisites``. The pipeline catalog offers it as unavailable while one is
unmet, and validation refuses a step running it.
The table names every feature that declares one.

A feature that runs its tool as a program in a separate environment declares an
interpreter prerequisite, which locates the interpreter as starting the program
does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from mosaic.behavior.feature_library.external.vame_protocol import VAME_VERSION
from mosaic.behavior.feature_library.feral_feature import FERAL
from mosaic.behavior.feature_library.kpms import KPMS_ENV, KpmsLicense
from mosaic.behavior.feature_library.lightning_action_feature import (
    LIGHTNING_ACTION,
)
from mosaic.behavior.feature_library.movement.convert import MOVEMENT
from mosaic.behavior.feature_library.registry import FEATURES
from mosaic.behavior.feature_library.runner_client import RunnerInterpreter
from mosaic.behavior.feature_library.sequence_classifier import TORCH as SEQUENCE_TORCH
from mosaic.behavior.feature_library.vame_feature import VAME_ENV
from mosaic.behavior.model_library.timm_backbone import TIMM, TORCH
from mosaic.core.narrowing import is_object_tuple
from mosaic.core.pipeline.toolenv import ToolEnv
from mosaic.optional_dependency import Prerequisite
from tests.helpers import (
    KPMS_TOOL,
    VAME_TOOL,
    SeparatedEnvironment,
    isolate_tool_location,
)

EXPECTED: dict[str, tuple[Prerequisite, ...]] = {
    "feral": (FERAL,),
    "global-identity-dinov2-temporal": (TORCH,),
    "global-identity-embedding": (TIMM,),
    "global-identity-model": (TIMM,),
    "kpms": (KpmsLicense(), RunnerInterpreter(KPMS_ENV)),
    "lightning-action": (LIGHTNING_ACTION,),
    "movement-filter-interpolate": (MOVEMENT,),
    "movement-smooth": (MOVEMENT,),
    "sequence-classifier": (SEQUENCE_TORCH,),
    "vame": (RunnerInterpreter(VAME_ENV),),
}

FEATURE_CLASSES = sorted(FEATURES.values(), key=lambda cls: cls.__name__)


def test_the_features_that_need_more_than_the_base_install() -> None:
    declared: dict[str, tuple[Prerequisite, ...]] = {
        cls.name: getattr(cls, "prerequisites", ()) for cls in FEATURE_CLASSES
    }

    assert {name: needs for name, needs in declared.items() if needs} == EXPECTED


@pytest.mark.parametrize(
    "cls", FEATURE_CLASSES, ids=[cls.__name__ for cls in FEATURE_CLASSES]
)
def test_every_declared_prerequisite_answers_missing(cls: type) -> None:
    """Each declaration is a tuple of ``Prerequisite``, the shape the catalog reads."""
    declared: object = getattr(cls, "prerequisites", ())

    assert is_object_tuple(declared)
    assert all(isinstance(item, Prerequisite) for item in declared)


# --- interpreter prerequisites ----------------------------------------------

(_KPMS_ENVIRONMENT,) = KPMS_TOOL.environments
(_VAME_ENVIRONMENT,) = VAME_TOOL.environments


@dataclass(frozen=True, slots=True)
class InterpreterTool:
    """A tool that a feature runs as a program in a separate environment.

    Attributes:
        prerequisite: The feature's prerequisite that the interpreter meets.
        environment: The tool's environment, as the registry of separated tools
            names it. The separation checks compare its ``ToolEnv`` with its
            directory.
        help_phrases: What the install help names besides the two variables.
    """

    prerequisite: Prerequisite
    environment: SeparatedEnvironment
    help_phrases: tuple[str, ...]

    @property
    def env(self) -> ToolEnv:
        """Returns the location ladder of the tool's interpreter."""
        return self.environment.located_by()


INTERPRETER_TOOLS: Final = (
    InterpreterTool(
        RunnerInterpreter(KPMS_ENV),
        _KPMS_ENVIRONMENT,
        ("external/README.md", "never bundled"),
    ),
    InterpreterTool(
        RunnerInterpreter(VAME_ENV),
        _VAME_ENVIRONMENT,
        ("external/vame-env/README.md", f"vame-py {VAME_VERSION}"),
    ),
)

INTERPRETER_IDS: Final = [tool.env.tool for tool in INTERPRETER_TOOLS]


@pytest.mark.parametrize("tool", INTERPRETER_TOOLS, ids=INTERPRETER_IDS)
def test_an_interpreter_prerequisite_says_how_to_provide_the_interpreter(
    tool: InterpreterTool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The help names the tool's variables and how to build its environment.

    The help does not name an extra, because mosaic's extras do not install the
    tool. An interpreter that the tool's variable names meets the prerequisite.
    """
    isolate_tool_location(monkeypatch, tool.env, found={})

    missing = tool.prerequisite.missing()

    assert missing is not None
    for phrase in (tool.env.bin_var, tool.env.conda_env_var, *tool.help_phrases):
        assert phrase in missing
    assert "mosaic-behavior[" not in missing
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setenv(tool.env.bin_var, str(interpreter))
    assert tool.prerequisite.missing() is None
