"""A feature that this installation cannot run, as the catalog and validation read it.

A feature declares its optional packages and setups in ``prerequisites``.
The declaration catalog evaluates them once into ``Declaration.unavailable``, and
three answers read that one sentence: the editor catalog lists it, validation
reports it at the step's ``feature``, and an attach candidate is refused with it.

The tests here make ``speed-angvel`` unavailable with a prerequisite that no
installation meets. Their outcome does not depend on the packages installed here.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, NoReturn

import pytest

from mosaic.core.pipeline.graph import (
    TRACKS_INPUT,
    Attachment,
    Problem,
    RecipeInvalid,
    attach_candidates,
    check_recipe,
    declaration_catalog,
    editor_catalog,
    feature_class_for_slug,
    read_recipe,
    reject_unless_valid,
    validation_report,
)
from mosaic.core.pipeline.graph import validate as validate_module
from mosaic.optional_dependency import Prerequisite, unmet

type Document = dict[str, object]

_UNMET: Final = "frobnicate is required for this test, and is not installed here."


@dataclass(frozen=True, slots=True)
class _Unmet:
    """A prerequisite that no installation meets."""

    def missing(self) -> str | None:
        return _UNMET


@pytest.fixture
def speed_unavailable() -> Iterator[None]:
    """``speed-angvel`` declares an unmet prerequisite for the length of one test.

    Both catalogs are memoized. Each is cleared on the way in and on the way
    out, and a catalog built during the test is not read by another test.
    """
    cls = feature_class_for_slug("speed-angvel")
    assert cls is not None
    assert not hasattr(cls, "prerequisites")
    setattr(cls, "prerequisites", (_Unmet(),))
    declaration_catalog.cache_clear()
    editor_catalog.cache_clear()
    try:
        yield
    finally:
        delattr(cls, "prerequisites")
        declaration_catalog.cache_clear()
        editor_catalog.cache_clear()


def _speed(**fields: object) -> Document:
    """A ``speed-angvel`` step reading the tracks, with *fields* added."""
    return {
        "id": "speed",
        "type": "feature",
        "feature": "speed-angvel",
        "inputs": ["tracks"],
        **fields,
    }


SMOOTH: Document = {
    "id": "smooth",
    "type": "feature",
    "feature": "trajectory-smooth",
    "inputs": ["tracks"],
}


# --- the declaration and the catalog ----------------------------------------------


def test_every_declaration_reads_the_prerequisites_of_its_class() -> None:
    """``unavailable`` is the sentence of the class's unmet prerequisites."""
    catalog = declaration_catalog()
    for name in catalog.names():
        declared = catalog.get(name)
        assert declared is not None
        if declared.produces.kind != "feature":
            assert declared.unavailable is None, name
            continue
        cls = feature_class_for_slug(name)
        prerequisites: tuple[Prerequisite, ...] = getattr(cls, "prerequisites", ())
        assert declared.unavailable == unmet(prerequisites), name


def test_the_catalog_lists_an_unavailable_feature_with_its_sentence(
    speed_unavailable: None,
) -> None:
    entries = {entry.name: entry for entry in editor_catalog().entries}

    assert entries["speed-angvel"].unavailable == _UNMET
    assert entries["speed-angvel"].declaration.unavailable == _UNMET
    assert entries["trajectory-smooth"].unavailable is None
    assert entries[TRACKS_INPUT].unavailable is None


def test_building_the_catalog_imports_none_of_the_modules_it_checks() -> None:
    """The catalog finds each prerequisite's module and does not import it.

    A fresh interpreter, because this process has imported some of them already.
    """
    checked = ("feral", "lightning_action", "movement", "timm", "torch")
    script = (
        "import sys\n"
        "from mosaic.core.pipeline.graph import editor_catalog\n"
        "_ = editor_catalog()\n"
        f"print(sorted(set({checked!r}) & set(sys.modules)))\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]"


# --- validation -------------------------------------------------------------------


def test_validation_reports_the_sentence_and_checks_the_rest(
    speed_unavailable: None,
) -> None:
    """The step's params and the steps below it are still checked."""
    stack: Document = {
        "id": "stack",
        "type": "feature",
        "feature": "temporal-stack",
        "inputs": [{"step": "speed"}],
        "params": {"not_a_field": 1},
    }

    alone = validation_report({"steps": [_speed(params={"not_a_field": 1})]})
    chained = validation_report({"steps": [_speed(), stack]})

    assert [(problem.step, problem.where) for problem in alone.problems] == [
        ("speed", "feature"),
        ("speed", "params.not_a_field"),
    ]
    assert alone.problems[0].message == _UNMET
    assert [(problem.step, problem.where) for problem in chained.problems] == [
        ("speed", "feature"),
        ("stack", "params.not_a_field"),
    ]


def test_a_plan_refuses_an_unavailable_step(speed_unavailable: None) -> None:
    with pytest.raises(RecipeInvalid) as raised:
        reject_unless_valid(read_recipe({"steps": [_speed()]}))

    assert raised.value.problems == (Problem("speed", "feature", _UNMET),)


def test_a_construction_that_fails_for_the_same_reason_is_reported_once(
    speed_unavailable: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature that imports a missing package on construction raises ImportError."""

    def _raises(_spec: object) -> NoReturn:
        raise ImportError("frobnicate is not installed")

    monkeypatch.setattr(validate_module, "build_step_feature", _raises)

    assert check_recipe(read_recipe({"steps": [_speed()]})) == (
        Problem("speed", "feature", _UNMET),
    )


# --- attach candidates ------------------------------------------------------------


_REFUSED: Final = Attachment(allowed=False, reason=_UNMET, template=None)


def test_attach_refuses_an_unavailable_feature_with_the_same_sentence(
    speed_unavailable: None,
) -> None:
    at_the_top = attach_candidates(read_recipe({"steps": []}), None)
    below = attach_candidates(read_recipe({"steps": [SMOOTH]}), "smooth")

    assert at_the_top.candidates["speed-angvel"] == _REFUSED
    assert below.candidates["speed-angvel"] == _REFUSED
    assert at_the_top.candidates["trajectory-smooth"].allowed
    assert below.candidates["nearest-neighbor"].allowed
