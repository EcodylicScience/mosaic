"""An optional module declared once, found without importing it, and loaded by name.

``OptionalModule`` is the one declaration that a feature's prerequisites and its
import guard share. The pipeline catalog's sentence and a run's error name the
same extra.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.optional_dependency import (
    MissingOptionalDependency,
    OptionalModule,
    RestrictedUse,
    license_notes,
    unmet,
)

_ABSENT = OptionalModule("mosaic_no_such_module", "movement", "this test")


def test_an_absent_module_is_missing_with_the_error_s_sentence() -> None:
    expected = str(
        MissingOptionalDependency("mosaic_no_such_module", "movement", "this test")
    )

    assert _ABSENT.missing() == expected
    assert 'pip install "mosaic-behavior[movement]"' in expected
    with pytest.raises(MissingOptionalDependency) as raised:
        _ = _ABSENT.load()
    assert str(raised.value) == expected


def test_a_module_is_found_without_being_imported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A package whose import raises is found, and is not imported by the search."""
    package = tmp_path / "mosaic_probe_package"
    package.mkdir()
    _ = (package / "__init__.py").write_text("raise RuntimeError('imported')\n")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    importlib.invalidate_caches()

    assert OptionalModule("mosaic_probe_package", "movement", "x").missing() is None
    assert "mosaic_probe_package" not in sys.modules


def test_a_module_built_in_place_is_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """``find_spec`` raises for a stub in ``sys.modules``, whose spec is None."""
    monkeypatch.setitem(
        sys.modules, "mosaic_stub_module", types.ModuleType("mosaic_stub_module")
    )

    assert OptionalModule("mosaic_stub_module", "movement", "x").missing() is None


@dataclass(frozen=True, slots=True)
class _Says:
    message: str | None

    def missing(self) -> str | None:
        return self.message


def test_unmet_joins_what_is_missing_in_declaration_order() -> None:
    assert unmet(()) is None
    assert unmet((_Says(None), _Says(None))) is None
    assert unmet((_Says("First."), _Says(None), _Says("Second."))) == "First. Second."


@dataclass(frozen=True, slots=True)
class _Restricted:
    note: str

    def missing(self) -> str | None:
        return None

    def license_note(self) -> str:
        return self.note


def test_license_notes_join_the_restricted_prerequisites_in_declaration_order() -> None:
    """A met prerequisite still lists its note, and an unrestricted one lists none."""
    first, second = _Restricted("First."), _Restricted("Second.")

    assert isinstance(first, RestrictedUse)
    assert not isinstance(_Says(None), RestrictedUse)
    assert license_notes(()) is None
    assert license_notes((_Says(None), _ABSENT)) is None
    assert license_notes((first, _Says("Missing."), second)) == "First. Second."
