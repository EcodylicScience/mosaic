"""``mosaic.entry_points`` finds the entry points of a group and loads one of them.

Each test installs entry points that name objects in this module, or runs a child
process that sees a distribution declaring some.
"""

from __future__ import annotations

import sys
import textwrap
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from mosaic.entry_points import (
    EntryPointRefused,
    InstalledOrigin,
    installed_entry_points,
    load_entry_point,
)
from tests.helpers import (
    CountingEntryPoint,
    FailingEntryPoint,
    install_entry_points,
    run_with_distribution,
)

GROUP = "mosaic.test_group"


class Base:
    """The type that an entry point is expected to hold."""


class Derived(Base):
    """A subclass of :class:`Base`."""


class Unrelated:
    """A class that is not a :class:`Base`."""


INSTANCE = Base()
DERIVED_INSTANCE = Derived()
NOT_A_BASE = "a string"


def _entry_point(
    name: str, attribute: str, kind: type[EntryPoint] = EntryPoint
) -> EntryPoint:
    return kind(name=name, value=f"{__name__}:{attribute}", group=GROUP)


def test_the_entry_points_of_a_group_are_listed_by_name_and_none_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    CountingEntryPoint.loads = 0
    twin_one = _entry_point("twin", "INSTANCE", CountingEntryPoint)
    twin_two = _entry_point("twin", "DERIVED_INSTANCE", CountingEntryPoint)
    first = _entry_point("alpha", "INSTANCE", CountingEntryPoint)
    elsewhere = EntryPoint(name="other", value=f"{__name__}:INSTANCE", group="x.other")
    groups = install_entry_points(monkeypatch, twin_one, twin_two, first, elsewhere)

    found = installed_entry_points(GROUP)

    assert list(found) == ["alpha", "twin"]
    assert found["alpha"] == (first,)
    assert found["twin"] == (twin_one, twin_two)
    assert groups == [GROUP]
    assert CountingEntryPoint.loads == 0


def test_a_group_without_entry_points_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = install_entry_points(monkeypatch)

    assert installed_entry_points(GROUP) == {}


def test_an_entry_point_without_a_distribution_has_a_blank_origin() -> None:
    found = _entry_point("alpha", "INSTANCE")

    origin = InstalledOrigin.from_entry_point(found)

    assert origin == InstalledOrigin(
        group=GROUP,
        name="alpha",
        distribution="",
        version="",
        value=f"{__name__}:INSTANCE",
    )
    assert origin.describe() == "entry point 'alpha' of an unknown distribution"


def test_an_origin_reads_the_distribution_that_declares_the_entry_point(
    tmp_path: Path,
) -> None:
    script = f"""
        from mosaic.entry_points import InstalledOrigin, installed_entry_points

        (found,) = installed_entry_points({GROUP!r})["alpha"]
        origin = InstalledOrigin.from_entry_point(found)
        print(origin)
        print(origin.describe())
    """

    proc = run_with_distribution(
        tmp_path,
        f"[{GROUP}]\nalpha = x_module:OBJECT\n",
        {"x_module": "OBJECT = 1\n"},
        [sys.executable, "-c", textwrap.dedent(script)],
    )

    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert lines[0] == (
        f"InstalledOrigin(group={GROUP!r}, name='alpha', distribution='x', "
        "version='1.0', value='x_module:OBJECT')"
    )
    assert lines[1] == "entry point 'alpha' of distribution 'x'"


def test_an_instance_of_the_expected_type_is_returned() -> None:
    assert load_entry_point(_entry_point("a", "INSTANCE"), Base) is INSTANCE
    assert load_entry_point(_entry_point("a", "DERIVED_INSTANCE"), Base) is (
        DERIVED_INSTANCE
    )


def test_a_class_is_refused_unless_a_subclass_is_asked_for() -> None:
    found = _entry_point("a", "Derived")

    with pytest.raises(EntryPointRefused, match=r"is a type, not a Base$"):
        _ = load_entry_point(found, Base)
    assert load_entry_point(found, Base, subclass=True) is Derived


def test_the_expected_class_itself_counts_as_its_own_subclass() -> None:
    assert load_entry_point(_entry_point("a", "Base"), Base, subclass=True) is Base


def test_a_refusal_is_a_value_error_that_names_the_origin() -> None:
    with pytest.raises(ValueError, match=r"entry point 'a' of an unknown") as raised:
        _ = load_entry_point(_entry_point("a", "NOT_A_BASE"), Base)

    assert isinstance(raised.value, EntryPointRefused)


def test_a_load_error_is_refused_with_its_type_and_message() -> None:
    found = _entry_point("a", "INSTANCE", FailingEntryPoint)

    with pytest.raises(
        EntryPointRefused,
        match=r"^entry point 'a' of an unknown distribution could not be loaded: "
        r"ImportError: the module is broken$",
    ) as raised:
        _ = load_entry_point(found, Base)

    assert isinstance(raised.value.__cause__, ImportError)


def test_an_object_of_another_type_is_refused() -> None:
    with pytest.raises(
        EntryPointRefused,
        match=r"^entry point 'a' of an unknown distribution is a str, not a Base$",
    ):
        _ = load_entry_point(_entry_point("a", "NOT_A_BASE"), Base)


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("INSTANCE", r"is a Base, not a Base subclass$"),
        ("Unrelated", r"is the class Unrelated, not a Base subclass$"),
        ("NOT_A_BASE", r"is a str, not a Base subclass$"),
    ],
    ids=["instance", "class-outside-the-hierarchy", "not-a-class"],
)
def test_a_subclass_is_refused_for_what_is_not_one(
    attribute: str, message: str
) -> None:
    with pytest.raises(EntryPointRefused, match=message):
        _ = load_entry_point(_entry_point("a", attribute), Base, subclass=True)
