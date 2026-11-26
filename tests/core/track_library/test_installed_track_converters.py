"""The ``mosaic.track_converters`` group: an installed package's converters.

An entry point of the group names a ``TrackConverter`` subclass under the converter's
``src_format``. ``ensure_track_converters_registered`` loads every one once per
process, after the built-ins, and a refused entry point is recorded and not
registered, so a package that fails does not take the built-ins down with it.
"""

from __future__ import annotations

import re
import sys
import textwrap
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import override

import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.convert_ops import ConvertTracksParams
from mosaic.core.pipeline.graph.resolve import (
    fill_builtin_registries,
    registry_snapshot,
)
from mosaic.core.pipeline.tracks_identity import (
    convert_variant_payload,
    converter_op,
    tracks_run_id,
)
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.pipeline.tracks_raw_index import (
    load_tracks_raw_index_frame,
    write_tracks_raw_index_rows,
)
from mosaic.core.track_converter import (
    INSTALLED_STATE,
    TRACK_CONVERTER_GROUP,
    TRACK_CONVERTERS,
    TrackConverter,
    builtin_track_converters,
    ensure_track_converters_registered,
    get_track_converter,
    refused_track_converters,
    track_converter_origin,
)
from mosaic.core.track_library.trex import TrexNpzConverter
from mosaic.entry_points import InstalledOrigin
from tests.helpers import (
    AcmeEntryPoint,
    CountingEntryPoint,
    installed_track_converters,
    make_dataset,
    plugin_module,
    run_with_distribution,
    write_trex_npz,
)

HEADER = """
from mosaic.core.track_converter import TrackConverter, register_track_converter
"""

ROWS = "acme__rows"


def _class_source(
    class_name: str,
    src_format: object,
    *,
    version: object = "1.0",
    decorate: bool = False,
) -> str:
    """The source of a converter class, decorated for registration when asked."""
    decorator = "@register_track_converter\n" if decorate else ""
    return f"""
{decorator}class {class_name}(TrackConverter):
    src_format = {src_format!r}
    version = {version!r}
    output_schema = "mosaic_v1"
    carries_tracklets = True

    def convert(self, path, params, hints):
        raise NotImplementedError
"""


ACME_SOURCE = HEADER + _class_source("AcmeConverter", ROWS)
NOT_A_CLASS_SOURCE = (
    ACME_SOURCE + "\nNOT_A_CLASS = 'text'\n\nclass NotAConverter:\n    pass\n"
)


def _entry_point(
    name: str,
    module: str,
    attribute: str,
    kind: type[EntryPoint] = AcmeEntryPoint,
) -> EntryPoint:
    return kind(name=name, value=f"{module}:{attribute}", group=TRACK_CONVERTER_GROUP)


class BrokenAcmeEntryPoint(AcmeEntryPoint):
    """An entry point of ``acme-tracks`` whose load raises."""

    @override
    def load(self) -> object:
        msg = "the module is broken"
        raise ImportError(msg)


def _refusal_message(name: str) -> str:
    """The recorded refusal of *name*, which must exist."""
    return refused_track_converters()[name]


def _assert_refused(
    name: str, pattern: str, builtins: Mapping[str, type[object]]
) -> None:
    """Check that *name* is recorded, absent, raised, and the built-ins are intact."""
    recorded = _refusal_message(name)
    assert re.search(pattern, recorded), recorded
    assert name not in TRACK_CONVERTERS
    with pytest.raises(ValueError) as caught:
        _ = get_track_converter(name)
    assert str(caught.value) == recorded
    for built_in, cls in builtins.items():
        assert TRACK_CONVERTERS[built_in] is cls


def _assert_every_installed_key_has_an_origin() -> None:
    """Check that no key of the test's namespace is registered without an entry point."""
    for key in TRACK_CONVERTERS:
        if str(key).startswith("acme__"):
            assert track_converter_origin(key) is not None, key


def _imported(module: str, attribute: str) -> type[TrackConverter]:
    cls = getattr(sys.modules[module], attribute)
    assert isinstance(cls, type)
    return cls


# --- A converter that an installed package declares ---------------------------


def test_an_installed_converter_is_registered_and_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, ACME_SOURCE)
    found = _entry_point(ROWS, module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        converter = get_track_converter(ROWS)
        dataset = make_dataset(tmp_path / "ds")

        assert type(converter) is _imported(module, "AcmeConverter")
        assert type(converter) is TRACK_CONVERTERS[ROWS]
        assert dataset.list_converters()[ROWS] is type(converter)
        assert ROWS in registry_snapshot()["TRACK_CONVERTERS"]
        assert track_converter_origin(ROWS) == InstalledOrigin(
            group=TRACK_CONVERTER_GROUP,
            name=ROWS,
            distribution="acme-tracks",
            version="1.2.0",
            value=f"{module}:AcmeConverter",
        )
        assert track_converter_origin("trex_npz") is None
        assert ROWS not in builtin_track_converters()
        assert refused_track_converters() == {}
    assert ROWS not in TRACK_CONVERTERS
    assert track_converter_origin(ROWS) is None


# --- Refusals ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Refusal:
    """A plugin module, the entry points that name it, and what is refused."""

    source: str
    entries: tuple[tuple[str, str, type[EntryPoint]], ...]
    refused: str
    message: str
    also_absent: tuple[object, ...] = ()
    accepted: str = ""


def _entry(
    name: str, attribute: str, kind: type[EntryPoint] = AcmeEntryPoint
) -> tuple[str, str, type[EntryPoint]]:
    return (name, attribute, kind)


def _mismatched_format(decorate: bool) -> Refusal:
    return Refusal(
        source=HEADER
        + _class_source("AcmeConverter", "acme__other", decorate=decorate),
        entries=(_entry(ROWS, "AcmeConverter"),),
        refused=ROWS,
        message=r"src_format is 'acme__other', not 'acme__rows'",
        also_absent=("acme__other",),
    )


def _bad_version(version: object, message: str, *, decorate: bool) -> Refusal:
    return Refusal(
        source=HEADER
        + _class_source("AcmeConverter", ROWS, version=version, decorate=decorate),
        entries=(_entry(ROWS, "AcmeConverter"),),
        refused=ROWS,
        message=message,
    )


def _bad_format(src_format: object, message: str, *, decorate: bool) -> Refusal:
    return Refusal(
        source=HEADER + _class_source("AcmeConverter", src_format, decorate=decorate),
        entries=(_entry(ROWS, "AcmeConverter"),),
        refused=ROWS,
        message=message,
        also_absent=(src_format,) if isinstance(src_format, int) else (),
    )


def _two_converters_source(second_format: str) -> str:
    return (
        HEADER
        + _class_source("AcmeConverter", ROWS, decorate=True)
        + _class_source("MoreConverter", second_format, decorate=True)
    )


STRAY_SOURCE = (
    HEADER
    + _class_source("AcmeConverter", ROWS, decorate=True)
    + _class_source("Stray", "acme__stray", decorate=True)
)

REFUSALS = {
    "two-entry-points-of-a-name": Refusal(
        source=ACME_SOURCE,
        entries=(_entry(ROWS, "AcmeConverter"), _entry(ROWS, "AcmeConverter")),
        refused=ROWS,
        message=(
            r"declared twice.*entry point 'acme__rows' of distribution 'acme-tracks'"
            r" and entry point 'acme__rows' of distribution 'acme-tracks'"
        ),
    ),
    "no-namespace": Refusal(
        source=ACME_SOURCE,
        entries=(_entry("nonamespace", "AcmeConverter"),),
        refused="nonamespace",
        message=r"entry point 'nonamespace' of distribution 'acme-tracks'.*namespace",
    ),
    "load-error": Refusal(
        source=ACME_SOURCE,
        entries=(_entry(ROWS, "AcmeConverter", BrokenAcmeEntryPoint),),
        refused=ROWS,
        message=(
            r"entry point 'acme__rows' of distribution 'acme-tracks' could not be "
            r"loaded: ImportError: the module is broken"
        ),
    ),
    "not-a-class": Refusal(
        source=NOT_A_CLASS_SOURCE,
        entries=(_entry(ROWS, "NOT_A_CLASS"),),
        refused=ROWS,
        message=r"entry point 'acme__rows' of distribution 'acme-tracks' is a str",
    ),
    "outside-the-hierarchy": Refusal(
        source=NOT_A_CLASS_SOURCE,
        entries=(_entry(ROWS, "NotAConverter"),),
        refused=ROWS,
        message=(
            r"entry point 'acme__rows' of distribution 'acme-tracks' is the class "
            r"NotAConverter, not a TrackConverter subclass"
        ),
    ),
    "format-of-another-name": _mismatched_format(decorate=False),
    "format-of-another-name-decorated": _mismatched_format(decorate=True),
    "version-outside-the-pattern": _bad_version("v1", r"version 'v1'", decorate=False),
    "version-outside-the-pattern-decorated": _bad_version(
        "v1", r"version 'v1'", decorate=True
    ),
    "version-that-is-not-a-string": _bad_version(
        1, r"version 1 does not match", decorate=False
    ),
    "version-that-is-not-a-string-decorated": _bad_version(
        1, r"version 1 does not match", decorate=True
    ),
    "format-that-is-not-a-string": _bad_format(
        5, r"src_format is 5, not 'acme__rows'", decorate=False
    ),
    "format-that-is-not-a-string-decorated": _bad_format(
        5, r"src_format is 5, not 'acme__rows'", decorate=True
    ),
    "format-that-is-a-list": _bad_format(
        ["x"], r"src_format is \['x'\], not 'acme__rows'", decorate=False
    ),
    "format-that-is-a-list-decorated": _bad_format(
        ["x"], r"could not be loaded: TypeError", decorate=True
    ),
    "a-key-no-entry-point-names": Refusal(
        source=STRAY_SOURCE,
        entries=(_entry(ROWS, "AcmeConverter"),),
        refused="acme__stray",
        message=(
            r"entry point 'acme__rows' of distribution 'acme-tracks'.*"
            r"'acme__stray'"
        ),
        accepted=ROWS,
    ),
    "a-key-that-only-a-differently-named-entry-point-names": Refusal(
        source=_two_converters_source("acme__more"),
        entries=(_entry(ROWS, "AcmeConverter"), _entry("more", "MoreConverter")),
        refused="more",
        message=r"entry point 'more' of distribution 'acme-tracks'.*namespace",
        also_absent=("acme__more",),
        accepted=ROWS,
    ),
}


@pytest.mark.parametrize("case", REFUSALS.values(), ids=REFUSALS.keys())
def test_a_refused_entry_point_is_recorded_and_registers_nothing(
    case: Refusal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builtins = builtin_track_converters()
    module = plugin_module(tmp_path, monkeypatch, case.source)
    found = [
        _entry_point(name, module, attr, kind) for name, attr, kind in case.entries
    ]

    with installed_track_converters(monkeypatch, *found):
        ensure_track_converters_registered()

        _assert_refused(case.refused, case.message, builtins)
        for name in case.also_absent:
            assert name not in TRACK_CONVERTERS
        if case.accepted:
            accepted = get_track_converter(case.accepted)
            assert type(accepted) is _imported(module, "AcmeConverter")
        else:
            assert ROWS not in TRACK_CONVERTERS
        _assert_every_installed_key_has_an_origin()


def test_an_entry_point_named_like_a_built_in_leaves_the_built_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, ACME_SOURCE)
    found = _entry_point("trex_npz", module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        assert "trex_npz" in refused_track_converters()
        assert "distribution 'acme-tracks'" in _refusal_message("trex_npz")
        assert type(get_track_converter("trex_npz")) is TrexNpzConverter
        assert track_converter_origin("trex_npz") is None
        assert TRACK_CONVERTERS["trex_npz"] is TrexNpzConverter


# --- A module that decorates several converters ------------------------------


@pytest.mark.parametrize("second_format", ["acme__more", "acme__zmore"])
def test_a_module_that_decorates_two_converters_registers_both_in_either_order(
    second_format: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, _two_converters_source(second_format))
    found = (
        _entry_point(ROWS, module, "AcmeConverter"),
        _entry_point(second_format, module, "MoreConverter"),
    )

    with installed_track_converters(monkeypatch, *found):
        ensure_track_converters_registered()

        assert TRACK_CONVERTERS[ROWS] is _imported(module, "AcmeConverter")
        assert TRACK_CONVERTERS[second_format] is _imported(module, "MoreConverter")
        assert refused_track_converters() == {}
        assert track_converter_origin(ROWS) is not None
        assert track_converter_origin(second_format) is not None
        _assert_every_installed_key_has_an_origin()


@pytest.mark.parametrize("misnamed", ["acme__amore", "acme__zmore"])
def test_a_misnamed_entry_point_removes_its_key_and_leaves_the_other(
    misnamed: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, _two_converters_source("acme__more"))
    found = (
        _entry_point(ROWS, module, "AcmeConverter"),
        _entry_point(misnamed, module, "MoreConverter"),
    )

    with installed_track_converters(monkeypatch, *found):
        ensure_track_converters_registered()

        assert TRACK_CONVERTERS[ROWS] is _imported(module, "AcmeConverter")
        assert "acme__more" not in TRACK_CONVERTERS
        assert misnamed not in TRACK_CONVERTERS
        assert misnamed in refused_track_converters()
        assert track_converter_origin(ROWS) is not None
        _assert_every_installed_key_has_an_origin()


def test_a_name_declared_twice_stays_refused_when_a_later_module_decorates_its_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builtins = builtin_track_converters()
    module_x = plugin_module(
        tmp_path,
        monkeypatch,
        HEADER
        + _class_source("AcmeA", "acme__a", decorate=True)
        + _class_source("AcmeB", "acme__b", decorate=True),
    )
    module_y = plugin_module(
        tmp_path, monkeypatch, HEADER + _class_source("AcmeA", "acme__a", decorate=True)
    )
    found = (
        _entry_point("acme__a", module_x, "AcmeA"),
        _entry_point("acme__a", module_y, "AcmeA"),
        _entry_point("acme__b", module_x, "AcmeB"),
    )

    with installed_track_converters(monkeypatch, *found):
        ensure_track_converters_registered()

        _assert_refused("acme__a", r"declared twice.*Neither is registered", builtins)
        assert TRACK_CONVERTERS["acme__b"] is _imported(module_x, "AcmeB")
        assert set(refused_track_converters()) == {"acme__a"}
        _assert_every_installed_key_has_an_origin()


def test_a_name_without_a_namespace_stays_refused_when_a_later_module_decorates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builtins = builtin_track_converters()
    module = plugin_module(
        tmp_path,
        monkeypatch,
        HEADER
        + _class_source("Aaa", "aaa", decorate=True)
        + _class_source("AcmeB", "acme__b", decorate=True),
    )
    found = (
        _entry_point("aaa", module, "Aaa"),
        _entry_point("acme__b", module, "AcmeB"),
    )

    with installed_track_converters(monkeypatch, *found):
        ensure_track_converters_registered()

        _assert_refused("aaa", r"entry point 'aaa'.*namespace", builtins)
        assert TRACK_CONVERTERS["acme__b"] is _imported(module, "AcmeB")
        assert set(refused_track_converters()) == {"aaa"}
        _assert_every_installed_key_has_an_origin()


# --- A refusal never ends the load -----------------------------------------

UNFORESEEN_SOURCE = (
    HEADER
    + """
class Hostile(type):
    @property
    def version(cls):
        raise RuntimeError("no version")


@register_track_converter
class BadConverter(TrackConverter, metaclass=Hostile):
    src_format = "acme__bad"
    output_schema = "mosaic_v1"
    carries_tracklets = True

    def convert(self, path, params, hints):
        raise NotImplementedError
"""
    + _class_source("AcmeConverter", ROWS)
)

LATER_ENTRY_POINTS = {
    "version-that-is-not-a-string": (
        HEADER
        + _class_source("BadConverter", "acme__bad", version=1, decorate=True)
        + _class_source("AcmeConverter", ROWS),
        r"version 1 does not match",
    ),
    "format-that-is-not-a-string": (
        HEADER
        + _class_source("BadConverter", 5)
        + _class_source("AcmeConverter", ROWS),
        r"src_format is 5, not 'acme__bad'",
    ),
    "an-error-the-load-did-not-foresee": (
        UNFORESEEN_SOURCE,
        r"RuntimeError: no version",
    ),
}


@pytest.mark.parametrize(
    ("source", "message"), LATER_ENTRY_POINTS.values(), ids=LATER_ENTRY_POINTS.keys()
)
def test_the_entry_points_after_a_refused_one_still_load(
    source: str, message: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    builtins = builtin_track_converters()
    module = plugin_module(tmp_path, monkeypatch, source)
    found = (
        _entry_point("acme__bad", module, "BadConverter"),
        _entry_point(ROWS, module, "AcmeConverter"),
    )

    with installed_track_converters(monkeypatch, *found):
        assert type(get_track_converter("trex_npz")) is TrexNpzConverter

        _assert_refused("acme__bad", rf"entry point 'acme__bad'.*{message}", builtins)
        assert type(get_track_converter(ROWS)) is _imported(module, "AcmeConverter")
        assert set(refused_track_converters()) == {"acme__bad"}
        _assert_every_installed_key_has_an_origin()


# --- Loading once ------------------------------------------------------------

SLOW_SOURCE = "import time\n\ntime.sleep(0.2)\n" + ACME_SOURCE


def test_the_group_loads_once_across_calls_and_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    CountingEntryPoint.loads = 0
    module = plugin_module(tmp_path, monkeypatch, SLOW_SOURCE)
    found = _entry_point(ROWS, module, "AcmeConverter", CountingEntryPoint)
    barrier = threading.Barrier(8)

    def lookup() -> None:
        _ = barrier.wait(timeout=30)
        assert type(get_track_converter(ROWS)) is _imported(module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        ensure_track_converters_registered()
        ensure_track_converters_registered()
        assert CountingEntryPoint.loads == 1
    CountingEntryPoint.loads = 0
    with installed_track_converters(monkeypatch, found):
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(lookup) for _ in range(8)]
            for future in futures:
                future.result(timeout=60)
        assert CountingEntryPoint.loads == 1


def test_a_plugin_module_that_looks_up_a_built_in_at_import_loads_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = (
        "from mosaic.core.track_converter import get_track_converter\n"
        "BUILT_IN = get_track_converter('trex_npz')\n" + ACME_SOURCE
    )
    module = plugin_module(tmp_path, monkeypatch, source)
    found = _entry_point(ROWS, module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        ensure_track_converters_registered()

        assert type(sys.modules[module].BUILT_IN) is TrexNpzConverter
        assert TRACK_CONVERTERS[ROWS] is _imported(module, "AcmeConverter")
        assert refused_track_converters() == {}


# --- A definition's snapshot -------------------------------------------------


def test_the_built_in_registries_hold_the_installed_formats_before_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, ACME_SOURCE)
    found = _entry_point(ROWS, module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        fill_builtin_registries()
        before = registry_snapshot()
        fill_builtin_registries()

        assert ROWS in before["TRACK_CONVERTERS"]
        assert registry_snapshot() == before


# --- The distribution a conversion records -----------------------------------

CONVERTING_SOURCE = (
    "import pandas as pd\n"
    + HEADER
    + """

class AcmeConverter(TrackConverter):
    src_format = "acme__rows"
    version = "1.0"
    output_schema = "mosaic_v1"
    carries_tracklets = True

    def convert(self, path, params, hints):
        rows = pd.read_csv(path)
        return rows.assign(
            group=hints.group, time=rows["frame"] / 30.0, X=1.0, Y=2.0
        )
"""
)


def _converted_row(tmp_path: Path, src_format: str) -> pd.Series:
    """Index a raw file of *src_format*, convert it, and return its tracks row."""
    dataset = make_dataset(tmp_path / "ds", roots=("tracks_raw", "tracks"))
    dataset.meta["fps_default"] = 30.0
    raw = dataset.get_root("tracks_raw")
    _ = (raw / "s.csv").write_text("frame,id\n0,0\n1,0\n")
    _ = dataset.index_tracks_raw([raw], patterns="*.csv", src_format=src_format)
    outcome = dataset.convert_all_tracks()
    assert (outcome.converted, outcome.failed) == (1, 0)
    index = read_tracks_index(dataset)
    assert len(index) == 1
    return index.iloc[0]


def test_a_conversion_records_the_distribution_and_leaves_the_variant_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, CONVERTING_SOURCE)
    found = _entry_point(ROWS, module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        cls = type(get_track_converter(ROWS))
        minted = tracks_run_id(
            converter_op(ROWS),
            cls.version,
            convert_variant_payload(cls.Params().identity_dump()),
        )

        first = _converted_row(tmp_path / "first", ROWS)
        origin = track_converter_origin(ROWS)
        assert origin is not None
        INSTALLED_STATE.origins[ROWS] = replace(origin, version="9.9.9")
        second = _converted_row(tmp_path / "second", ROWS)

    assert first["converter_distribution"] == "acme-tracks 1.2.0"
    assert second["converter_distribution"] == "acme-tracks 9.9.9"
    assert first["run_id"] == second["run_id"] == minted


def test_a_built_in_conversion_records_a_blank_distribution(tmp_path: Path) -> None:
    raw = tmp_path / "ds" / "raw"
    write_trex_npz(raw / "myseq_fish0.npz", individual=0)
    dataset = make_dataset(tmp_path / "ds", roots=("tracks_raw", "tracks"))
    dataset.meta["fps_default"] = 30.0
    _ = dataset.index_tracks_raw([raw], patterns="*.npz", src_format="trex_npz")
    _ = dataset.convert_all_tracks()

    row = read_tracks_index(dataset).iloc[0]

    assert row["converter_distribution"] == ""


# --- A format no converter holds ---------------------------------------------


def test_a_step_naming_an_uninstalled_format_is_refused_naming_the_group() -> None:
    with pytest.raises(ValidationError) as caught:
        _ = ConvertTracksParams(source_format="acme__absent")

    message = str(caught.value)
    assert "acme__absent" in message
    assert TRACK_CONVERTER_GROUP in message


def test_a_step_naming_a_refused_format_carries_the_recorded_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = plugin_module(tmp_path, monkeypatch, ACME_SOURCE)
    found = _entry_point("nonamespace", module, "AcmeConverter")

    with installed_track_converters(monkeypatch, found):
        recorded = _refusal_message("nonamespace")
        with pytest.raises(ValidationError) as caught:
            _ = ConvertTracksParams(source_format="nonamespace")

        assert recorded in str(caught.value)


def test_indexing_an_uninstalled_format_raises_the_same_message(
    tmp_path: Path,
) -> None:
    dataset = make_dataset(tmp_path / "ds")
    raw = dataset.get_root("tracks_raw")
    write_trex_npz(raw / "vid_id0.npz", individual=0)

    with pytest.raises(ValueError) as caught:
        _ = dataset.index_tracks_raw([raw], patterns="*.npz", src_format="acme__absent")

    with pytest.raises(ValueError) as expected:
        _ = get_track_converter("acme__absent")
    assert str(caught.value) == str(expected.value)
    assert TRACK_CONVERTER_GROUP in str(caught.value)


def _indexed_with_a_ghost_row(tmp_path: Path) -> Dataset:
    """A dataset whose raw index holds a TRex sequence and a row of an absent format."""
    dataset = make_dataset(tmp_path / "ds")
    raw = dataset.get_root("tracks_raw")
    write_trex_npz(raw / "vid_id0.npz", individual=0)
    _ = dataset.index_tracks_raw([raw], patterns="*.npz", src_format="trex_npz")
    index_path = raw / "index.csv"
    frame = load_tracks_raw_index_frame(index_path)
    ghost = frame.iloc[[0]].copy()
    ghost["src_format"] = "acme__absent"
    ghost["sequence"] = "ghost"
    ghost["abs_path"] = str(raw / "ghost.bin")
    ghost["md5"] = ""
    combined = pd.concat([frame, ghost], ignore_index=True)
    write_tracks_raw_index_rows(index_path, combined)
    return dataset


def test_a_raw_row_of_an_uninstalled_format_fails_alone(tmp_path: Path) -> None:
    dataset = _indexed_with_a_ghost_row(tmp_path)
    failures: list[tuple[str, BaseException]] = []

    def record(key: str, error: BaseException) -> None:
        failures.append((key, error))

    outcome = dataset.convert_all_tracks(on_failure=record)

    assert (outcome.converted, outcome.failed) == (1, 1)
    assert len(failures) == 1
    assert "acme__absent" in str(failures[0][1])
    assert TRACK_CONVERTER_GROUP in str(failures[0][1])


# --- A fresh interpreter -----------------------------------------------------

DISTRIBUTION_MODULE = """
from mosaic.core.track_converter import TrackConverter


class Rows(TrackConverter):
    src_format = "x__rows"
    version = "1.0"
    output_schema = "mosaic_v1"
    carries_tracklets = True

    def convert(self, path, params, hints):
        raise NotImplementedError
"""


def test_a_fresh_interpreter_finds_a_converter_that_a_distribution_installs(
    tmp_path: Path,
) -> None:
    script = """
        import sys

        from mosaic.core.track_converter import (
            get_track_converter,
            track_converter_origin,
        )

        assert "x_rows" not in sys.modules, "an entry point loaded on import"
        converter = get_track_converter("x__rows")
        assert type(converter).__name__ == "Rows"
        origin = track_converter_origin("x__rows")
        assert origin is not None
        print(origin.distribution, origin.version)
    """

    proc = run_with_distribution(
        tmp_path,
        f"[{TRACK_CONVERTER_GROUP}]\nx__rows = x_rows:Rows\n",
        {"x_rows": DISTRIBUTION_MODULE},
        [sys.executable, "-c", textwrap.dedent(script)],
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "x 1.0"
