"""The window file names a source dataset and the windows cut from its entries.

One TOML file holds the source, its tracks variant and a list of windows. The
reader refuses a file no conversion could honor before any source is opened, and
each refusal names the field it concerns.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

import mosaic.core.track_library as track_library
from mosaic.core.track_library.window import (
    WINDOW_FORMAT,
    WindowEntry,
    WindowSpec,
    read_window_spec,
)
from tests.helpers import run_fresh, write_window_spec

SOURCE = Path("guppies") / "dataset.yaml"
RUN_ID = "convert-trex_npz.0.1-abc123"


def window(sequence: str = "trial_01", **fields: object) -> dict[str, object]:
    """One window table of the file, with *fields* beside its sequence."""
    return {"sequence": sequence, **fields}


def write(
    tmp_path: Path,
    windows: Sequence[Mapping[str, object]] | None = None,
    *,
    source: Path = SOURCE,
    tracks_run_id: str = RUN_ID,
) -> Path:
    return write_window_spec(
        tmp_path / "windows.toml",
        source=source,
        tracks_run_id=tracks_run_id,
        windows=[window()] if windows is None else windows,
    )


def refusal(path: Path) -> str:
    """The message of the ValueError that reading *path* raises."""
    with pytest.raises(ValueError) as caught:
        read_window_spec(path)
    return str(caught.value)


def test_a_file_reads_with_every_default(tmp_path: Path) -> None:
    path = write(tmp_path)

    spec = read_window_spec(path)

    assert spec.format == WINDOW_FORMAT == "mosaic-window/1"
    assert spec.tracks_run_id == RUN_ID
    assert len(spec.windows) == 1
    entry = spec.windows[0]
    assert (entry.group, entry.sequence) == ("", "trial_01")
    assert (entry.start, entry.stop) == (None, None)
    assert (entry.as_group, entry.as_sequence) == (None, None)
    assert entry.output_entry() == ("", "trial_01")


def test_a_file_reads_every_window_in_order(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        [
            window("a", group="g1", start=0, stop=10, as_group="out", as_sequence="x"),
            window("b", start=10),
        ],
    )

    spec = read_window_spec(path)

    first, second = spec.windows
    assert (first.group, first.sequence, first.start, first.stop) == ("g1", "a", 0, 10)
    assert (first.as_group, first.as_sequence) == ("out", "x")
    assert (second.sequence, second.start, second.stop) == ("b", 10, None)


def test_a_relative_source_resolves_against_the_files_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "files"
    folder.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = write_window_spec(
        folder / "windows.toml",
        source=Path("..") / "guppies" / "dataset.yaml",
        tracks_run_id=RUN_ID,
        windows=[window()],
    )
    monkeypatch.chdir(elsewhere)

    spec = read_window_spec(path)

    assert spec.source == folder / ".." / "guppies" / "dataset.yaml"
    assert spec.source.is_absolute()


def test_an_absolute_source_is_kept(tmp_path: Path) -> None:
    source = tmp_path / "other" / "dataset.yaml"
    path = write(tmp_path, source=source)

    assert read_window_spec(path).source == source


def test_a_leading_tilde_in_the_source_expands_to_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    path = write(tmp_path, source=Path("~") / "data" / "x" / "dataset.yaml")

    assert read_window_spec(path).source == home / "data" / "x" / "dataset.yaml"


def test_a_source_directory_is_resolved_as_given_and_not_opened(
    tmp_path: Path,
) -> None:
    path = write(tmp_path, source=Path("missing_dataset"))

    assert read_window_spec(path).source == tmp_path / "missing_dataset"


def test_a_syntax_error_names_the_path(tmp_path: Path) -> None:
    path = tmp_path / "windows.toml"
    path.write_text("format = \n[[windows\n")

    message = refusal(path)

    assert str(path) in message
    assert "not valid TOML" in message


def test_a_format_other_than_the_window_format_is_refused(tmp_path: Path) -> None:
    path = write(tmp_path)
    path.write_text(path.read_text().replace("mosaic-window/1", "mosaic-synthetic/1"))

    message = refusal(path)

    assert str(path) in message
    assert "format" in message
    assert "mosaic-window/1" in message


def test_an_unknown_key_is_refused_naming_it(tmp_path: Path) -> None:
    message = refusal(write(tmp_path, [window(stride=2)]))

    assert "windows.0.stride" in message


@pytest.mark.parametrize("missing", ["source", "tracks_run_id", "windows"])
def test_a_missing_top_level_key_is_refused_naming_it(
    tmp_path: Path, missing: str
) -> None:
    path = write(tmp_path)
    lines = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(f"{missing} =")
    ]
    if missing == "windows":
        lines = [line for line in lines if not line.startswith(("[[", "sequence"))]
    path.write_text("\n".join(lines) + "\n")

    assert missing in refusal(path)


def test_an_empty_tracks_run_id_is_refused(tmp_path: Path) -> None:
    assert "tracks_run_id" in refusal(write(tmp_path, tracks_run_id=""))


def test_an_empty_windows_list_is_refused(tmp_path: Path) -> None:
    assert "windows" in refusal(write(tmp_path, []))


def test_a_window_needs_a_sequence(tmp_path: Path) -> None:
    assert "windows.0.sequence" in refusal(write(tmp_path, [{"group": "g"}]))


@pytest.mark.parametrize(("start", "stop"), [(10, 10), (11, 10)])
def test_a_window_whose_start_reaches_its_stop_is_refused(
    tmp_path: Path, start: int, stop: int
) -> None:
    message = refusal(write(tmp_path, [window(start=start, stop=stop)]))

    assert "windows.0" in message
    assert "start" in message
    assert "stop" in message


def test_a_window_with_one_bound_is_accepted(tmp_path: Path) -> None:
    spec = read_window_spec(write(tmp_path, [window(start=5), window("b", stop=5)]))

    assert [(w.start, w.stop) for w in spec.windows] == [(5, None), (None, 5)]


@pytest.mark.parametrize("field", ["as_group", "as_sequence"])
@pytest.mark.parametrize("name", ["a/b", "a\\b"])
def test_an_output_name_that_is_not_one_path_component_is_refused(
    tmp_path: Path, field: str, name: str
) -> None:
    message = refusal(write(tmp_path, [window(**{field: name})]))

    assert f"windows.0: {field} may not contain" in message


def test_a_default_output_name_built_from_a_refused_sequence_is_refused(
    tmp_path: Path,
) -> None:
    message = refusal(write(tmp_path, [window("a/b", start=0, stop=5)]))

    assert "windows.0" in message
    assert "as_sequence" in message
    assert "a/b" in message


def test_a_default_output_group_built_from_a_refused_group_is_refused(
    tmp_path: Path,
) -> None:
    message = refusal(write(tmp_path, [window(group="x/y")]))

    assert "windows.0" in message
    assert "as_group" in message


def test_an_explicit_name_replaces_a_default_the_validator_would_refuse(
    tmp_path: Path,
) -> None:
    spec = read_window_spec(
        write(tmp_path, [window("a/b", group="x/y", as_group="g", as_sequence="s")])
    )

    assert spec.windows[0].output_entry() == ("g", "s")


def test_the_default_output_entry_is_the_source_entry_without_bounds() -> None:
    entry = WindowEntry(group="g", sequence="trial_01")

    assert entry.output_entry() == ("g", "trial_01")


@pytest.mark.parametrize(
    ("start", "stop", "expected"),
    [
        (0, 200000, "trial_01_f0-200000"),
        (100, 250, "trial_01_f100-250"),
        (None, 250, "trial_01_ffirst-250"),
        (100, None, "trial_01_f100-end"),
    ],
)
def test_the_default_output_sequence_names_the_bounds(
    start: int | None, stop: int | None, expected: str
) -> None:
    entry = WindowEntry(group="g", sequence="trial_01", start=start, stop=stop)

    assert entry.output_entry() == ("g", expected)


def test_an_explicit_output_name_replaces_the_default() -> None:
    entry = WindowEntry(
        group="g", sequence="s", start=0, stop=5, as_group="out", as_sequence="mine"
    )

    assert entry.output_entry() == ("out", "mine")


def test_an_explicit_empty_output_group_replaces_the_source_group() -> None:
    entry = WindowEntry(group="g", sequence="s", as_group="")

    assert entry.output_entry() == ("", "s")


def test_two_windows_writing_one_entry_are_refused(tmp_path: Path) -> None:
    windows = [window(as_sequence="same"), window("other", as_sequence="same")]

    message = refusal(write(tmp_path, windows))

    assert message.count("windows: ") == 1
    assert "same" in message
    assert "windows[0]" in message


def test_two_windows_with_one_default_name_are_refused(tmp_path: Path) -> None:
    windows = [window(start=0, stop=5), window(start=0, stop=5)]

    message = refusal(write(tmp_path, windows))

    assert "trial_01_f0-5" in message


def test_a_window_whose_default_collides_with_an_explicit_name_is_refused(
    tmp_path: Path,
) -> None:
    windows = [window("a"), window("b", as_sequence="a")]

    assert "windows[0]" in refusal(write(tmp_path, windows))


def test_windows_of_one_source_sequence_with_different_bounds_are_accepted(
    tmp_path: Path,
) -> None:
    spec = read_window_spec(
        write(tmp_path, [window(start=0, stop=5), window(start=5, stop=10)])
    )

    assert [w.output_entry()[1] for w in spec.windows] == [
        "trial_01_f0-5",
        "trial_01_f5-10",
    ]


def test_windows_in_different_groups_may_share_a_sequence_name(
    tmp_path: Path,
) -> None:
    spec = read_window_spec(write(tmp_path, [window(group="g1"), window(group="g2")]))

    assert len(spec.windows) == 2


def test_the_model_is_the_file_read_without_resolving(tmp_path: Path) -> None:
    document = tomllib.loads(write(tmp_path).read_text())

    spec = WindowSpec.model_validate(document)

    assert spec.source == SOURCE


def test_write_window_spec_serializes_the_arguments_as_arrays_of_tables(
    tmp_path: Path,
) -> None:
    path = write(
        tmp_path,
        [window("a", start=0, stop=5, as_group=""), window("b", group="g")],
    )

    assert tomllib.loads(path.read_text()) == {
        "format": "mosaic-window/1",
        "source": str(SOURCE),
        "tracks_run_id": RUN_ID,
        "windows": [
            {"sequence": "a", "start": 0, "stop": 5, "as_group": ""},
            {"sequence": "b", "group": "g"},
        ],
    }


LIBRARY = Path(track_library.__file__).parent
WINDOW_MODULE_IMPORTS = {
    "pydantic",
    "mosaic.core.strict_model",
    "mosaic.core.entry",
    "mosaic.user_paths",
}


def test_the_module_level_imports_stay_inside_the_allowed_modules() -> None:
    tree = ast.parse((LIBRARY / "window.py").read_text())
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "window.py imports relatively"
            imported.add(node.module or "")

    allowed = WINDOW_MODULE_IMPORTS | set(sys.stdlib_module_names) | {"__future__"}
    outside = {
        module
        for module in imported
        if module not in allowed and module.split(".")[0] not in allowed
    }
    assert not outside, f"window.py imports {sorted(outside)} at module level"


def test_importing_the_window_module_loads_neither_the_dataset_nor_the_pipeline() -> (
    None
):
    loaded = run_fresh(
        "import sys\n"
        "import mosaic.core.track_library.window\n"
        "print(sorted(m for m in sys.modules if m in ('mosaic.core.dataset', "
        "'mosaic.core.pipeline')) or 'neither')"
    )

    assert loaded == "neither"
