"""The window file names a source dataset and the windows cut from its entries.

One TOML file holds the source, its tracks variant and a list of windows. The
reader refuses a file no conversion could honor before any source is opened, and
each refusal names the field it concerns. The window converters write each window
as one entry: the source rows of its frames, as stored, under its name.
"""

from __future__ import annotations

import sys
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import pytest

import mosaic.core.track_library as track_library
from mosaic.core.dataset import Dataset
from mosaic.core.entry import make_entry_key
from mosaic.core.pipeline.input_size import tracks_covariates
from mosaic.core.pipeline.tracks_index import (
    frame_measures,
    read_carries_tracklets,
    read_frame_extent,
    read_ids_per_frame,
    read_pair_frames,
    read_tracks_index,
    variant_entry_rows,
)
from mosaic.core.track_library.window import (
    WINDOW_FORMAT,
    WindowEntry,
    WindowSpec,
    read_window_spec,
)
from tests.helpers import (
    assert_tree_unchanged,
    imports_of,
    make_dataset,
    modules_loaded_by,
    set_tracks_cell,
    synthetic_dataset,
    tree_snapshot,
    write_synthetic_spec,
    write_trex_npz,
    write_window_spec,
)

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
    "pandas",
    "pydantic",
    "mosaic.core.params",
    "mosaic.core.schema",
    "mosaic.core.track_converter",
    "mosaic.core.strict_model",
    "mosaic.core.entry",
    "mosaic.user_paths",
}


def test_the_module_level_imports_stay_inside_the_allowed_modules() -> None:
    allowed = WINDOW_MODULE_IMPORTS | set(sys.stdlib_module_names) | {"__future__"}
    outside = {
        imported.module
        for imported in imports_of(LIBRARY / "window.py", scope="module", relative=True)
        if imported.module not in allowed
        and imported.module.split(".")[0] not in allowed
    }
    assert not outside, f"window.py imports {sorted(outside)} at module level"


def test_importing_the_track_library_loads_neither_the_dataset_nor_the_pipeline() -> (
    None
):
    loaded = modules_loaded_by(
        "import mosaic.core.track_library",
        ["mosaic.core.dataset", "mosaic.core.pipeline"],
    )

    assert loaded == []


# --- conversion -------------------------------------------------------------------

BODY = {"keypoints": 2, "length": 40.0, "keypoint_sd": 1.0}


def synthetic_source(
    base: Path,
    names: Sequence[str] = ("a", "b"),
    *,
    src_format: Literal["synthetic", "synthetic_3d"] = "synthetic",
    params: dict[str, object] | None = None,
) -> tuple[Dataset, str]:
    """A dataset converting one synthetic file per name, and its tracks variant."""
    ds = synthetic_dataset(base, names, src_format=src_format, body=BODY)
    outcome = ds.convert_all_tracks(params)
    assert outcome.failed == 0
    return ds, ds.track_conversion_variant(src_format, params)


def trex_source(base: Path, src_format: str) -> tuple[Dataset, str]:
    """A dataset converting one TRex sequence of two individuals under *src_format*."""
    ds = make_dataset(base)
    raw = ds.base_dir / "raw"
    for individual in (0, 1):
        write_trex_npz(raw / f"rec_fish{individual}.npz", n=8, cm_per_pixel=0.5)
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format=src_format)
    outcome = ds.convert_all_tracks()
    assert outcome.failed == 0
    return ds, ds.track_conversion_variant(src_format)


def window_dataset(
    base: Path,
    source: Dataset,
    run_id: str,
    windows: Sequence[Mapping[str, object]],
    *,
    src_format: str = "window_mosaic_v1",
    multi_sequences_per_file: bool = True,
    group_from_path: Callable[[Path], str] | None = None,
) -> Dataset:
    """A dataset indexing one window file of *source*, by default as a multi-sequence source."""
    ds = make_dataset(base / "dataset")
    files = base / "files"
    _ = write_window_spec(
        files / "windows.toml",
        source=source.manifest_path,
        tracks_run_id=run_id,
        windows=windows,
    )
    _ = ds.index_tracks_raw(
        [files],
        patterns=["*.toml"],
        src_format=src_format,
        multi_sequences_per_file=multi_sequences_per_file,
        group_from_path=group_from_path,
    )
    return ds


def entry_table(ds: Dataset, run_id: str, entry: tuple[str, str]) -> pd.DataFrame:
    (path,) = variant_entry_rows(ds, run_id, entry)["abs_path"]
    return pd.read_parquet(ds.resolve_path(str(path)))


def written(ds: Dataset) -> dict[tuple[str, str], pd.Series]:
    """The tracks-index row of each written entry."""
    return {
        (str(row["group"]), str(row["sequence"])): row
        for _, row in read_tracks_index(ds).iterrows()
    }


def without_entry(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=["group", "sequence"])


def source_slice(
    table: pd.DataFrame, start: int | None, stop: int | None
) -> pd.DataFrame:
    """The rows of *table* with ``start <= frame < stop``, in their stored order."""
    keep = pd.Series(True, index=table.index)
    if start is not None:
        keep &= table["frame"] >= start
    if stop is not None:
        keep &= table["frame"] < stop
    return without_entry(table[keep].reset_index(drop=True))


def failures_of(ds: Dataset, params: dict[str, object] | None = None) -> list[str]:
    """The failure messages of converting every window of *ds* under *params*."""
    messages: list[str] = []
    _ = ds.convert_all_tracks(
        params, on_failure=lambda _key, error: messages.append(str(error))
    )
    return messages


SPANS = [(None, 50), (50, 120), (120, None)]
TRACKLETS: dict[str, object] = {"carries_tracklets": True}


def bounded(sequence: str, start: int | None, stop: int | None) -> dict[str, object]:
    """A window of *sequence*, writing each bound that is set."""
    bounds = {"start": start, "stop": stop}
    return window(sequence, **{k: v for k, v in bounds.items() if v is not None})


def test_each_window_is_its_sources_rows_in_their_stored_order(tmp_path: Path) -> None:
    source, run_id = synthetic_source(tmp_path / "source")
    windows = [bounded("a", start, stop) for start, stop in SPANS]
    ds = window_dataset(tmp_path / "dest", source, run_id, [*windows, window("b")])

    outcome = ds.convert_all_tracks(TRACKLETS)

    assert (outcome.converted, outcome.failed) == (4, 0)
    rows = written(ds)
    assert sorted(rows) == sorted(
        [("", "a_ffirst-50"), ("", "a_f50-120"), ("", "a_f120-end"), ("", "b")]
    )
    a = entry_table(source, run_id, ("", "a"))
    assert any(column.startswith("poseP") for column in a.columns)
    for spec, (start, stop) in zip(windows, SPANS, strict=True):
        entry = WindowEntry.model_validate(spec).output_entry()
        table = pd.read_parquet(ds.resolve_path(str(rows[entry]["abs_path"])))
        assert set(table["group"]) == {""}
        assert set(table["sequence"]) == {entry[1]}
        pd.testing.assert_frame_equal(
            without_entry(table), source_slice(a, start, stop)
        )
    whole = pd.read_parquet(ds.resolve_path(str(rows["", "b"]["abs_path"])))
    pd.testing.assert_frame_equal(whole, entry_table(source, run_id, ("", "b")))


def test_each_window_records_its_own_measures_and_covariates(tmp_path: Path) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    windows = [bounded("a", start, stop) for start, stop in SPANS]
    ds = window_dataset(tmp_path / "dest", source, run_id, windows)
    _ = ds.convert_all_tracks(TRACKLETS)
    rows = written(ds)
    (variant,) = {str(row["run_id"]) for row in rows.values()}

    covariates = tracks_covariates(ds, list(rows), tracks_run_id=variant, variants=None)

    for entry, row in rows.items():
        measures = frame_measures(ds.resolve_path(str(row["abs_path"])))
        assert measures is not None
        table = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
        assert int(row["n_rows"]) == len(table)
        assert read_frame_extent(row) == measures.extent
        assert measures.extent == (int(table["frame"].min()), int(table["frame"].max()))
        assert read_pair_frames(row) == measures.pair_frames
        assert measures.ids_per_frame is not None
        assert read_ids_per_frame(row) == pytest.approx(
            measures.ids_per_frame, abs=5e-5
        )
        first, last = measures.extent
        assert covariates[entry].frames == last - first + 1
        assert covariates[entry].pair_frames == measures.pair_frames
    assert read_frame_extent(rows["", "a_f50-120"]) == (50, 119)


def test_a_source_without_tracklets_needs_the_param_that_says_so(
    tmp_path: Path,
) -> None:
    source, run_id = synthetic_source(
        tmp_path / "source", ["a"], params={"relabel_after_gap": True}
    )
    ds = window_dataset(tmp_path / "dest", source, run_id, [window("a", stop=100)])

    (message,) = failures_of(ds)
    outcome = ds.convert_all_tracks({"carries_tracklets": False})

    assert "carries_tracklets = false" in message
    assert "a" in message
    assert outcome.converted == 1
    assert [read_carries_tracklets(row) for row in written(ds).values()] == [False]


def test_a_source_with_tracklets_records_them_when_told(tmp_path: Path) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    ds = window_dataset(tmp_path / "dest", source, run_id, [window("a", stop=100)])

    (message,) = failures_of(ds)
    outcome = ds.convert_all_tracks(TRACKLETS)

    assert "carries_tracklets = true" in message
    assert outcome.converted == 1
    assert [read_carries_tracklets(row) for row in written(ds).values()] == [True]


def test_the_world_format_copies_a_3d_source(tmp_path: Path) -> None:
    source, run_id = synthetic_source(
        tmp_path / "source", ["a"], src_format="synthetic_3d"
    )
    ds = window_dataset(
        tmp_path / "dest",
        source,
        run_id,
        [window("a", start=20, stop=80)],
        src_format="window_mosaic_world_v1",
    )

    outcome = ds.convert_all_tracks(TRACKLETS)

    assert outcome.converted == 1
    (row,) = written(ds).values()
    assert row["std_format"] == "mosaic_world_v1"
    table = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
    assert {"Z", "poseZ0", "poseZ1"} <= set(table.columns)
    pd.testing.assert_frame_equal(
        without_entry(table),
        source_slice(entry_table(source, run_id, ("", "a")), 20, 80),
    )


def test_the_pixel_format_refuses_a_3d_source_naming_the_world_format(
    tmp_path: Path,
) -> None:
    source, run_id = synthetic_source(
        tmp_path / "source", ["a"], src_format="synthetic_3d"
    )
    ds = window_dataset(tmp_path / "dest", source, run_id, [window("a")])

    (message,) = failures_of(ds, TRACKLETS)

    assert "window_mosaic_world_v1" in message
    assert "'a'" in message


@pytest.mark.parametrize(
    ("source_format", "window_format", "schema"),
    [
        ("trex_npz", "window_trex_v2", "trex_v2"),
        ("trex_npz_cm", "window_mosaic_cm_v1", "mosaic_cm_v1"),
    ],
)
def test_a_trex_source_copies_with_its_columns_in_stored_order(
    tmp_path: Path, source_format: str, window_format: str, schema: str
) -> None:
    source, run_id = trex_source(tmp_path / "source", source_format)
    stored = entry_table(source, run_id, ("", "rec"))
    assert not stored["frame"].is_monotonic_increasing
    ds = window_dataset(
        tmp_path / "dest",
        source,
        run_id,
        [window("rec", start=2, stop=6)],
        src_format=window_format,
    )

    outcome = ds.convert_all_tracks(TRACKLETS)

    assert (outcome.converted, outcome.failed) == (1, 0)
    (row,) = written(ds).values()
    assert row["std_format"] == schema
    table = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))
    assert {"X#head", "Y#head"} <= set(table.columns)
    pd.testing.assert_frame_equal(without_entry(table), source_slice(stored, 2, 6))


def test_a_source_recording_no_schema_is_refused_naming_upgrade_tracks(
    tmp_path: Path,
) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    set_tracks_cell(source, "std_format", "")
    ds = window_dataset(tmp_path / "dest", source, run_id, [window("a")])

    (message,) = failures_of(ds, TRACKLETS)

    assert "mosaic upgrade-tracks --apply" in message
    assert "'a'" in message


def test_each_failed_window_is_its_own_failed_entry(tmp_path: Path) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    raw = source.base_dir / "raw"
    _ = write_synthetic_spec(raw / "c.toml", seed=9)
    _ = source.index_tracks_raw([raw], patterns=["*.toml"], src_format="synthetic")
    assert source.convert_all_tracks({"relabel_after_gap": True}).failed == 0
    windows = [
        window("missing"),
        window("c"),
        window("a", start=500, stop=600),
        window("a", start=10, stop=40),
    ]
    ds = window_dataset(tmp_path / "dest", source, run_id, windows)
    failed: dict[str, str] = {}

    outcome = ds.convert_all_tracks(
        TRACKLETS,
        on_failure=lambda key, error: failed.update({key: str(error)}),
    )

    assert (outcome.converted, outcome.failed) == (1, 3)
    assert list(failed) == [
        make_entry_key("", "missing"),
        make_entry_key("", "c"),
        make_entry_key("", "a_f500-600"),
    ]
    assert "'missing'" in failed[make_entry_key("", "missing")]
    assert run_id in failed[make_entry_key("", "c")]
    assert "500" in failed[make_entry_key("", "a_f500-600")]
    assert list(written(ds)) == [("", "a_f10-40")]


@pytest.mark.parametrize(
    ("bounds", "condition"),
    [({"stop": 0}, "frame < 0"), ({"start": 500}, "500 <= frame")],
)
def test_an_empty_window_with_one_bound_names_only_that_bound(
    tmp_path: Path, bounds: dict[str, int], condition: str
) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    ds = window_dataset(tmp_path / "dest", source, run_id, [window("a", **bounds)])

    (message,) = failures_of(ds, TRACKLETS)

    assert f"no row with {condition} in" in message
    assert "None" not in message


def test_a_conversion_leaves_the_source_dataset_unchanged(tmp_path: Path) -> None:
    source, run_id = synthetic_source(tmp_path / "source")
    before = tree_snapshot(source.base_dir)
    ds = window_dataset(
        tmp_path / "dest", source, run_id, [window("a", stop=60), window("b")]
    )

    outcome = ds.convert_all_tracks(TRACKLETS)

    assert outcome.converted == 2
    assert_tree_unchanged(source.base_dir, before)


def test_two_datasets_converting_one_file_agree_to_the_byte(tmp_path: Path) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    outputs: list[tuple[str, bytes]] = []
    for name in ("first", "second"):
        ds = window_dataset(
            tmp_path / name, source, run_id, [window("a", start=30, stop=90)]
        )
        _ = ds.convert_all_tracks(TRACKLETS)
        (row,) = written(ds).values()
        outputs.append(
            (str(row["run_id"]), ds.resolve_path(str(row["abs_path"])).read_bytes())
        )

    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]


def in_trials(_path: Path) -> str:
    """The group of every file, for a source that files each under one group."""
    return "trials"


@pytest.mark.parametrize(
    "group_from_path", [None, in_trials], ids=["ungrouped", "grouped"]
)
def test_a_window_file_indexed_as_one_sequence_says_to_declare_many(
    tmp_path: Path, group_from_path: Callable[[Path], str] | None
) -> None:
    source, run_id = synthetic_source(tmp_path / "source", ["a"])
    ds = window_dataset(
        tmp_path / "dest",
        source,
        run_id,
        [window("a")],
        multi_sequences_per_file=False,
        group_from_path=group_from_path,
    )

    (message,) = failures_of(ds, TRACKLETS)

    assert "--multi-sequences-per-file" in message
    assert "windows.toml" in message
