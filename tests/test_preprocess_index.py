"""The media variant index, and where a variant's files sit in a dataset.

A variant's row is what a consumer reads instead of probing the file: the
placement that maps the file back to its entry's source, the file's probed facts,
and what the entry's media was when the file was written. The index lives in the
``preprocess`` kind directory under the media root, a directory every media scan
steps over, so a variant is never indexed as source media.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import MediaFacts

from mosaic.core.dataset import Dataset
from mosaic.core.manifest import MediaScanSource
from mosaic.core.media.facts_columns import FACTS_COLUMNS, store_facts
from mosaic.core.media.preprocess import CropStep, Placement, TrimStep
from mosaic.core.pipeline.composition import compositions_disagree
from mosaic.core.pipeline.dataset_indexes import iter_dataset_indexes
from mosaic.core.pipeline.preprocess_index import (
    MediaVariantDriftedError,
    MediaVariantMissingError,
    MediaVariantRow,
    media_variant_index,
    media_variant_row,
    read_media_variant_index,
    variant_facts,
    variant_placement,
    variant_row,
    write_media_variant_row,
)
from mosaic.core.pipeline.preprocess_layout import (
    PREPROCESS_KIND_DIRECTORY,
    media_variant_index_path,
    media_variant_path,
    media_variant_run_root,
)

from tests.helpers import make_dataset

_RUN = "preprocess.0.1-aaaaaaaaaa"


def _facts(frame_count: int = 300, video_uuid: str = "uuid-variant") -> MediaFacts:
    return store_facts(
        320, 240, 30.0, frame_count, "av1", frame_count / 30.0, video_uuid, ""
    )


def _placement() -> Placement:
    placement = Placement.identity(640, 480, 600, 30.0)
    for step in (
        CropStep(x=120, y=40, width=320, height=240),
        TrimStep(start=100, stop=400),
    ):
        placement = step.place(placement)
    return placement


def _write(
    ds: Dataset,
    *,
    sequence: str = "s",
    camera: str = "",
    composition: str = "composition-1",
    facts: MediaFacts | None = None,
) -> MediaVariantRow:
    """Write a variant file and its row, as the op does: the file first."""
    path = media_variant_path(ds, _RUN, "g", sequence, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"variant")
    row = media_variant_row(
        ds,
        path=path,
        run_id=_RUN,
        group="g",
        sequence=sequence,
        camera=camera,
        upstream="",
        upstream_video_uuid="",
        placement=_placement(),
        facts=_facts() if facts is None else facts,
        encoder="libsvtav1",
        consumed_media_composition=composition,
    )
    write_media_variant_row(ds, row)
    return row


def _found(ds: Dataset, sequence: str = "s", camera: str = "") -> dict[str, str]:
    row = variant_row(ds, _RUN, "g", sequence, camera)
    assert row is not None
    return row


# --- the path rule -----------------------------------------------------------


def test_a_variant_file_sits_under_its_run_in_the_media_kind_directory(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    run_root = ds.get_root("media") / PREPROCESS_KIND_DIRECTORY / _RUN

    assert media_variant_run_root(ds, _RUN) == run_root
    assert media_variant_path(ds, _RUN, "g", "s", "") == run_root / "g__s.mp4"
    assert media_variant_index_path(ds) == run_root.parent / "index.csv"


def test_a_named_camera_adds_a_directory_level(tmp_path: Path) -> None:
    """The rule frame extraction follows, so two cameras never collide."""
    ds = make_dataset(tmp_path / "ds")

    path = media_variant_path(ds, _RUN, "g", "s", "cam0")

    assert path == media_variant_run_root(ds, _RUN) / "g__s" / "cam0.mp4"


# --- the writer and the reader -----------------------------------------------


def test_an_absent_index_reads_as_the_full_schema(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    frame = read_media_variant_index(ds)

    columns = [field.name for field in dataclasses.fields(MediaVariantRow)]
    assert frame.empty
    assert list(frame.columns) == columns
    assert set(FACTS_COLUMNS) <= set(columns)
    assert variant_row(ds, _RUN, "g", "s", "") is None


def test_a_rewritten_entry_replaces_its_row_and_leaves_the_others(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _write(ds, sequence="s")
    _ = _write(ds, sequence="t")

    _ = _write(ds, sequence="s", composition="composition-2")

    frame = read_media_variant_index(ds)
    assert sorted(frame["sequence"]) == ["s", "t"]
    assert _found(ds, "s")["consumed_media_composition"] == "composition-2"
    assert _found(ds, "t")["consumed_media_composition"] == "composition-1"


def test_two_cameras_of_one_entry_are_two_rows(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _write(ds, camera="cam0")
    _ = _write(ds, camera="cam1")

    assert len(read_media_variant_index(ds)) == 2
    assert (
        _found(ds, camera="cam0")["abs_path"] != _found(ds, camera="cam1")["abs_path"]
    )


def test_the_path_is_stored_relative_and_resolves_after_a_move(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _write(ds)
    stored = _found(ds)["abs_path"]

    _ = shutil.move(tmp_path / "ds", tmp_path / "moved")
    moved = Dataset(manifest_path=tmp_path / "moved" / "dataset.yaml").load()

    assert not Path(stored).is_absolute()
    assert stored == f"media/{PREPROCESS_KIND_DIRECTORY}/{_RUN}/g__s.mp4"
    assert moved.resolve_path(_found(moved)["abs_path"]).read_bytes() == b"variant"


def test_the_placement_and_the_facts_round_trip(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    facts = _facts(frame_count=150, video_uuid="uuid-150")
    _ = _write(ds, facts=facts)

    row = _found(ds)

    assert variant_placement(row) == _placement()
    assert variant_facts(row) == facts
    assert row["video_uuid"] == "uuid-150"
    assert (row["width"], row["height"], row["codec"]) == ("320", "240", "av1")
    assert row["encoder"] == "libsvtav1"


def test_an_entry_name_that_is_not_one_path_component_is_refused(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path / "ds")
    row = _write(ds)

    with pytest.raises(ValueError, match="sequence"):
        write_media_variant_row(ds, dataclasses.replace(row, sequence="a/b"))


def test_a_missing_variant_is_a_missing_file_and_a_drifted_one_a_wrong_value() -> None:
    """The two failures have different remedies, so they are different classes."""
    assert issubclass(MediaVariantMissingError, FileNotFoundError)
    assert issubclass(MediaVariantDriftedError, ValueError)
    assert not issubclass(MediaVariantDriftedError, FileNotFoundError)


# --- compositions ------------------------------------------------------------


@pytest.mark.parametrize(
    ("recorded", "current", "disagree"),
    [
        ("", "", False),
        ("", "now", False),
        ("was", "", False),
        ("same", "same", False),
        ("was", "now", True),
    ],
)
def test_only_two_known_compositions_that_differ_disagree(
    recorded: str, current: str, disagree: bool
) -> None:
    """A blank on either side is unknown, and unknown is not drift."""
    assert compositions_disagree(recorded, current) is disagree


# --- the dataset-wide passes -------------------------------------------------


def test_the_index_is_enumerated_under_its_kind(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")

    found = {i.path: i.root_key for i in iter_dataset_indexes(ds)}

    assert found[media_variant_index_path(ds)] == PREPROCESS_KIND_DIRECTORY
    assert found[ds.get_root("media") / "index.csv"] == "media"


def test_loading_a_dataset_registers_the_index() -> None:
    """The passes run from a dataset, which never imports the index module itself."""
    probe = (
        "from mosaic.core.dataset import Dataset\n"
        "from mosaic.core.pipeline.dataset_indexes import reconcilable_index\n"
        f"print(reconcilable_index({PREPROCESS_KIND_DIRECTORY!r}) is not None)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert completed.stdout.strip() == "True", completed.stderr


def test_reindex_drops_a_row_whose_file_is_gone(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    _ = _write(ds, sequence="s")
    _ = _write(ds, sequence="t")
    media_variant_path(ds, _RUN, "g", "t", "").unlink()

    dropped = ds.reindex(PREPROCESS_KIND_DIRECTORY, dry_run=False)

    assert dropped == {str(media_variant_index_path(ds)): 1}
    assert list(read_media_variant_index(ds)["sequence"]) == ["s"]


def test_reindex_leaves_the_media_index_alone(tmp_path: Path) -> None:
    """The derivative index has no ``IndexCSV`` behind it and needs its own pass."""
    ds = make_dataset(tmp_path / "ds")

    assert ds.reindex("media", dry_run=False) == {}


def test_make_portable_relativizes_an_absolute_variant_path(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    row = _write(ds)
    absolute = media_variant_path(ds, _RUN, "g", "s", "").resolve()
    index = media_variant_index(media_variant_index_path(ds))
    index.append([dataclasses.replace(row, abs_path=absolute)])

    changed = ds.make_portable()

    assert changed[str(media_variant_index_path(ds))] == 1
    assert not Path(_found(ds)["abs_path"]).is_absolute()


# --- the scan exclusion ------------------------------------------------------


def _scanned_names(ds: Dataset) -> list[str]:
    index = pd.read_csv(ds.scan_media(), keep_default_na=False)
    return sorted(Path(str(p)).name for p in index["abs_path"])


def test_a_media_scan_indexes_no_variant_and_every_user_folder(
    tmp_path: Path,
    write_cfr_mp4: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A dataset whose originals are in ``media/``, scanned recursively from there.

    Without the exclusion a variant would be indexed as an original, earning a
    ``video_uuid`` and a place in its entry's media composition, which would then
    move and read every variant as drifted.
    """
    ds = make_dataset(tmp_path / "ds", roots=["media", "tracks"])
    media = ds.get_root("media")
    write_cfr_mp4(media / "clips" / "seq_a.mp4", frames=6)
    write_cfr_mp4(media / "lab" / PREPROCESS_KIND_DIRECTORY / "seq_b.mp4", frames=7)
    write_cfr_mp4(media_variant_path(ds, _RUN, "", "seq_a", ""), frames=8)
    ds.add_scan_source(MediaScanSource(id="media", path="media"))

    names = _scanned_names(ds)

    assert ds.resolve_media_root() == "media"
    assert names == ["seq_a.mp4", "seq_b.mp4"]
    assert "[INFO] skipped 1 media variant file(s)" in capsys.readouterr().err


def test_a_folder_named_media_preprocess_outside_the_dataset_is_scanned(
    tmp_path: Path, write_cfr_mp4: Callable[..., None]
) -> None:
    """Only this dataset's own variants directory is stepped over.

    A lab share laid out as ``media/preprocess/`` holds originals, and a source
    may point anywhere, so matching the directory names would drop every file in
    it.
    """
    ds = make_dataset(tmp_path / "ds", roots=["media", "tracks"])
    outside = tmp_path / "lab" / "media" / PREPROCESS_KIND_DIRECTORY
    write_cfr_mp4(outside / "trial.mp4", frames=6)
    ds.add_scan_source(MediaScanSource(id="lab", path=str(outside.parent)))

    assert _scanned_names(ds) == ["trial.mp4"]


def test_a_variant_under_a_renamed_media_root_is_not_scanned(
    tmp_path: Path, write_cfr_mp4: Callable[..., None]
) -> None:
    """The variants directory is found through the media root, whatever its name."""
    ds = make_dataset(tmp_path / "ds", roots=["media", "tracks"])
    ds.set_root("media", tmp_path / "ds" / "videos")
    videos = ds.get_root("media")
    write_cfr_mp4(videos / "clips" / "seq_a.mp4", frames=6)
    write_cfr_mp4(media_variant_path(ds, _RUN, "", "seq_b", ""), frames=8)
    ds.add_scan_source(MediaScanSource(id="videos", path="videos"))

    assert media_variant_path(ds, _RUN, "", "seq_b", "").is_relative_to(videos)
    assert _scanned_names(ds) == ["seq_a.mp4"]
