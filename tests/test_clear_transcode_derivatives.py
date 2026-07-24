"""Tests for the transcode-derivative clearing sweep."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.media_index import (
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from scripts.clear_transcode_derivatives import (
    ClearReport,
    clear_transcode_derivatives,
)


@pytest.fixture
def transcoded_dataset(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> Dataset:
    """A two-root dataset holding one derivative: an index link, a row, and a file.

    ``media_raw/index.csv`` carries an original whose ``analysis_derivative_path``
    forward-links to the derivative; ``media/index.csv`` carries the one
    derivative row, whose ``abs_path`` resolves to a stub file under
    ``media/transcode/``. The stub is written rather than transcoded: the sweep
    never reads a derivative's content, so a real encode buys nothing. One
    derivative file and one derivative row pin ``files_removed`` and
    ``rows_removed`` to 1.
    """
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)

    original = base / "media_raw" / "entry.mp4"
    original.touch()

    transcode_root = base / "media" / "transcode"
    transcode_root.mkdir(parents=True, exist_ok=True)
    derivative = transcode_root / "U.recipe.analysis.mp4"
    derivative.write_bytes(b"stub")

    originals: list[dict[str, object]] = [
        {
            "name": "entry.mp4",
            "group": "g1",
            "sequence": "entry",
            "abs_path": dataset.relative_to_root(str(original)),
            "video_uuid": "U",
            # Curated cells the sweep must preserve. video_order and the
            # comma-bearing media_facts JSON are the ones a careless rewrite
            # would drop; the test pins them to these seeded values.
            "video_order": 3,
            "media_facts": '{"video_uuid": "U", "frame_count": 6}',
            "analysis_derivative_path": "transcode/U.recipe.analysis.mp4",
        }
    ]
    write_media_index_rows(
        dataset.get_root("media_raw") / "index.csv", frame_from_rows(originals)
    )

    derivatives: list[dict[str, object]] = [
        {
            "name": derivative.name,
            "group": "g1",
            "sequence": "entry",
            "abs_path": dataset.relative_to_root(str(derivative)),
            "source_video_uuid": "U",
            "recipe_hash": "recipe",
        }
    ]
    write_media_index_rows(
        dataset.get_root("media") / "index.csv", frame_from_rows(derivatives)
    )
    return dataset


@pytest.fixture
def single_root_dataset(tmp_path: Path) -> Dataset:
    """A ``media``-root-only dataset whose index carries a live link cell.

    With no ``media_raw`` root, ``media`` is the originals index and the sweep
    must decline. The seeded ``analysis_derivative_path`` is what makes the
    all-zeros report meaningful: were the gate not to fire, the sweep would
    strip this cell and rewrite the file, so a byte-identical index proves the
    gate declined rather than proving the fixture empty.
    """
    base = (tmp_path / "dataset").resolve()
    dataset = Dataset(
        manifest_path=base / "dataset.yaml",
        roots={"media": str(base / "media")},
    )
    dataset.ensure_roots()
    dataset.save()

    original = base / "media" / "entry.mp4"
    original.touch()
    rows: list[dict[str, object]] = [
        {
            "name": "entry.mp4",
            "group": "g1",
            "sequence": "entry",
            "abs_path": dataset.relative_to_root(str(original)),
            "video_uuid": "U",
            "analysis_derivative_path": "transcode/U.recipe.analysis.mp4",
        }
    ]
    write_media_index_rows(
        dataset.get_root("media") / "index.csv", frame_from_rows(rows)
    )
    return dataset


def test_it_clears_links_rows_and_files(transcoded_dataset: Dataset) -> None:
    ds = transcoded_dataset
    report = clear_transcode_derivatives(ds, apply=True)
    originals = read_media_index(ds.get_root("media_raw") / "index.csv")
    assert all(not row["analysis_derivative_path"] for row in originals)
    assert all(not row["playback_derivative_path"] for row in originals)
    # The row survives whole -- clearing the links is the only edit. Curated
    # cells, including the comma-bearing media_facts JSON, keep their seeded
    # values; a rewrite that dropped any would go red here.
    (surviving,) = originals
    assert surviving["group"] == "g1"
    assert surviving["sequence"] == "entry"
    assert surviving["video_order"] == "3"
    assert surviving["media_facts"] == '{"video_uuid": "U", "frame_count": 6}'
    assert read_media_index(ds.get_root("media") / "index.csv") == []
    assert not (ds.get_root("media") / "transcode").exists()
    assert report.files_removed == 1


def test_it_never_touches_the_frames_tree(transcoded_dataset: Dataset) -> None:
    ds = transcoded_dataset
    frames = ds.get_root("media") / "frames" / "kmeans" / "run" / "seq"
    frames.mkdir(parents=True)
    (frames / "frame_000001.png").write_bytes(b"x")
    _ = clear_transcode_derivatives(ds, apply=True)
    assert (frames / "frame_000001.png").exists()


def test_a_single_root_dataset_is_untouched(single_root_dataset: Dataset) -> None:
    ds = single_root_dataset
    before = (ds.get_root("media") / "index.csv").read_text()
    report = clear_transcode_derivatives(ds, apply=True)
    assert not report.considered
    assert (report.links_cleared, report.rows_removed, report.files_removed) == (
        0,
        0,
        0,
    )
    assert (ds.get_root("media") / "index.csv").read_text() == before


def test_dry_run_writes_nothing(transcoded_dataset: Dataset) -> None:
    ds = transcoded_dataset
    before = (ds.get_root("media") / "index.csv").read_text()
    report = clear_transcode_derivatives(ds, apply=False)
    assert report.rows_removed == 1
    assert (ds.get_root("media") / "index.csv").read_text() == before


def test_report_is_a_clear_report(transcoded_dataset: Dataset) -> None:
    report = clear_transcode_derivatives(transcoded_dataset, apply=False)
    assert isinstance(report, ClearReport)
    assert report.considered
