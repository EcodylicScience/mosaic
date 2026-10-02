"""Rescaling a centimetre-era TRex table whose raw export is gone.

Reconverting is the ordinary route and the better one. This covers the case
where it is not available: ``sweep-tracking`` reclaims a finished tracker run
past its retention window, leaving tables whose ``.npz`` files no longer exist.

It is possible at all because the factor survived in the table -- TRex writes
``cm_per_pixel`` into every export, the old conversion copied every field
through, and the flattener pads a one-element array rather than dropping it. The
tests below therefore also guard that inheritance: if a table stops carrying the
factor, this stops working, and the failure would otherwise look like a
migration bug rather than a lost input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.tracks_identity import tracks_variant_root
from mosaic.core.pipeline.tracks_index import read_tracks_index, write_tracks_row
from mosaic.core.pipeline.upgrade_tracks import upgrade_trex_tables
from mosaic.core.pipeline.writers import write_parquet_atomic

from tests.helpers import make_dataset


def _legacy_trex_table(
    ds: Dataset,
    sequence: str,
    *,
    cm_per_pixel: float,
    pixels: np.ndarray,
    variant: str = "convert-trex_npz.0.1-aaaaaaaaaa",
    producer: str = "convert-trex_npz",
    std_format: str = "trex_v1",
    carry_calibration: bool = True,
) -> Path:
    """A table as the centimetre-era conversion left it: cm, with X at the head."""
    n = len(pixels)
    columns: dict[str, object] = {
        "frame": np.arange(n, dtype=np.int64),
        "time": np.arange(n, dtype=float) / 30.0,
        "id": np.zeros(n, dtype=np.int64),
        "group": [""] * n,
        "sequence": [sequence] * n,
        # Everything positional scaled, exactly as TRex reported it.
        "X": pixels * cm_per_pixel + 5.0,  # the head, offset from the centre
        "Y": pixels * cm_per_pixel + 5.0,
        "X#wcentroid": pixels * cm_per_pixel,
        "Y#wcentroid": pixels * cm_per_pixel,
        "SPEED": pixels * cm_per_pixel,
        # Keypoints were always pixels, even when the rest was not.
        "poseX0": pixels,
        "poseY0": pixels,
    }
    if carry_calibration:
        padded = np.full(n, np.nan)
        padded[0] = cm_per_pixel
        columns["cm_per_pixel"] = padded

    root = tracks_variant_root(ds.get_root("tracks"), variant)
    out_path = root / f"{sequence}.parquet"
    _ = write_parquet_atomic(pd.DataFrame(columns), out_path)
    write_tracks_row(
        ds,
        run_id=variant,
        group="",
        sequence=sequence,
        out_path=out_path,
        producer=producer,
        std_format=std_format,
        n_rows=n,
        consumed_source_roots=("tracks_raw",),
    )
    return out_path


def test_a_dry_run_reports_without_writing(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    _ = _legacy_trex_table(
        ds, "seq_a", cm_per_pixel=0.25, pixels=np.linspace(0.0, 100.0, 6)
    )

    report = upgrade_trex_tables(ds, apply=False)

    assert len(report.upgraded) == 1
    assert len(read_tracks_index(ds)) == 1, "nothing written on a dry run"


def test_an_upgrade_recovers_the_original_pixels(tmp_path: Path) -> None:
    """The point of the whole exercise, end to end."""
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    pixels = np.linspace(0.0, 100.0, 6)
    _ = _legacy_trex_table(ds, "seq_a", cm_per_pixel=0.25, pixels=pixels)

    report = upgrade_trex_tables(ds, apply=True)
    assert len(report.upgraded) == 1

    index = read_tracks_index(ds)
    upgraded = index[index["run_id"] == report.target_variant]
    assert len(upgraded) == 1
    table = pd.read_parquet(ds.resolve_path(str(upgraded.iloc[0]["abs_path"])))

    assert table["X"].to_numpy() == pytest.approx(pixels)
    assert table["SPEED"].to_numpy() == pytest.approx(pixels)
    # Keypoints were already pixels and must not be divided a second time.
    assert table["poseX0"].to_numpy() == pytest.approx(pixels)


def test_an_upgrade_moves_x_to_the_body_centre(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    pixels = np.linspace(0.0, 100.0, 6)
    _ = _legacy_trex_table(ds, "seq_a", cm_per_pixel=0.25, pixels=pixels)

    report = upgrade_trex_tables(ds, apply=True)
    index = read_tracks_index(ds)
    row = index[index["run_id"] == report.target_variant].iloc[0]
    table = pd.read_parquet(ds.resolve_path(str(row["abs_path"])))

    assert table["X"].to_numpy() == pytest.approx(pixels)
    # The head was 5cm off the centre, which is 20px at 0.25 cm/px.
    assert table["X#head"].to_numpy() == pytest.approx(pixels + 20.0)


def test_the_upgraded_row_records_the_new_schema(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    _ = _legacy_trex_table(
        ds, "seq_a", cm_per_pixel=0.5, pixels=np.linspace(0.0, 10.0, 4)
    )

    report = upgrade_trex_tables(ds, apply=True)
    index = read_tracks_index(ds)
    row = index[index["run_id"] == report.target_variant].iloc[0]
    assert str(row["std_format"]) == "trex_v2"


def test_a_table_without_a_calibration_is_refused_not_assumed(tmp_path: Path) -> None:
    """Centimetres and pixels are indistinguishable once the number is lost."""
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    _ = _legacy_trex_table(
        ds,
        "seq_a",
        cm_per_pixel=0.25,
        pixels=np.linspace(0.0, 10.0, 4),
        carry_calibration=False,
    )

    report = upgrade_trex_tables(ds, apply=True)

    assert len(report.refused) == 1
    assert "cm_per_pixel" in report.refused[0].detail
    assert len(read_tracks_index(ds)) == 1, "the refused table is left alone"


def test_a_non_trex_table_is_skipped(tmp_path: Path) -> None:
    """Another converter's table is already pixels; its columns change, not its scale."""
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    _ = _legacy_trex_table(
        ds,
        "seq_a",
        cm_per_pixel=1.0,
        pixels=np.linspace(0.0, 10.0, 4),
        variant="convert-sleap_analysis_h5.0.1-bbbbbbbbbb",
        producer="convert-sleap_analysis_h5",
    )

    report = upgrade_trex_tables(ds, apply=True)

    assert len(report.skipped) == 1
    assert not report.upgraded


def test_an_already_converted_table_is_skipped(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    _ = _legacy_trex_table(
        ds,
        "seq_a",
        cm_per_pixel=1.0,
        pixels=np.linspace(0.0, 10.0, 4),
        variant="convert-trex_npz.0.2-cccccccccc",
        std_format="trex_v2",
    )

    report = upgrade_trex_tables(ds, apply=True)

    assert len(report.skipped) == 1


def test_the_target_is_where_a_reconversion_would_have_written(tmp_path: Path) -> None:
    """So converting properly later finds it, rather than writing a second table."""
    from mosaic.core.track_converter import get_track_converter

    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    _ = _legacy_trex_table(
        ds, "seq_a", cm_per_pixel=0.5, pixels=np.linspace(0.0, 10.0, 4)
    )
    report = upgrade_trex_tables(ds, apply=False)

    converter = get_track_converter("trex_npz")
    assert report.target_variant.startswith(
        f"convert-trex_npz.{type(converter).version}-"
    )


def test_one_bad_table_does_not_abort_the_migration(tmp_path: Path) -> None:
    """A refusal is per entry, not per run.

    The strict validation here is the only one in production, and unguarded it
    ended the whole migration on the first table that failed it -- after the
    tables before it had already been written and indexed. That left a
    half-upgraded dataset, a traceback instead of a report, and no record of
    which entry was responsible.
    """
    ds = make_dataset(tmp_path / "dataset", name="upgrade", save=False)
    pixels = np.linspace(0.0, 100.0, 6)
    good = _legacy_trex_table(ds, "seq_ok", cm_per_pixel=0.25, pixels=pixels)
    bad = _legacy_trex_table(ds, "seq_bad", cm_per_pixel=0.25, pixels=pixels)
    # Drop a column mosaic_v1 requires, leaving a table that converts and then
    # fails validation -- which is exactly the shape that used to abort.
    broken = pd.read_parquet(bad).drop(columns=["time"])
    _ = write_parquet_atomic(broken, bad)
    assert good.exists()

    report = upgrade_trex_tables(ds, apply=True)

    upgraded = {outcome.sequence for outcome in report.upgraded}
    refused = {outcome.sequence for outcome in report.refused}
    assert upgraded == {"seq_ok"}
    assert refused == {"seq_bad"}
    assert any("time" in outcome.detail for outcome in report.refused)
