"""``mosaic convert-tracks`` and ``mosaic convert-labels`` report what they did."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.dataset import Dataset
from tests.helpers import calms21_dataset, invoke_json, make_dataset, write_trex_npz


def test_convert_labels_json_reports_variant_and_counts(tmp_path: Path) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4, "s2": 3}}})
    planned = ds.label_conversion_variant("behavior", "calms21_npy")

    payload = invoke_json(
        [
            "convert-labels",
            "-m",
            str(ds.manifest_path),
            "--source-format",
            "calms21_npy",
            "--json",
        ]
    )

    assert payload == {
        "status": "ok",
        "variant": planned,
        "written": 2,
        "skipped": 0,
    }


def test_convert_tracks_json_reports_variants(tmp_path: Path) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})
    planned = ds.track_conversion_variant("calms21_npy")

    payload = invoke_json(
        [
            "convert-tracks",
            "-m",
            str(ds.manifest_path),
            "--source-format",
            "calms21_npy",
            "--json",
        ]
    )

    assert payload == {
        "status": "ok",
        "converted": 1,
        "failed": 0,
        "variants": [planned],
    }


def _trex_dataset(
    tmp_path: Path, good: tuple[str, ...], bad: tuple[str, ...]
) -> Dataset:
    """A dataset indexing a TRex file for each name, the *bad* ones unreadable."""
    ds = make_dataset(tmp_path / "ds")
    raw = ds.base_dir / "raw"
    for name in good:
        write_trex_npz(raw / f"{name}_fish0.npz")
    for name in bad:
        raw.mkdir(parents=True, exist_ok=True)
        _ = (raw / f"{name}_fish0.npz").write_bytes(b"not an npz")
    _ = ds.index_tracks_raw([raw], patterns=["*.npz"], src_format="trex_npz")
    return ds


def test_convert_tracks_fails_when_every_entry_fails(tmp_path: Path) -> None:
    ds = _trex_dataset(tmp_path, good=(), bad=("a", "b"))

    result = CliRunner().invoke(
        app, ["convert-tracks", "-m", str(ds.manifest_path), "--json"]
    )

    assert result.exit_code != 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert (payload["converted"], payload["failed"]) == (0, 2)


def test_convert_tracks_reports_partial_and_exits_zero_when_some_entries_convert(
    tmp_path: Path,
) -> None:
    ds = _trex_dataset(tmp_path, good=("good",), bad=("bad",))

    result = CliRunner().invoke(
        app, ["convert-tracks", "-m", str(ds.manifest_path), "--json"]
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "partial"
    assert (payload["converted"], payload["failed"]) == (1, 1)


def test_a_failed_conversion_says_so_in_text_and_exits_non_zero(
    tmp_path: Path,
) -> None:
    ds = _trex_dataset(tmp_path, good=(), bad=("a",))

    result = CliRunner().invoke(app, ["convert-tracks", "-m", str(ds.manifest_path)])

    assert result.exit_code != 0
    assert "failed" in result.stderr
