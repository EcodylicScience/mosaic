"""``mosaic convert-tracks`` and ``mosaic convert-labels`` report what they did."""

from __future__ import annotations

from pathlib import Path

from tests.helpers import calms21_dataset, invoke_json


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
