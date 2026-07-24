from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media_raw", "media"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
        },
    )


def test_a_legacy_index_reads_new_columns_as_empty(tmp_path: Path) -> None:
    # An index written before the identity columns must read them back as "",
    # through match_media_rows -- the same normalization routing uses -- not
    # raise KeyError and not surface a float NaN.
    dataset = _make_dataset(tmp_path)
    legacy = pd.DataFrame(
        [{"name": "a.mp4", "group": "g", "sequence": "s", "abs_path": "a.mp4"}]
    )
    legacy.to_csv(tmp_path / "media_raw" / "index.csv", index=False)

    matched = dataset.match_media_rows("g", "s")

    for column in ("video_uuid", "content_digest", "source_video_uuid"):
        assert column in matched.columns, column
        assert matched.iloc[0][column] == "", column


def test_present_text_columns_are_nan_filled_to_empty(tmp_path: Path) -> None:
    # A column present but with a NaN cell (an empty CSV value) must come back as
    # "" not float NaN, the same guarantee the hardcoded loops gave.
    dataset = _make_dataset(tmp_path)
    legacy = pd.DataFrame(
        [
            {
                "name": "a.mp4",
                "group": "g",
                "sequence": "s",
                "abs_path": "a.mp4",
                "video_uuid": None,
            }
        ]
    )
    legacy.to_csv(tmp_path / "media_raw" / "index.csv", index=False)

    matched = dataset.match_media_rows("g", "s")
    assert matched.iloc[0]["video_uuid"] == ""
