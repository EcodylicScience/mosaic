from collections.abc import Callable
from pathlib import Path

import pandas as pd

from mosaic.core.dataset import Dataset

MakeDataset = Callable[[Path], Dataset]


def test_a_legacy_index_reads_new_columns_as_empty(
    tmp_path: Path, make_media_dataset: MakeDataset
) -> None:
    # An index written before the identity columns must read them back as "",
    # through match_media_rows -- the same normalization routing uses -- not
    # raise KeyError and not surface a float NaN.
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)
    legacy = pd.DataFrame(
        [{"name": "a.mp4", "group": "g", "sequence": "s", "abs_path": "a.mp4"}]
    )
    legacy.to_csv(base / "media_raw" / "index.csv", index=False)

    matched = dataset.match_media_rows("g", "s")

    for column in ("video_uuid", "content_digest", "source_video_uuid"):
        assert column in matched.columns, column
        assert matched.iloc[0][column] == "", column


def test_present_text_columns_are_nan_filled_to_empty(
    tmp_path: Path, make_media_dataset: MakeDataset
) -> None:
    # A column present but with a NaN cell (an empty CSV value) must come back as
    # "" not float NaN, the same guarantee the hardcoded loops gave.
    base = (tmp_path / "dataset").resolve()
    dataset = make_media_dataset(base)
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
    legacy.to_csv(base / "media_raw" / "index.csv", index=False)

    matched = dataset.match_media_rows("g", "s")
    assert matched.iloc[0]["video_uuid"] == ""
