"""End-to-end: extract_frames over an imgstore-backed sequence."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("imgstore")

from mosaic.core.dataset import Dataset  # noqa: E402
from mosaic.tracking import extract_frames  # noqa: E402


def _make_dataset(tmp_path: Path) -> Dataset:
    for sub in ("media", "tracks", "frames"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    return Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media": str(tmp_path / "media"),
            "tracks": str(tmp_path / "tracks"),
            "frames": str(tmp_path / "frames"),
        },
    )


@pytest.mark.slow
@pytest.mark.parametrize("method", ["uniform", "kmeans"])
def test_extract_frames_from_imgstore(tmp_path, make_imgstore, method):
    ds = _make_dataset(tmp_path)
    search = tmp_path / "raw"
    make_imgstore(name="rec", nframes=20, parent=search)
    ds.index_media([search])

    extract_frames(ds, n_frames=4, method=method)

    pngs = list((tmp_path / "frames").rglob("*.png"))
    assert len(pngs) == 4
