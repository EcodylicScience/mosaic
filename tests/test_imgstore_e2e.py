"""End-to-end: extract_frames over an imgstore-backed sequence."""

from __future__ import annotations

import pytest

from tests.helpers import make_dataset

pytest.importorskip("imgstore")

from mosaic.tracking import extract_frames  # noqa: E402


@pytest.mark.slow
@pytest.mark.parametrize("method", ["uniform", "kmeans"])
def test_extract_frames_from_imgstore(tmp_path, make_imgstore, method):
    # No ``media_raw``: ``index_media`` resolves through ``resolve_media_root``,
    # so the store is indexed into ``media/index.csv``.
    ds = make_dataset(tmp_path, roots=("media", "tracks", "frames"), save=False)
    search = tmp_path / "raw"
    make_imgstore(name="rec", nframes=20, parent=search)
    ds.index_media([search])

    extract_frames(ds, n_frames=4, method=method)

    pngs = list((tmp_path / "frames").rglob("*.png"))
    assert len(pngs) == 4
