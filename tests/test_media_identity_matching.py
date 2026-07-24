from collections.abc import Callable
from pathlib import Path

import numpy as np

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import (
    media_row_path_key,
    media_row_uuid,
    read_link_cell,
)
from mosaic.core.media.probe_row import probe_video_metadata
from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)

MakeDataset = Callable[[Path], Dataset]
MakeImgstore = Callable[..., tuple[Path, list[np.ndarray]]]
WriteVideo = Callable[..., None]


def test_read_link_cell_treats_every_absent_form_as_empty() -> None:
    assert read_link_cell({"c": ""}, "c") == ""
    assert read_link_cell({"c": "nan"}, "c") == ""
    assert read_link_cell({"c": "NaN"}, "c") == ""
    assert read_link_cell({"c": float("nan")}, "c") == ""
    assert read_link_cell({}, "c") == ""
    assert read_link_cell({"c": "  a.mp4  "}, "c") == "a.mp4"


def test_media_row_uuid_reads_the_column_or_empty() -> None:
    assert media_row_uuid({"video_uuid": "abc"}) == "abc"
    assert media_row_uuid({"video_uuid": ""}) == ""
    assert media_row_uuid({"video_uuid": float("nan")}) == ""
    assert media_row_uuid({}) == ""


def test_media_row_path_key_is_the_two_leaf_components() -> None:
    assert media_row_path_key({"abs_path": "/data/seqA/clip.mp4"}) == (
        "seqA",
        "clip.mp4",
    )
    assert media_row_path_key({"abs_path": "seqA/clip.mp4"}) == ("seqA", "clip.mp4")


def _write_prior_index(ds: Dataset, rows: list[dict[str, object]]) -> None:
    """Seed the media index a reindex reads before overwriting it.

    ``frame_from_rows`` widens each partial mapping to the full schema, so a cell
    left out arrives empty rather than absent -- which is what an unminted
    ``video_uuid`` has to be for the path fallback to be under test.
    """
    index_path = ds.get_root(ds.resolve_media_root()) / "index.csv"
    write_media_index_rows(index_path, frame_from_rows(rows))


def _links_by_name(ds: Dataset) -> dict[str, str]:
    return {
        record["name"]: record["analysis_derivative_path"]
        for record in read_media_index(
            ds.get_root(ds.resolve_media_root()) / "index.csv"
        )
    }


def _links_by_stored_path(ds: Dataset) -> dict[str, str]:
    """Links keyed by the stored ``abs_path``, which stays unique when names do not."""
    return {
        record["abs_path"]: record["analysis_derivative_path"]
        for record in read_media_index(
            ds.get_root(ds.resolve_media_root()) / "index.csv"
        )
    }


def test_carry_forward_matches_a_renamed_original_by_uuid(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    # The prior index knows the video as old_name.mp4; on disk it is now
    # renamed.mp4. The two path keys differ, so only the uuid map can carry the
    # link -- which is the point of minting an identity in the first place.
    # Driven through write_media_index, the public reindex that carries links.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    renamed = sequence_dir / "renamed.mp4"
    write_cfr_mp4(renamed)
    # A real derivative file so the dangling-link check passes.
    derivative = base / "media" / "clip.analysis.mp4"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    _ = derivative.write_bytes(b"x")

    minted = probe_video_metadata(renamed)["video_uuid"]
    _write_prior_index(
        ds,
        [
            {
                "name": "old_name.mp4",
                "abs_path": "media_raw/seqA/old_name.mp4",
                "video_uuid": minted,
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="seqA")]
    )

    assert _links_by_name(ds) == {"renamed.mp4": "clip.analysis.mp4"}


def test_carry_forward_falls_back_to_path_for_an_unminted_pair(
    tmp_path: Path, make_media_dataset: MakeDataset, make_imgstore: MakeImgstore
) -> None:
    # An imgstore is the one media type that can never carry an identity, so
    # both the prior row and the freshly probed one are unminted and the link
    # can only be carried by the two-leaf path key.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    store_dir, _frames = make_imgstore(name="store", parent=sequence_dir)
    derivative = base / "media" / "clip.analysis.mp4"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    _ = derivative.write_bytes(b"x")

    _write_prior_index(
        ds,
        [
            {
                "name": store_dir.name,
                "abs_path": f"media_raw/seqA/{store_dir.name}",
                "media_type": "imgstore",
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="seqA")]
    )

    records = {
        record["name"]: record
        for record in read_media_index(
            ds.get_root(ds.resolve_media_root()) / "index.csv"
        )
    }
    # Both sides really are unminted: the fallback is under test, not a uuid hit.
    assert records[store_dir.name]["video_uuid"] == ""
    assert records[store_dir.name]["analysis_derivative_path"] == "clip.analysis.mp4"


def test_carry_forward_survives_a_relocated_absolute_path(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    # The relocation robustness the two-leaf key exists for: a prior row whose
    # abs_path was recorded absolute on another machine still matches the fresh
    # root-relative row for the same file. A collision guard must not cost this.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    write_cfr_mp4(sequence_dir / "clip.mp4")
    derivative = base / "media" / "clip.analysis.mp4"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    _ = derivative.write_bytes(b"x")

    _write_prior_index(
        ds,
        [
            {
                "name": "clip.mp4",
                "abs_path": "/nas/archive/seqA/clip.mp4",
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="seqA")]
    )

    links = _links_by_stored_path(ds)
    assert links["media_raw/seqA/clip.mp4"] == "clip.analysis.mp4"


def test_carry_forward_drops_a_link_two_prior_rows_could_claim(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    # Two recordings whose per-camera layout gives them the same two-leaf key.
    # The key identifies neither row, so carrying either link would attach one
    # recording's derivative to the other's row -- silently routing analysis
    # reads at the wrong video. Losing the link is recoverable; this is not.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "rec1" / "cam0"
    write_cfr_mp4(sequence_dir / "video.mp4")
    media_root = base / "media"
    media_root.mkdir(parents=True, exist_ok=True)
    for name in ("rec1.analysis.mp4", "rec2.analysis.mp4"):
        _ = (media_root / name).write_bytes(b"x")

    _write_prior_index(
        ds,
        [
            {
                "name": "video.mp4",
                "abs_path": "media_raw/rec1/cam0/video.mp4",
                "analysis_derivative_path": "rec1.analysis.mp4",
            },
            {
                "name": "video.mp4",
                "abs_path": "media_raw/rec2/cam0/video.mp4",
                "analysis_derivative_path": "rec2.analysis.mp4",
            },
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="rec1")]
    )

    links = _links_by_stored_path(ds)
    assert links["media_raw/rec1/cam0/video.mp4"] == ""
    # The out-of-scope row is preserved verbatim, link included.
    assert links["media_raw/rec2/cam0/video.mp4"] == "rec2.analysis.mp4"
