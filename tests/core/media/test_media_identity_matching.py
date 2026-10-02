from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import yaml
from mosaic_media.transcode import TranscodeError

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import (
    media_row_path_key,
    media_row_uuid,
    read_link_cell,
)
from mosaic.core.media.imgstore_io import imgstore_store_identity
from mosaic.core.media.probe_row import probe_video_metadata
from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.pipeline.transcode import set_forward_link

MakeDataset = Callable[[Path], Dataset]
MakeImgstore = Callable[..., tuple[Path, list[np.ndarray]]]
WriteVideo = Callable[..., None]


def _strip_store_uuid(store_dir: Path) -> None:
    """Remove ``__store.uuid`` from a store's metadata, leaving it unminted.

    The shape of a store written before Motif recorded a uuid, and the only way
    to build an unminted store now that :func:`imgstore_probe` reads one.
    """
    metadata = store_dir / "metadata.yaml"
    loaded = yaml.safe_load(metadata.read_text())
    _ = loaded["__store"].pop("uuid", None)
    _ = metadata.write_text(yaml.safe_dump(loaded))


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
    ``video_uuid`` has to be for the no-carry contract to be under test.
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


def test_carry_forward_drops_a_link_on_a_store_with_no_uuid(
    tmp_path: Path, make_media_dataset: MakeDataset, make_imgstore: MakeImgstore
) -> None:
    # Since open item O5 a store mints its __store.uuid, so "an imgstore can
    # never carry an identity" -- the premise this test was written on -- is no
    # longer true, and the store this fixture builds is minted. What is still
    # true, and is what the test is actually for, is that matching is by
    # video_uuid alone with no path fallback: a *prior* row carrying no uuid
    # matches nothing, so its link is dropped rather than mis-attached. Stripping
    # the store's uuid from its metadata reproduces exactly that row, and is also
    # the real shape of a store written before Motif recorded one.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    store_dir, _frames = make_imgstore(name="store", parent=sequence_dir)
    _strip_store_uuid(store_dir)
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
    # The store carries no uuid, so nothing matches it and the link is dropped
    # rather than attached to some other row by path.
    assert records[store_dir.name]["video_uuid"] == ""
    assert records[store_dir.name]["analysis_derivative_path"] == ""


def test_carry_forward_matches_a_minted_store_by_its_uuid(
    tmp_path: Path, make_media_dataset: MakeDataset, make_imgstore: MakeImgstore
) -> None:
    """A store now names itself, so the uuid map reaches it like anything else.

    Dormant in practice -- no store holds a derivative link, and transcoding one
    is refused -- but it is the visible half of open item O5, and the assertion
    that a mint really did land in the same column a derived value uses.
    """
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    store_dir, _frames = make_imgstore(name="store", parent=sequence_dir)
    minted = imgstore_store_identity(store_dir).store_uuid
    assert minted, "the fixture's store must carry a __store.uuid"
    derivative = base / "media" / "clip.analysis.mp4"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    _ = derivative.write_bytes(b"x")

    _write_prior_index(
        ds,
        [
            {
                "name": "renamed_store",
                "abs_path": "media_raw/seqA/renamed_store",
                "media_type": "imgstore",
                "video_uuid": minted,
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="seqA")]
    )

    assert _links_by_name(ds) == {store_dir.name: "clip.analysis.mp4"}


def test_carry_forward_does_not_cross_a_link_on_a_colliding_path_key_with_a_different_uuid(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    # The mis-attribution the fallback removal exists to kill. The prior index
    # holds an unminted row for a different recording, rec2/cam0/video.mp4,
    # carrying a link; its two-leaf path key (cam0, video.mp4) collides with the
    # freshly probed rec1/cam0/video.mp4, which mints its own distinct uuid. The
    # old path fallback matched on that shared key and crossed rec2's link onto
    # rec1's row -- routing rec1's analysis reads at rec2's derivative. Matching
    # by video_uuid alone, the two identities differing, carries nothing.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "rec1" / "cam0"
    write_cfr_mp4(sequence_dir / "video.mp4")
    # A real derivative file, so the pre-fix fallback would clear its
    # dangling-link guard and actually cross the link rather than drop it.
    derivative = base / "media" / "rec2.analysis.mp4"
    derivative.parent.mkdir(parents=True, exist_ok=True)
    _ = derivative.write_bytes(b"x")

    _write_prior_index(
        ds,
        [
            {
                "name": "video.mp4",
                "abs_path": "media_raw/rec2/cam0/video.mp4",
                "analysis_derivative_path": "rec2.analysis.mp4",
            }
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="rec1")]
    )

    links = _links_by_stored_path(ds)
    assert links["media_raw/rec1/cam0/video.mp4"] == ""


def test_carry_forward_drops_a_link_from_an_unminted_prior_row(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    # An index written before the identity columns existed: the prior row carries
    # a link but no video_uuid. The re-probe mints an identity on the fresh row,
    # but a link matches only by video_uuid on both sides, so a link sitting on
    # an unminted prior row carries nothing forward. The old path key that stood
    # in here was not unique -- rec1/cam0/video.mp4 and rec2/cam0/video.mp4
    # answer to the same key -- so carrying by it could route analysis reads at
    # the wrong recording; losing the link only costs a re-transcode.
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
                "abs_path": "media_raw/seqA/clip.mp4",
                "analysis_derivative_path": "clip.analysis.mp4",
            }
        ],
    )

    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="seqA")]
    )

    links = _links_by_stored_path(ds)
    assert links["media_raw/seqA/clip.mp4"] == ""


def test_a_forward_link_raises_when_no_row_carries_the_identity(
    tmp_path: Path, make_media_dataset: MakeDataset, write_cfr_mp4: WriteVideo
) -> None:
    # After the mandatory re-probe every row carries a video_uuid, so a forward
    # link for a uuid no row holds means the caller measured a file the index
    # does not describe. There is no path fallback to attach it silently, so it
    # raises rather than writing the link onto a mismatched row.
    base = (tmp_path / "dataset").resolve()
    ds = make_media_dataset(base)
    sequence_dir = base / "media_raw" / "seqA"
    source = sequence_dir / "clip.mp4"
    write_cfr_mp4(source)
    _ = ds.write_media_index(
        [MediaIndexScope(directory=sequence_dir, group="g", sequence="seqA")]
    )

    with pytest.raises(TranscodeError, match="no media_raw row carries video_uuid"):
        set_forward_link(ds, source, "absent-uuid", "transcode/x.mp4", "analysis")
