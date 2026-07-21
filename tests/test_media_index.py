"""Tests for the media-index writer: the ``video_order`` densifier, the
assignment-driven ``Dataset.write_media_index`` projection, and its invariants
(root-relative-in-tree ``abs_path``, preserved rows, derivative-link carry
forward). The densifier cases mirror the upload finalize contract the API drives
through ``write_media_index``.
"""

from pathlib import Path

import cv2
import numpy as np

from mosaic.core.dataset import Dataset
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    build_prior_order,
    densify_video_order,
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)


def _cfr_mp4(path: Path, n: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(n):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


def _make_dataset(tmp_path: Path) -> Dataset:
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
        },
    )
    ds.ensure_roots()
    # Seed the manifest so base_dir resolves to the manifest's parent (the API
    # always seeds it before writing the index); without it _dataset_base_dir
    # would treat the missing manifest path itself as the base directory.
    ds.save()
    return ds


def _order_row(
    *, group: str, sequence: str, filename: str, prior_video_order: str = ""
) -> dict[str, object]:
    """Minimal index row carrying only the keys the densifier reads."""
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "group": group,
            "sequence": sequence,
            "name": Path(filename).stem,
            "abs_path": f"/media_raw/{sequence}/{filename}",
            "video_order": prior_video_order,
        }
    )
    return row


# --- densify_video_order: the video_order numbering contract ---------------


def test_densify_fresh_sequence_follows_position() -> None:
    # A fresh sequence: every video is a session upload, no prior order. The
    # arranged position drives video_order (position 0 -> video_order 0).
    rows = [
        _order_row(group="", sequence="s", filename="a.mp4"),
        _order_row(group="", sequence="s", filename="b.mp4"),
        _order_row(group="", sequence="s", filename="c.mp4"),
    ]
    session_positions = {("s", "a.mp4"): 2, ("s", "b.mp4"): 0, ("s", "c.mp4"): 1}
    result = densify_video_order(
        rows, session_positions=session_positions, prior_order={}
    )
    order_by_name = {row["name"]: row["video_order"] for row in result}
    assert order_by_name == {"b": 0, "c": 1, "a": 2}


def test_densify_append_keeps_prior_then_position() -> None:
    # Append: two pre-existing videos keep their prior order, then the two
    # session uploads follow in arranged-position order after them.
    rows = [
        _order_row(group="", sequence="s", filename="old0.mp4", prior_video_order="0"),
        _order_row(group="", sequence="s", filename="old1.mp4", prior_video_order="1"),
        _order_row(group="", sequence="s", filename="new0.mp4"),
        _order_row(group="", sequence="s", filename="new1.mp4"),
    ]
    session_positions = {("s", "new0.mp4"): 0, ("s", "new1.mp4"): 1}
    prior_order = {("s", "old0.mp4"): 0, ("s", "old1.mp4"): 1}
    result = densify_video_order(
        rows, session_positions=session_positions, prior_order=prior_order
    )
    order_by_name = {row["name"]: row["video_order"] for row in result}
    assert order_by_name == {"old0": 0, "old1": 1, "new0": 2, "new1": 3}


def test_densify_preserved_sequence_keeps_prior_not_name() -> None:
    # An untouched sequence (no session uploads): video_order follows the prior
    # order, not the filename order. Names sort x,y,z but prior order is z,y,x.
    rows = [
        _order_row(group="", sequence="s", filename="z.mp4", prior_video_order="0"),
        _order_row(group="", sequence="s", filename="y.mp4", prior_video_order="1"),
        _order_row(group="", sequence="s", filename="x.mp4", prior_video_order="2"),
    ]
    prior_order = {("s", "z.mp4"): 0, ("s", "y.mp4"): 1, ("s", "x.mp4"): 2}
    result = densify_video_order(rows, session_positions={}, prior_order=prior_order)
    order_by_name = {row["name"]: row["video_order"] for row in result}
    assert order_by_name == {"z": 0, "y": 1, "x": 2}


def test_densify_blank_prior_order_falls_back_to_filename() -> None:
    # An existing row with a blank video_order is not in prior_order, so it sorts
    # after every recorded prior order, then by filename.
    rows = [
        _order_row(group="", sequence="s", filename="b.mp4"),
        _order_row(group="", sequence="s", filename="a.mp4", prior_video_order="0"),
    ]
    prior_order = build_prior_order(rows)  # only a.mp4 has an order
    result = densify_video_order(rows, session_positions={}, prior_order=prior_order)
    order_by_name = {row["name"]: row["video_order"] for row in result}
    assert order_by_name == {"a": 0, "b": 1}


def test_densify_independent_sequences_number_from_zero() -> None:
    # Each (group, sequence) is its own dense counter.
    rows = [
        _order_row(group="", sequence="s1", filename="a.mp4"),
        _order_row(group="", sequence="s2", filename="a.mp4"),
        _order_row(group="", sequence="s2", filename="b.mp4"),
    ]
    session_positions = {
        ("s1", "a.mp4"): 0,
        ("s2", "a.mp4"): 0,
        ("s2", "b.mp4"): 1,
    }
    result = densify_video_order(
        rows, session_positions=session_positions, prior_order={}
    )
    by_seq = {(row["sequence"], row["name"]): row["video_order"] for row in result}
    assert by_seq[("s1", "a")] == 0
    assert by_seq[("s2", "a")] == 0
    assert by_seq[("s2", "b")] == 1


# --- write_media_index: the projection ------------------------------------


def test_write_media_index_orders_by_position_and_stores_relative(
    tmp_path: Path,
) -> None:
    # Resolve away the macOS /var -> /private/var symlink so the dataset base_dir
    # (manifest parent) and abs_path.resolve() agree; production roots are real
    # paths, so in-tree files are stored root-relative there without this.
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    _cfr_mp4(seq_dir / "a.mp4")
    _cfr_mp4(seq_dir / "b.mp4")

    ds.write_media_index(
        [
            MediaIndexScope(
                directory=seq_dir,
                group="",
                sequence="seqA",
                order_by_name={"a.mp4": 1, "b.mp4": 0},
            )
        ],
        extensions=(".mp4",),
    )

    rows = ds.read_media_index()
    order = {row["name"]: row["video_order"] for row in rows}
    assert order == {"a.mp4": "1", "b.mp4": "0"}
    # media_raw is inside the dataset tree -> abs_path is root-relative.
    for row in rows:
        assert row["abs_path"] == f"media_raw/seqA/{row['name']}"
        assert ds.resolve_path(row["abs_path"]).exists()


def test_write_media_index_preserves_other_and_external_rows(
    tmp_path: Path,
) -> None:
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    index_path = tmp_path / "media_raw" / "index.csv"

    # Seed: one already-indexed sequence (seqB) and one external NAS reference.
    seeded: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    other = dict(seeded)
    other.update(
        {
            "name": "x.mp4",
            "sequence": "seqB",
            "abs_path": "media_raw/seqB/x.mp4",
            "video_order": "0",
        }
    )
    external = dict(seeded)
    external.update(
        {
            "name": "clip.mp4",
            "sequence": "remote",
            "abs_path": "/mnt/nas/clip.mp4",
            "video_order": "0",
        }
    )
    write_media_index_rows(index_path, frame_from_rows([other, external]))

    seq_dir = tmp_path / "media_raw" / "seqA"
    _cfr_mp4(seq_dir / "a.mp4")
    ds.write_media_index(
        [
            MediaIndexScope(
                directory=seq_dir,
                group="",
                sequence="seqA",
                order_by_name={"a.mp4": 0},
            )
        ],
        extensions=(".mp4",),
    )

    rows = ds.read_media_index()
    abs_paths = {row["abs_path"] for row in rows}
    assert "media_raw/seqA/a.mp4" in abs_paths  # freshly probed, relative
    assert "media_raw/seqB/x.mp4" in abs_paths  # other sequence preserved
    assert "/mnt/nas/clip.mp4" in abs_paths  # external row preserved, still absolute


def test_write_media_index_appends_keeping_prior_order(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    _cfr_mp4(seq_dir / "a.mp4")

    # First import: a.mp4 at position 0.
    ds.write_media_index(
        [
            MediaIndexScope(
                directory=seq_dir, group="", sequence="seqA", order_by_name={"a.mp4": 0}
            )
        ],
        extensions=(".mp4",),
    )
    # Append b.mp4; a.mp4 keeps prior order 0, b.mp4 follows.
    _cfr_mp4(seq_dir / "b.mp4")
    ds.write_media_index(
        [
            MediaIndexScope(
                directory=seq_dir, group="", sequence="seqA", order_by_name={"b.mp4": 0}
            )
        ],
        extensions=(".mp4",),
    )

    rows = ds.read_media_index()
    order = {row["name"]: int(row["video_order"]) for row in rows}
    assert order == {"a.mp4": 0, "b.mp4": 1}


def test_write_media_index_carries_forward_derivative_links(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    seq_dir = tmp_path / "media_raw" / "seqA"
    _cfr_mp4(seq_dir / "a.mp4")
    scope = MediaIndexScope(
        directory=seq_dir, group="", sequence="seqA", order_by_name={"a.mp4": 0}
    )
    ds.write_media_index([scope], extensions=(".mp4",))

    # Simulate a transcode: point a.mp4's row at a derivative that exists.
    derivative = ds.get_root("media") / "seqA.analysis.mp4"
    derivative.write_bytes(b"stub")
    rows = ds.read_media_index()
    for row in rows:
        row["analysis_derivative_path"] = "seqA.analysis.mp4"
    write_media_index_rows(
        tmp_path / "media_raw" / "index.csv", frame_from_rows(list(rows))
    )

    # A re-finalize of seqA must not drop the routing link.
    ds.write_media_index([scope], extensions=(".mp4",))
    after = ds.read_media_index()
    assert after[0]["analysis_derivative_path"] == "seqA.analysis.mp4"


def test_read_media_index_round_trips_fact_columns(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row["media_facts"] = '{"width": 1920}'
    row["analysis_derivative_path"] = "analysis/clip.mp4"
    write_media_index_rows(index_path, frame_from_rows([row]))
    written = read_media_index(index_path)
    assert written[0]["media_facts"] == '{"width": 1920}'
    assert written[0]["analysis_derivative_path"] == "analysis/clip.mp4"
