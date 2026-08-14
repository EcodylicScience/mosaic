"""Tests for the media-index writer: the ``video_order`` densifier, the
assignment-driven ``Dataset.write_media_index`` projection, and its invariants
(root-relative-in-tree ``abs_path``, preserved rows, derivative-link carry
forward). The densifier cases mirror the upload finalize contract the API drives
through ``write_media_index``.

Every dataset here is *saved* before it is written to, as the API seeds a manifest
before finalizing an upload: without one, ``base_dir`` resolves to the manifest
path rather than its directory and every root-relative ``abs_path`` lands a level
too deep.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS
from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    VideoOrderKey,
    assign_video_order,
    build_prior_order,
    densify_video_order,
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)

from tests.helpers import make_dataset, write_mpeg4_mp4

# ``media_raw`` is declared so ``resolve_media_root`` lands there and the
# originals index is written to ``media_raw/index.csv``; ``media`` is the
# derivative root the carry-forward case points a routing link into. Two tests
# add ``tracks`` themselves, because a keymap is what they are exercising.
_ROOTS = ("media_raw", "media")


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
    session_positions = {
        ("", "s", "a.mp4"): 2,
        ("", "s", "b.mp4"): 0,
        ("", "s", "c.mp4"): 1,
    }
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
    session_positions = {("", "s", "new0.mp4"): 0, ("", "s", "new1.mp4"): 1}
    prior_order = {("", "s", "old0.mp4"): 0, ("", "s", "old1.mp4"): 1}
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
    prior_order = {("", "s", "z.mp4"): 0, ("", "s", "y.mp4"): 1, ("", "s", "x.mp4"): 2}
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
        ("", "s1", "a.mp4"): 0,
        ("", "s2", "a.mp4"): 0,
        ("", "s2", "b.mp4"): 1,
    }
    result = densify_video_order(
        rows, session_positions=session_positions, prior_order={}
    )
    by_seq = {(row["sequence"], row["name"]): row["video_order"] for row in result}
    assert by_seq[("s1", "a")] == 0
    assert by_seq[("s2", "a")] == 0
    assert by_seq[("s2", "b")] == 1


def test_same_named_sequences_in_different_groups_keep_separate_orders() -> None:
    # Two groups hold a "trial1" with the same basenames in opposite orders. The
    # prior-order key carries the group, so neither group's table overwrites the
    # other's and each keeps its own numbering.
    rows = [
        _order_row(
            group="control", sequence="trial1", filename="a.mp4", prior_video_order="0"
        ),
        _order_row(
            group="control", sequence="trial1", filename="b.mp4", prior_video_order="1"
        ),
        _order_row(
            group="exp", sequence="trial1", filename="a.mp4", prior_video_order="1"
        ),
        _order_row(
            group="exp", sequence="trial1", filename="b.mp4", prior_video_order="0"
        ),
    ]
    result = densify_video_order(
        rows, session_positions={}, prior_order=build_prior_order(rows)
    )
    order = {(row["group"], row["name"]): row["video_order"] for row in result}
    assert order == {
        ("control", "a"): 0,
        ("control", "b"): 1,
        ("exp", "a"): 1,
        ("exp", "b"): 0,
    }


# --- assign_video_order: the shared ranking core ---------------------------


def _vk(
    *,
    name: str,
    group: str = "",
    sequence: str = "s",
    camera: str = "",
    prior_order: int | None = None,
    session_position: int | None = None,
) -> VideoOrderKey:
    return VideoOrderKey(
        group=group,
        sequence=sequence,
        camera=camera,
        name=name,
        prior_order=prior_order,
        session_position=session_position,
    )


def test_assign_video_order_fresh_orders_by_session_position() -> None:
    keys = [
        _vk(name="a", session_position=2),
        _vk(name="b", session_position=0),
        _vk(name="c", session_position=1),
    ]
    result = assign_video_order(keys, lambda k: k)
    assert [(k.name, order) for k, order in result] == [("b", 0), ("c", 1), ("a", 2)]


def test_assign_video_order_append_prior_then_session() -> None:
    keys = [
        _vk(name="new0", session_position=0),
        _vk(name="old1", prior_order=1),
        _vk(name="old0", prior_order=0),
        _vk(name="new1", session_position=1),
    ]
    result = assign_video_order(keys, lambda k: k)
    assert [(k.name, order) for k, order in result] == [
        ("old0", 0),
        ("old1", 1),
        ("new0", 2),
        ("new1", 3),
    ]


def test_assign_video_order_order_zero_is_not_the_missing_sentinel() -> None:
    # prior_order 0 must be kept, not replaced by the unindexed sentinel: an
    # unindexed prior (prior_order None) sorts after every recorded order, then by
    # name. Guards the `prior_order if is not None else sentinel` branch against a
    # truthiness bug that would demote order 0.
    keys = [
        _vk(name="z", prior_order=None),
        _vk(name="indexed", prior_order=0),
        _vk(name="a", prior_order=None),
    ]
    result = assign_video_order(keys, lambda k: k)
    assert [(k.name, order) for k, order in result] == [
        ("indexed", 0),
        ("a", 1),
        ("z", 2),
    ]


def test_assign_video_order_session_position_zero_is_a_session_video() -> None:
    # session_position 0 must classify as a session addition (sorts after an
    # unindexed prior), not be mistaken for "no session position". Guards the
    # `session_position is not None` branch against a truthiness bug.
    keys = [
        _vk(name="aaa_upload", session_position=0),
        _vk(name="zzz_ghost", prior_order=None),
    ]
    result = assign_video_order(keys, lambda k: k)
    assert [k.name for k, _ in result] == ["zzz_ghost", "aaa_upload"]


def test_assign_video_order_tie_break_on_name() -> None:
    # Same rank class and value -> the name breaks the tie deterministically.
    keys = [
        _vk(name="b", session_position=0),
        _vk(name="a", session_position=0),
    ]
    result = assign_video_order(keys, lambda k: k)
    assert [k.name for k, _ in result] == ["a", "b"]


def test_assign_video_order_counts_per_camera_and_sequence() -> None:
    # The dense counter restarts per (group, sequence, camera): two cameras of one
    # recording are numbered independently, never as chunks of one timeline.
    keys = [
        _vk(name="cam0_a", sequence="rec", camera="c0", session_position=0),
        _vk(name="cam0_b", sequence="rec", camera="c0", session_position=1),
        _vk(name="cam1_a", sequence="rec", camera="c1", session_position=0),
        _vk(name="other", sequence="rec2", camera="", session_position=0),
    ]
    result = assign_video_order(keys, lambda k: k)
    order_by_name = {k.name: order for k, order in result}
    assert order_by_name == {"cam0_a": 0, "cam0_b": 1, "cam1_a": 0, "other": 0}


# --- write_media_index: the projection ------------------------------------


def test_write_media_index_orders_by_position_and_stores_relative(
    tmp_path: Path,
) -> None:
    # Resolve away the macOS /var -> /private/var symlink so the dataset base_dir
    # (manifest parent) and abs_path.resolve() agree; production roots are real
    # paths, so in-tree files are stored root-relative there without this.
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    seq_dir = tmp_path / "media_raw" / "seqA"
    write_mpeg4_mp4(seq_dir / "a.mp4")
    write_mpeg4_mp4(seq_dir / "b.mp4")

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
    ds = make_dataset(tmp_path, roots=_ROOTS)
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
    write_mpeg4_mp4(seq_dir / "a.mp4")
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
    ds = make_dataset(tmp_path, roots=_ROOTS)
    seq_dir = tmp_path / "media_raw" / "seqA"
    write_mpeg4_mp4(seq_dir / "a.mp4")

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
    write_mpeg4_mp4(seq_dir / "b.mp4")
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


def test_write_media_index_writes_a_doubly_scoped_file_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two scopes over one directory are caller error, not two rows.

    Left in, the same file carries two sequence names and its uid lands in two
    sequences' media compositions.
    """
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    shared = tmp_path / "media_raw" / "shared"
    write_mpeg4_mp4(shared / "a.mp4")

    ds.write_media_index(
        [
            MediaIndexScope(directory=shared, group="", sequence="seqA"),
            MediaIndexScope(directory=shared, group="", sequence="seqB"),
        ],
        extensions=(".mp4",),
    )

    rows = ds.read_media_index()
    assert [(row["name"], row["sequence"]) for row in rows] == [("a.mp4", "seqA")]
    assert "is under two scopes" in capsys.readouterr().err


def test_write_media_index_leaves_an_unnamed_sequence_alone(tmp_path: Path) -> None:
    """Naming seqA must not renumber seqB, even where seqB's order is blank.

    ``build_prior_order`` skips a blank ``video_order`` cell, so a preserved row
    carrying one used to reach the densifier as an unknown-order prior and be
    renumbered by filename -- during someone else's upload.
    """
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    index_path = tmp_path / "media_raw" / "index.csv"

    seeded: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    untouched: list[dict[str, object]] = []
    # Curated cells the write must return verbatim: a blank order, and a "1" on
    # the file whose name sorts first.
    for name, order in (("z.mp4", "1"), ("y.mp4", ""), ("x.mp4", "0")):
        row = dict(seeded)
        row.update(
            {
                "name": name,
                "sequence": "seqB",
                "abs_path": f"media_raw/seqB/{name}",
                "video_order": order,
            }
        )
        untouched.append(row)
    write_media_index_rows(index_path, frame_from_rows(untouched))

    seq_dir = tmp_path / "media_raw" / "seqA"
    write_mpeg4_mp4(seq_dir / "a.mp4")
    ds.write_media_index(
        [
            MediaIndexScope(
                directory=seq_dir, group="", sequence="seqA", order_by_name={"a.mp4": 0}
            )
        ],
        extensions=(".mp4",),
    )

    after = {
        row["name"]: row["video_order"]
        for row in ds.read_media_index()
        if row["sequence"] == "seqB"
    }
    assert after == {"z.mp4": "1", "y.mp4": "", "x.mp4": "0"}


def test_index_media_keeps_the_order_an_arranged_write_gave(tmp_path: Path) -> None:
    """A rescan reads the order it is about to overwrite instead of re-sorting.

    The filenames deliberately sort the *opposite* way from the arranged order,
    so a rescan that discarded the prior order would swap them -- and the media
    composition hash, which is computed over ``video_order``, would move with no
    content change.
    """
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    ds.set_root("tracks", str(tmp_path / "tracks"))
    ds.ensure_roots()
    # index_media derives (group, sequence) from the tracks keymap, so the two
    # files must resolve to the sequence the arranged write named. Prefix mode
    # maps both stems onto "seqA".
    (tmp_path / "tracks" / "index.csv").write_text(
        "run_id,group,sequence,abs_path\n,,seqA,tracks/seqA.parquet\n"
    )

    seq_dir = tmp_path / "media_raw" / "seqA"
    write_mpeg4_mp4(seq_dir / "seqA_0.mp4")
    write_mpeg4_mp4(seq_dir / "seqA_1.mp4")
    ds.write_media_index(
        [
            MediaIndexScope(
                directory=seq_dir,
                group="",
                sequence="seqA",
                order_by_name={"seqA_1.mp4": 0, "seqA_0.mp4": 1},
            )
        ],
        extensions=(".mp4",),
    )
    arranged = {row["name"]: row["video_order"] for row in ds.read_media_index()}
    assert arranged == {"seqA_1.mp4": "0", "seqA_0.mp4": "1"}

    ds.index_media([seq_dir], extensions=(".mp4",), sequence_match_mode="prefix")

    rescanned = {row["name"]: row["video_order"] for row in ds.read_media_index()}
    assert rescanned == arranged, "a rescan renumbered an arranged sequence by name"


def test_write_media_index_carries_forward_derivative_links(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    seq_dir = tmp_path / "media_raw" / "seqA"
    write_mpeg4_mp4(seq_dir / "a.mp4")
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


def test_a_row_records_how_it_learned_its_name(tmp_path: Path) -> None:
    """Assigned, keymap-matched and stem-derived identities are distinguishable.

    Item 4.7's question -- whether the scan path may be authoritative -- is
    answered per row rather than by a break: a sequence whose rows say
    ``scan-keymap`` derived a *source* root's identity from a *derived* one, so
    the per-sequence composition can decline to compute a value for it instead of
    recording a confident wrong one.

    This is also the suite's first coverage of a keymap *hit*: every other media
    test has an empty or absent tracks index and takes the fallback path.
    """
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    ds.set_root("tracks", str(tmp_path / "tracks"))
    ds.ensure_roots()
    (tmp_path / "tracks" / "index.csv").write_text(
        "run_id,group,sequence,abs_path\n,,known,tracks/known.parquet\n"
    )

    scanned = tmp_path / "media_raw" / "scanned"
    write_mpeg4_mp4(scanned / "known.mp4")
    write_mpeg4_mp4(scanned / "stranger.mp4")
    ds.index_media([scanned], extensions=(".mp4",))

    by_name = {row["name"]: row["assignment_source"] for row in ds.read_media_index()}
    assert by_name == {"known.mp4": "scan-keymap", "stranger.mp4": "scan-stem"}

    assigned_dir = tmp_path / "media_raw" / "seqA"
    write_mpeg4_mp4(assigned_dir / "a.mp4")
    ds.write_media_index(
        [MediaIndexScope(directory=assigned_dir, group="", sequence="seqA")],
        extensions=(".mp4",),
    )
    after = {row["name"]: row["assignment_source"] for row in ds.read_media_index()}
    assert after["a.mp4"] == "assigned"
    # The scan's rows are outside this write's scope, so they keep their cells.
    assert after["known.mp4"] == "scan-keymap"


_FINALIZE_PROBE = """
import sys
from pathlib import Path
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.media_index import MediaIndexScope

manifest, sequence, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ds = Dataset(manifest_path=manifest).load()
seq_dir = ds.get_root("media_raw") / sequence

barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
# Both processes enter the probe phase together. Unlocked, both read the index
# before either writes, so each computes its preserved set from the same empty
# starting state and the second erases the first.
ds.write_media_index(
    [MediaIndexScope(directory=seq_dir, group="", sequence=sequence)],
    extensions=(".mp4",),
)
"""


def test_two_finalizes_of_different_sequences_do_not_lose_rows(tmp_path: Path) -> None:
    """`write_media_index` is a whole-file rewrite, so it needs the lock.

    Two uploads finalizing different sequences is the ordinary case, not a
    contrived one: each preserves "every row not under my scope", and without
    serialization the second preserves a snapshot taken before the first wrote.
    """
    tmp_path = tmp_path.resolve()
    ds = make_dataset(tmp_path, roots=_ROOTS)
    for sequence in ("seqA", "seqB"):
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            write_mpeg4_mp4(tmp_path / "media_raw" / sequence / name)

    gate = tmp_path / "gate"
    gate.mkdir()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _FINALIZE_PROBE,
                str(ds.manifest_path),
                sequence,
                str(gate / f"{sequence}.ready"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for sequence in ("seqA", "seqB")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode()[-800:]

    written = {row["sequence"] for row in ds.read_media_index()}
    assert written == {"seqA", "seqB"}, f"a concurrent finalize was lost: {written}"


def test_read_media_index_round_trips_fact_columns(tmp_path: Path) -> None:
    index_path = tmp_path / "index.csv"
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row["media_facts"] = '{"width": 1920}'
    row["analysis_derivative_path"] = "analysis/clip.mp4"
    write_media_index_rows(index_path, frame_from_rows([row]))
    written = read_media_index(index_path)
    assert written[0]["media_facts"] == '{"width": 1920}'
    assert written[0]["analysis_derivative_path"] == "analysis/clip.mp4"
