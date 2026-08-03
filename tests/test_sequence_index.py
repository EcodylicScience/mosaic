"""Tests for the per-sequence composition index (item 4.4's storage half).

The round trip a writer performs: commit an ``index.csv``, then project it into
``<root>/sequences.csv``. ``rebuild_sequence_index`` recomputes the same value
from disk, so it doubles as the oracle -- what a writer wrote and what a rebuild
produces must agree.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.composition import MEDIA_COMPOSITION_SCHEME
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic.core.pipeline.sequence_index import (
    SEQUENCE_INDEX_COLUMNS,
    encode_entry_composition,
    read_sequence_index,
    sequence_index_path,
)


def _cfr_mp4(path: Path, n: int = 6) -> None:
    """A short video whose *content* depends on its name.

    Deliberately not all-black: two byte-identical videos share one
    ``video_uuid`` by design, so a composition over them is unchanged by a
    reorder -- correctly, and it would make an ordering test pass vacuously.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    shade = sum(path.name.encode()) % 200 + 20
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(n):
        writer.write(np.full((48, 64, 3), shade, np.uint8))
    writer.release()


def _make_dataset(tmp_path: Path) -> Dataset:
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={
            "media_raw": str(tmp_path / "media_raw"),
            "media": str(tmp_path / "media"),
            "tracks_raw": str(tmp_path / "tracks_raw"),
            # index_media reads the tracks index to derive each file's identity.
            "tracks": str(tmp_path / "tracks"),
        },
    )
    ds.ensure_roots()
    ds.save()
    return ds


def _compositions(ds: Dataset, root: str) -> dict[str, str]:
    frame = read_sequence_index(ds, root)  # pyright: ignore[reportArgumentType]
    return dict(zip(frame["sequence"], frame["composition"]))


def _arrange(ds: Dataset, sequence: str, order: dict[str, int]) -> None:
    ds.write_media_index(
        [
            MediaIndexScope(
                directory=ds.get_root("media_raw") / sequence,
                group="",
                sequence=sequence,
                order_by_name=order,
            )
        ],
        extensions=(".mp4",),
    )


# --- media_raw --------------------------------------------------------------


def test_a_media_write_projects_one_row_per_sequence(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        _cfr_mp4(tmp_path / "media_raw" / "seqA" / name)

    _arrange(ds, "seqA", {"a.mp4": 0, "b.mp4": 1})

    frame = read_sequence_index(ds, "media_raw")
    assert list(frame.columns) == SEQUENCE_INDEX_COLUMNS
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["sequence"] == "seqA"
    assert row["composition"] != ""
    assert int(row["member_count"]) == 2
    assert row["identity_scheme"] == MEDIA_COMPOSITION_SCHEME
    assert row["computed_at"] != ""


def test_an_identical_rewrite_leaves_the_composition_alone(tmp_path: Path) -> None:
    """A churning digest is the failure mode; only ``computed_at`` may move."""
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        _cfr_mp4(tmp_path / "media_raw" / "seqA" / name)

    _arrange(ds, "seqA", {"a.mp4": 0, "b.mp4": 1})
    first = _compositions(ds, "media_raw")
    _arrange(ds, "seqA", {"a.mp4": 0, "b.mp4": 1})
    assert _compositions(ds, "media_raw") == first


def test_reordering_a_sequence_moves_its_composition(tmp_path: Path) -> None:
    """The change a media composition exists to detect."""
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        _cfr_mp4(tmp_path / "media_raw" / "seqA" / name)

    _arrange(ds, "seqA", {"a.mp4": 0, "b.mp4": 1})
    before = _compositions(ds, "media_raw")["seqA"]
    _arrange(ds, "seqA", {"a.mp4": 1, "b.mp4": 0})
    assert _compositions(ds, "media_raw")["seqA"] != before


def test_an_unrelated_sequence_does_not_move(tmp_path: Path) -> None:
    """Scoped invalidation, at the level a composition is compared.

    This is the shape of H3 case 2: adding a source somewhere else must leave
    everything it did not touch exactly where it was.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    _cfr_mp4(tmp_path / "media_raw" / "seqA" / "a.mp4")
    _arrange(ds, "seqA", {"a.mp4": 0})
    before = _compositions(ds, "media_raw")["seqA"]

    _cfr_mp4(tmp_path / "media_raw" / "seqB" / "b.mp4")
    _arrange(ds, "seqB", {"b.mp4": 0})

    after = _compositions(ds, "media_raw")
    assert after["seqA"] == before
    assert after["seqB"] != ""


def test_a_sequence_that_goes_away_leaves_the_index(tmp_path: Path) -> None:
    """A projection, not an accumulation -- which is why ``replace`` is used."""
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    scanned = tmp_path / "media_raw" / "scanned"
    _cfr_mp4(scanned / "a.mp4")
    _cfr_mp4(scanned / "b.mp4")
    ds.index_media([scanned], extensions=(".mp4",))
    assert set(_compositions(ds, "media_raw")) == {"a", "b"}

    # Rescan the same directory with one file gone. It is claimed by the scan,
    # so its row goes -- and the projection has to follow it out.
    (scanned / "b.mp4").unlink()
    ds.index_media([scanned], extensions=(".mp4",))

    assert set(_compositions(ds, "media_raw")) == {"a"}


def test_a_sequence_outside_the_scan_keeps_its_composition(tmp_path: Path) -> None:
    """The other half: a projection must not drop what the scan never claimed.

    ``write_sequence_compositions`` replaces the per-sequence index wholesale, so
    a scan that projected only the rows *it* walked would delete the composition
    of every sequence it had just carefully preserved -- silently, and only
    visible later as a moved identity hash.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    _cfr_mp4(tmp_path / "media_raw" / "seqA" / "a.mp4")
    _arrange(ds, "seqA", {"a.mp4": 0})
    assert set(_compositions(ds, "media_raw")) == {"seqA"}

    _cfr_mp4(tmp_path / "media_raw" / "seqB" / "b.mp4")
    ds.index_media([tmp_path / "media_raw" / "seqB"], extensions=(".mp4",))

    assert set(_compositions(ds, "media_raw")) == {"seqA", "b"}


# --- tracks_raw -------------------------------------------------------------


def test_a_tracks_raw_scan_projects_its_checksums(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    src = tmp_path / "tracks_raw" / "seqA"
    src.mkdir(parents=True)
    (src / "seqA.npy").write_bytes(b"payload")

    ds.index_tracks_raw([src], patterns=["*.npy"], src_format="calms21_npy")

    frame = read_sequence_index(ds, "tracks_raw")
    assert dict(zip(frame["sequence"], frame["member_count"].astype(int))) == {
        "seqA": 1
    }
    assert frame.iloc[0]["composition"] != ""


def test_checksums_off_leaves_the_composition_unestablishable(tmp_path: Path) -> None:
    """An honest empty, with the count still saying how much was there."""
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    src = tmp_path / "tracks_raw" / "seqA"
    src.mkdir(parents=True)
    (src / "seqA.npy").write_bytes(b"payload")

    ds.index_tracks_raw(
        [src], patterns=["*.npy"], src_format="calms21_npy", compute_md5=False
    )

    frame = read_sequence_index(ds, "tracks_raw")
    assert frame.iloc[0]["composition"] == ""
    assert int(frame.iloc[0]["member_count"]) == 1


def test_a_changed_source_file_moves_the_composition(tmp_path: Path) -> None:
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    src = tmp_path / "tracks_raw" / "seqA"
    src.mkdir(parents=True)
    (src / "seqA.npy").write_bytes(b"payload")
    ds.index_tracks_raw([src], patterns=["*.npy"], src_format="calms21_npy")
    before = _compositions(ds, "tracks_raw")["seqA"]

    (src / "seqA.npy").write_bytes(b"a different payload entirely")
    ds.index_tracks_raw([src], patterns=["*.npy"], src_format="calms21_npy")

    assert _compositions(ds, "tracks_raw")["seqA"] != before


# --- absence, rebuild, and the roots that have none -------------------------


def test_an_unindexed_root_reads_as_a_full_schema_empty_frame(tmp_path: Path) -> None:
    """Absence and emptiness are one state, and the columns are load-bearing.

    A column-less frame would turn "this root has never been indexed" into
    ``KeyError: 'group'`` at the first filter.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    frame = read_sequence_index(ds, "tracks_raw")
    assert frame.empty
    assert list(frame.columns) == SEQUENCE_INDEX_COLUMNS
    assert not sequence_index_path(ds, "tracks_raw").exists()


def test_an_unset_root_reads_as_empty_rather_than_raising(tmp_path: Path) -> None:
    """``get_root`` raises on an unset root; every dataset predating this is one."""
    tmp_path = tmp_path.resolve()
    ds = Dataset(manifest_path=tmp_path / "dataset.yaml", roots={})
    assert read_sequence_index(ds, "media_raw").empty


def test_rebuild_reproduces_what_the_writer_wrote(tmp_path: Path) -> None:
    """The oracle: projecting from committed rows and from disk must agree."""
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        _cfr_mp4(tmp_path / "media_raw" / "seqA" / name)
    _arrange(ds, "seqA", {"a.mp4": 1, "b.mp4": 0})
    written = _compositions(ds, "media_raw")

    sequence_index_path(ds, "media_raw").unlink()
    _ = ds.rebuild_sequence_index("media_raw")

    assert _compositions(ds, "media_raw") == written


def test_a_legacy_media_only_dataset_gets_no_media_composition(
    tmp_path: Path,
) -> None:
    """``media/`` holds derivatives, and a derivative has no composition (P6)."""
    tmp_path = tmp_path.resolve()
    ds = Dataset(
        manifest_path=tmp_path / "dataset.yaml",
        roots={"media": str(tmp_path / "media")},
    )
    ds.ensure_roots()
    ds.save()
    _cfr_mp4(tmp_path / "media" / "seqA" / "a.mp4")

    ds.write_media_index(
        [
            MediaIndexScope(
                directory=tmp_path / "media" / "seqA", group="", sequence="seqA"
            )
        ],
        extensions=(".mp4",),
    )

    assert not (tmp_path / "media" / "sequences.csv").exists()


def test_a_blank_sequence_row_composes_nothing(tmp_path: Path) -> None:
    """``multi_sequences_per_file`` puts the grouping in ``group`` and blanks the
    sequence, so one file covers many. Composing them under ``(group, "")`` would
    mint a value describing no sequence in particular.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    src = tmp_path / "tracks_raw" / "batch"
    src.mkdir(parents=True)
    (src / "everything.npy").write_bytes(b"payload")

    ds.index_tracks_raw(
        [src],
        patterns=["*.npy"],
        src_format="calms21_npy",
        multi_sequences_per_file=True,
        group_from="filename",
    )

    assert read_sequence_index(ds, "tracks_raw").empty


# --- the two-file write order -----------------------------------------------


_ORDER_PROBE = """
import sys
from pathlib import Path
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.media_index import MediaIndexScope

manifest, sequence, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ds = Dataset(manifest_path=manifest).load()
barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
ds.write_media_index(
    [MediaIndexScope(directory=ds.get_root("media_raw") / sequence, group="", sequence=sequence)],
    extensions=(".mp4",),
)
"""


def test_two_concurrent_writes_leave_a_consistent_projection(tmp_path: Path) -> None:
    """Two locks, taken in sequence, never nested.

    The projection may lag the index by one write -- stale, self-healing, and
    argued for where ``replace`` is defined -- but it must never be torn, and
    every sequence the index holds must have a row.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    for sequence in ("seqA", "seqB"):
        for name in ("a.mp4", "b.mp4"):
            _cfr_mp4(tmp_path / "media_raw" / sequence / name)

    gate = tmp_path / "gate"
    gate.mkdir()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _ORDER_PROBE,
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

    indexed = {row["sequence"] for row in ds.read_media_index()}
    assert indexed == {"seqA", "seqB"}
    frame = read_sequence_index(ds, "media_raw")
    # Never torn: one row per sequence, and the file parses as the full schema.
    assert list(frame.columns) == SEQUENCE_INDEX_COLUMNS
    assert len(frame) == len(set(frame["sequence"]))
    # A rebuild always reconciles a lagging projection with the index.
    _ = ds.rebuild_sequence_index("media_raw")
    assert set(read_sequence_index(ds, "media_raw")["sequence"]) == indexed


def test_the_projection_is_written_after_the_index(tmp_path: Path) -> None:
    """Order is the crash-safety argument, so it is pinned rather than assumed.

    Reversed, a crash between the two would leave a composition describing an
    index state that never committed -- a confident value nothing supports.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    _cfr_mp4(tmp_path / "media_raw" / "seqA" / "a.mp4")
    _arrange(ds, "seqA", {"a.mp4": 0})

    index_mtime = (tmp_path / "media_raw" / "index.csv").stat().st_mtime_ns
    projection_mtime = sequence_index_path(ds, "media_raw").stat().st_mtime_ns
    assert projection_mtime >= index_mtime


def test_the_projection_survives_a_csv_round_trip_with_integer_counts(
    tmp_path: Path,
) -> None:
    """``member_count`` must read back as an int, not ``2.0``.

    The dtype trap ``adopt_sequence_columns`` builds every column as ``object``
    to avoid: an empty column lands ``float64`` and a later concat widens the
    integer.
    """
    tmp_path = tmp_path.resolve()
    ds = _make_dataset(tmp_path)
    for name in ("a.mp4", "b.mp4"):
        _cfr_mp4(tmp_path / "media_raw" / "seqA" / name)
    _arrange(ds, "seqA", {"a.mp4": 0, "b.mp4": 1})

    raw = pd.read_csv(sequence_index_path(ds, "media_raw"), keep_default_na=False)
    assert list(raw["member_count"].astype(str)) == ["2"]


# --- the cell encoding (item 6.2 reads it, so its shape is load-bearing) -------


class TestEncodeEntryComposition:
    """The one minter for the ``consumed_composition`` cell.

    Its output is compared against itself across time -- a row written months ago
    against a value encoded now -- so the shape has to be a function of the
    declaration alone. A shape that varied with what happened to be recorded
    would make two different states encode alike.
    """

    def test_no_declared_root_records_nothing(self) -> None:
        assert encode_entry_composition({}, []) == ""
        assert encode_entry_composition({"media_raw": "abc"}, []) == ""

    def test_one_declared_root_records_a_bare_digest(self) -> None:
        """The form every cell on every dataset carries today."""
        assert encode_entry_composition({"media_raw": "abc"}, ["media_raw"]) == "abc"

    def test_one_declared_root_with_nothing_recorded_is_empty(self) -> None:
        assert encode_entry_composition({}, ["media_raw"]) == ""
        assert encode_entry_composition({"tracks_raw": "abc"}, ["media_raw"]) == ""

    def test_two_declared_roots_are_labelled_and_sorted(self) -> None:
        assert (
            encode_entry_composition(
                {"tracks_raw": "def", "media_raw": "abc"},
                ["tracks_raw", "media_raw"],
            )
            == "media_raw=abc,tracks_raw=def"
        )

    def test_a_declared_root_that_recorded_nothing_still_appears(self) -> None:
        """The case this encoding exists for.

        Emitting only what was found would return the bare ``"abc"`` here -- the
        same cell a consumer declaring ``media_raw`` alone writes. The two say
        different things and must not compare equal.
        """
        cell = encode_entry_composition(
            {"media_raw": "abc"}, ["media_raw", "tracks_raw"]
        )
        assert cell == "media_raw=abc,tracks_raw="
        assert cell != encode_entry_composition({"media_raw": "abc"}, ["media_raw"])

    def test_two_declared_roots_recording_nothing_is_empty(self) -> None:
        """Not ``media_raw=,tracks_raw=``: a cell carrying no digest is unknown.

        Spelling it out would make it compare unequal to the ``""`` a legacy row
        holds, turning "nothing recorded, then and now" into drift.
        """
        assert encode_entry_composition({}, ["media_raw", "tracks_raw"]) == ""

    def test_the_declaration_order_does_not_reach_the_cell(self) -> None:
        """Two spellings of one declaration are one answer."""
        recorded = {"media_raw": "abc", "tracks_raw": "def"}
        assert encode_entry_composition(
            recorded, ["media_raw", "tracks_raw"]
        ) == encode_entry_composition(
            recorded, ["tracks_raw", "media_raw", "media_raw"]
        )
