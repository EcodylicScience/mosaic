"""A changed source under an unchanged recipe must not serve a stale feature output.

The per-entry cache check compared compositions only for roots a feature *declares*
in ``consumed_roots``, and two of forty-four declarations are non-empty -- both
``media_raw``. No feature declares tracks, because tracks is a derived root and
``SourceRoot`` is closed over the three raw ones. So both sides of the comparison were
empty, the disposition was ``undetectable``, and the entry was served.

Meanwhile the tracks row *did* record the source composition its table was converted
from, and a re-conversion correctly updates it. Nothing carried that to the feature.

What this closes, precisely:

* **source bytes or sequence membership changed** -> the tracks row's composition
  moves -> the feature recomputes. That is the defect.
* **params changed** -> a new variant id, a different ``tracks/<variant>/``
  directory, and the downstream ``_tracks`` hash term already moves. Never was the
  defect.
* **a re-run over unchanged sources** -> composition unchanged -> still a cache hit,
  including for a nondeterministic tracker. Deliberately unaffected.

A row written before the column reads empty and is served with a warning rather than
recomputed, so no existing dataset is forced to redo work.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.index import (
    FEATURE_INDEX_COLUMNS,
    feature_index_path,
    recorded_consumption,
    recorded_tracks_reads,
)
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.run import cached_entry_disposition, run_feature
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.index_csv import index_records
from mosaic.core.pipeline.sequence_index import read_sequence_index
from mosaic.core.pipeline.tracks_index import (
    read_tracks_index,
    tracks_compositions,
    write_tracks_row,
)
from mosaic.core.pipeline.writers import write_parquet_atomic
from mosaic.core.pipeline.types import Feature, Inputs, Result, TrackInput
from mosaic.core.pipeline.types.feature import EmitsLevel
from tests.helpers import (
    CropLike,
    PlainFeature,
    add_tracks_variant,
    write_media_index,
    write_trex_npz,
)

VARIANT = "convert-trex_npz.0.1-aaaaaaaaaa"
"""One labelled tracks variant, so ``scope.tracks_variants`` is non-empty.

Pinned on every ``run_feature`` call below. Without the pin the scope also takes
``seq_b``, whose only row is the unlabelled one ``scenario_dataset`` writes, and
an assertion about one entry's provenance would be answered by another's.
"""

BLIND_TRACKS = "tracks tables record no source composition"
"""What the tracks channel says when neither side recorded a composition."""

SAID_ANYTHING = "composition"
"""The word every provenance report contains, whatever it goes on to claim.

The negative assertions test for *this* rather than for the sentence they expect
to be absent. A silence test worded against the message it is meant to suppress
passes the moment that message is reworded, which is how a regression test for a
warning quietly stops testing anything.
"""


def _cache_serving_stderr(
    ds: Dataset, feature: Feature, capsys: pytest.CaptureFixture[str]
) -> str:
    """What *feature* says on the run that serves its outputs from cache.

    The first run computes and is drained: it reports what it wrote, and only the
    second run reaches the pre-pass whose provenance accounting is under test.
    """
    _ = run_feature(ds, feature, tracks_run_id=VARIANT)
    _ = capsys.readouterr()
    _ = run_feature(ds, feature, tracks_run_id=VARIANT)
    return capsys.readouterr().err


def test_the_tracks_rows_composition_is_readable_per_entry(
    scenario_dataset: Dataset,
) -> None:
    """The comparison's right-hand side comes from the tracks index, not a digest."""
    ds = scenario_dataset
    add_tracks_variant(ds, "conv.0.1-aaaaaaaaaa", "seq_a")

    # Through the typed reader, not read_csv: an empty cell comes back as NaN from
    # a raw read and ``str(NaN)`` is the word "nan", which is the trap
    # ``text_cell`` exists to close.
    index = read_tracks_index(ds)
    recorded = text_cell(index.iloc[0].get("consumed_composition", ""))

    per_entry = tracks_compositions(ds, ("conv.0.1-aaaaaaaaaa",))

    # Whatever the row recorded is what a consumer sees -- including "" on a dataset
    # with no source projection, which is the honest unknown rather than a claim.
    assert per_entry.get(("", "seq_a"), "") == recorded


def test_a_moved_tracks_composition_is_a_recompute() -> None:
    """The disposition rule, applied to the tracks cell.

    Two known and different values is the only combination that means "the source
    moved under this output"; the rest are the same three-valued answer the roots
    comparison already gives.
    """
    assert cached_entry_disposition("aaa", "bbb") == "recompute"
    assert cached_entry_disposition("aaa", "aaa") == "serve"
    # One side known is still a refusal: an absent record is not evidence of safety.
    assert cached_entry_disposition("", "bbb") == "recompute"
    assert cached_entry_disposition("aaa", "") == "recompute"


def test_a_row_predating_the_column_is_not_forced_to_recompute(
    scenario_dataset: Dataset,
) -> None:
    """Both sides empty stays ``undetectable``, which is served.

    This is what makes the new column free: every row written before it exists reads
    ``""``, and a dataset is not made to redo every tracks-consuming feature.
    """
    ds = scenario_dataset
    assert cached_entry_disposition("", "") == "undetectable"
    # And nothing is recorded for a run that never wrote the column.
    assert recorded_tracks_reads(ds, "no-such-feature", "0.1-x") == {}


def test_a_computed_row_records_the_tracks_cell() -> None:
    """The column exists, so the next run has something to compare against."""
    assert "consumed_tracks_composition" in FEATURE_INDEX_COLUMNS


class TestWhichChannelIsBlind:
    """A channel a run does not read is absent, not unverifiable.

    The comparison is two channels wide -- the source roots a feature declares,
    and the tracks tables it reads -- and each can be blind alone. Counting them
    in one set reported the wrong thing in both directions. ``disposition`` is
    unconditionally ``undetectable`` for a feature declaring no source root,
    which is forty-four of forty-seven and the declaration the protocol documents
    as correct, so those runs announced an unverifiable source on every cache hit
    while naming ``tracks`` -- the one channel that was in fact recorded and
    compared. The mirror is the same line: a genuinely unrecorded tracks channel
    under a declared root that *was* recorded was served in silence.
    """

    def test_a_recorded_tracks_composition_says_nothing(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Provenance established on both sides is not something to report.

        The defect, in the shape a user meets it: a per-frame feature over
        converted tracks, on a dataset whose raw sources are indexed and whose
        projection is written, warned about the source it could check.
        """
        ds = scenario_dataset
        write_trex_npz(ds.get_root("tracks_raw") / "seq_a.npz", n=8)
        _ = ds.index_tracks_raw(
            [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
        )
        add_tracks_variant(ds, VARIANT, "seq_a")

        # The precondition the assertion rests on: both sides hold the same real
        # digest, so the tracks channel serves rather than being undetectable.
        recorded = tracks_compositions(ds, (VARIANT,)).get(("", "seq_a"), "")
        assert recorded, "the scenario did not record a tracks composition"

        assert SAID_ANYTHING not in _cache_serving_stderr(ds, PlainFeature(), capsys)

    def test_an_unrecorded_tracks_composition_is_reported(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Blind is still blind, and the tracks channel now says so itself.

        Same run as above with the raw indexing left out, so no
        ``tracks_raw/sequences.csv`` exists and the tracks row's cell is empty.
        Silence here would be the over-correction.
        """
        ds = scenario_dataset
        add_tracks_variant(ds, VARIANT, "seq_a")
        assert tracks_compositions(ds, (VARIANT,)) == {}

        err = _cache_serving_stderr(ds, PlainFeature(), capsys)

        assert BLIND_TRACKS in err
        # The repair has to be one that reaches this cell. VARIANT is a conversion,
        # whose cell comes from ``tracks_raw``, so a scan alone leaves it empty --
        # and `mosaic reindex`, which only drops rows naming files that are gone,
        # never touches it at all.
        assert "`mosaic scan --kind tracks`" in err
        assert "`mosaic convert-tracks --overwrite`" in err
        assert "mosaic reindex" not in err

    def test_a_bridged_table_is_not_told_to_re_convert(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A tracker's table takes its cell from the video, not from ``tracks_raw``.

        So the conversion pair cannot reach it: a scan of ``tracks_raw`` writes a
        projection that contributed nothing, and ``convert-tracks`` walks
        ``tracks_raw/index.csv``, where a ``_tracking/``-bridged entry has no row
        at all. Naming them would be two commands that change nothing, and on a
        dataset still holding the raw file the second could mint a second variant
        for an entry that already has one.
        """
        ds = scenario_dataset
        bridged = "trex.0.2-bbbbbbbbbb"
        add_tracks_variant(ds, bridged, "seq_a", consumed_source_roots=("media_raw",))
        assert tracks_compositions(ds, (bridged,)) == {}

        _ = run_feature(ds, PlainFeature(), tracks_run_id=bridged)
        _ = capsys.readouterr()
        _ = run_feature(ds, PlainFeature(), tracks_run_id=bridged)
        err = capsys.readouterr().err

        assert BLIND_TRACKS in err
        assert "trex" in err, "the message did not say which producer wrote it"
        assert "`mosaic convert-tracks --overwrite`" not in err
        assert "`mosaic scan --kind tracks`" not in err

    def test_both_channels_blind_are_reported_independently(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two channels, two reports -- neither one standing in for the other.

        A dataset with no projection at all is blind on both, and the two are
        separate statements about separate repairs. Collapsing them into one
        message, or reaching the second only when the first did not fire, would
        tell a reader that the channel they were not told about is fine.
        """
        ds = scenario_dataset
        add_tracks_variant(ds, VARIANT, "seq_a")

        err = _cache_serving_stderr(ds, CropLike(), capsys)

        assert "media_raw composition is unrecorded" in err
        assert BLIND_TRACKS in err

    def test_a_run_reading_no_tracks_variant_says_nothing(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An unlabelled dataset is not told about a projection that cannot help.

        ``tracks_compositions`` filters on the variant, so an entry whose only row
        predates variants contributes nothing whatever the projection holds --
        writing one would not change the answer, and neither would re-converting.
        Reporting it would name a repair that does not exist.
        """
        ds = scenario_dataset
        _ = run_feature(ds, PlainFeature())
        _ = capsys.readouterr()
        _ = run_feature(ds, PlainFeature())

        assert SAID_ANYTHING not in capsys.readouterr().err

    def test_a_blind_declared_root_names_its_own_repair(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The roots channel reports the root it declared, not ``tracks``.

        ``CropLike`` declares ``media_raw`` and this dataset has no media, so the
        roots channel is genuinely blind while the tracks channel is recorded. The
        message must name ``media_raw`` and the scan that writes its projection --
        the fallback that used to spell this ``tracks`` is what made the warning
        point at the wrong root.
        """
        ds = scenario_dataset
        write_trex_npz(ds.get_root("tracks_raw") / "seq_a.npz", n=8)
        _ = ds.index_tracks_raw(
            [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
        )
        add_tracks_variant(ds, VARIANT, "seq_a")

        err = _cache_serving_stderr(ds, CropLike(), capsys)

        assert "media_raw composition is unrecorded" in err
        # The whole command, not a prefix of it: `--kind media_raw` also contains
        # "--kind media", and it is a command the CLI rejects outright. The suffix
        # strip that turns the root into the kind has no other test.
        assert "`mosaic scan --kind media`" in err
        # The tracks channel is recorded here, so it must not be reported beside it.
        assert BLIND_TRACKS not in err

    def test_a_recorded_declared_root_says_nothing(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The roots half of the same rule: recorded on both sides is not news.

        The mirror of the tracks case, and the one that keeps the fix from being
        "declaring a root means you get warned". ``CropLike`` declares ``media_raw``
        and here both channels are projected, so the run says nothing at all.
        """
        ds = scenario_dataset
        write_media_index(ds, ["seq_a"], uids={"seq_a": "uid-seq-a"})
        _ = ds.rebuild_sequence_index("media_raw")
        write_trex_npz(ds.get_root("tracks_raw") / "seq_a.npz", n=8)
        _ = ds.index_tracks_raw(
            [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
        )
        add_tracks_variant(ds, VARIANT, "seq_a")

        # Both preconditions, so a silence caused by the channels being absent
        # rather than satisfied cannot pass for the answer under test.
        projected = index_records(read_sequence_index(ds, "media_raw"))
        assert projected and projected[0]["composition"]
        assert tracks_compositions(ds, (VARIANT,)).get(("", "seq_a"), "")

        assert SAID_ANYTHING not in _cache_serving_stderr(ds, CropLike(), capsys)


class TestTheFirstFrameATableWasNumberedFrom:
    """A table published again at a moved first frame reads as another composition.

    A tracker's table of a later sequence in a continuous group is numbered from
    the sequence's first frame. When a member before it changes length, the
    table is published again at the new first frame from the same media. Its
    source composition alone is unchanged.
    """

    @staticmethod
    def _composition(ds: Dataset, first_frame: int | None) -> str:
        write_trex_npz(ds.get_root("tracks_raw") / "seq_a.npz", n=8)
        _ = ds.index_tracks_raw(
            [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
        )
        out = ds.get_root("tracks") / VARIANT / "seq_a.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        _ = write_parquet_atomic(pd.DataFrame({"frame": [0, 1], "id": [0, 0]}), out)
        write_tracks_row(
            ds,
            run_id=VARIANT,
            group="",
            sequence="seq_a",
            out_path=out,
            producer="convert-trex_npz",
            std_format="mosaic_v1",
            n_rows=2,
            consumed_source_roots=("tracks_raw",),
            first_frame=first_frame,
            carries_tracklets=None,
        )
        return tracks_compositions(ds, (VARIANT,))[("", "seq_a")]

    @pytest.mark.parametrize("first_frame", [None, 0], ids=["blank", "zero"])
    def test_a_table_at_frame_0_keeps_the_source_composition(
        self, scenario_dataset: Dataset, first_frame: int | None
    ) -> None:
        composition = self._composition(scenario_dataset, first_frame)

        rows = read_tracks_index(scenario_dataset)
        (recorded,) = rows[rows["run_id"] == VARIANT]["consumed_composition"]
        assert recorded
        assert composition == recorded

    def test_another_first_frame_is_another_composition(
        self, scenario_dataset: Dataset
    ) -> None:
        at_30 = self._composition(scenario_dataset, 30)
        at_40 = self._composition(scenario_dataset, 40)

        assert at_30 != at_40
        assert cached_entry_disposition(at_30, at_40) == "recompute"


class _Upstream(PlainFeature):
    """A per-entry feature over tracks whose output another feature reads."""

    name = "prov-upstream"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"


class _Reader(_Upstream):
    """A per-entry feature that reads the inputs it is given, a feature's among them."""

    name = "prov-reader"


def _convert(ds: Dataset, *sequences: str, n: int = 8) -> None:
    """Convert *sequences* from raw files of *n* frames, recording their composition."""
    for sequence in sequences:
        write_trex_npz(ds.get_root("tracks_raw") / f"{sequence}.npz", n=n)
    _ = ds.index_tracks_raw(
        [ds.get_root("tracks_raw")], patterns=["*.npz"], src_format="trex_npz"
    )
    add_tracks_variant(ds, VARIANT, *sequences)


def _tracks_cell(ds: Dataset, sequence: str) -> str:
    return tracks_compositions(ds, (VARIANT,))[("", sequence)]


def _recorded_cell(ds: Dataset, result: Result, sequence: str) -> str:
    assert result.run_id is not None
    return recorded_tracks_reads(ds, result.feature, result.run_id)[
        ("", sequence)
    ].composition


def _recomputed(err: str) -> set[str]:
    """The sequences a run refused to serve from its cache."""
    return {
        sequence
        for sequence in ("seq_a", "seq_b")
        if f"cannot serve cached output for (,{sequence})" in err
    }


class TestTracksProvenanceThroughFeatureInputs:
    """A run reading a feature's output compares the tracks its input rows read."""

    def test_a_reader_recomputes_after_its_upstream_recomputes_under_one_run_id(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        _convert(ds, "seq_a", "seq_b")
        upstream = run_feature(ds, _Upstream(), tracks_run_id=VARIANT)
        reader = _Reader(
            Inputs((Result(feature=upstream.feature, run_id=upstream.run_id),))
        )
        _ = run_feature(ds, reader)
        before = _tracks_cell(ds, "seq_a")

        _convert(ds, "seq_a", n=9)
        assert _tracks_cell(ds, "seq_a") != before
        again = run_feature(ds, _Upstream(), tracks_run_id=VARIANT)
        assert again.run_id == upstream.run_id
        _ = capsys.readouterr()

        _ = run_feature(ds, reader)

        assert _recomputed(capsys.readouterr().err) == {"seq_a"}

    def test_a_reader_row_records_its_inputs_compositions_in_input_order(
        self, scenario_dataset: Dataset
    ) -> None:
        ds = scenario_dataset
        _convert(ds, "seq_a", "seq_b")
        upstream = run_feature(ds, _Upstream(), tracks_run_id=VARIANT)
        upstream_read = _tracks_cell(ds, "seq_a")
        # Converted again without running the upstream, so its row and the
        # tracks row record different compositions and the order shows.
        _convert(ds, "seq_a", n=9)
        tracks_now = _tracks_cell(ds, "seq_a")
        assert tracks_now != upstream_read
        result_input = Result(feature=upstream.feature, run_id=upstream.run_id)

        upstream_first = run_feature(
            ds, _Reader(Inputs((result_input, "tracks"))), tracks_run_id=VARIANT
        )
        tracks_first = run_feature(
            ds, _Reader(Inputs(("tracks", result_input))), tracks_run_id=VARIANT
        )

        assert (
            _recorded_cell(ds, upstream_first, "seq_a")
            == f"{upstream_read}+{tracks_now}"
        )
        assert (
            _recorded_cell(ds, tracks_first, "seq_a") == f"{tracks_now}+{upstream_read}"
        )

    def test_a_reader_row_written_before_the_change_recomputes_once(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        _convert(ds, "seq_a", "seq_b")
        upstream = run_feature(ds, _Upstream(), tracks_run_id=VARIANT)
        reader = _Reader(
            Inputs((Result(feature=upstream.feature, run_id=upstream.run_id),))
        )
        written = run_feature(ds, reader)
        # A row written before feature inputs carried their tracks composition
        # recorded none for a reader of feature output.
        index_path = feature_index_path(ds, written.feature)
        rows = pd.read_csv(index_path, dtype=str, keep_default_na=False)
        rows["consumed_tracks_composition"] = ""
        rows.to_csv(index_path, index=False)
        _ = capsys.readouterr()

        _ = run_feature(ds, reader)
        first = capsys.readouterr().err
        _ = run_feature(ds, reader)
        second = capsys.readouterr().err

        assert _recomputed(first) == {"seq_a", "seq_b"}
        assert _recomputed(second) == set()

    def test_a_tracks_only_run_records_the_cell_as_before(
        self, scenario_dataset: Dataset
    ) -> None:
        ds = scenario_dataset
        _convert(ds, "seq_a", "seq_b")

        result = run_feature(ds, _Upstream(), tracks_run_id=VARIANT)
        _, resolved = build_manifest(ds, Inputs(("tracks",)), tracks_run_id=VARIANT)

        assert resolved.input_tracks_compositions is None
        assert _recorded_cell(ds, result, "seq_a") == _tracks_cell(ds, "seq_a")


class _RootUpstream(_Upstream):
    """A per-entry feature over tracks that declares ``media_raw``, as ``scale-to-cm`` does."""

    name = "prov-root-upstream"
    consumed_roots: tuple[str, ...] = ("media_raw",)


class _Third(_Upstream):
    """A per-entry feature at the end of a three-feature chain."""

    name = "prov-third"


def _index_media(ds: Dataset, seq_a: str, seq_b: str) -> None:
    """Index one video per sequence under the given uids, and project ``media_raw``."""
    write_media_index(ds, ["seq_a", "seq_b"], uids={"seq_a": seq_a, "seq_b": seq_b})
    _ = ds.rebuild_sequence_index("media_raw")


def _reader_of(*inputs: TrackInput | Result) -> _Reader:
    return _Reader(Inputs(inputs))


def _input_of(result: Result) -> Result:
    return Result(feature=result.feature, run_id=result.run_id)


def _recorded_input_cell(ds: Dataset, result: Result, sequence: str) -> str:
    assert result.run_id is not None
    return recorded_tracks_reads(ds, result.feature, result.run_id)[
        ("", sequence)
    ].inputs


def _recorded_roots_cell(ds: Dataset, result: Result, sequence: str) -> str:
    assert result.run_id is not None
    return recorded_consumption(ds, result.feature, result.run_id)[("", sequence)][1]


class TestDeclaredRootsThroughFeatureInputs:
    """A run reading a feature's output compares the roots its upstream declared."""

    @staticmethod
    def _chain(ds: Dataset) -> tuple[Result, _Reader]:
        _convert(ds, "seq_a", "seq_b")
        _index_media(ds, "uid-a-1", "uid-b-1")
        upstream = run_feature(ds, _RootUpstream(), tracks_run_id=VARIANT)
        reader = _reader_of(_input_of(upstream))
        _ = run_feature(ds, reader)
        return upstream, reader

    def test_a_reader_recomputes_after_its_upstreams_declared_root_moves(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        upstream, reader = self._chain(ds)

        _index_media(ds, "uid-a-2", "uid-b-1")
        _ = capsys.readouterr()
        again = run_feature(ds, _RootUpstream(), tracks_run_id=VARIANT)
        assert again.run_id == upstream.run_id
        assert _recomputed(capsys.readouterr().err) == {"seq_a"}

        result = run_feature(ds, reader)
        err = capsys.readouterr().err

        assert _recomputed(err) == {"seq_a"}
        assert "its upstream's source moved" in err
        assert not result.cache_hit

    def test_a_third_feature_recomputes_after_the_first_ones_root_moves(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        _, reader = self._chain(ds)
        middle = run_feature(ds, reader)
        third = _Third(Inputs((_input_of(middle),)))
        written = run_feature(ds, third)
        # The middle feature declares no root, so the third inherits the first's.
        first_part = _recorded_input_cell(ds, middle, "seq_a")
        assert first_part.endswith("/")
        assert _recorded_input_cell(ds, written, "seq_a") == f"/{first_part}"

        _index_media(ds, "uid-a-2", "uid-b-1")
        _ = run_feature(ds, _RootUpstream(), tracks_run_id=VARIANT)
        _ = run_feature(ds, reader)
        _ = capsys.readouterr()

        result = run_feature(ds, third)

        assert _recomputed(capsys.readouterr().err) == {"seq_a"}
        assert not result.cache_hit

    def test_an_unchanged_rerun_serves_the_reader_without_a_warning(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        _, reader = self._chain(ds)
        _ = run_feature(ds, _RootUpstream(), tracks_run_id=VARIANT)
        _ = capsys.readouterr()

        result = run_feature(ds, reader)

        assert result.cache_hit
        assert SAID_ANYTHING not in capsys.readouterr().err

    def test_a_reader_row_records_its_upstreams_root_cell_in_input_order(
        self, scenario_dataset: Dataset
    ) -> None:
        ds = scenario_dataset
        upstream, _ = self._chain(ds)
        own = _recorded_roots_cell(ds, upstream, "seq_a")
        assert own

        upstream_first = run_feature(
            ds, _reader_of(_input_of(upstream), "tracks"), tracks_run_id=VARIANT
        )
        tracks_first = run_feature(
            ds, _reader_of("tracks", _input_of(upstream)), tracks_run_id=VARIANT
        )

        assert _recorded_input_cell(ds, upstream_first, "seq_a") == f"{own}/+"
        assert _recorded_input_cell(ds, tracks_first, "seq_a") == f"+{own}/"

    def test_a_reader_row_written_before_the_cell_recomputes_once(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        _, reader = self._chain(ds)
        written = run_feature(ds, reader)
        # An index written before the column existed has no such column at all.
        index_path = feature_index_path(ds, written.feature)
        rows = pd.read_csv(index_path, dtype=str, keep_default_na=False)
        rows = rows.drop(columns=["consumed_input_composition"])
        rows.to_csv(index_path, index=False)
        _ = capsys.readouterr()

        _ = run_feature(ds, reader)
        first = capsys.readouterr().err
        _ = run_feature(ds, reader)
        second = capsys.readouterr().err

        assert _recomputed(first) == {"seq_a", "seq_b"}
        assert _recomputed(second) == set()

    def test_a_reader_over_an_upstream_recording_no_tracks_composition_names_it(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        add_tracks_variant(ds, VARIANT, "seq_a")
        assert tracks_compositions(ds, (VARIANT,)) == {}
        upstream = run_feature(ds, _Upstream(), tracks_run_id=VARIANT)
        assert upstream.run_id is not None
        reader = _reader_of(_input_of(upstream))
        _ = run_feature(ds, reader)
        _ = capsys.readouterr()

        _ = run_feature(ds, reader)
        err = capsys.readouterr().err

        assert BLIND_TRACKS in err
        assert upstream.feature in err
        assert upstream.run_id in err
        assert VARIANT not in err

    def test_a_reader_over_an_upstream_of_the_unlabelled_variant_says_nothing(
        self, scenario_dataset: Dataset, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ds = scenario_dataset
        upstream = run_feature(ds, _Upstream())
        reader = _reader_of(_input_of(upstream))
        _ = run_feature(ds, reader)
        _ = capsys.readouterr()

        _ = run_feature(ds, reader)

        assert SAID_ANYTHING not in capsys.readouterr().err
