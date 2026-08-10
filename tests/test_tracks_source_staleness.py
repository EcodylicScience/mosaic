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

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.index import (
    FEATURE_INDEX_COLUMNS,
    recorded_tracks_composition,
)
from mosaic.core.pipeline.run import cached_entry_disposition, run_feature
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.provenance import index_records
from mosaic.core.pipeline.sequence_index import read_sequence_index
from mosaic.core.pipeline.tracks_index import read_tracks_index, tracks_compositions
from mosaic.core.pipeline.types import Feature
from tests.conftest import add_tracks_variant, write_media_index, write_trex_npz
from tests.test_provenance import CropLike, PlainFeature

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
    scenario_dataset: object,
) -> None:
    """The comparison's right-hand side comes from the tracks index, not a digest."""
    ds = scenario_dataset
    add_tracks_variant(ds, "conv.0.1-aaaaaaaaaa", "seq_a")  # pyright: ignore[reportArgumentType]

    # Through the typed reader, not read_csv: an empty cell comes back as NaN from
    # a raw read and ``str(NaN)`` is the word "nan", which is the trap
    # ``text_cell`` exists to close.
    index = read_tracks_index(ds)  # pyright: ignore[reportArgumentType]
    recorded = text_cell(index.iloc[0].get("consumed_composition", ""))

    per_entry = tracks_compositions(ds, ("conv.0.1-aaaaaaaaaa",))  # pyright: ignore[reportArgumentType]

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
    scenario_dataset: object,
) -> None:
    """Both sides empty stays ``undetectable``, which is served.

    This is what makes the new column free: every row written before it exists reads
    ``""``, and a dataset is not made to redo every tracks-consuming feature.
    """
    ds = scenario_dataset
    assert cached_entry_disposition("", "") == "undetectable"
    # And nothing is recorded for a run that never wrote the column.
    assert recorded_tracks_composition(ds, "no-such-feature", "0.1-x") == {}  # pyright: ignore[reportArgumentType]


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
