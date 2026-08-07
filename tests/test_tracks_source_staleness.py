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

from mosaic.core.pipeline.index import (
    FEATURE_INDEX_COLUMNS,
    recorded_tracks_composition,
)
from mosaic.core.pipeline.run import cached_entry_disposition
from mosaic.core.helpers import text_cell
from mosaic.core.pipeline.tracks_index import read_tracks_index, tracks_compositions
from tests.conftest import add_tracks_variant


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
