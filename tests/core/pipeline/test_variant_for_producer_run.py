"""Which tracks variant an op run produced.

TracksIndexRow.run_id is the tracks *variant* and producer_run_id is the op run
behind it. Nothing could ask the reverse, so anything holding a tracker's run
identifier passed it straight through as a variant -- which works only because
tracker_variant_payload is a passthrough, and stops working the moment an op
version bump moves one identity and not the other.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mosaic.core.pipeline.tracks_index import variant_for_producer_run


def _index(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"run_id": variant, "producer_run_id": producer} for variant, producer in rows]
    )


def test_it_finds_the_variant_a_producer_wrote() -> None:
    frame = _index([("trex.0.2-aaaaaaaaaa", "trex.0.1-bbbbbbbbbb")])

    assert variant_for_producer_run(frame, "trex.0.1-bbbbbbbbbb") == (
        "trex.0.2-aaaaaaaaaa"
    )


def test_the_two_identities_are_allowed_to_differ() -> None:
    """The case the passthrough coincidence hides. An op version bump moves the
    op run and not the variant, and a reader passing one for the other resolves
    silently to a directory that does not exist."""
    frame = _index([("trex.0.2-samedigest", "trex.0.3-samedigest")])

    assert variant_for_producer_run(frame, "trex.0.3-samedigest") == (
        "trex.0.2-samedigest"
    )


def test_an_unknown_producer_reads_none_rather_than_raising() -> None:
    frame = _index([("trex.0.2-aaaaaaaaaa", "trex.0.1-bbbbbbbbbb")])

    assert variant_for_producer_run(frame, "trex.0.1-cccccccccc") is None


def test_an_empty_producer_matches_nothing() -> None:
    """An empty producer cell means *a conversion*, which is every converted row
    -- not one run to be found. Matching it would return an arbitrary variant."""
    frame = _index([("convert-trex.0.2-aaaaaaaaaa", ""), ("", "")])

    assert variant_for_producer_run(frame, "") is None


def test_two_variants_for_one_producer_raise_rather_than_guess() -> None:
    """The same refusal select_variant_rows makes for two recipes on one entry:
    there is no defensible way to pick, and guessing is a silent wrong answer."""
    frame = _index(
        [
            ("trex.0.2-aaaaaaaaaa", "trex.0.1-shared"),
            ("trex.0.2-bbbbbbbbbb", "trex.0.1-shared"),
        ]
    )

    with pytest.raises(ValueError, match="more than one"):
        _ = variant_for_producer_run(frame, "trex.0.1-shared")


def test_an_absent_index_reads_none(tmp_path) -> None:
    """Absent is empty everywhere in this package; it is not an error here either."""
    assert variant_for_producer_run(pd.DataFrame(), "trex.0.1-aaaaaaaaaa") is None
