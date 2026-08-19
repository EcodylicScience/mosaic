import dataclasses
from pathlib import Path

import pytest
from mosaic_media import CHROME_149, MediaFacts, Verdict, derive

from mosaic.core.media.facts_columns import (
    FLAT_FACTS_COLUMNS,
    MEDIA_INDEX_COLUMNS,
    facts_to_row,
    store_facts,
)
from mosaic.core.media.imgstore_io import STORE_IDENTITY_SCHEME
from mosaic.core.pipeline.media_index import build_media_index_row
from mosaic.media_probe_config import media_thresholds


@pytest.fixture
def sample_facts() -> MediaFacts:
    # store_facts builds a valid MediaFacts cheaply (no probe needed); replace
    # the empty identity with explicit non-empty values so the test can assert
    # facts_to_row actually carries them through.
    base = store_facts(
        320, 240, 25.0, 250, "h264", 10.0, video_uuid="", identity_scheme=""
    )
    return dataclasses.replace(
        base, video_uuid="video-uuid-abc123", content_digest="content-digest-def456"
    )


@pytest.fixture
def sample_verdict(sample_facts: MediaFacts) -> Verdict:
    return derive(sample_facts, CHROME_149, media_thresholds())


def test_the_three_identity_columns_are_in_the_schema() -> None:
    for column in ("video_uuid", "content_digest", "source_video_uuid"):
        assert column in FLAT_FACTS_COLUMNS, column
        assert column in MEDIA_INDEX_COLUMNS, column


def test_facts_to_row_carries_the_two_fact_values(
    sample_facts: MediaFacts, sample_verdict: Verdict
) -> None:
    row = facts_to_row(sample_facts, sample_verdict)
    assert row["video_uuid"] == "video-uuid-abc123"
    assert row["content_digest"] == "content-digest-def456"
    # source_video_uuid and encoder are transcode edges, not facts, so
    # facts_to_row leaves them empty for build_media_index_row to override.
    assert row["source_video_uuid"] == ""
    assert row["encoder"] == ""


def test_the_encoder_column_is_in_the_schema() -> None:
    assert "encoder" in FLAT_FACTS_COLUMNS
    assert "encoder" in MEDIA_INDEX_COLUMNS


def test_build_media_index_row_overrides_source_video_uuid(
    tmp_path: Path, sample_facts: MediaFacts, sample_verdict: Verdict
) -> None:
    target = tmp_path / "clip.mp4"
    _ = target.write_bytes(b"x")
    probe = {
        **facts_to_row(sample_facts, sample_verdict),
        "width": sample_facts.width,
        "height": sample_facts.height,
        "fps": sample_facts.fps,
        "codec": sample_facts.codec_name,
    }
    row = build_media_index_row(
        path=target,
        stat=target.stat(),
        to_store_path=lambda p: p.name,
        group="g",
        sequence="s",
        group_safe="g",
        sequence_safe="s",
        probe=probe,
        source_video_uuid="the-source-uuid",
    )
    assert row["source_video_uuid"] == "the-source-uuid"


def test_store_facts_carries_a_mint_and_never_a_content_digest() -> None:
    facts = store_facts(
        320,
        240,
        25.0,
        10,
        "h264",
        10.0,
        video_uuid="store-uuid-abc123",
        identity_scheme=STORE_IDENTITY_SCHEME,
    )
    # Since open item O5 a store carries an identity, but a *declared* one: its
    # __store.uuid, minted at creation, which the namespaced scheme marks as a
    # mint rather than a measurement. There is still no elementary stream to
    # hash, so content_digest stays empty, and no ffprobe runs, so
    # prober_version does too. Both remain claims store_facts states rather than
    # defaults it inherits.
    assert facts.video_uuid == "store-uuid-abc123"
    assert facts.identity_scheme == "imgstore/1"
    assert facts.content_digest == ""
    assert facts.timing_source == "presentation"
    assert facts.prober_version == ""


def test_store_facts_states_an_empty_identity_when_there_is_no_uuid() -> None:
    """A store whose metadata carries no uuid is honest about it, not guessed."""
    facts = store_facts(
        320, 240, 25.0, 10, "h264", 10.0, video_uuid="", identity_scheme=""
    )
    assert facts.video_uuid == ""
    assert facts.identity_scheme == ""
