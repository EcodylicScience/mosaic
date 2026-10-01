"""The stores of one entry share a rate by one rule, whatever their length.

A store's rate is estimated from its timestamps, and its frames are read by
index. So two stores measured at 30.0 and 30.002 fps are one rate measured twice,
at 60 frames or at an hour of them, and 30 and 31 fps are two rates at any length.
The reader of a store sequence and the check that refuses one before any work
starts ask :func:`~mosaic.core.media.store_rate.store_rate_mismatch`, and a grid
of rates and lengths pins that they agree.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import mosaic.core.media.video_io as video_io
from mosaic.core.media.store_rate import STORE_RATE_TOLERANCE, store_rate_mismatch
from mosaic.core.media.video_io import MultiVideoReader, VideoMetadata
from mosaic.cli import app
from mosaic.core.dataset import Dataset
from mosaic.tracking.common.scope import (
    JoinedSourceMismatchError,
    build_work_items,
    refuse_unjoinable,
)
from tests.helpers import clip_facts


class TestTheRule:
    @pytest.mark.parametrize(
        ("rates", "found"),
        [
            ((30.0,), None),
            ((30.0, 30.002), None),
            ((30.0, 30.0, 30.002), None),
            ((30.0, 29.97), 1),
            ((30.0, 31.0), 1),
            ((30.0, 30.0, 31.0), 2),
            ((30.0, 0.0), 1),
            ((0.0, 30.0), 1),
        ],
    )
    def test_a_store_at_another_rate_is_found(
        self, rates: tuple[float, ...], found: int | None
    ) -> None:
        assert store_rate_mismatch(rates) == found

    def test_the_tolerance_separates_one_rate_from_the_closest_two(self) -> None:
        """30.002 against 30 is one rate, and 29.97 against 30 is two."""
        assert abs(30.002 - 30.0) / 30.0 < STORE_RATE_TOLERANCE
        assert abs(30000 / 1001 - 30.0) / 30.0 > STORE_RATE_TOLERANCE


_STORE_SIZE = (64, 48)


def _stores(tmp_path: Path, count: int) -> list[Path]:
    """Store directories that ``is_imgstore`` recognizes, holding no frames."""
    stores: list[Path] = []
    for position in range(count):
        store = tmp_path / f"s{position}.store"
        store.mkdir()
        _ = (store / "metadata.yaml").write_text("__store: {}\n")
        stores.append(store)
    return stores


def _read_as(
    monkeypatch: pytest.MonkeyPatch, measured: dict[Path, tuple[float, int]]
) -> None:
    """Make the store reader measure each store at its ``(fps, frames)``."""

    def metadata(path: Path | str) -> VideoMetadata:
        fps, frames = measured[Path(path)]
        width, height = _STORE_SIZE
        return VideoMetadata(
            path=Path(path), frame_count=frames, fps=fps, width=width, height=height
        )

    monkeypatch.setattr(video_io, "get_video_metadata", metadata)


type Verdict = Callable[[], None]


def _accepted(verdict: Verdict) -> bool:
    try:
        verdict()
    except (ValueError, JoinedSourceMismatchError):
        return False
    return True


@pytest.mark.parametrize("frames", [60, 7_600, 108_000])
@pytest.mark.parametrize(
    ("second", "one_rate"),
    [(30.002, True), (29.995, True), (29.97, False), (31.0, False)],
)
def test_the_reader_and_the_early_check_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frames: int,
    second: float,
    one_rate: bool,
) -> None:
    """At every length, including an hour at 30 fps."""
    stores = _stores(tmp_path, 2)
    rates = (30.0, second)
    _read_as(monkeypatch, {store: (rate, frames) for store, rate in zip(stores, rates)})
    facts = [clip_facts(fps=rate, frame_count=frames) for rate in rates]

    def read() -> None:
        MultiVideoReader(stores, target="analysis").close()

    def check() -> None:
        refuse_unjoinable(
            "infer-localizer", "", "sess", stores, facts, hands_over_path=False
        )

    assert _accepted(read) is one_rate
    assert _accepted(check) is one_rate


def test_stores_at_two_rates_are_refused_as_unjoinable(tmp_path: Path) -> None:
    """The refusal says stores cannot be joined and names the preprocess remedy."""
    stores = _stores(tmp_path, 2)
    facts = [clip_facts(fps=30.0), clip_facts(fps=31.0)]

    with pytest.raises(JoinedSourceMismatchError) as refused:
        refuse_unjoinable(
            "infer-localizer", "", "sess", stores, facts, hands_over_path=False
        )

    message = str(refused.value)
    assert "s1.store was recorded at 31 fps" in message
    assert "Stores cannot be joined" in message
    assert '--kind preprocess --entries ":sess"' in message


def test_a_consumer_reading_stores_itself_takes_them_at_one_rate(
    tmp_path: Path,
) -> None:
    """The localizer reads stores in its own process, as one reader."""
    stores = _stores(tmp_path, 2)
    facts = [clip_facts(fps=30.0), clip_facts(fps=30.0)]

    refuse_unjoinable(
        "infer-localizer", "", "sess", stores, facts, hands_over_path=False
    )


def test_a_consumer_handing_over_a_path_refuses_stores_at_one_rate(
    tmp_path: Path,
) -> None:
    """Its tool is handed one file, and no file joins stores."""
    stores = _stores(tmp_path, 2)
    facts = [clip_facts(fps=30.0, frame_count=40), clip_facts(fps=30.0, frame_count=60)]

    with pytest.raises(JoinedSourceMismatchError) as refused:
        refuse_unjoinable("infer-pose", "g", "s", stores, facts, hands_over_path=True)

    message = str(refused.value)
    assert "export-joined does not join stores" in message
    trim = """--params '{"steps":[{"step":"trim","start":0,"stop":100}]}'"""
    assert f'--entries "g:s" {trim}' in message


def test_the_remedy_for_stores_of_unmeasured_length_says_to_measure_them(
    tmp_path: Path,
) -> None:
    stores = _stores(tmp_path, 2)
    facts = [clip_facts(fps=30.0, frame_count=0), clip_facts(fps=30.0, frame_count=60)]

    with pytest.raises(JoinedSourceMismatchError, match="reprobe-media --apply"):
        refuse_unjoinable("trex", "", "s", stores, facts, hands_over_path=True)


@pytest.mark.parametrize("fmt", ["npy"])
def test_long_real_stores_measured_apart_are_read_as_one(
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]], fmt: str
) -> None:
    """Eight thousand frames each, long enough to fail the plain half-frame rule."""
    first, _ = make_imgstore(
        name="a", nframes=8_000, fps=30.0, fmt=fmt, shape=(4, 4, 1), chunksize=2_000
    )
    second, _ = make_imgstore(
        name="b", nframes=8_000, fps=30.002, fmt=fmt, shape=(4, 4, 1), chunksize=2_000
    )

    reader = MultiVideoReader([first, second], target="analysis")

    assert reader.total_frames == 16_000
    reader.close()


@pytest.mark.media
def test_the_variant_the_refusal_names_is_made_and_read(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    make_imgstore: Callable[..., tuple[Path, list[np.ndarray]]],
) -> None:
    """Stores at 30 and 31 fps: the command runs, and a tracker reads its file."""
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    search = ds.get_root("media_raw") / "recordings"
    search.mkdir(parents=True)
    for name, fps in (("a", 30.0), ("b", 31.0)):
        _ = make_imgstore(
            name=name, nframes=30, chunksize=5, parent=search, fill=True, fps=fps
        )
    ds.index_media([search])
    index = ds.get_root("media_raw") / "index.csv"
    table = pd.read_csv(index, keep_default_na=False).sort_values("abs_path")
    table = table.assign(group="", sequence="sess", video_order=range(len(table)))
    table.to_csv(index, index=False)

    with pytest.raises(JoinedSourceMismatchError) as refused:
        _ = build_work_items(ds, ds.resolve_media_scope(None), kind="sleap")
    (command,) = [
        line.strip()
        for line in str(refused.value).splitlines()
        if line.strip().startswith("mosaic run")
    ]
    argv = shlex.split(command.replace("<manifest>", str(ds.manifest_path)))[1:]

    result = CliRunner().invoke(app, [*argv, "--json"])

    assert result.exit_code == 0, result.output
    variant = str(json.loads(result.stdout)["run_id"])
    (item,) = build_work_items(
        ds, ds.resolve_media_scope(None), kind="sleap", media=variant
    ).items
    assert item.source_facts[0].frame_count == 60
