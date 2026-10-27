"""``mosaic media``: mosaic-media's commands, with a probe that also reads an imgstore."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from mosaic_media.cli import media_app as upstream_media_app
from typer.testing import CliRunner

from mosaic.cli import app
from mosaic.core.media import imgstore_io
from mosaic.core.media.facts_columns import STORE_CONTAINER
from tests.helpers import invoke_ok, video_store_maker, write_h264_mp4

pytestmark = pytest.mark.media

SERIAL = "23036146"


def _store(parent: Path) -> Path:
    """A video store of chunks ``[5, 5, 2]`` at 20 fps whose chunks declare 25 fps."""
    make = video_store_maker([5, 5, 2], fps=20.0, container_fps=25.0)
    store, _ = make(
        name=f"{SERIAL}_20210219_112253",
        parent=parent,
        extra_metadata={"camera_serial": SERIAL, "title": "tank one"},
    )
    return store


def _probe_document(path: Path) -> dict[str, dict[str, object]]:
    document: dict[str, dict[str, object]] = json.loads(
        invoke_ok(["media", "probe", str(path)]).stdout
    )
    return document


def test_probing_a_store_prints_its_facts_verdict_and_store(tmp_path: Path) -> None:
    store = _store(tmp_path)

    document = _probe_document(store)

    assert sorted(document) == ["facts", "store", "verdict"]
    facts = document["facts"]
    assert facts["container"] == STORE_CONTAINER
    assert facts["frame_count"] == 12
    assert facts["fps"] == pytest.approx(20.0)
    assert (facts["width"], facts["height"]) == (64, 48)
    verdict = document["verdict"]
    assert verdict["analysis_transcode"] is None
    assert verdict["analysis_reasons"] == []
    assert document["store"] == {
        "camera_serial": SERIAL,
        "sync_uuid": "",
        "synchronization": "",
        "title": "tank one",
        "store_uuid": facts["video_uuid"],
        "chunksize": 5,
        "chunk_count": 3,
        "header_codec": "h264",
    }


def test_a_store_verdict_writes_its_reasons_as_sorted_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    verdict = dataclasses.replace(
        imgstore_io.STORE_VERDICT,
        stream_reasons=frozenset({"rotated", "interlaced"}),
        analysis_reasons=frozenset({"variable_frame_rate", "rotated"}),
    )
    monkeypatch.setattr(imgstore_io, "STORE_VERDICT", verdict)

    document = _probe_document(store)

    assert document["verdict"]["stream_reasons"] == ["interlaced", "rotated"]
    assert document["verdict"]["analysis_reasons"] == ["rotated", "variable_frame_rate"]


def test_probing_a_store_by_its_metadata_file_reads_the_store(tmp_path: Path) -> None:
    store = _store(tmp_path)

    by_metadata = _probe_document(store / "metadata.yaml")

    assert by_metadata == _probe_document(store)


def test_probing_an_image_store_reports_no_header_codec(tmp_path: Path) -> None:
    store = tmp_path / "rec.store"
    store.mkdir()
    _ = (store / "metadata.yaml").write_text(
        "__store: {class: DirectoryImgStore, format: npy, imgshape: [48, 64, 3]}\n"
    )

    document = _probe_document(store)

    assert document["facts"]["codec_name"] == "npy"
    assert document["store"]["chunk_count"] == 0
    assert document["store"]["chunksize"] == 0
    assert document["store"]["header_codec"] == ""


def test_a_store_that_does_not_open_exits_one_naming_it(tmp_path: Path) -> None:
    store = tmp_path / "broken.store"
    store.mkdir()
    _ = (store / "metadata.yaml").write_text(
        "__store: {class: NoSuchStore, imgshape: [48, 64, 3]}\n"
    )

    result = CliRunner().invoke(app, ["media", "probe", str(store)])

    assert result.exit_code == 1
    assert f"probe failed for {store}" in result.stderr
    assert "unsupported imgstore class" in result.stderr


def test_probing_a_video_prints_what_mosaic_media_prints(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    write_h264_mp4(video, frames=6)

    ours = invoke_ok(["media", "probe", str(video)]).stdout
    upstream = CliRunner().invoke(upstream_media_app, ["probe", str(video)])

    assert upstream.exit_code == 0, upstream.stderr
    assert ours == upstream.stdout


def test_compare_and_transcode_run_as_mosaic_media_runs_them(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    write_h264_mp4(video, frames=6)

    compared = CliRunner().invoke(app, ["media", "compare", str(video), str(video)])
    transcoded = invoke_ok(["media", "transcode", "--help"])

    assert compared.exit_code == 0, compared.stderr
    assert json.loads(compared.stdout)["verdict"] == "duplicate"
    assert "--target" in transcoded.stdout
