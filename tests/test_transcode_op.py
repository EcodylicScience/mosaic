"""End-to-end tests for the multi-video transcode job and derivative routing."""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from mosaic_media import (
    CHROME_149,
    DEFAULT_THRESHOLDS,
    MediaProbeError,
    derive,
    probe_media,
)
from mosaic_media.transcode import ANALYSIS_ENCODING, TranscodeError

from mosaic.core.dataset import Dataset
from mosaic.core.helpers import to_safe_name
from mosaic.core.media.facts_columns import MEDIA_INDEX_COLUMNS, row_to_facts
from mosaic.core.pipeline.media_index import (
    MediaIndexScope,
    frame_from_rows,
    read_media_index,
    write_media_index_rows,
)
from mosaic.core.pipeline.ops import list_ops, run_op
from mosaic.core.pipeline.transcode import (
    TranscodeParams,
    transcode_recipe_hash,
    transcode_run_id,
)
from mosaic.media_probe_config import media_thresholds

# Every transcode here leaks one file descriptor, and it is not this suite's doing.
# `mosaic_media.transcode.convert._run_ffmpeg` starts ffmpeg with
# `stdout=subprocess.PIPE, text=True`, hands the pipe to a draining thread, and
# never closes it -- `process.wait()` does not close stdout -- so the
# `TextIOWrapper` is finalized by the garbage collector and Python reports the
# leak. Harmless across a test run; a long-lived transcode worker walks toward its
# descriptor limit.
#
# Scoped to this module rather than allowed in the suite-wide `filterwarnings`, so
# that an unclosed file anywhere else still fails. Remove once mosaic-media closes
# the pipe.
pytestmark = pytest.mark.filterwarnings("ignore:unclosed file:ResourceWarning")


def _write_analysis_required_mp4(path: Path, duration: str = "2") -> None:
    """Write a variable-frame-rate mp4 that ``derive`` marks analysis-required.

    Height/width are >= 64 so the analysis (SVT-AV1) encoder accepts the source.
    *duration* varies the content, which is how a caller that needs two sources
    gets two distinct ``video_uuid``s rather than one shared by identical bytes.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not available")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration}:size=128x128:rate=30",
        "-vf",
        "setpts=N/(30+8*sin(N))/TB",
        "-fps_mode",
        "vfr",
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not path.exists():
        pytest.skip("this environment could not produce an mp4 fixture with ffmpeg")


def _require_analysis_required(path: Path) -> None:
    facts = probe_media(path)
    verdict = derive(facts, CHROME_149, DEFAULT_THRESHOLDS)
    if verdict.analysis_transcode != "required":
        pytest.skip(
            "this environment's ffmpeg did not produce an analysis-required fixture"
        )


def _indexed_entry(ds: Dataset, source_dir: Path) -> tuple[str, str]:
    ds.index_media([source_dir])
    df = pd.read_csv(ds.get_root("media_raw") / "index.csv")
    row = df.iloc[0]
    group = "" if pd.isna(row["group"]) else str(row["group"])
    return group, str(row["sequence"])


def _strip_identity_columns(ds: Dataset) -> None:
    """Rewrite the media_raw index without the identity columns.

    The shape of an index written before per-file identity existed: the columns
    are absent from the header, so every row reads back unminted.
    """
    index_path = ds.get_root("media_raw") / "index.csv"
    df = pd.read_csv(index_path)
    df = df.drop(
        columns=[
            column
            for column in ("video_uuid", "content_digest", "media_facts")
            if column in df.columns
        ]
    )
    df.to_csv(index_path, index=False)


def _analysis_required_dataset(
    tmp_path: Path,
    make_media_dataset: Callable[[Path], Dataset],
    *,
    minted: bool = True,
) -> tuple[Dataset, str]:
    """A dataset with one indexed media_raw video whose verdict requires an
    analysis transcode, and that video's minted uuid."""
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    original = ds.get_root("media_raw") / "s" / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)
    ds.index_media([ds.get_root("media_raw") / "s"])
    index_path = ds.get_root("media_raw") / "index.csv"
    # index_media derives (group, sequence) from the absent track keymap, so the
    # single row lands under an empty group and the file stem; the tests address
    # it as ("g", "s"). Read every cell as a string so the empty group never
    # round-trips as a float NaN when the sequence key is rewritten.
    raw_df = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    raw_df["group"] = "g"
    raw_df["sequence"] = "s"
    raw_df["group_safe"] = to_safe_name("g")
    raw_df["sequence_safe"] = to_safe_name("s")
    video_uuid = str(raw_df.iloc[0]["video_uuid"])
    raw_df.to_csv(index_path, index=False)
    if not minted:
        _strip_identity_columns(ds)
        return ds, ""
    return ds, video_uuid


def _clear_forward_links(ds: Dataset) -> None:
    """Blank both forward-link columns on every ``media_raw`` row.

    The state an interrupted registration leaves behind: the back-link row and
    the derivative file both exist, but the forward link that would mark the
    registration complete was never written.
    """
    index_path = ds.get_root("media_raw") / "index.csv"
    rows: list[dict[str, object]] = [
        dict(record) for record in read_media_index(index_path)
    ]
    for row in rows:
        row["analysis_derivative_path"] = ""
        row["playback_derivative_path"] = ""
    write_media_index_rows(index_path, frame_from_rows(rows))


def test_transcode_op_writes_derivative_and_links(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    source_dir = tmp_path / "raw_src"
    source_dir.mkdir()
    original = source_dir / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)

    group, sequence = _indexed_entry(ds, source_dir)

    # Before transcoding, the required row has no derivative -> routing errors.
    with pytest.raises(MediaProbeError, match="requires an analysis transcode"):
        _ = ds.resolve_media(group, sequence)

    run_id = run_op(
        ds, "transcode", TranscodeParams(entry=(group, sequence), target="analysis")
    )
    assert run_id.startswith("transcode-")
    transcode_dir = ds.get_root("media") / "transcode"
    assert list(transcode_dir.glob("*.mp4"))  # a derivative was written

    # A derivative exists under the transcode kind directory and re-probes clean.
    derivative_files = list(transcode_dir.glob("*.mp4"))
    assert len(derivative_files) == 1
    derivative = derivative_files[0]
    derivative_verdict = derive(probe_media(derivative), CHROME_149, DEFAULT_THRESHOLDS)
    assert derivative_verdict.analysis_transcode is None

    # Forward link: the original's media_raw row now names the analysis
    # derivative, and only that per-target column is set.
    raw_df = pd.read_csv(ds.get_root("media_raw") / "index.csv")
    assert str(raw_df.iloc[0]["analysis_derivative_path"]).strip() not in ("", "nan")
    assert str(raw_df.iloc[0]["playback_derivative_path"]).strip() in ("", "nan")

    # Back link: the media index row carries source_path and reconstructable facts.
    media_df = pd.read_csv(ds.get_root("media") / "index.csv")
    assert len(media_df) == 1
    deriv_row = {str(k): v for k, v in media_df.iloc[0].items()}
    assert str(deriv_row["source_path"]).strip() not in ("", "nan")
    reconstructed = row_to_facts(deriv_row)
    assert reconstructed.frame_count == probe_media(derivative).frame_count

    # Routing end-to-end: resolve_media now returns the derivative + its facts.
    resolved = ds.resolve_media(group, sequence)
    assert resolved.paths[0].resolve() == derivative.resolve()
    routed_verdict = derive(resolved.facts[0], CHROME_149, DEFAULT_THRESHOLDS)
    assert routed_verdict.analysis_transcode is None
    assert resolved.facts[0].frame_count == probe_media(derivative).frame_count


def test_analysis_facts_not_crossed_when_playback_transcoded_first(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    source_dir = tmp_path / "raw_src"
    source_dir.mkdir()
    original = source_dir / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)

    group, sequence = _indexed_entry(ds, source_dir)

    # Transcode PLAYBACK first, then ANALYSIS. The playback derivative row (which
    # shares source_path with the analysis row) then precedes the analysis row in
    # the media index, so a source_path-first lookup would return playback facts
    # for the analysis route.
    _ = run_op(
        ds, "transcode", TranscodeParams(entry=(group, sequence), target="playback")
    )
    _ = run_op(
        ds, "transcode", TranscodeParams(entry=(group, sequence), target="analysis")
    )

    transcode_dir = ds.get_root("media") / "transcode"
    analysis_files = list(transcode_dir.glob("*.analysis.mp4"))
    playback_files = list(transcode_dir.glob("*.playback.mp4"))
    assert len(analysis_files) == 1
    assert len(playback_files) == 1
    analysis_derivative = analysis_files[0]

    # Reconstruct each per-target derivative's stored facts from its media-index row.
    media_df = pd.read_csv(ds.get_root("media") / "index.csv")

    def _stored_facts(suffix: str):
        rows = [
            {str(k): v for k, v in row.items()}
            for _, row in media_df.iterrows()
            if str(row["abs_path"]).endswith(suffix)
        ]
        assert len(rows) == 1
        return row_to_facts(rows[0])

    analysis_stored = _stored_facts(".analysis.mp4")
    playback_stored = _stored_facts(".playback.mp4")
    if analysis_stored == playback_stored:
        pytest.skip(
            "this environment produced indistinguishable analysis/playback derivatives"
        )

    # Routing opens the analysis derivative and must carry ITS facts, not the
    # playback derivative's, even though playback was transcoded (and indexed) first.
    resolved = ds.resolve_media(group, sequence)
    assert resolved.paths[0].resolve() == analysis_derivative.resolve()
    assert resolved.facts[0] == analysis_stored
    assert resolved.facts[0] != playback_stored
    assert resolved.facts[0].frame_count == probe_media(analysis_derivative).frame_count


def test_transcode_op_is_idempotent(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    source_dir = tmp_path / "raw_src"
    source_dir.mkdir()
    original = source_dir / "vfr.mp4"
    _write_analysis_required_mp4(original)
    _require_analysis_required(original)

    group, sequence = _indexed_entry(ds, source_dir)

    first = run_op(
        ds, "transcode", TranscodeParams(entry=(group, sequence), target="analysis")
    )
    second = run_op(
        ds, "transcode", TranscodeParams(entry=(group, sequence), target="analysis")
    )
    assert first == second  # deterministic run_id
    # The real invariant: one derivative, one media row, one raw row -- no duplication.
    assert len(list((ds.get_root("media") / "transcode").glob("*.mp4"))) == 1
    media_df = pd.read_csv(ds.get_root("media") / "index.csv")
    assert len(media_df) == 1
    raw_df = pd.read_csv(ds.get_root("media_raw") / "index.csv")
    assert len(raw_df) == 1


def test_an_existing_linked_derivative_is_reused(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds, video_uuid = _analysis_required_dataset(tmp_path, make_media_dataset)
    params = TranscodeParams(entry=("g", "s"), target="analysis")
    _ = run_op(ds, "transcode", params)
    recipe = transcode_recipe_hash(
        params, ANALYSIS_ENCODING, CHROME_149, media_thresholds()
    )
    dest = ds.get_root("media") / "transcode" / f"{video_uuid}.{recipe}.analysis.mp4"
    first = dest.stat().st_mtime_ns
    _ = run_op(ds, "transcode", params)
    assert dest.stat().st_mtime_ns == first


def test_an_existing_but_unlinked_derivative_is_relinked(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds, _ = _analysis_required_dataset(tmp_path, make_media_dataset)
    params = TranscodeParams(entry=("g", "s"), target="analysis")
    _ = run_op(ds, "transcode", params)
    _clear_forward_links(ds)
    _ = run_op(ds, "transcode", params)
    row = read_media_index(ds.get_root("media_raw") / "index.csv")[0]
    assert row["analysis_derivative_path"]


def test_the_derivative_is_named_after_its_source_and_recipe(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds, video_uuid = _analysis_required_dataset(tmp_path, make_media_dataset)
    params = TranscodeParams(entry=("g", "s"), target="analysis")
    _ = run_op(ds, "transcode", params)
    recipe = transcode_recipe_hash(
        params, ANALYSIS_ENCODING, CHROME_149, media_thresholds()
    )
    expected = (
        ds.get_root("media") / "transcode" / f"{video_uuid}.{recipe}.analysis.mp4"
    )
    assert expected.exists()
    assert not list(ds.get_root("media").glob("*.mp4"))


def test_the_derivative_row_records_the_recipe(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds, _ = _analysis_required_dataset(tmp_path, make_media_dataset)
    params = TranscodeParams(entry=("g", "s"), target="analysis")
    _ = run_op(ds, "transcode", params)
    rows = read_media_index(ds.get_root("media") / "index.csv")
    recipe = transcode_recipe_hash(
        params, ANALYSIS_ENCODING, CHROME_149, media_thresholds()
    )
    assert rows[0]["recipe_hash"] == recipe


def test_a_source_with_no_uuid_refuses_to_transcode(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    ds, _ = _analysis_required_dataset(tmp_path, make_media_dataset, minted=False)
    with pytest.raises(TranscodeError, match="no video_uuid"):
        _ = run_op(
            ds, "transcode", TranscodeParams(entry=("g", "s"), target="analysis")
        )


def test_an_imgstore_refuses_to_transcode(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The refusal is explicit now that a store has a uuid to pass the other one.

    Until open item O5 a store carried no ``video_uuid``, so the empty-uuid check
    was the only thing keeping it out of this path -- an accident, and one that
    stopped being true the moment a store started naming itself. Left implicit,
    ffmpeg would be handed a directory.
    """
    ds, _ = _analysis_required_dataset(tmp_path, make_media_dataset, minted=True)
    index_path = ds.get_root(ds.resolve_media_root()) / "index.csv"
    rows = list(read_media_index(index_path))
    for row in rows:
        row["media_type"] = "imgstore"
    write_media_index_rows(index_path, frame_from_rows(list(rows)))

    with pytest.raises(TranscodeError, match="imgstore"):
        _ = run_op(
            ds, "transcode", TranscodeParams(entry=("g", "s"), target="analysis")
        )


def test_a_reorder_produces_zero_re_encodes(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The M3 acceptance gate, and the whole promise of content-addressed names.

    Under the positional scheme a derivative was named by its rank within the
    sequence, so swapping two videos renamed both and re-encoded both -- in
    place, non-transactionally, which is what made a cancel mid-loop leave a row
    pointing at another video's frames. Named by source uuid, a reorder is
    metadata: the two files keep their names, the two links keep naming them, and
    the op's reuse gate skips every source.

    The two sources carry different content on purpose. Byte-identical videos
    share one ``video_uuid`` by design, so a derivative of either has one name
    and the assertion would hold without proving anything.
    """
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    directory = ds.get_root("media_raw") / "s"
    for name, duration in (("a.mp4", "2"), ("b.mp4", "3")):
        _write_analysis_required_mp4(directory / name, duration=duration)
        _require_analysis_required(directory / name)

    def arrange(first: str, second: str) -> None:
        _ = ds.write_media_index(
            [
                MediaIndexScope(
                    directory=directory,
                    group="g",
                    sequence="s",
                    order_by_name={first: 0, second: 1},
                )
            ],
            extensions=(".mp4",),
        )

    arrange("a.mp4", "b.mp4")
    params = TranscodeParams(entry=("g", "s"), target="analysis")
    _ = run_op(ds, "transcode", params)

    transcode_root = ds.get_root("media") / "transcode"
    before_files = {p.name: p.stat().st_mtime_ns for p in transcode_root.iterdir()}
    assert len(before_files) == 2, f"expected two derivatives, got {before_files}"
    before_links = {
        row["name"]: row["analysis_derivative_path"]
        for row in read_media_index(ds.get_root("media_raw") / "index.csv")
    }
    before_order = {
        row["name"]: row["video_order"]
        for row in read_media_index(ds.get_root("media_raw") / "index.csv")
    }

    arrange("b.mp4", "a.mp4")
    _ = run_op(ds, "transcode", params)

    after_order = {
        row["name"]: row["video_order"]
        for row in read_media_index(ds.get_root("media_raw") / "index.csv")
    }
    assert after_order != before_order, "the arrangement did not actually change"
    after_files = {p.name: p.stat().st_mtime_ns for p in transcode_root.iterdir()}
    assert after_files == before_files, "a reorder re-encoded a derivative"
    after_links = {
        row["name"]: row["analysis_derivative_path"]
        for row in read_media_index(ds.get_root("media_raw") / "index.csv")
    }
    assert after_links == before_links, "a reorder moved a derivative link"


def test_reverting_the_break_leaves_every_other_cell_and_link_intact(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """The reverse migration for item 7.1's ``[break]``, which is: revert the code.

    7.1 renamed and relocated every derivative, and the rollback rule says a
    ``[break]`` ships its reverse migration with a test. The migration the
    milestone sketched -- carrying the old positional name in a column -- is not
    what shipped, and should not: it would be dead weight on the one index the
    pruner has to classify. What shipped is additive, and this is the check that
    says so.

    Reverted code reads this index through a schema with no ``recipe_hash``, so
    a round-trip through it drops that cell and nothing else. Every other cell
    survives byte-identical, and the forward link still resolves -- a link cell
    is a media-root-relative path string, which reverted code anchors the same
    way, and no file is moved by a revert.

    What a revert does *not* undo is the files: they keep their content-addressed
    names, which reverted code will not produce. `scripts/clear_transcode_
    derivatives.py` plus a re-run is the migration in that direction, and a
    revert also reintroduces 7.2's path-keyed link matching -- a known defect,
    which is the one way this differs from the additive break in M2.
    """
    ds = make_media_dataset((tmp_path / "dataset").resolve())
    transcode_root = ds.get_root("media") / "transcode"
    transcode_root.mkdir(parents=True, exist_ok=True)
    derivative = transcode_root / "some-uuid.recipe01.analysis.mp4"
    _ = derivative.write_bytes(b"stub")
    media_index = ds.get_root("media") / "index.csv"
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "name": derivative.name,
            "group": "g",
            "sequence": "s",
            "abs_path": ds.relative_to_root(str(derivative)),
            "source_path": "s/vfr.mp4",
            "source_video_uuid": "some-uuid",
            "recipe_hash": "recipe01",
            "video_order": 2,
            # A comma-bearing JSON cell is what a careless rewrite mangles.
            "media_facts": '{"video_uuid": "some-uuid", "frame_count": 6}',
        }
    )
    write_media_index_rows(media_index, frame_from_rows([row]))
    before = {k: v for k, v in read_media_index(media_index)[0].items()}

    # What reverted code sees: the same schema minus the column 7.1 added.
    reverted_columns = [c for c in MEDIA_INDEX_COLUMNS if c != "recipe_hash"]
    frame = frame_from_rows([dict(before)])[reverted_columns]
    frame.to_csv(media_index, index=False)

    after = {k: v for k, v in read_media_index(media_index)[0].items()}
    assert "recipe_hash" not in after, "the round-trip did not model a revert"
    assert after == {k: v for k, v in before.items() if k != "recipe_hash"}
    assert ds.resolve_path(after["abs_path"]).exists(), (
        "the derivative stopped resolving"
    )


def test_a_dataset_with_no_media_raw_refuses_to_transcode(tmp_path: Path) -> None:
    """With one media root the derivative index *is* the originals index.

    The op used to run here and quietly do harm: the back-link appended a
    derivative row into the originals index, the forward link went to the same
    file, and ``route_derivatives`` was then False -- so nothing ever read what
    the encode produced. The refusal is checked before any row is matched, so a
    dataset in this shape fails immediately rather than after an encode.
    """
    base = (tmp_path / "dataset").resolve()
    ds = Dataset(
        manifest_path=base / "dataset.yaml", roots={"media": str(base / "media")}
    )
    ds.ensure_roots()
    ds.save()
    index_path = ds.get_root("media") / "index.csv"
    original = base / "media" / "entry.mp4"
    original.touch()
    row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
    row.update(
        {
            "name": "entry.mp4",
            "group": "g",
            "sequence": "s",
            "abs_path": ds.relative_to_root(str(original)),
            "video_uuid": "U",
        }
    )
    write_media_index_rows(index_path, frame_from_rows([row]))
    before = index_path.read_bytes()

    with pytest.raises(TranscodeError, match="no media_raw root"):
        _ = run_op(
            ds, "transcode", TranscodeParams(entry=("g", "s"), target="analysis")
        )

    assert index_path.read_bytes() == before, "the originals index was written to"
    assert not (ds.get_root("media") / "transcode").exists()


def test_the_run_identity_ignores_the_source_order() -> None:
    assert transcode_run_id("abc123", ["b", "a"]) == transcode_run_id(
        "abc123", ["a", "b"]
    )


def test_the_recipe_hash_ignores_the_entry() -> None:
    thresholds = media_thresholds()
    here = TranscodeParams(entry=("g", "s"), target="analysis")
    elsewhere = TranscodeParams(entry=("other", "sequence"), target="analysis")
    assert transcode_recipe_hash(here, ANALYSIS_ENCODING, CHROME_149, thresholds) == (
        transcode_recipe_hash(elsewhere, ANALYSIS_ENCODING, CHROME_149, thresholds)
    )


def test_the_recipe_hash_ignores_the_hardware_permission() -> None:
    thresholds = media_thresholds()
    plain = TranscodeParams(entry=("g", "s"), target="analysis")
    hardware = TranscodeParams(entry=("g", "s"), target="analysis", allow_hardware=True)
    assert transcode_recipe_hash(plain, ANALYSIS_ENCODING, CHROME_149, thresholds) == (
        transcode_recipe_hash(hardware, ANALYSIS_ENCODING, CHROME_149, thresholds)
    )


def test_the_recipe_hash_separates_the_targets_under_one_encoding() -> None:
    thresholds = media_thresholds()
    analysis = TranscodeParams(entry=("g", "s"), target="analysis")
    playback = TranscodeParams(entry=("g", "s"), target="playback")
    assert transcode_recipe_hash(
        analysis, ANALYSIS_ENCODING, CHROME_149, thresholds
    ) != transcode_recipe_hash(playback, ANALYSIS_ENCODING, CHROME_149, thresholds)


def test_the_recipe_hash_moves_with_the_encoding_parameters() -> None:
    params = TranscodeParams(entry=("g", "s"), target="analysis")
    thresholds = media_thresholds()
    tuned = dataclasses.replace(
        ANALYSIS_ENCODING, quality=ANALYSIS_ENCODING.quality + 1
    )
    assert transcode_recipe_hash(
        params, ANALYSIS_ENCODING, CHROME_149, thresholds
    ) != transcode_recipe_hash(params, tuned, CHROME_149, thresholds)


def test_the_recipe_hash_is_stable_across_processes() -> None:
    # The profile carries frozensets, and json_ready serializes a set without
    # sorting. Two subprocesses under different hash seeds must agree, or the
    # derivative filename changes run to run.
    script = (
        "from mosaic.core.pipeline.transcode import ("
        "TranscodeParams, transcode_recipe_hash)\n"
        "from mosaic.media_probe_config import media_thresholds\n"
        "from mosaic_media import CHROME_149\n"
        "from mosaic_media.transcode import ANALYSIS_ENCODING\n"
        "print(transcode_recipe_hash("
        "TranscodeParams(entry=('g','s'), target='analysis'), "
        "ANALYSIS_ENCODING, CHROME_149, media_thresholds()))\n"
    )
    digests = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "random")
    }
    assert len(digests) == 1


def test_transcode_registered_as_media_op():
    from mosaic.core.pipeline.ops import OPS, describe_op

    assert "transcode" in OPS
    info = describe_op("transcode")
    assert info["domain"] == "media"
    assert info["category"] == "transcode"
    assert "params_schema" in info
    schema = TranscodeParams.model_json_schema()
    assert {"entry", "target", "allow_hardware"} <= set(schema["properties"])


def test_transcode_excluded_from_tracking_listing():
    kinds = {entry["kind"] for entry in list_ops(domain="tracking")}
    assert "transcode" not in kinds
    assert "transcode" in {entry["kind"] for entry in list_ops()}


def test_transcode_resource_class_is_cpu():
    from mosaic.core.pipeline.ops import op_resource_class

    assert op_resource_class("transcode") == "cpu"


def test_transcode_params_reject_unknown_key():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TranscodeParams.model_validate({"entry": ("", "vid1"), "bogus": 1})


# Two processes, each linking a different source. Both read the index before
# either writes, so an unserialized writer computes its whole output frame from a
# starting state the other has already moved past.
_LINK_PROBE = """
import sys
from pathlib import Path

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.transcode import set_forward_link

manifest, uuid, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
ds = Dataset(manifest_path=manifest).load()

barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
set_forward_link(
    ds, Path(f"{uuid}.mp4"), uuid, f"transcode/{uuid}.r.analysis.mp4", "analysis"
)
"""


def test_two_concurrent_forward_links_both_survive(
    tmp_path: Path, make_media_dataset: Callable[[Path], Dataset]
) -> None:
    """``set_forward_link`` is a whole-file rewrite, so it needs the lock.

    Two sources of one entry transcoding in parallel is the ordinary case: the op
    links each source after its own iteration, and a queue running two entries at
    once does the same across entries. Each writer reads the whole index, sets one
    cell, and writes the whole index back -- so without serialization the second
    writes a frame that never contained the first's cell, and the link is lost
    with no error on either side. The derivative file survives, unreferenced,
    which is precisely the state the pruner would then read as garbage.
    """
    tmp_path = tmp_path.resolve()
    ds = make_media_dataset(tmp_path)
    index_path = ds.get_root("media_raw") / "index.csv"
    rows: list[dict[str, object]] = []
    for uuid in ("uuid-a", "uuid-b"):
        row: dict[str, object] = {column: "" for column in MEDIA_INDEX_COLUMNS}
        row.update(
            {
                "name": uuid,
                "group": "g",
                "sequence": "s",
                "abs_path": f"media_raw/{uuid}.mp4",
                "video_uuid": uuid,
            }
        )
        rows.append(row)
    write_media_index_rows(index_path, frame_from_rows(rows))

    gate = tmp_path / "gate"
    gate.mkdir()
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _LINK_PROBE,
                str(ds.manifest_path),
                uuid,
                str(gate / f"{uuid}.ready"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for uuid in ("uuid-a", "uuid-b")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err.decode()[-800:]

    linked = {
        str(row["video_uuid"]): str(row["analysis_derivative_path"])
        for row in read_media_index(index_path)
    }
    assert linked == {
        "uuid-a": "transcode/uuid-a.r.analysis.mp4",
        "uuid-b": "transcode/uuid-b.r.analysis.mp4",
    }, f"a concurrent forward link was lost: {linked}"
