"""Tracker reuse: what counts as done, who holds a directory, and on what.

Three defects, all of which report success while producing a wrong result:

* **8.2** -- reuse was gated on ``seq_dir.exists()``, and the directory was
  created before the first subprocess started. A timeout or cancellation left
  one behind that made every later identical run a silent no-op with zero
  individuals.
* **8.3** -- nothing distinguished an active multi-hour job from an abandoned
  directory, and two attempts resolving to one directory did not serialize.
* **8.8** -- the run identity hashes settings only, so changing which video a
  sequence resolves to left the identifier, run root and working directory
  unchanged: both phases were skipped, the stale parquet stayed, and the index
  row was appended with the freshly resolved video beside the old run's counts.

None of this path had any test coverage, so these are the first regression net
over it as well as the proof of the fixes.

TREx itself never runs: ``run_trex_convert`` / ``run_trex_track`` are replaced
with recording fakes, the established shape in ``test_tracking_ops.py``.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive

import mosaic.tracking.trex.dataset_runs as dr
from mosaic.tracking.common.bridge import BridgeCounts
from mosaic.tracking.common.scope import JoinedSourceMismatchError
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts
from mosaic.core.pipeline.markers import (
    InflightMarker,
    inflight_marker_path,
    phase_fields,
    read_inflight,
    read_phase_marker,
    write_inflight,
)
from mosaic.core.pipeline.types import Declared
from mosaic.tracking.common.params import PhasedTrackerOpParams
from mosaic.tracking.trex.conversion_cache import CONVERT_KIND
from mosaic.tracking.trex.dataset_runs import trex_index_path
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.trex.run import TRexConvertResult, TRexTrackResult

# --- fixtures --------------------------------------------------------------


def clean_facts_cells(
    width: int = 640, height: int = 480, video_uuid: str = ""
) -> dict[str, object]:
    """Flat + JSON facts cells for one analysis-clean media row.

    ``video_uuid`` defaults empty, which is the state of every media index
    written before the identity columns existed -- so the tests that do not pass
    one exercise the reuse guard's *path* fallback, deliberately.
    """
    facts: MediaFacts = store_facts(
        width=width,
        height=height,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid=video_uuid,
        identity_scheme="video/1" if video_uuid else "",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


@dataclass
class MediaEntry:
    """One row of the synthetic media index."""

    sequence: str
    filename: str
    camera: str = ""
    video_uuid: str = ""


def write_media_index(ds: Dataset, entries: list[MediaEntry]) -> None:
    """Rewrite the media index from *entries*, storing root-relative paths."""
    media_root = ds.get_root(ds.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for entry in entries:
        video = media_root / entry.filename
        if not video.exists():
            _ = video.write_bytes(b"fake")
        rows.append(
            {
                "name": entry.filename,
                "group": "",
                "sequence": entry.sequence,
                "group_safe": "",
                "sequence_safe": entry.sequence,
                "camera": entry.camera,
                "abs_path": ds.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
                **clean_facts_cells(video_uuid=entry.video_uuid),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    """A dataset with one sequence, ``vid1``, backed by ``vid1.mp4``."""
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(dataset, [MediaEntry(sequence="vid1", filename="vid1.mp4")])
    return dataset


@dataclass
class FakeTrex:
    """Recording stand-ins for the two TREx phases."""

    converted: list[Path] = field(default_factory=list)
    tracked: list[Path] = field(default_factory=list)
    npz_per_track: int = 1
    pv_beside_the_video: bool = False
    on_convert: Callable[[Path], None] | None = None
    sources: list[list[Path]] = field(default_factory=list)
    """Every conversion's *whole* source list, so a joined run is inspectable.

    ``converted`` keeps recording one path per call -- clip 0 -- because that is
    what every single-video assertion in this file reads.
    """

    def convert(
        self,
        video_path: Path | Sequence[Path],
        seq_dir: Path,
        *,
        output_name: str | None = None,
        **_kwargs: object,
    ) -> TRexConvertResult:
        # Mirrors run_trex_convert's own normalisation: one source or many.
        given = (
            [Path(video_path)]
            if isinstance(video_path, (str, Path))
            else [Path(p) for p in video_path]
        )
        self.sources.append(given)
        self.converted.append(given[0])
        if self.on_convert is not None:
            self.on_convert(Path(seq_dir))
        stem = output_name if output_name is not None else given[0].stem
        # `pv_beside_the_video` models TREx choosing its own location, which it
        # only does when nothing pinned the name. Given `filename`, it writes
        # where it was told -- the same order `run_trex_convert` looks in.
        home = (
            given[0].parent
            if self.pv_beside_the_video and output_name is None
            else Path(seq_dir)
        )
        home.mkdir(parents=True, exist_ok=True)
        pv_path = home / f"{stem}.pv"
        _ = pv_path.write_bytes(b"pv")
        # TREx writes a settings file beside every conversion, and it is not
        # decorative: re-opening a `.pv` recovers only seven fields from the file
        # itself, so this is the only thing carrying the detection parameters
        # into a later tracking run. Written here for the same reason the npz
        # below is real rather than a stub -- a fake that omits what the tool
        # always produces exercises a path that cannot happen.
        settings_path = home / f"{stem}.settings"
        _ = settings_path.write_text("detect_type = yolo\n")
        return TRexConvertResult(
            pv_path=pv_path,
            settings_path=settings_path,
            background_path=None,
            stdout="",
            stderr="",
        )

    def track(self, pv_path: Path, seq_dir: Path, **_kwargs: object) -> TRexTrackResult:
        self.tracked.append(Path(pv_path))
        data_dir = Path(seq_dir) / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.npz_per_track):
            # A real, convertible export rather than a stub. These tests are
            # about markers and reuse, not about conversion -- but a stub made
            # every bridge in this suite fail, which used to be swallowed and is
            # now recorded as a lost entry. Writing what TREx writes keeps the
            # suite exercising the real publish path instead of a broken one.
            np.savez(
                data_dir / f"fish{i}.npz",
                frame=np.arange(4),
                time=np.arange(4) / 30.0,
                cm_per_pixel=np.array([1.0]),
                **{
                    "X#wcentroid": np.arange(4, dtype=float),
                    "Y#wcentroid": np.arange(4, dtype=float),
                },
            )
        _ = (Path(seq_dir) / f"{Path(pv_path).stem}.results").write_bytes(b"results")
        return TRexTrackResult()


@pytest.fixture
def trex(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeTrex]:
    fake = FakeTrex()
    monkeypatch.setattr(dr, "run_trex_convert", fake.convert)
    monkeypatch.setattr(dr, "run_trex_track", fake.track)
    yield fake


def seq_dir_of(ds: Dataset, run_id: str, key: str = "vid1") -> Path:
    return dr.trex_run_root(ds, run_id) / key


def index_rows(ds: Dataset) -> pd.DataFrame:
    return pd.read_csv(trex_index_path(ds))


# --- 8.2: completion, not attempt ------------------------------------------


def test_a_complete_run_is_reused(ds: Dataset, trex: FakeTrex) -> None:
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 1, "a complete run must not recompute"
    assert len(trex.tracked) == 1
    assert read_phase_marker(seq_dir_of(ds, run_id), "convert") is not None
    assert read_phase_marker(seq_dir_of(ds, run_id), "track") is not None


def test_a_cancelled_run_recomputes(ds: Dataset, trex: FakeTrex) -> None:
    """The defect: an empty directory from a killed run read as complete forever."""
    run_id = dr.run_trex(
        ds, TrexParams(entries=[("", "vid1")], convert_to_tracks=False)
    )
    partial = seq_dir_of(ds, run_id)
    shutil.rmtree(partial)
    partial.mkdir(parents=True)

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 2, "a directory with no outputs is not a result"


def test_an_interrupted_track_reruns_only_the_track_phase(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Retaining the .pv is the entire point of keeping the intermediate."""
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    work_dir = seq_dir_of(ds, run_id)
    # A track phase killed after convert: its marker and outputs are gone.
    (work_dir / ".mosaic-track.json").unlink()
    shutil.rmtree(work_dir / "data")

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")], overwrite=False))

    assert len(trex.converted) == 1, "convert completed; it must not run again"
    assert len(trex.tracked) == 2


def test_the_track_phase_receives_the_recorded_pv(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TREx may leave the .pv beside the source video, where no seq_dir glob finds it."""
    fake = FakeTrex(pv_beside_the_video=True)
    monkeypatch.setattr(dr, "run_trex_convert", fake.convert)
    monkeypatch.setattr(dr, "run_trex_track", fake.track)

    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    work_dir = seq_dir_of(ds, run_id)
    (work_dir / ".mosaic-track.json").unlink()

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(fake.converted) == 1, "convert completed even though seq_dir has no .pv"
    assert [p.name for p in fake.tracked] == ["vid1.pv", "vid1.pv"]


def test_a_track_finding_no_individuals_still_counts_as_complete(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completion is "the phase returned", not "the outputs are non-empty"."""
    fake = FakeTrex(npz_per_track=0)
    monkeypatch.setattr(dr, "run_trex_convert", fake.convert)
    monkeypatch.setattr(dr, "run_trex_track", fake.track)

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(fake.tracked) == 1, "an empty result is a result, not a missing one"


def test_overwrite_clears_the_markers(ds: Dataset, trex: FakeTrex) -> None:
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")], overwrite=True))

    assert len(trex.converted) == 2
    assert len(trex.tracked) == 2


# --- 8.2: adopting directories that predate the markers --------------------


def test_a_legacy_complete_directory_is_adopted(ds: Dataset, trex: FakeTrex) -> None:
    """Without adoption, every already-tracked sequence re-tracks -- hours each."""
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    work_dir = seq_dir_of(ds, run_id)
    for marker in work_dir.glob(".mosaic-*.json"):
        marker.unlink()

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 1, "a directory holding a finished run is finished"
    adopted = read_phase_marker(work_dir, "track")
    assert adopted is not None
    assert adopted.backfilled is True
    assert adopted.source == "", "an adopted marker cannot know what it consumed"


def test_a_legacy_directory_without_results_recomputes(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The .pv and data files are written as processing proceeds; .results is not.

    So neither proves completion on its own, and a directory holding only those
    is indistinguishable from one killed partway.
    """
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    work_dir = seq_dir_of(ds, run_id)
    for marker in work_dir.glob(".mosaic-*.json"):
        marker.unlink()
    for results in work_dir.glob("*.results"):
        results.unlink()

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 2


# --- 8.3: in-flight claims -------------------------------------------------


def test_the_claim_is_held_during_the_phase_and_released_after(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[InflightMarker | None] = []
    fake = FakeTrex(on_convert=lambda work_dir: seen.append(read_inflight(work_dir)))
    monkeypatch.setattr(dr, "run_trex_convert", fake.convert)
    monkeypatch.setattr(dr, "run_trex_track", fake.track)

    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(seen) == 1
    held = seen[0]
    assert held is not None and held.execution_id
    assert read_inflight(seq_dir_of(ds, run_id)) is None, "the claim must be released"


def test_the_activity_callback_re_stamps_the_running_claim(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live phase refreshes its own claim as TREx prints progress.

    The claim's expiry is an inactivity bound, so a long healthy run would let
    it lapse -- and a concurrent execution would read the directory as
    abandoned -- unless output activity re-stamps it. The callback the phase
    receives must write the claim back to *this* working directory.
    """
    fake = FakeTrex()

    def convert_and_refresh(
        video_path: Path,
        seq_dir: Path,
        *,
        on_output: Callable[[str], None] | None = None,
        **kwargs: object,
    ) -> TRexConvertResult:
        assert on_output is not None, "the phase must receive the activity callback"
        # Drop the claim, then fire one progress line; the callback must restore
        # it -- proof the closure captured this seq_dir and its claim.
        inflight_marker_path(Path(seq_dir)).unlink(missing_ok=True)
        on_output("[Statistics] Progress: 10%")
        assert read_inflight(Path(seq_dir)) is not None, (
            "output activity re-stamped the claim for this directory"
        )
        return fake.convert(video_path, seq_dir, **kwargs)

    monkeypatch.setattr(dr, "run_trex_convert", convert_and_refresh)
    monkeypatch.setattr(dr, "run_trex_track", fake.track)

    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(fake.converted) == 1
    assert read_inflight(seq_dir_of(ds, run_id)) is None, (
        "the mid-phase refresh must not defeat the final release"
    )


def test_the_claim_is_released_when_a_phase_raises(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim outliving its execution would block the retry that fixes it."""

    def explode(video_path: Path, seq_dir: Path, **_: object) -> TRexConvertResult:
        raise RuntimeError("convert died")

    monkeypatch.setattr(dr, "run_trex_convert", explode)

    with pytest.raises(RuntimeError, match="convert died"):
        dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert list(ds.get_root("trex").rglob(".mosaic-inflight.json")) == []


def test_a_live_foreign_claim_skips_the_entry(ds: Dataset, trex: FakeTrex) -> None:
    """One contended sequence must not kill a batch, and must not be clobbered.

    A skipped entry also writes **no** index row: a row would claim this
    execution produced a result it never touched.
    """
    # An empty scope returns the identifier without doing any work.
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "no-such-sequence")]))
    work_dir = seq_dir_of(ds, run_id)
    work_dir.mkdir(parents=True)
    write_inflight(
        work_dir,
        InflightMarker(
            execution_id="SOMEONE-ELSE",
            expires_at="2099-01-01T00:00:00+00:00",
        ),
    )

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert trex.converted == [], "the entry belongs to another execution"
    assert read_inflight(work_dir) is not None, "someone else's claim must survive"
    assert len(index_rows(ds)) == 0, "no row for work this execution did not do"


def test_a_held_entry_does_not_stop_the_rest_of_the_batch(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The reason the skip is a `continue` and not a `break`.

    A batch of fifty sequences must not end because one of them is being
    worked by a concurrent job. Every other test here runs a single-entry
    batch, which cannot tell the two apart.
    """
    write_media_index(
        ds,
        [
            MediaEntry(sequence="vid1", filename="vid1.mp4"),
            MediaEntry(sequence="vid2", filename="vid2.mp4"),
        ],
    )
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "no-such-sequence")]))
    held = seq_dir_of(ds, run_id, key="vid1")
    held.mkdir(parents=True)
    write_inflight(
        held,
        InflightMarker(
            execution_id="SOMEONE-ELSE",
            expires_at="2099-01-01T00:00:00+00:00",
        ),
    )

    dr.run_trex(ds, TrexParams(entries=[("", "vid1"), ("", "vid2")]))

    assert [p.name for p in trex.converted] == ["vid2.mp4"], (
        "the free entry must still run"
    )
    rows = index_rows(ds)
    assert list(rows["sequence"]) == ["vid2"], "one row, for the entry actually done"


def test_overwrite_does_not_destroy_a_live_claim(ds: Dataset, trex: FakeTrex) -> None:
    """The claim check must precede every destructive step, rmtree included."""
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "no-such-sequence")]))
    work_dir = seq_dir_of(ds, run_id)
    work_dir.mkdir(parents=True)
    _ = (work_dir / "in-progress.pv").write_bytes(b"someone else's work")
    write_inflight(
        work_dir,
        InflightMarker(
            execution_id="SOMEONE-ELSE",
            expires_at="2099-01-01T00:00:00+00:00",
        ),
    )

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")], overwrite=True))

    assert trex.converted == []
    assert (work_dir / "in-progress.pv").exists(), "overwrite clobbered a live run"
    assert read_inflight(work_dir) is not None


def test_an_expired_foreign_claim_is_reclaimed(ds: Dataset, trex: FakeTrex) -> None:
    """A claim whose execution died must not hold the directory forever."""
    run_id = dr.run_trex(
        ds, TrexParams(entries=[("", "vid1")], convert_to_tracks=False)
    )
    work_dir = seq_dir_of(ds, run_id)
    # A run killed partway: markers gone, outputs incomplete, claim left behind.
    shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    write_inflight(
        work_dir,
        InflightMarker(
            execution_id="LONG-GONE",
            expires_at="2000-01-01T00:00:00+00:00",
        ),
    )

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 2, "the entry was reclaimed and recomputed"
    assert not inflight_marker_path(work_dir).exists()


# --- 8.8: the source video is part of the reuse key ------------------------


def test_a_changed_source_video_forces_a_recompute(ds: Dataset, trex: FakeTrex) -> None:
    """The identity hashes settings only, so nothing else would notice."""
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    write_media_index(ds, [MediaEntry(sequence="vid1", filename="vid2.mp4")])

    second = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert second == run_id, "settings did not change, so neither does the identity"
    assert [p.name for p in trex.converted] == ["vid1.mp4", "vid2.mp4"]
    marker = read_phase_marker(seq_dir_of(ds, run_id), "convert")
    assert marker is not None and marker.source.endswith("vid2.mp4")


def test_a_forced_recompute_refreshes_the_tracks_parquet(
    ds: Dataset, trex: FakeTrex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the stale parquet stays and the recompute is invisible downstream."""
    written: list[Path] = []

    def record_bridge(
        dataset: Dataset,
        group: str,
        sequence: str,
        npz_paths: list[Path],
        *,
        tracks_variant: str,
        producer_run_id: str,
        video_paths: Sequence[Path],
        timeline: object,
        overwrite: bool,
    ) -> BridgeCounts | None:
        written.append(Path(f"{group}__{sequence}"))
        assert overwrite is True, "a recomputed entry must overwrite its parquet"
        # The bridge is handed the variant it belongs to and the tracker run
        # that produced it, minted once for the whole run rather than per entry.
        assert tracks_variant.startswith("trex.")
        assert producer_run_id.startswith("trex.")
        return BridgeCounts(n_rows=1, n_ids=1)

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    write_media_index(ds, [MediaEntry(sequence="vid1", filename="vid2.mp4")])

    monkeypatch.setattr(dr, "_bridge_npz_to_tracks", record_bridge)
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert written == [Path("__vid1")]


def test_a_moved_dataset_does_not_recompute(
    ds: Dataset, trex: FakeTrex, tmp_path: Path
) -> None:
    """The guard compares resolved paths, so relocation is not a source change."""
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    moved = tmp_path.parent / f"{tmp_path.name}-moved"
    shutil.copytree(tmp_path, moved)
    relocated = Dataset(manifest_path=moved / "dataset.yaml").load()

    dr.run_trex(relocated, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 1, "a moved dataset holds the same result"


def test_the_reuse_path_reports_the_video_that_produced_the_data(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The row is the only recorded edge from tracks back to media."""
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    rows = index_rows(ds)
    assert len(rows) == 1, "one row per (run_id, group, sequence)"
    row = rows.iloc[0]
    assert str(row["video_abs_path"]).endswith("vid1.mp4")
    assert int(row["n_ids"]) == 1
    assert str(row["run_id"]) == run_id


# --- one working directory, one work item ----------------------------------


def test_two_cameras_produce_one_work_item(ds: Dataset, trex: FakeTrex) -> None:
    """Both cameras of a sequence resolve to one seq_dir until item 8.1 splits them.

    Left alone, the second work item would see the first's source, call it a
    change, recompute over the first's outputs, replace its index row -- and do
    it again on every run.
    """
    write_media_index(
        ds,
        [
            MediaEntry(sequence="vid1", filename="vid1.mp4", camera="cam0"),
            MediaEntry(sequence="vid1", filename="vid2.mp4", camera="cam1"),
        ],
    )

    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert len(trex.converted) == 1
    assert len(index_rows(ds)) == 1


# --- the identity itself does not move -------------------------------------


def declare_unphased_model() -> type[PhasedTrackerOpParams]:
    """Declare a model whose own field names no phase, for the guard to refuse."""

    class Unphased(PhasedTrackerOpParams):
        forgotten: Annotated[int | None, Declared("a knob nobody phased")] = None

    return Unphased


def test_a_field_that_names_no_phase_is_refused_at_class_creation() -> None:
    """A settings key in neither phase would stop invalidating anything.

    ``phase_fields`` returns an empty tuple both for a phase no field names and
    for a model that declares no ``Phase`` at all, so a ``TrexParams`` that lost
    its markers would project an empty settings dictionary and TREx would run on
    its own defaults. The base class refuses such a model where it is written.
    """
    with pytest.raises(TypeError, match="without a Phase"):
        _ = declare_unphased_model()


def test_every_tool_facing_field_reaches_a_phase() -> None:
    """The other half: the phases between them consume all sixteen fields."""
    phased = set(phase_fields(TrexParams, "convert")) | set(
        phase_fields(TrexParams, "track")
    )
    inherited = set(PhasedTrackerOpParams.model_fields)

    assert phased == set(TrexParams.model_fields) - inherited
    assert "track_max_individuals" in phase_fields(TrexParams, "convert"), (
        "it is a convert input despite its name -- changing it must re-convert"
    )
    assert "track_max_individuals" in phase_fields(TrexParams, "track")


def test_a_partial_run_still_records_its_index_row(ds: Dataset, trex: FakeTrex) -> None:
    """A skipped entry writes no row; a completed one in the same batch still does."""
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    rows = index_rows(ds)

    assert list(rows["run_id"]) == [run_id]
    assert not str(rows.iloc[0]["video_abs_path"]).startswith("/"), (
        "stored root-relative so a move does not read as a source change"
    )


# --- 8.5: the reuse comparison is uid-first, with the path as fallback -------


def test_a_video_replaced_in_place_forces_a_recompute(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The case the path compare cannot see at all.

    Same sequence, same filename, different bytes. ``_same_video`` compares
    resolved paths and calls this unchanged, so before item 8.5 the second run
    reused a conversion of content that no longer existed. The uid is what
    notices.
    """
    write_media_index(
        ds, [MediaEntry(sequence="vid1", filename="vid1.mp4", video_uuid="uid-aaa")]
    )
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    write_media_index(
        ds, [MediaEntry(sequence="vid1", filename="vid1.mp4", video_uuid="uid-bbb")]
    )
    second = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert second == run_id, "settings did not change, so neither does the identity"
    assert len(trex.converted) == 2, "the replaced video was not re-converted"


def test_the_same_video_under_a_new_name_is_not_a_recompute(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The other direction, and the one item 8.5 exists for.

    A rearrangement can change which file a sequence resolves to without
    changing the bytes. The path compare calls that a source change and throws
    away hours of conversion; the uid says it is the same video.
    """
    write_media_index(
        ds, [MediaEntry(sequence="vid1", filename="vid1.mp4", video_uuid="uid-aaa")]
    )
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    write_media_index(
        ds, [MediaEntry(sequence="vid1", filename="renamed.mp4", video_uuid="uid-aaa")]
    )
    second = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert second == run_id
    assert len(trex.converted) == 1, "the same bytes were converted twice"


def test_an_absent_uid_still_falls_back_to_the_path(
    ds: Dataset, trex: FakeTrex
) -> None:
    """Item 8.8's protection survives for the datasets it was written for.

    Media indexed before the identity columns carries no uid, and neither does a
    marker adopted from a pre-marker directory. Dropping the path compare would
    remove the guard from exactly those.
    """
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    write_media_index(ds, [MediaEntry(sequence="vid1", filename="vid2.mp4")])

    second = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))

    assert second == run_id
    assert [p.name for p in trex.converted] == ["vid1.mp4", "vid2.mp4"]


# --- a session's clips are one conversion -----------------------------------

# The clip boundary a recorder leaves is a filesystem artifact, not an event, so
# TREx is handed the whole entry: `source` is a PathArray and VideoSource sums
# the clip lengths into one continuous frame index. What the reuse gate then has
# to notice is any change to *the arrangement*, which the first clip's uid
# cannot see.


def _session(ds: Dataset, *names: str, widths: dict[str, int] | None = None) -> None:
    """Put *names* in one sequence, in the order given, each with an identity.

    *widths* overrides a clip's frame width, for the one case that needs clips
    which cannot be read as one video.
    """
    media_root = ds.get_root(ds.resolve_media_root())
    sizes = widths or {}
    rows: list[dict[str, object]] = []
    for order, name in enumerate(names):
        width = sizes.get(name, 640)
        video = media_root / name
        if not video.exists():
            _ = video.write_bytes(b"fake")
        rows.append(
            {
                "name": name,
                "group": "",
                "sequence": "sess",
                "group_safe": "",
                "sequence_safe": "sess",
                "camera": "",
                "abs_path": ds.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": width,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": order,
                **clean_facts_cells(width=width, video_uuid=f"uid-{name}"),
            }
        )
    pd.DataFrame(rows).to_csv(media_root / "index.csv", index=False)


def test_a_session_converts_once_with_every_clip(ds: Dataset, trex: FakeTrex) -> None:
    _session(ds, "c0.mp4", "c1.mp4", "c2.mp4")
    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))

    assert len(trex.sources) == 1, "three clips, one conversion"
    assert [p.name for p in trex.sources[0]] == ["c0.mp4", "c1.mp4", "c2.mp4"]


def test_the_pv_is_named_for_the_entry_not_the_first_clip(
    ds: Dataset, trex: FakeTrex
) -> None:
    """TREx would otherwise name a multi-source .pv after the shared parent.

    The name is pinned harder now that a conversion is shared: every slot spells
    it ``conversion.pv``, because the source file that happened to be first must
    not decide what a directory several runs read is called. What this guards is
    unchanged -- the name is chosen, never inherited from the clips.
    """
    _session(ds, "c0.mp4", "c1.mp4")
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    seq_dir = seq_dir_of(ds, run_id, "sess")
    marker = read_phase_marker(seq_dir, "convert")
    assert marker is not None
    pv_path = ds.resolve_path(marker.recorded_output)
    assert pv_path.name == "conversion.pv"
    assert pv_path.exists()
    assert "c0" not in pv_path.name


def test_an_unchanged_session_is_reused(ds: Dataset, trex: FakeTrex) -> None:
    _session(ds, "c0.mp4", "c1.mp4")
    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    assert len(trex.sources) == 1


def test_adding_a_clip_forces_a_recompute(ds: Dataset, trex: FakeTrex) -> None:
    """What the first clip's uid cannot see, and the composition can."""
    _session(ds, "c0.mp4", "c1.mp4")
    first = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    _session(ds, "c0.mp4", "c1.mp4", "c2.mp4")
    second = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))

    assert len(trex.sources) == 2, "the added clip must invalidate the conversion"
    assert first == second, "the settings did not change, so the run_id must not"


def test_reordering_the_clips_forces_a_recompute(ds: Dataset, trex: FakeTrex) -> None:
    _session(ds, "c0.mp4", "c1.mp4")
    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    _session(ds, "c1.mp4", "c0.mp4")
    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    assert len(trex.sources) == 2


def test_the_row_records_the_whole_arrangement(ds: Dataset, trex: FakeTrex) -> None:
    _session(ds, "c0.mp4", "c1.mp4")
    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    # Through the typed reader, which is the documented read path: it projects
    # to the schema and reads a blank cell as "", where a raw read_csv gives NaN.
    row = dr.list_trex_runs(ds).iloc[0]

    assert int(row["n_source_videos"]) == 2
    assert row["video_uuids"] == "uid-c0.mp4,uid-c1.mp4"
    assert str(row["media_composition"]) != ""
    sources = json.loads(str(row["video_sources"]))
    assert [Path(p).name for p in sources] == ["c0.mp4", "c1.mp4"]
    assert not any(Path(p).is_absolute() for p in sources), "portable by construction"
    # video_abs_path keeps naming the first clip, as every tracker's row does.
    assert str(row["video_abs_path"]).endswith("c0.mp4")


def test_a_single_video_row_says_so(ds: Dataset, trex: FakeTrex) -> None:
    """One clip is not a session, and its cells must not imply one."""
    _ = dr.run_trex(ds, TrexParams(entries=[("", "vid1")]))
    row = dr.list_trex_runs(ds).iloc[0]
    assert int(row["n_source_videos"]) == 1
    assert str(row["media_composition"]) == ""


def test_a_resolution_mismatch_is_refused_before_trex_runs(
    ds: Dataset, trex: FakeTrex
) -> None:
    _session(ds, "c0.mp4", "c1.mp4", widths={"c1.mp4": 1280})

    with pytest.raises(JoinedSourceMismatchError, match="c1.mp4"):
        _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    assert trex.sources == [], "nothing may be converted before the refusal"


def test_a_joined_entry_is_not_adopted(ds: Dataset, trex: FakeTrex) -> None:
    """A pre-marker directory cannot say how many clips it covered.

    Its shape is identical to a single-video one, so adopting would keep one
    clip's tracks for a whole session under a marker asserting it was done.

    The conversion cache is cleared here so the directory really is the only
    evidence available. A surviving slot would supply the conversion legitimately
    -- it is addressed by the composition digest of the ordered clips, which is
    exactly the thing a marker-less directory cannot demonstrate -- and that is a
    different question, covered separately.
    """
    _session(ds, "c0.mp4", "c1.mp4")
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    seq_dir = seq_dir_of(ds, run_id, "sess")
    for marker in seq_dir.glob(".mosaic-*.json"):
        marker.unlink()
    shutil.rmtree(ds.get_root(CONVERT_KIND))

    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    assert len(trex.sources) == 2, "a joined entry must recompute, never adopt"
    convert_marker = read_phase_marker(seq_dir_of(ds, run_id, "sess"), "convert")
    assert convert_marker is not None
    assert not convert_marker.backfilled, "the directory must not have been adopted"


def test_a_joined_session_reuses_a_slot_that_proves_its_composition(
    ds: Dataset, trex: FakeTrex
) -> None:
    """The evidence adoption lacks, a slot has by construction.

    A marker-less directory cannot say which clips it covered. A slot can: it is
    addressed by the ordered composition digest, so a hit *is* the proof. The
    conversion is therefore reused and only the tracking redone -- the opposite
    of adoption, which would have skipped both on no evidence at all.
    """
    _session(ds, "c0.mp4", "c1.mp4")
    run_id = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    seq_dir = seq_dir_of(ds, run_id, "sess")
    for marker in seq_dir.glob(".mosaic-*.json"):
        marker.unlink()
    trex.tracked.clear()

    _ = dr.run_trex(ds, TrexParams(entries=[("", "sess")]))
    assert len(trex.sources) == 1, "the slot's conversion covers this clip set"
    assert len(trex.tracked) == 1, "tracking is redone, never adopted"
