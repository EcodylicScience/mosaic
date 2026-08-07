"""What each tracker leaves on disk, pinned as a shape rather than as behavior.

The three integrations write the same kind of tree -- a run root holding
``run_params.json`` and one working directory per entry, each carrying a phase
marker and the tool's outputs -- and the run index and tracks table beside it.
The marker suites assert what a run *decides* (reuse, recompute, refuse); nothing
asserted what it *leaves*, so a change to the layout was invisible until a real
dataset stopped resolving.

That is the gap a consolidation has to be measured against: shared machinery may
change how the loop is written, and must not change a single path, marker field,
or index column. So the snapshot is normalized rather than hashed -- volatile
values (timestamps, execution ids, host, pid) are masked and the run identifier
is replaced with ``<run>``, because identity is the golden corpus's subject and
this file's subject is structure. A diff here is a layout change, and a layout
change is either intended and re-pinned in the same commit, or a defect.
"""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mosaic_media import CHROME_149, DEFAULT_THRESHOLDS, MediaFacts, derive

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.facts_columns import facts_to_row, store_facts

# Marker fields this file does not own. Masked rather than dropped, so a field
# that stops being written is still a visible diff. The first group differs
# between two identical runs; ``params_hash`` is stable but is identity, and
# identity is pinned by ``tests/data/op_identity_golden.json`` -- pinning a
# digest here too would mean two files to update for one intended change, and
# the weaker one would be updated by reflex.
_MASKED: frozenset[str] = frozenset(
    {
        "completed_at",
        "execution_id",
        "expires_at",
        "host",
        "params_hash",
        "pid",
        "started_at",
    }
)


# --- dataset fixture (shared shape with the three marker suites) ------------


def _clean_facts_cells() -> dict[str, object]:
    facts: MediaFacts = store_facts(
        width=640,
        height=480,
        fps=30.0,
        frame_count=100,
        codec="h264",
        duration=100 / 30.0,
        video_uuid="uid-vid1",
        identity_scheme="",
    )
    facts = dataclasses.replace(
        facts,
        container="mov,mp4,m4a,3gp,3g2,mj2",
        pixel_format="yuv420p",
        moov_at_start=True,
    )
    return dict(facts_to_row(facts, derive(facts, CHROME_149, DEFAULT_THRESHOLDS)))


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("layout", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    media_root = dataset.get_root(dataset.resolve_media_root())
    media_root.mkdir(parents=True, exist_ok=True)
    video = media_root / "vid1.mp4"
    video.write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "name": "vid1.mp4",
                "group": "",
                "sequence": "vid1",
                "group_safe": "",
                "sequence_safe": "vid1",
                "abs_path": dataset.relative_to_root(video),
                "size_bytes": 4,
                "mtime_iso": "",
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "codec": "h264",
                "media_type": "video",
                "video_order": 0,
                **_clean_facts_cells(),
            }
        ]
    ).to_csv(media_root / "index.csv", index=False)
    return dataset


# --- the snapshot ----------------------------------------------------------


def _normalize(value: object, run_id: str) -> object:
    if isinstance(value, str):
        return value.replace(run_id, "<run>")
    return value


def snapshot(ds: Dataset, kind: str, run_id: str) -> dict[str, object]:
    """The normalized on-disk shape one tracker run left, as comparable data."""
    root = ds.get_root(kind)
    paths = sorted(
        str(p.relative_to(root)).replace(run_id, "<run>")
        for p in root.rglob("*")
        if p.is_file()
    )

    markers: dict[str, dict[str, object]] = {}
    for marker_path in sorted(root.rglob(".mosaic-*.json")):
        payload: dict[str, object] = json.loads(marker_path.read_text())
        markers[marker_path.name] = {
            key: ("<masked>" if key in _MASKED else _normalize(value, run_id))
            for key, value in sorted(payload.items())
        }

    index = pd.read_csv(root / "index.csv")
    params_path = root / run_id / "run_params.json"
    tracks_parquet = sorted(ds.get_root("tracks").rglob("*.parquet"))

    return {
        "files": paths,
        "markers": markers,
        "index_columns": list(index.columns),
        "index_rows": int(len(index)),
        "run_params_keys": sorted(json.loads(params_path.read_text())),
        "tracks_files": [
            str(p.relative_to(ds.get_root("tracks"))).replace(run_id, "<run>")
            for p in tracks_parquet
        ],
        "tracks_columns": (
            sorted(pd.read_parquet(tracks_parquet[0]).columns) if tracks_parquet else []
        ),
    }


# --- Lightning Pose --------------------------------------------------------


def _write_dlc_csv(path: Path, *, n: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    scorer = ["scorer"]
    bodyparts = ["bodyparts"]
    coords = ["coords"]
    for bodypart in ("nose", "tail"):
        scorer += ["heatmap_tracker"] * 3
        bodyparts += [bodypart] * 3
        coords += ["x", "y", "likelihood"]
    lines = [",".join(scorer), ",".join(bodyparts), ",".join(coords)]
    for i in range(n):
        row = [str(i)]
        for _bodypart in ("nose", "tail"):
            x, y = rng.uniform(0, 100, 2)
            row += [f"{x:.6f}", f"{y:.6f}", f"{rng.uniform(0.5, 1.0):.6f}"]
        lines.append(",".join(row))
    path.write_text("\n".join(lines))


@dataclass
class _FakeLitpose:
    predicted: list[Path] = field(default_factory=list)

    def predict(self, video_path: Path, out_csv: Path, **_kwargs: object) -> object:
        from mosaic.tracking.litpose.run import LitposePredictResult

        self.predicted.append(Path(video_path))
        _write_dlc_csv(Path(out_csv))
        return LitposePredictResult(csv_path=Path(out_csv), stdout="", stderr="")


@pytest.fixture
def litpose_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "lp_model"
    ckpt = model_dir / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"weights")
    (model_dir / "config.yaml").write_text(
        "model:\n  model_type: heatmap\ndata:\n  keypoint_names: [nose, tail]\n"
    )
    return model_dir


@pytest.fixture
def fake_litpose(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeLitpose]:
    import mosaic.tracking.litpose.dataset_runs as litpose_runs

    fake = _FakeLitpose()
    monkeypatch.setattr(litpose_runs, "run_litpose_predict", fake.predict)
    yield fake


def test_litpose_leaves_this_shape(
    ds: Dataset, litpose_model: Path, fake_litpose: _FakeLitpose
) -> None:
    import mosaic.tracking.litpose.dataset_runs as litpose_runs

    run_id = litpose_runs.run_litpose(ds, model_path=str(litpose_model))
    got = snapshot(ds, "litpose", run_id)

    assert got["files"] == [
        "<run>/.identity_scheme",
        "<run>/run_params.json",
        "<run>/vid1/.mosaic-track.json",
        "<run>/vid1/vid1.predictions.csv",
        "index.csv",
    ]
    assert got["markers"] == {
        ".mosaic-track.json": {
            "backfilled": False,
            "completed_at": "<masked>",
            "execution_id": "<masked>",
            "params_hash": "<masked>",
            "phase": "track",
            "recorded_output": "_tracking/litpose/<run>/vid1/vid1.predictions.csv",
            "run_id": "<run>",
            "schema_version": 1,
            "source": "media_raw/vid1.mp4",
            "source_uid": "uid-vid1",
        }
    }
    assert got["index_columns"] == [
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "group",
        "sequence",
        "video_abs_path",
        "params_hash",
        "n_ids",
        "model_id",
        "model_type",
        "csv_path",
    ]
    assert got["index_rows"] == 1
    assert got["run_params_keys"] == ["litpose_overrides", "model"]
    assert got["tracks_files"] == ["<run>/vid1.parquet"]
    assert got["tracks_columns"] == [
        "ANGLE",
        "SPEED",
        "VX",
        "VY",
        "X",
        "X#wcentroid",
        "Y",
        "Y#wcentroid",
        "frame",
        "group",
        "id",
        "poseP0",
        "poseP1",
        "poseX0",
        "poseX1",
        "poseY0",
        "poseY1",
        "sequence",
        "time",
    ]


# --- SLEAP -----------------------------------------------------------------


def _write_analysis_h5(path: Path, *, n_frames: int = 6) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    # matlab layout: (track, xy, node, frame)
    tracks = rng.uniform(0, 100, (1, 2, 2, n_frames))
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("tracks", data=tracks)
        dataset.attrs["dims"] = json.dumps(["track", "xy", "node", "frame"])
        handle.create_dataset(
            "node_names", data=np.array([b"nose", b"tail"], dtype="S")
        )
        handle.create_dataset("track_names", data=np.array([b"track_0"], dtype="S"))


@dataclass
class _FakeSleap:
    tracked: list[Path] = field(default_factory=list)
    converted: list[Path] = field(default_factory=list)

    def track(self, video_path: Path, output_slp: Path, **_kwargs: object) -> object:
        from mosaic.tracking.sleap.run import SleapTrackResult

        self.tracked.append(Path(video_path))
        Path(output_slp).parent.mkdir(parents=True, exist_ok=True)
        Path(output_slp).write_bytes(b"slp")
        return SleapTrackResult(slp_path=Path(output_slp), stdout="", stderr="")

    def convert(self, slp_path: Path, output_h5: Path, **_kwargs: object) -> object:
        from mosaic.tracking.sleap.run import SleapConvertResult

        self.converted.append(Path(slp_path))
        _write_analysis_h5(Path(output_h5))
        return SleapConvertResult(
            analysis_h5_path=Path(output_h5), stdout="", stderr=""
        )


@pytest.fixture
def sleap_model(tmp_path: Path) -> Path:
    model_dir = tmp_path / "sleap_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "best.ckpt").write_bytes(b"weights")
    (model_dir / "training_config.yaml").write_text("head: single_instance\n")
    return model_dir


@pytest.fixture
def fake_sleap(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeSleap]:
    import mosaic.tracking.sleap.dataset_runs as sleap_runs

    fake = _FakeSleap()
    monkeypatch.setattr(sleap_runs, "run_sleap_track", fake.track)
    monkeypatch.setattr(sleap_runs, "run_sleap_convert", fake.convert)
    yield fake


def test_sleap_leaves_this_shape(
    ds: Dataset, sleap_model: Path, fake_sleap: _FakeSleap
) -> None:
    _ = pytest.importorskip("h5py")
    import mosaic.tracking.sleap.dataset_runs as sleap_runs

    run_id = sleap_runs.run_sleap(ds, model_paths=[str(sleap_model)])
    got = snapshot(ds, "sleap", run_id)

    # The .h5 sits beside the .slp with no marker of its own: it is the ungated,
    # atomically published export, re-run only when missing or when the inference
    # it derives from was recomputed.
    assert got["files"] == [
        "<run>/.identity_scheme",
        "<run>/run_params.json",
        "<run>/vid1/.mosaic-track.json",
        "<run>/vid1/vid1.analysis.h5",
        "<run>/vid1/vid1.predictions.slp",
        "index.csv",
    ]
    assert got["markers"] == {
        ".mosaic-track.json": {
            "backfilled": False,
            "completed_at": "<masked>",
            "execution_id": "<masked>",
            "params_hash": "<masked>",
            "phase": "track",
            "recorded_output": "_tracking/sleap/<run>/vid1/vid1.predictions.slp",
            "run_id": "<run>",
            "schema_version": 1,
            "source": "media_raw/vid1.mp4",
            "source_uid": "uid-vid1",
        }
    }
    assert got["index_columns"] == [
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "group",
        "sequence",
        "video_abs_path",
        "params_hash",
        "n_ids",
        "model_id",
        "model_type",
        "slp_path",
        "analysis_h5_path",
    ]
    assert got["index_rows"] == 1
    assert got["run_params_keys"] == [
        "analysis_range",
        "match",
        "max_instances",
        "max_tracking",
        "model",
        "peak_threshold",
        "similarity",
        "sleap_extra_settings",
        "track_window",
        "tracker",
        "tracking",
    ]
    assert got["tracks_files"] == ["<run>/vid1.parquet"]


# --- TREx ------------------------------------------------------------------


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ whose member names need not be Python identifiers.

    ``np.savez`` takes its array names as keyword arguments, and TREx column
    names carry a ``#`` -- so they can only be passed by unpacking a mapping,
    which collides with the ``allow_pickle`` keyword in the same signature. This
    writes the same container directly: an uncompressed zip of ``.npy`` members,
    which is exactly what ``np.savez`` produces and what ``np.load`` reads back.
    """
    with zipfile.ZipFile(path, "w") as archive:
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", buffer.getvalue())


@dataclass
class _FakeTrex:
    converted: list[Path] = field(default_factory=list)
    tracked: list[Path] = field(default_factory=list)

    def convert(self, video_path: Path, output_dir: Path, **_kwargs: object) -> object:
        from mosaic.tracking.trex.run import TRexConvertResult

        self.converted.append(Path(video_path))
        stem = Path(video_path).stem
        pv = Path(output_dir) / f"{stem}.pv"
        pv.parent.mkdir(parents=True, exist_ok=True)
        pv.write_bytes(b"pv")
        # settings_path is never None in production: the real wrapper falls back
        # to the video's sibling, which is a Path whether or not it exists.
        return TRexConvertResult(
            pv_path=pv,
            settings_path=Path(output_dir) / f"{stem}.settings",
            background_path=None,
            stdout="",
            stderr="",
        )

    def track(self, pv_path: Path, output_dir: Path, **_kwargs: object) -> object:
        from mosaic.tracking.trex.run import TRexTrackResult

        self.tracked.append(Path(pv_path))
        out = Path(output_dir)
        stem = Path(pv_path).stem
        data = out / "data"
        data.mkdir(parents=True, exist_ok=True)
        npz = data / f"{stem}_fish0.npz"
        _write_npz(
            npz,
            {
                "frame": np.arange(6),
                "time": np.arange(6) / 30.0,
                # TREx records the factor it scaled positions by in every
                # export; the conversion refuses a file that does not say.
                "cm_per_pixel": np.array([1.0]),
                "X#wcentroid": np.arange(6, dtype=float),
                "Y#wcentroid": np.arange(6, dtype=float),
                "poseX0": np.arange(6, dtype=float),
                "poseY0": np.arange(6, dtype=float),
            },
        )
        results = out / f"{stem}.results"
        results.write_bytes(b"results")
        return TRexTrackResult(
            npz_paths=[npz],
            results_path=results,
            settings_path=out / f"{stem}.settings",
            stdout="",
            stderr="",
        )


@pytest.fixture
def fake_trex(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeTrex]:
    import mosaic.tracking.trex.dataset_runs as trex_runs

    fake = _FakeTrex()
    monkeypatch.setattr(trex_runs, "run_trex_convert", fake.convert)
    monkeypatch.setattr(trex_runs, "run_trex_track", fake.track)
    yield fake


def test_trex_leaves_this_shape(ds: Dataset, fake_trex: _FakeTrex) -> None:
    import mosaic.tracking.trex.dataset_runs as trex_runs

    run_id = trex_runs.run_trex(ds)
    got = snapshot(ds, "trex", run_id)

    assert got["files"] == [
        "<run>/.identity_scheme",
        "<run>/run_params.json",
        "<run>/vid1/.mosaic-convert.json",
        "<run>/vid1/.mosaic-track.json",
        "<run>/vid1/data/vid1_fish0.npz",
        "<run>/vid1/vid1.pv",
        "<run>/vid1/vid1.results",
        "index.csv",
    ]
    assert got["markers"] == {
        ".mosaic-convert.json": {
            "backfilled": False,
            "completed_at": "<masked>",
            "execution_id": "<masked>",
            "params_hash": "<masked>",
            "phase": "convert",
            "recorded_output": "_tracking/trex/<run>/vid1/vid1.pv",
            "run_id": "<run>",
            "schema_version": 1,
            "source": "media_raw/vid1.mp4",
            "source_uid": "uid-vid1",
        },
        ".mosaic-track.json": {
            "backfilled": False,
            "completed_at": "<masked>",
            "execution_id": "<masked>",
            "params_hash": "<masked>",
            "phase": "track",
            "recorded_output": "_tracking/trex/<run>/vid1/vid1.results",
            "run_id": "<run>",
            "schema_version": 1,
            "source": "media_raw/vid1.mp4",
            "source_uid": "uid-vid1",
        },
    }
    assert got["index_columns"] == [
        "abs_path",
        "run_id",
        "started_at",
        "finished_at",
        "group",
        "sequence",
        "video_abs_path",
        "params_hash",
        "n_ids",
        "pv_path",
    ]
    assert got["index_rows"] == 1
    assert got["tracks_files"] == ["<run>/vid1.parquet"]


def test_trex_gates_its_two_phases_on_different_parameter_subsets(
    ds: Dataset, fake_trex: _FakeTrex
) -> None:
    """The property a shared driver must not collapse into one hash.

    ``CONVERT_KEYS`` and ``TRACK_KEYS`` project the settings onto what each phase
    actually consumes, so retuning a track-only knob reuses the conversion. A
    driver that hashed the whole settings dict per phase would agree with itself
    and mismatch every marker already on disk, re-converting every sequence once.
    Asserted from the markers rather than from the projection, because the
    markers are what a later run compares against.
    """
    import mosaic.tracking.trex.dataset_runs as trex_runs

    run_id = trex_runs.run_trex(ds)
    seq_dir = trex_runs.trex_run_root(ds, run_id) / "vid1"

    convert = json.loads((seq_dir / ".mosaic-convert.json").read_text())
    track = json.loads((seq_dir / ".mosaic-track.json").read_text())

    assert convert["params_hash"] != ""
    assert track["params_hash"] != ""
    assert convert["params_hash"] != track["params_hash"]


def test_an_extra_trex_column_survives_into_the_tracks_table(
    ds: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whatever TREx exports reaches the parquet, including fields mosaic never names.

    TREx's ``output_fields`` decides what its NPZ holds, and mosaic does not set
    it -- so which columns exist is the user's choice, made through
    ``track_extra_settings``. What this pins is that the choice is honoured all
    the way through: the converter flattens every NPZ key rather than a known
    list, ``ensure_track_schema`` accepts unknown columns, and the bridge
    concatenates per-individual frames on the *union* of their columns.

    ``tracklet_id`` is the case this exists for. It identifies consecutively
    tracked frame segments, it is absent from TREx's default ``output_fields``,
    and it is what future identity work needs -- so the question "will mosaic
    keep it once TREx emits it" has a recorded answer rather than an assumption.
    """
    import mosaic.tracking.trex.dataset_runs as trex_runs
    from mosaic.tracking.trex.run import TRexTrackResult

    fake = _FakeTrex()

    def track_with_extra_fields(
        pv_path: Path, output_dir: Path, **_kwargs: object
    ) -> TRexTrackResult:
        out = Path(output_dir)
        stem = Path(pv_path).stem
        data = out / "data"
        data.mkdir(parents=True, exist_ok=True)
        npz = data / f"{stem}_fish0.npz"
        _write_npz(
            npz,
            {
                "frame": np.arange(6),
                "time": np.arange(6) / 30.0,
                # TREx records the factor it scaled positions by in every
                # export; the conversion refuses a file that does not say.
                "cm_per_pixel": np.array([1.0]),
                "X#wcentroid": np.arange(6, dtype=float),
                "Y#wcentroid": np.arange(6, dtype=float),
                "poseX0": np.arange(6, dtype=float),
                "poseY0": np.arange(6, dtype=float),
                # The two fields a user adds to output_fields beyond TREx's
                # defaults, and the reason this test exists.
                "tracklet_id": np.array([0, 0, 0, 1, 1, 1]),
                "blobid": np.arange(6),
            },
        )
        results = out / f"{stem}.results"
        results.write_bytes(b"results")
        return TRexTrackResult(
            npz_paths=[npz],
            results_path=results,
            settings_path=out / f"{stem}.settings",
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(trex_runs, "run_trex_convert", fake.convert)
    monkeypatch.setattr(trex_runs, "run_trex_track", track_with_extra_fields)

    run_id = trex_runs.run_trex(ds)

    table = pd.read_parquet(next(ds.get_root("tracks").rglob("*.parquet")))
    assert "tracklet_id" in table.columns, "an extra TREx field was dropped"
    assert "blobid" in table.columns
    assert sorted(table["tracklet_id"]) == [0, 0, 0, 1, 1, 1]
    assert run_id.startswith("trex.")
