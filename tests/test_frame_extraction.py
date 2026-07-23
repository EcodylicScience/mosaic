"""Tests for the tracking frame-extraction subsystem.

Covers the frames index (moved here from test_index_csv.py when frame extraction
relocated from core to mosaic.tracking.frame_extraction) and the selection
algorithms.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.tracking.frame_extraction import (
    FramesIndexRow,
    frames_index,
    select_kmeans_frames,
    select_uniform_frames,
)


# --- Frames Index ---


class TestFramesIndex:
    def test_schema_has_required_columns(self) -> None:
        names = {f.name for f in dataclasses.fields(FramesIndexRow)}
        assert "method" in names
        assert "n_frames_extracted" in names

    def test_factory_returns_index_csv(self, tmp_path: Path) -> None:
        idx = frames_index(tmp_path / "index.csv")
        assert isinstance(idx, IndexCSV)

    def test_ensure_creates(self, tmp_path: Path) -> None:
        idx = frames_index(tmp_path / "index.csv")
        idx.ensure()
        df = pd.read_csv(tmp_path / "index.csv")
        assert "method" in df.columns

    def test_dedup(self, tmp_path: Path) -> None:
        idx = frames_index(tmp_path / "index.csv")
        p = tmp_path / "frames_dir"
        p.mkdir()
        row = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="g",
            sequence="s",
            camera="",
            abs_path=str(p),
            video_abs_path=str(p),
            params_hash="h",
            n_frames_extracted=10,
        )
        idx.append([row])
        row2 = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="g",
            sequence="s",
            camera="",
            abs_path=str(p),
            video_abs_path=str(p),
            params_hash="h",
            n_frames_extracted=20,
        )
        idx.append([row2])
        df = idx.read()
        assert len(df) == 1

    def test_distinct_cameras_are_not_deduped(self, tmp_path: Path) -> None:
        # Two cameras of one recording share (run_id, group, sequence); camera is
        # part of the dedup key so a partial re-run of one never drops the other.
        idx = frames_index(tmp_path / "index.csv")
        p = tmp_path / "frames_dir"
        p.mkdir()

        def _row(camera: str) -> FramesIndexRow:
            cam_dir = p / camera
            cam_dir.mkdir(exist_ok=True)
            return FramesIndexRow(
                run_id="r1",
                method="uniform",
                group="g",
                sequence="s",
                camera=camera,
                abs_path=str(cam_dir),
                video_abs_path=str(cam_dir),
                params_hash="h",
                n_frames_extracted=10,
            )

        idx.append([_row("CAMA")])
        idx.append([_row("CAMB")])
        df = idx.read()
        assert len(df) == 2
        assert set(df["camera"]) == {"CAMA", "CAMB"}


class TestFramesIndexRow:
    def test_fields_match_schema(self, tmp_path: Path) -> None:
        p = tmp_path / "G1__S1"
        p.mkdir()
        v = tmp_path / "v1.mp4"
        v.touch()
        row = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="G1",
            sequence="S1",
            camera="",
            abs_path=str(p),
            n_frames_extracted=50,
            n_frames_requested=50,
            video_abs_path=str(v),
            params_hash="abc",
        )
        df = pd.DataFrame([row])
        assert set(df.columns) == {f.name for f in dataclasses.fields(FramesIndexRow)}
        assert df.iloc[0]["method"] == "uniform"
        assert df.iloc[0]["n_frames_extracted"] == 50

    def test_finished_at_default(self, tmp_path: Path) -> None:
        p = tmp_path / "frames"
        p.mkdir()
        row = FramesIndexRow(
            run_id="r",
            method="m",
            group="",
            sequence="s",
            camera="",
            abs_path=str(p),
            n_frames_extracted=0,
            n_frames_requested=0,
            video_abs_path=str(p),
            params_hash="h",
        )
        assert row.finished_at == ""

    def test_appendable_to_frames_index(self, tmp_path: Path) -> None:
        idx = frames_index(tmp_path / "index.csv")
        p = tmp_path / "frames"
        p.mkdir()
        row = FramesIndexRow(
            run_id="r1",
            method="uniform",
            group="G1",
            sequence="S1",
            camera="",
            abs_path=str(p),
            n_frames_extracted=10,
            n_frames_requested=50,
            video_abs_path=str(p),
            params_hash="h",
        )
        idx.append([row])
        df = idx.read()
        assert len(df) == 1
        assert df.iloc[0]["method"] == "uniform"


# --- Selection algorithms ---


class TestSelectUniformFrames:
    def test_count_and_membership(self) -> None:
        candidates = np.arange(0, 100, dtype=np.int32)
        selected = select_uniform_frames(candidates, 5)
        assert len(selected) == 5
        assert set(selected).issubset(set(candidates.tolist()))
        assert len(set(selected.tolist())) == 5  # unique
        assert list(selected) == sorted(selected)  # ordered

    def test_n_ge_candidates_returns_all(self) -> None:
        candidates = np.array([3, 7, 9], dtype=np.int32)
        selected = select_uniform_frames(candidates, 10)
        assert sorted(selected.tolist()) == [3, 7, 9]


class TestSelectKmeansFrames:
    def test_count_and_membership(self) -> None:
        rng = np.random.default_rng(0)
        candidates = np.arange(0, 40, dtype=np.int32)
        features = rng.standard_normal((40, 8)).astype(np.float32)
        selected = select_kmeans_frames(candidates, features, 6, random_state=42)
        assert len(selected) == 6
        assert set(selected.tolist()).issubset(set(candidates.tolist()))
        assert len(set(selected.tolist())) == 6  # unique

    def test_deterministic(self) -> None:
        rng = np.random.default_rng(1)
        candidates = np.arange(0, 30, dtype=np.int32)
        features = rng.standard_normal((30, 5)).astype(np.float32)
        a = select_kmeans_frames(candidates, features, 4, random_state=7)
        b = select_kmeans_frames(candidates, features, 4, random_state=7)
        assert a.tolist() == b.tolist()
