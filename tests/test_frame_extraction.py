"""Tests for the tracking frame-extraction subsystem.

Covers the frames index (moved here from test_index_csv.py when frame extraction
relocated from core to mosaic.tracking.frame_extraction) and the selection
algorithms.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

import numpy as np
import pandas as pd

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.index_csv import IndexCSV
from mosaic.tracking.frame_extraction import (
    FramesIndexRow,
    frames_index,
    select_kmeans_frames,
    select_uniform_frames,
)
from mosaic.tracking.frame_extraction.dataset_runs import _source_identity_maps


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


# --- item 5.1's frames half: what a frame set was cut from -------------------


class TestSourceIdentity:
    """The uids and composition a frames row records.

    The interesting property is *where the uids come from*. They are read from
    the media index, not from the ``ResolvedMedia.facts`` already in hand at the
    call site, because a routed entry carries the analysis derivative's facts and
    those name the transcode rather than the source.
    """

    def test_uids_follow_video_order_and_are_not_sorted(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """The cell is the arrangement, so its order is semantic."""
        ds = scenario_dataset_with_media
        uids, _ = _source_identity_maps(ds, [("", "seq_a")])

        by_order = {
            int(row["video_order"]): str(row["video_uuid"])
            for row in ds.read_media_index()
            if str(row["sequence"]) == "seq_a"
        }
        expected = [by_order[i] for i in sorted(by_order)]
        assert uids[("", "seq_a", "")] == ",".join(expected)
        assert len(expected) == 2, "the fixture must hold two videos to order"

    def test_a_reorder_moves_the_recorded_uids(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """A rearrangement is exactly what this cell exists to make visible."""
        from mosaic.core.pipeline.media_index import MediaIndexScope

        ds = scenario_dataset_with_media
        before, _ = _source_identity_maps(ds, [("", "seq_a")])

        _ = ds.write_media_index(
            [
                MediaIndexScope(
                    directory=ds.get_root("media_raw") / "seq_a",
                    group="",
                    sequence="seq_a",
                    order_by_name={"b.mp4": 0, "a.mp4": 1},
                )
            ],
            extensions=(".mp4",),
        )
        after, _ = _source_identity_maps(ds, [("", "seq_a")])

        assert after[("", "seq_a", "")] != before[("", "seq_a", "")]
        assert sorted(after[("", "seq_a", "")].split(",")) == sorted(
            before[("", "seq_a", "")].split(",")
        ), "a reorder must move the order, not the membership"

    def test_the_composition_is_the_projected_one(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """Recorded, never recomputed -- so it cannot disagree with the baseline."""
        from mosaic.core.pipeline.sequence_index import read_sequence_index

        ds = scenario_dataset_with_media
        _, compositions = _source_identity_maps(ds, [("", "seq_a")])

        projected = read_sequence_index(ds, "media_raw")
        expected = {
            (str(r["group"]), str(r["sequence"])): str(r["composition"])
            for _, r in projected.iterrows()
        }
        assert compositions[("", "seq_a")] == expected[("", "seq_a")]
        assert compositions[("", "seq_a")]

    def test_a_sequence_with_no_media_records_nothing(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """Absent, not "composed of nothing" -- seq_b is track-only."""
        ds = scenario_dataset_with_media
        uids, compositions = _source_identity_maps(ds, [("", "seq_b")])

        assert uids == {}
        assert compositions == {("", "seq_b"): ""}

    def test_a_legacy_media_rooted_dataset_records_nothing(
        self, tmp_path: Path
    ) -> None:
        """That root holds derivatives, and a derivative has no composition (P6).

        The same carve-out ``Dataset._write_media_compositions`` makes, so the
        two never disagree about which dataset shapes have an answer.
        """
        from mosaic.core.dataset import Dataset as _Dataset

        ds = _Dataset(
            manifest_path=tmp_path / "dataset.yaml",
            roots={"media": str(tmp_path / "media")},
        )
        ds.ensure_roots()

        assert _source_identity_maps(ds, [("", "seq_a")]) == ({}, {})

    def test_one_identityless_member_unestablishes_the_whole_camera(
        self, scenario_dataset_with_media: Dataset
    ) -> None:
        """A partial list would compare equal to a genuinely different sequence.

        ``composition.py`` applies this rule to the digest; the uid cell has to
        apply it too, or the two answers disagree about the same sequence. An
        imgstore chunk or a row written before identity existed is how this state
        arises in the wild.
        """
        ds = scenario_dataset_with_media
        established, _ = _source_identity_maps(ds, [("", "seq_a")])
        assert established[("", "seq_a", "")], "the fixture must start established"

        index_path = ds.get_root("media_raw") / "index.csv"
        frame = pd.read_csv(index_path, keep_default_na=False, dtype=str)
        frame.loc[frame.index[0], "video_uuid"] = ""
        frame.to_csv(index_path, index=False)

        uids, _ = _source_identity_maps(ds, [("", "seq_a")])
        assert uids[("", "seq_a", "")] == "", (
            "one member without an identity must unestablish the whole camera"
        )


def test_an_extracted_frame_set_records_what_it_was_cut_from(
    scenario_dataset_with_media: Dataset,
) -> None:
    """End to end, because the interesting part is the wiring, not the values.

    ``_source_identity_maps`` is tested above; what this adds is that the answer
    survives the spec being pickled into a worker and reaching the index row. A
    unit test of the resolver plus an unexercised assignment is exactly the shape
    that looks right and is not.
    """
    from mosaic.tracking import extract_frames
    from mosaic.tracking.frame_extraction import list_frame_runs

    ds = scenario_dataset_with_media
    expected_uids, expected_compositions = _source_identity_maps(ds, [("", "seq_a")])

    _ = extract_frames(ds, n_frames=2, method="uniform", sequences=["seq_a"])

    runs = list_frame_runs(ds, method="uniform")
    rows = runs[runs["sequence"] == "seq_a"]
    assert len(rows) == 1
    row = rows.iloc[0]
    assert str(row["video_uuids"]) == expected_uids[("", "seq_a", "")]
    assert str(row["media_composition"]) == expected_compositions[("", "seq_a")]
    assert str(row["video_uuids"]), "a frame set recorded no source identity"


def test_a_frames_row_stores_its_video_paths_root_relative(
    scenario_dataset_with_media: Dataset,
) -> None:
    """The one path cell in this row that no portability pass ever reached.

    ``"frames"`` is absent from ``_INDEX_PATH_COLUMNS``, so a moved or synced
    dataset kept a frames index pointing at the old tree. Registering it there
    would not help -- those passes do prefix substring replacement and this cell
    is multi-valued -- so it is made portable by construction instead.
    """
    from mosaic.tracking import extract_frames
    from mosaic.tracking.frame_extraction import list_frame_runs

    ds = scenario_dataset_with_media
    _ = extract_frames(ds, n_frames=2, method="uniform", sequences=["seq_a"])

    rows = list_frame_runs(ds, method="uniform")
    stored = str(rows[rows["sequence"] == "seq_a"].iloc[0]["video_abs_path"])
    assert stored.startswith("["), "seq_a holds two clips, so the cell stays a list"
    for path in json.loads(stored):
        assert not Path(path).is_absolute(), f"not portable: {path}"
        assert ds.resolve_path(path).exists(), f"does not resolve: {path}"


def _spec(*, seq_dir, overwrite: bool):
    """A minimal work spec: only ``seq_dir`` and ``overwrite`` are read here."""
    from mosaic.tracking.frame_extraction.dataset_runs import _ExtractSpec

    return _ExtractSpec(
        group="",
        sequence="seq_a",
        camera="",
        video_paths=(),
        facts=(),
        video_uuids="",
        media_composition="",
        seq_dir=seq_dir,
        run_id="uniform-aaaaaaaaaa",
        params_hash="aaaaaaaaaa",
        n_frames=1,
        method="uniform",
        start_frame=None,
        end_frame=None,
        candidate_step=1,
        crop=None,
        random_state=42,
        kmeans_resize=(64, 64),
        kmeans_grayscale=True,
        kmeans_max_candidates=None,
        kmeans_batch_size=1024,
        kmeans_max_iter=100,
        kmeans_n_init="auto",
        overwrite=overwrite,
    )


# --- item 6.4's first carve-out: frames a rearrangement must not destroy -------


class TestOverwriteRefuses:
    """P4's carve-out, enforced at the destroyer rather than at the deleter.

    ``Pipeline.clean`` never reached ``frames``; ``overwrite=True`` did, and
    because ``overwrite`` is ``HASH_EXCLUDE`` it removed the *same* directory
    ``AnnotationFrame.image_path`` names.
    """

    def test_an_existing_directory_refuses_rather_than_being_removed(
        self, tmp_path: Path
    ) -> None:
        from mosaic.tracking.frame_extraction.dataset_runs import (
            AnnotatedFramesWouldBeDestroyed,
            _refuse_to_overwrite,
        )

        occupied = tmp_path / "run" / "seq_a"
        occupied.mkdir(parents=True)
        (occupied / "frame_000000.png").write_bytes(b"x")

        spec = _spec(seq_dir=occupied, overwrite=True)
        with pytest.raises(AnnotatedFramesWouldBeDestroyed, match="revision"):
            _refuse_to_overwrite([spec])

        assert (occupied / "frame_000000.png").exists(), "the frame set went"

    def test_it_refuses_before_any_work_starts(self, tmp_path: Path) -> None:
        """Naming every conflict at once, rather than aborting partway.

        Raising inside the worker would leave some sequences extracted and the
        index carrying rows for them -- a half-applied run whose report is a
        traceback.
        """
        from mosaic.tracking.frame_extraction.dataset_runs import (
            AnnotatedFramesWouldBeDestroyed,
            _refuse_to_overwrite,
        )

        first = tmp_path / "run" / "a"
        second = tmp_path / "run" / "b"
        for directory in (first, second):
            directory.mkdir(parents=True)

        with pytest.raises(AnnotatedFramesWouldBeDestroyed) as caught:
            _refuse_to_overwrite(
                [
                    _spec(seq_dir=first, overwrite=True),
                    _spec(seq_dir=second, overwrite=True),
                ]
            )

        message = str(caught.value)
        assert str(first) in message and str(second) in message

    def test_an_absent_directory_is_no_conflict(self, tmp_path: Path) -> None:
        from mosaic.tracking.frame_extraction.dataset_runs import _refuse_to_overwrite

        _refuse_to_overwrite([_spec(seq_dir=tmp_path / "nothing", overwrite=True)])

    def test_overwrite_false_never_refuses(self, tmp_path: Path) -> None:
        """Skipping an existing directory is the ordinary, untouched path."""
        from mosaic.tracking.frame_extraction.dataset_runs import _refuse_to_overwrite

        occupied = tmp_path / "run" / "seq_a"
        occupied.mkdir(parents=True)
        _refuse_to_overwrite([_spec(seq_dir=occupied, overwrite=False)])


class TestTheRevisionTerm:
    """The only term allowed to move the frozen extraction identifier."""

    def test_the_default_reproduces_the_frozen_identifier(self) -> None:
        """Byte-identical at revision 0, which the op corpus also pins."""
        from mosaic.core.pipeline._utils import hash_params
        from mosaic.tracking.frame_extraction.dataset_runs import (
            ExtractFramesParams,
            frames_run_id,
        )

        params = ExtractFramesParams(n_frames=100)
        assert frames_run_id("uniform", params) == (
            f"uniform-{hash_params(params.identity_dump())}"
        )

    def test_a_bumped_revision_mints_a_new_run(self) -> None:
        from mosaic.tracking.frame_extraction.dataset_runs import (
            ExtractFramesParams,
            frames_run_id,
        )

        base = frames_run_id("uniform", ExtractFramesParams(n_frames=100))
        bumped = frames_run_id("uniform", ExtractFramesParams(n_frames=100, revision=1))

        assert bumped != base
        assert bumped.startswith("uniform-")

    def test_two_revisions_differ_from_each_other(self) -> None:
        from mosaic.tracking.frame_extraction.dataset_runs import (
            ExtractFramesParams,
            frames_run_id,
        )

        first = frames_run_id("uniform", ExtractFramesParams(n_frames=100, revision=1))
        second = frames_run_id("uniform", ExtractFramesParams(n_frames=100, revision=2))
        assert first != second
