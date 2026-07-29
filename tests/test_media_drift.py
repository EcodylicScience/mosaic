"""Item 5.2: an ordinary index write reports a source that moved under it.

Drift detection existed only as the operator command ``mosaic reprobe-media``.
This is the other half of milestone M3's gate -- "drift detected at reindex" --
and the tests here are about *which* path notices, and at what cost, rather than
about the comparison itself (that is ``classify_identity``, exercised through the
reprobe suite).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.drift import classify_identity
from mosaic.core.pipeline.media_index import MediaIndexScope
from mosaic.core.pipeline.types import Inputs, Params


class _P(Params):
    pass


def _dataset(base: Path) -> Dataset:
    manifest = new_dataset_manifest(name="drift", base_dir=base / "dataset")
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def _write(ds: Dataset, sequence: str, directory: Path) -> object:
    return ds.write_media_index(
        [MediaIndexScope(directory=directory, group="", sequence=sequence)],
        extensions=(".mp4",),
    )


def _clip(path: Path, shade: int, frames: int = 6) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (64, 48))
    for _ in range(frames):
        writer.write(np.full((48, 64, 3), shade, np.uint8))
    writer.release()


@pytest.mark.usefixtures("requires_ffprobe")
class TestWritePathDrift:
    def test_a_replaced_file_is_reported_as_drift(self, tmp_path: Path) -> None:
        """Different bytes under a stable path, found because the write re-probed."""
        ds = _dataset(tmp_path)
        directory = ds.get_root("media_raw") / "seq_a"
        _clip(directory / "a.mp4", shade=40)
        first = _write(ds, "seq_a", directory)
        assert first.drift == []

        # Replaced in place. The size and mtime both move, so the measurement
        # cache misses and the file is re-probed -- which is the moment the
        # stored identity and the fresh one are both in hand.
        _clip(directory / "a.mp4", shade=200, frames=9)
        second = _write(ds, "seq_a", directory)

        assert len(second.drift) == 1, "a replaced file was not reported"
        moved = second.drift[0]
        assert moved.change in ("content_digest_changed", "video_uuid_changed")
        assert moved.recorded_uuid and moved.measured_uuid
        assert moved.recorded_uuid != moved.measured_uuid
        # Both sides ride on the report: the fresh measurement wins and the row is
        # rewritten, so nothing else remembers the old identity.
        assert moved.recorded_digest != moved.measured_digest
        assert second.disagreements == [], "a re-probe is drift, not a stale override"

    def test_an_unchanged_file_is_neither_reported_nor_probed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cost-profile guard, and it is the point of the cache.

        A second write over an untouched directory must reuse every stored
        measurement. Asserting only "no drift" would pass just as well if the
        cache had been removed and every file re-probed, which is the change this
        test exists to reject -- so the probe itself is made to raise.
        """
        ds = _dataset(tmp_path)
        directory = ds.get_root("media_raw") / "seq_a"
        _clip(directory / "a.mp4", shade=40)
        _ = _write(ds, "seq_a", directory)

        import mosaic.core.dataset as dataset_module

        def _refuse(path: Path) -> object:
            raise AssertionError(f"re-probed an unchanged file: {path}")

        monkeypatch.setattr(dataset_module, "probe_video_metadata", _refuse)
        second = _write(ds, "seq_a", directory)

        assert second.drift == []
        assert second.disagreements == []

    def test_two_files_of_one_name_are_each_compared(self, tmp_path: Path) -> None:
        """The bug the path key fixes: a shared basename used to report nothing.

        The prior map was keyed on basename and dropped any name it saw twice, so
        two sequences each holding an ``a.mp4`` were the one case that got no
        comparison at all -- while the measurement cache twenty lines below was
        already path-keyed for exactly this reason.
        """
        ds = _dataset(tmp_path)
        first_dir = ds.get_root("media_raw") / "seq_a"
        second_dir = ds.get_root("media_raw") / "seq_b"
        _clip(first_dir / "a.mp4", shade=40)
        _clip(second_dir / "a.mp4", shade=90)
        _ = _write(ds, "seq_a", first_dir)
        _ = _write(ds, "seq_b", second_dir)

        _clip(first_dir / "a.mp4", shade=210, frames=9)
        report = _write(ds, "seq_a", first_dir)

        assert len(report.drift) == 1
        assert report.drift[0].resolved_path.parent.name == "seq_a"

    def test_a_replacement_preserving_size_and_mtime_is_not_detected(
        self, tmp_path: Path
    ) -> None:
        """The residual, asserted so it is a known limit rather than a surprise.

        The write path serves the stored measurement whenever size and mtime both
        match, so it never probes and never compares. ``mosaic reprobe-media`` is
        the only detector, which is why its no-cache rule must not be
        "optimized" into agreement with this one.
        """
        ds = _dataset(tmp_path)
        directory = ds.get_root("media_raw") / "seq_a"
        clip = directory / "a.mp4"
        _clip(clip, shade=40)
        _ = _write(ds, "seq_a", directory)
        before = clip.stat()

        _clip(clip, shade=200)
        # Restore both stat fields the cache keys on. Size is only equal if the
        # two encodes happen to match; skip rather than assert a coincidence.
        if clip.stat().st_size != before.st_size:
            pytest.skip("the two encodes differ in size; the residual needs equal size")
        import os

        os.utime(clip, (before.st_atime, before.st_mtime))

        report = _write(ds, "seq_a", directory)
        assert report.drift == [], (
            "this case is known to escape the write path; if it is now caught, "
            "the cache changed and this test should become the assertion that "
            "it is caught"
        )


class TestClassifierIsShared:
    def test_the_write_path_and_the_audit_agree_by_construction(self) -> None:
        """One comparison, one home -- reprobe imports it rather than owning it."""
        from mosaic.core.media import reprobe

        assert reprobe.classify_identity is classify_identity


# --- item 5.2's other half: the chain runner shows a source that moved --------


class _CropLike:
    """A per-frame feature that opens video, the shape ``egocentric-crop`` has.

    Local rather than imported from the scenario suite: what matters here is only
    that it declares ``consumed_roots``, so its rows carry the composition each
    entry was built from. Borrowing a scenario stub would couple this file to the
    H1-H5 spec suite, which is about identity rather than about display.
    """

    name = "drift-crop-probe"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    consumed_roots: tuple[str, ...] = ("media_raw",)

    def __init__(
        self,
        inputs: Inputs | None = None,
        params: dict[str, object] | _P | None = None,
    ) -> None:
        self.inputs = inputs if inputs is not None else Inputs(("tracks",))
        self.params = params if isinstance(params, _P) else _P.from_overrides(params)

    def load_state(
        self, run_root: Path, artifact_paths: object, dependency_lookups: object
    ) -> bool:
        return True

    def fit(self, inputs: object) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df):
        return df


@pytest.mark.usefixtures("requires_ffprobe")
def test_the_status_display_reports_a_source_that_moved(
    scenario_dataset_with_media: Dataset,
) -> None:
    """A reorder under a finished run shows up as drift, and costs no probe.

    Both sides are already on disk -- the run's recorded ``consumed_composition``
    and what ``media_raw/sequences.csv`` holds now -- so this is two index reads.
    ``probe_video_metadata`` is made to raise for the duration, because a status
    display that measured the filesystem would be unusable on the corpora this
    exists for, and asserting only the drift count would not notice if it began
    to.
    """
    from mosaic.core.pipeline.media_index import MediaIndexScope
    from mosaic.core.pipeline.pipeline import FeatureStep, Pipeline

    ds = scenario_dataset_with_media
    pipeline = Pipeline()
    _ = pipeline.add(FeatureStep("crop", _CropLike, {}))
    _ = pipeline.run(ds)

    before = pipeline.status(ds)
    assert list(before["drift"]) == [""], "a freshly run pipeline must show no drift"

    # Reorder seq_a's two clips. No bytes change; the composition does.
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

    import mosaic.core.dataset as dataset_module

    def _refuse(path: Path) -> object:
        raise AssertionError(f"status() probed the filesystem: {path}")

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(dataset_module, "probe_video_metadata", _refuse)
        after = pipeline.status(ds)

    assert list(after["drift"]) == [1], "a reordered source was not reported"
    assert list(after["cached"]) == list(before["cached"]), (
        "drift is a source verdict; it must not change the cache verdict"
    )
