"""Item 6.5: would a reader open this sequence, under the order about to be committed.

The hazard is narrow and worth stating precisely, because a precheck that fired
more widely would be noise. ``uniform_properties`` compares each clip against
*the first*, and only its frame-rate arm is order-dependent: the drift allowance
is ``|other - reference| / reference`` scaled by the shorter clip's frame count,
so both the denominator and the multiplier move when position 0 does. The
dimension arm is exact equality against the reference, which is a total agreement
test -- if any clip disagrees then some clip disagrees with the first, in every
ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

from mosaic.core.media.uniformity import camera_uniformity


def _row(
    *,
    name: str,
    order: int,
    width: int = 64,
    height: int = 48,
    fps: float = 30.0,
    frame_count: int = 60,
    rotation: int = 0,
    facts: bool = True,
) -> dict[str, object]:
    """A media-index row carrying just what the precheck reads."""
    payload = {
        "container": "mp4",
        "codec_name": "h264",
        "pixel_format": "yuv420p",
        "color_range": "",
        "color_primaries": "",
        "color_transfer": "",
        "width": width,
        "height": height,
        "rotation_degrees": rotation,
        "square_pixels": True,
        "progressive": True,
        "has_audio": False,
        "video_stream_count": 1,
        "duration": frame_count / fps if fps else 0.0,
        "fps": fps,
        "frame_count": frame_count,
        "start_time": 0.0,
        "constant_frame_rate": True,
        "max_instantaneous_fps": None,
        "declared_duration": frame_count / fps if fps else 0.0,
        "declared_fps": fps,
        "declared_frame_count": frame_count,
        "moov_at_start": True,
        "max_keyframe_interval_frames": 1,
        "max_gop_bytes": 1,
        "timing_measured": True,
        "video_uuid": f"uuid-{name}",
        "content_digest": f"digest-{name}",
        "identity_scheme": "1",
        "prober_version": "test",
    }
    return {
        "name": name,
        "group": "",
        "sequence": "seq",
        "camera": "",
        "abs_path": f"media_raw/seq/{name}",
        "video_order": str(order),
        "media_facts": json.dumps(payload) if facts else "",
    }


class TestTheDimensionArm:
    def test_matching_clips_are_readable(self) -> None:
        verdict = camera_uniformity(
            [_row(name="a.mp4", order=0), _row(name="b.mp4", order=1)]
        )
        assert verdict.readable
        assert verdict.established

    def test_a_disagreement_is_reported(self) -> None:
        verdict = camera_uniformity(
            [_row(name="a.mp4", order=0), _row(name="b.mp4", order=1, width=128)]
        )
        assert not verdict.readable
        assert verdict.mismatch is not None
        assert verdict.mismatch.field == "width"

    def test_the_dimension_arm_cannot_flip_on_a_reorder(self) -> None:
        """Exact equality against a reference is order-independent.

        Asserted rather than assumed: it is the half of item 6.5's hazard that
        does *not* exist, and a precheck reporting it would fire on sequences a
        reorder cannot affect.
        """
        rows = [_row(name="a.mp4", order=0), _row(name="b.mp4", order=1, width=128)]
        forward = camera_uniformity(rows)
        reversed_order = camera_uniformity(rows, order_by_name={"b.mp4": 0, "a.mp4": 1})
        assert not forward.readable and not reversed_order.readable


class TestTheFrameRateArm:
    """The arm a reorder can flip, and the reason the precheck exists."""

    def test_a_tolerated_drift_is_readable(self) -> None:
        """Two clips of one rig at a nominal rate never fit identically.

        Exact equality would reject every multi-clip sequence in the project's
        own data, which is why the allowance exists at all.
        """
        verdict = camera_uniformity(
            [
                _row(name="a.mp4", order=0, fps=30.0, frame_count=60),
                _row(name="b.mp4", order=1, fps=30.00000004, frame_count=60),
            ]
        )
        assert verdict.readable

    def test_position_zero_decides_a_marginal_sequence(self) -> None:
        """The hazard, as an arrangement that is readable one way and not the other.

        The allowance divides by the *reference* clip's rate, so which clip leads
        moves the denominator while the frame-count multiplier stays put. The
        pair below straddles the half-frame threshold: 29 fps leading tolerates
        the 28 fps clip at 0.483 frames of drift, 28 fps leading rejects the same
        pair at exactly 0.500. Nothing is deleted and no artifact moves -- which
        is why this has to be checked before the arrangement is committed rather
        than discovered when a read later fails.

        The values are derived rather than chosen: the interval that straddles is
        only half a frame wide, so an arbitrary "obviously different" pair (30 vs
        10 fps) fails both ways and would prove nothing.
        """
        faster = _row(name="faster.mp4", order=0, fps=29.0, frame_count=14)
        slower = _row(name="slower.mp4", order=1, fps=28.0, frame_count=14)
        rows = [faster, slower]

        as_committed = camera_uniformity(rows)
        reordered = camera_uniformity(
            rows, order_by_name={"slower.mp4": 0, "faster.mp4": 1}
        )

        assert as_committed.readable, "the committed order should open"
        assert not reordered.readable, "leading with the slower clip should not"
        assert reordered.mismatch is not None
        assert reordered.mismatch.field == "fps"


class TestDisplayedRatherThanCoded:
    def test_a_quarter_turn_is_compared_after_the_swap(self) -> None:
        """The trap a coded comparison falls into.

        An upright 64x48 clip and a quarter-turned one whose *coded* size is also
        64x48 are uniform on the coded numbers and transposed on the displayed
        ones. The reader autorotates, so it refuses them; a precheck reading the
        index's flat width/height cells would pass them.
        """
        verdict = camera_uniformity(
            [
                _row(name="a.mp4", order=0),
                _row(name="b.mp4", order=1, rotation=90),
            ]
        )
        assert not verdict.readable, "the rotation swap was not applied"


class TestEvidence:
    def test_a_row_without_facts_is_named_rather_than_skipped(self) -> None:
        """Agreement over rows that were skipped is unknown, not agreement."""
        verdict = camera_uniformity(
            [_row(name="a.mp4", order=0), _row(name="b.mp4", order=1, facts=False)]
        )
        assert verdict.readable
        assert not verdict.established
        assert verdict.unmeasured == ("b.mp4",)

    def test_a_single_clip_is_trivially_readable(self) -> None:
        verdict = camera_uniformity([_row(name="a.mp4", order=0)])
        assert verdict.readable and verdict.established


class TestTheDatasetLayer:
    def test_only_cameras_with_something_to_report_appear(self, tmp_path: Path) -> None:
        """An empty mapping is the useful predicate: uniform and fully evidenced."""
        from mosaic.core.dataset import Dataset, new_dataset_manifest
        from mosaic.core.pipeline.media_index import (
            frame_from_rows,
            write_media_index_rows,
        )

        manifest = new_dataset_manifest(name="uni", base_dir=tmp_path / "dataset")
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        index_path = ds.get_root("media_raw") / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        write_media_index_rows(
            index_path,
            frame_from_rows([_row(name="a.mp4", order=0), _row(name="b.mp4", order=1)]),
        )

        assert ds.sequence_uniformity("", "seq") == {}

    def test_a_mismatched_camera_is_reported_under_its_name(
        self, tmp_path: Path
    ) -> None:
        from mosaic.core.dataset import Dataset, new_dataset_manifest
        from mosaic.core.pipeline.media_index import (
            frame_from_rows,
            write_media_index_rows,
        )

        manifest = new_dataset_manifest(name="uni", base_dir=tmp_path / "dataset")
        ds = Dataset(manifest_path=manifest).load(ensure_roots=True)
        index_path = ds.get_root("media_raw") / "index.csv"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        left = _row(name="a.mp4", order=0)
        right = _row(name="b.mp4", order=1, width=128)
        for row in (left, right):
            row["camera"] = "cam0"
        write_media_index_rows(index_path, frame_from_rows([left, right]))

        reported = ds.sequence_uniformity("", "seq")

        assert set(reported) == {"cam0"}
        assert reported["cam0"].mismatch is not None
