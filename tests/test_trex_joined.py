"""Correcting a joined conversion's time, and dropping what one rate spoiled.

TRex converts a session's clips into one ``.pv`` but times all of them at the
first clip's rate, because ``VideoSource`` reads ``_framerate`` from
``_files_in_seq.front()`` and never checks the others. For a real session
measuring 30 / 29.95 / 31 fps that is about 3% wrong for most of the recording,
and mosaic's converter *prefers* what TRex exported -- so the error reaches
``tracks/`` unless the bridge intervenes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from mosaic_media import MediaFacts

from mosaic.core.media.timeline import concatenated_timeline
from mosaic.tracking.trex.joined import retime_joined_frame

from tests.test_media_timeline import _facts

# Clip length matters here, and is not incidental. `rate_uniform` measures
# *accumulated* drift -- |other - first| / first, scaled by the shorter clip's
# frame count -- so 30 beside 31 fps is inside the half-frame allowance for ten
# frames and far outside it for three hundred. Real clips are minutes long; a
# ten-frame fixture would read as uniform and quietly test nothing.
CLIP = 300
SESSION = [
    _facts(fps=30.0, frame_count=CLIP),
    _facts(fps=31.0, frame_count=CLIP),
]
UNIFORM = [_facts(fps=30.0, frame_count=CLIP), _facts(fps=30.0, frame_count=CLIP)]


def _export(n: int = 2 * CLIP) -> pd.DataFrame:
    """What the trex_npz converter hands the bridge, with TRex's own timing."""
    frames = np.arange(n)
    return pd.DataFrame(
        {
            "frame": frames,
            # As TRex computed them: index over the FIRST clip's rate.
            "time": frames / 30.0,
            "timestamp": frames * (1e6 / 30.0),
            "frame_rate": np.full(n, 30.0),
            "X": np.linspace(0, 1, n),
            "Y": np.linspace(1, 0, n),
            "X#wcentroid": np.linspace(0, 1, n),
            "ANGLE": np.zeros(n),
            "SPEED": np.ones(n),
            "SPEED#wcentroid": np.ones(n),
            "ANGULAR_V": np.ones(n),
            "VX": np.ones(n),
            "poseX0": np.zeros(n),
        }
    )


class TestTime:
    def test_each_clip_is_timed_at_its_own_rate(self) -> None:
        out = retime_joined_frame(_export(), concatenated_timeline(SESSION))
        # Clip 0 is unchanged; clip 1 runs at 31 fps from where clip 0 ended.
        assert out["time"].iloc[0] == pytest.approx(0.0)
        assert out["time"].iloc[CLIP - 1] == pytest.approx((CLIP - 1) / 30.0)
        assert out["time"].iloc[CLIP] == pytest.approx(CLIP / 30.0)
        assert out["time"].iloc[-1] == pytest.approx(CLIP / 30.0 + (CLIP - 1) / 31.0)

    def test_it_disagrees_with_what_trex_exported(self) -> None:
        """The whole point: TRex's own column is wrong past the first clip."""
        exported = _export()
        out = retime_joined_frame(exported, concatenated_timeline(SESSION))
        assert out["time"].iloc[-1] != pytest.approx(exported["time"].iloc[-1])

    def test_frame_is_left_alone(self) -> None:
        """VideoSource sums the clip lengths, so the global index is already right."""
        exported = _export()
        out = retime_joined_frame(exported, concatenated_timeline(SESSION))
        assert list(out["frame"]) == list(exported["frame"])

    def test_frame_rate_becomes_the_rate_in_force(self) -> None:
        out = retime_joined_frame(_export(), concatenated_timeline(SESSION))
        assert set(out["frame_rate"].iloc[:CLIP]) == {30.0}
        assert set(out["frame_rate"].iloc[CLIP:]) == {31.0}

    def test_the_synthesised_timestamp_is_dropped_not_recomputed(self) -> None:
        """TRex minted it from the index and one rate; mosaic measured nothing."""
        out = retime_joined_frame(_export(), concatenated_timeline(SESSION))
        assert "timestamp" not in out.columns


class TestWhatARateSpoiled:
    def test_per_second_quantities_go(self) -> None:
        out = retime_joined_frame(_export(), concatenated_timeline(SESSION))
        for column in ("SPEED", "SPEED#wcentroid", "ANGULAR_V", "VX"):
            assert column not in out.columns

    def test_positions_and_angles_stay(self) -> None:
        """Neither depends on a rate, and X#wcentroid is where X comes from."""
        out = retime_joined_frame(_export(), concatenated_timeline(SESSION))
        for column in ("X", "Y", "X#wcentroid", "ANGLE", "poseX0"):
            assert column in out.columns

    def test_a_uniform_session_keeps_everything(self) -> None:
        """Nothing was wrong with it, so nothing is taken away."""
        out = retime_joined_frame(_export(), concatenated_timeline(UNIFORM))
        for column in ("SPEED", "ANGULAR_V", "VX"):
            assert column in out.columns

    def test_a_uniform_session_is_still_retimed(self) -> None:
        """Two clips is a concatenation even when one rate indexes both."""
        out = retime_joined_frame(_export(), concatenated_timeline(UNIFORM))
        assert out["time"].iloc[-1] == pytest.approx((2 * CLIP - 1) / 30.0)
        assert "timestamp" not in out.columns


class TestOneClip:
    def test_a_single_segment_timeline_changes_nothing(self) -> None:
        exported = _export(CLIP)
        timeline = concatenated_timeline([_facts(fps=30.0, frame_count=CLIP)])
        out = retime_joined_frame(exported, timeline)
        pd.testing.assert_frame_equal(out, exported)

    def test_a_frame_without_the_column_is_returned_as_is(self) -> None:
        frame = pd.DataFrame({"X": [1.0, 2.0]})
        out = retime_joined_frame(frame, concatenated_timeline(SESSION))
        pd.testing.assert_frame_equal(out, frame)


def test_the_facts_helper_is_the_shared_one() -> None:
    """Guards the cross-module import this file leans on."""
    assert isinstance(_facts(), MediaFacts)
