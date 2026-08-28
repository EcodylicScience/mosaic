"""A multi-camera sequence must not overwrite its own inference output.

``Dataset.resolve_media_scope`` yields one entry per
``(group, sequence, camera)``. Both the trackers and the ``infer-*`` ops key a
working directory on ``(group, sequence)`` alone. Two cameras of one sequence
therefore meet in one directory. The trackers reduced the scope and reported the
drop. The inference ops did not, and the second camera replaced the first
without printing anything.

Read directly against the reduction, which is why it was extracted: it used to
be reachable only by running an inference job.
"""

import pytest

from mosaic.core.dataset import ResolvedMedia, ResolvedScopeEntry
from mosaic.tracking.common import scope as scope_module
from mosaic.tracking.common.scope import one_camera_per_entry
from mosaic.tracking.ops import infer as infer_module
from tests.helpers import names_called_by


def _entry(group: str, sequence: str, camera: str) -> ResolvedScopeEntry:
    """One resolved entry, with the smallest media value the dataclass takes."""
    return ResolvedScopeEntry(
        group=group,
        sequence=sequence,
        camera=camera,
        resolved=ResolvedMedia(paths=[], facts=[]),
    )


class TestCameraReduction:
    def test_a_second_camera_is_dropped_and_reported(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two cameras of one sequence resolve to one working directory."""
        kept = one_camera_per_entry(
            "infer-pose",
            [_entry("A", "one", "cam0"), _entry("A", "one", "cam1")],
        )
        assert kept == [_entry("A", "one", "cam0")]
        reported = capsys.readouterr().err
        assert "cam1" in reported
        assert "skipping" in reported

    def test_two_sequences_each_with_one_camera_both_survive(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        kept = one_camera_per_entry(
            "infer-pose",
            [_entry("A", "one", "cam0"), _entry("A", "two", "cam0")],
        )
        assert len(kept) == 2
        assert capsys.readouterr().err == ""

    def test_one_sequence_name_under_two_groups_survives(self) -> None:
        """The key is the pair, and a repeated sequence name is two entries."""
        kept = one_camera_per_entry(
            "infer-pose",
            [_entry("A", "one", ""), _entry("B", "one", "")],
        )
        assert len(kept) == 2

    def test_an_unnamed_camera_is_reported_readably(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Single-camera media carries an empty camera cell."""
        _ = one_camera_per_entry(
            "infer-pose", [_entry("A", "one", ""), _entry("A", "one", "")]
        )
        assert "<unnamed>" in capsys.readouterr().err

    def test_the_message_names_the_op_the_user_invoked(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The shared machinery reports under the caller's kind."""
        _ = one_camera_per_entry(
            "infer-points", [_entry("A", "one", "cam0"), _entry("A", "one", "cam1")]
        )
        assert "[infer-points]" in capsys.readouterr().err

    def test_the_arrival_order_is_kept(self) -> None:
        """The first camera of each entry wins, in the order the scope gave."""
        kept = one_camera_per_entry(
            "infer-pose",
            [
                _entry("B", "two", "cam0"),
                _entry("A", "one", "cam0"),
                _entry("B", "two", "cam1"),
            ],
        )
        assert [(item.group, item.sequence) for item in kept] == [
            ("B", "two"),
            ("A", "one"),
        ]

    def test_an_empty_scope_reduces_to_nothing(self) -> None:
        kept = one_camera_per_entry("infer-pose", [])
        assert kept == []


class TestBothCallersReduce:
    """The rule has one implementation, and both callers reach it.

    Read from the parsed body, which is the only thing that fails when a caller
    stops reducing. Every test above would still pass with the call deleted.
    """

    def test_the_tracker_loop_reduces(self) -> None:
        called = names_called_by(scope_module, "build_work_items")
        assert "one_camera_per_entry" in called

    def test_the_inference_loop_reduces(self) -> None:
        called = names_called_by(infer_module, "_run_inference_op")
        assert "one_camera_per_entry" in called
