"""SLEAP inference runs ``sleap-nn track``, and nothing about it is left implicit.

The ``sleap`` distribution ships two inference scripts, ``sleap-track`` and
``sleap-nn-track``, and on sleap-nn 0.3.3 both fail: each imports
``sleap_nn.predict``, which sleap-nn removed, and reports the ``ImportError`` as
"sleap-nn is not installed" while exiting 0. ``sleap-nn`` itself ships with the
modules it imports. ``sleap-nn-track`` is the near-miss worth a test of its own:
it is present, it runs, and it fails with a message about a missing package.

Most of the argv carries over from the legacy CLI. What does not -- the
subcommand, the video named by ``-i``, the device spelling and every tracking
option -- is what the benchmark that motivated this never exercised, so those are
pinned hardest here. ``_sleap_invocation`` and ``_run_sleap`` are stubbed, which
asserts *which script is asked for* without a SLEAP install to locate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from mosaic.tracking.sleap import run as sleap_run
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.sleap.run import (
    SleapFeatures,
    SleapScoringMethod,
    run_sleap_convert,
    run_sleap_track,
    sleap_track_device_args,
)

_TRACKING_OPTIONS = (
    "-t",
    "--use_flow",
    "--candidates_method",
    "--features",
    "--scoring_method",
    "--track_matching_method",
    "--tracking_window_size",
    "--max_tracks",
)


@dataclass
class Launched:
    """What the stubbed launcher was asked to run."""

    script: str = ""
    args: list[str] = field(default_factory=list)


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> Launched:
    """Record the script and argv of each SLEAP launch, and write its ``-o``."""
    seen = Launched()

    def fake_invocation(script: str, **_kwargs: object) -> list[str]:
        seen.script = script
        return [f"/fake/bin/{script}"]

    def fake_run(
        _invocation: list[str], args: list[str], **_kwargs: object
    ) -> tuple[str, str]:
        seen.args = list(args)
        Path(args[args.index("-o") + 1]).write_bytes(b"")
        return ("", "")

    monkeypatch.setattr(sleap_run, "_sleap_invocation", fake_invocation)
    monkeypatch.setattr(sleap_run, "_run_sleap", fake_run)
    return seen


def _value_of(args: list[str], option: str) -> str:
    """The token following *option*, which must appear exactly once."""
    assert args.count(option) == 1, f"{option} appears {args.count(option)} times"
    return args[args.index(option) + 1]


def _assert_no_legacy_namespace(args: list[str]) -> None:
    assert not [a for a in args if a.startswith("--tracking.")], args


# --- which script ------------------------------------------------------------


def test_inference_asks_for_sleap_nns_own_cli(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(tmp_path / "v.mp4", tmp_path / "out.slp", model_paths=["m"])

    assert launched.script == "sleap-nn", (
        "not sleap-nn-track, the sleap wrapper that imports a module sleap-nn "
        "removed, and not the legacy sleap-track, which imports it too"
    )
    assert launched.args[0] == "track", "sleap-nn dispatches on a subcommand"


def test_the_export_still_asks_for_sleap_convert(
    tmp_path: Path, launched: Launched
) -> None:
    """``sleap-convert`` imports no sleap_nn module and works on 0.3.3.

    sleap-nn has no export writing the analysis-HDF5 layout the bridge reads, so
    moving this to "finish the migration" would break the half that works.
    """
    slp, h5 = tmp_path / "v.slp", tmp_path / "v.analysis.h5"

    run_sleap_convert(slp, h5)

    assert launched.script == "sleap-convert"
    assert launched.args == [str(slp), "--format", "analysis", "-o", str(h5)]


# --- the argv that carries over, and the part that does not ---------------------


def test_the_video_is_named_by_flag_and_never_positionally(
    tmp_path: Path, launched: Launched
) -> None:
    """A positional path reaches ``sleap-nn track`` as an unexpected argument."""
    video, out = tmp_path / "v.mp4", tmp_path / "out.slp"

    run_sleap_track(video, out, model_paths=["m"])

    assert launched.args[1:5] == ["-i", str(video), "-o", str(out)]
    assert launched.args.count(str(video)) == 1


def test_an_ordered_model_pair_survives_in_order(
    tmp_path: Path, launched: Launched
) -> None:
    """Top-down reads the centroid model first; reversed, it scores nothing."""
    centroid, instance = tmp_path / "centroid", tmp_path / "centered"

    run_sleap_track(
        tmp_path / "v.mp4", tmp_path / "out.slp", model_paths=[centroid, instance]
    )

    models = [launched.args[i + 1] for i, a in enumerate(launched.args) if a == "-m"]
    assert models == [str(centroid), str(instance)]


def test_peak_threshold_and_batch_size_always_reach_the_tool(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(
        tmp_path / "v.mp4",
        tmp_path / "out.slp",
        model_paths=["m"],
        peak_threshold=0.3,
        batch_size=8,
    )

    assert _value_of(launched.args, "--peak_threshold") == "0.3"
    assert _value_of(launched.args, "--batch_size") == "8"


def test_max_instances_and_frames_reach_the_tool_when_given(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(
        tmp_path / "v.mp4",
        tmp_path / "out.slp",
        model_paths=["m"],
        max_instances=3,
        frames="0-99",
    )

    assert _value_of(launched.args, "-n") == "3"
    assert _value_of(launched.args, "--frames") == "0-99"


def test_max_instances_and_frames_are_absent_when_not_given(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(tmp_path / "v.mp4", tmp_path / "out.slp", model_paths=["m"])

    assert "-n" not in launched.args
    assert "--frames" not in launched.args


# --- tracking ----------------------------------------------------------------


def test_tracking_off_sends_no_tracking_option(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(
        tmp_path / "v.mp4", tmp_path / "out.slp", model_paths=["m"], tracking=False
    )

    assert not [a for a in launched.args if a in _TRACKING_OPTIONS], launched.args
    _assert_no_legacy_namespace(launched.args)


def test_the_default_tracker_is_spelled_out_in_full(
    tmp_path: Path, launched: Launched
) -> None:
    """Every option is sent, so no sleap-nn default can move under a run_id.

    sleap-nn's ``track`` and ``predict`` already disagree on the default
    ``--candidates_method``. These defaults are what mosaic's legacy defaults ran
    under SLEAP 1.6: the flow tracker, keypoint OKS, hungarian, five frames.
    """
    run_sleap_track(tmp_path / "v.mp4", tmp_path / "out.slp", model_paths=["m"])

    tracking = launched.args[launched.args.index("-t") :]
    assert tracking == [
        "-t",
        "--use_flow",
        "--candidates_method",
        "fixed_window",
        "--features",
        "keypoints",
        "--scoring_method",
        "oks",
        "--track_matching_method",
        "hungarian",
        "--tracking_window_size",
        "5",
    ]


@pytest.mark.parametrize(
    ("features", "scoring_method"),
    [
        ("keypoints", "oks"),
        ("centroids", "euclidean_dist"),
        ("bboxes", "iou"),
        ("image", "cosine_sim"),
    ],
)
def test_each_tracking_option_reaches_the_tool_as_given(
    tmp_path: Path,
    launched: Launched,
    features: SleapFeatures,
    scoring_method: SleapScoringMethod,
) -> None:
    run_sleap_track(
        tmp_path / "v.mp4",
        tmp_path / "out.slp",
        model_paths=["m"],
        use_flow=False,
        features=features,
        scoring_method=scoring_method,
        track_matching_method="greedy",
        tracking_window_size=12,
    )

    args = launched.args
    assert "-t" in args
    assert "--use_flow" not in args
    assert _value_of(args, "--features") == features
    assert _value_of(args, "--scoring_method") == scoring_method
    assert _value_of(args, "--track_matching_method") == "greedy"
    assert _value_of(args, "--tracking_window_size") == "12"
    assert "--max_tracks" not in args
    _assert_no_legacy_namespace(args)


def test_a_track_cap_is_sent_with_local_queues(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(
        tmp_path / "v.mp4",
        tmp_path / "out.slp",
        model_paths=["m"],
        candidates_method="local_queues",
        max_tracks=4,
    )

    assert _value_of(launched.args, "--candidates_method") == "local_queues"
    assert _value_of(launched.args, "--max_tracks") == "4"


def test_extra_settings_are_sleap_nn_track_options_appended_last(
    tmp_path: Path, launched: Launched
) -> None:
    run_sleap_track(
        tmp_path / "v.mp4",
        tmp_path / "out.slp",
        model_paths=["m"],
        extra_settings={"of_img_scale": 0.5, "use_kalman": True, "gui": False},
    )

    assert launched.args[-3:] == ["--of_img_scale", "0.5", "--use_kalman"]


# --- the device ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("0", ["-d", "cuda:0"]),
        ("1", ["-d", "cuda:1"]),
        ("cuda:1", ["-d", "cuda:1"]),
        ("cuda", ["-d", "cuda"]),
        ("cpu", ["-d", "cpu"]),
        ("mps", ["-d", "mps"]),
        ("auto", []),
        ("", []),
        (None, []),
    ],
)
def test_the_device_is_respelled_not_dropped(
    device: str | None, expected: list[str]
) -> None:
    """mosaic spells a device as a family or a CUDA index; sleap-nn wants a torch
    device. Sent through bare, ``"0"`` selects nothing, and a run asked for GPU 1
    would land wherever torch chose. ``"cuda"`` is a request for CUDA, not a
    preference: it fails where CUDA is absent rather than falling back."""
    assert sleap_track_device_args(device) == expected


def test_the_device_reaches_the_tool(tmp_path: Path, launched: Launched) -> None:
    run_sleap_track(
        tmp_path / "v.mp4", tmp_path / "out.slp", model_paths=["m"], device="0"
    )

    assert _value_of(launched.args, "-d") == "cuda:0"


@pytest.mark.parametrize("device", ["0,1", "gpu0", "cuda:", "gpu"])
def test_an_unusable_device_is_refused_at_validation(device: str) -> None:
    """Refused at submit, not on a GPU node once the job is scheduled."""
    with pytest.raises(ValidationError, match="unusable device"):
        SleapParams(model_paths=["m"], device=device)


# --- params the tool would not honor as stated ------------------------------------


def test_a_track_cap_without_local_queues_is_refused() -> None:
    """sleap-nn enforces a cap only through local_queues; under fixed_window one
    release drops it and another switches candidates itself."""
    with pytest.raises(ValidationError, match="local_queues"):
        SleapParams(model_paths=["m"], max_tracks=4)


@pytest.mark.parametrize(
    "legacy",
    [
        {"tracker": "flow"},
        {"similarity": "instance"},
        {"match": "hungarian"},
        {"track_window": 5},
        {"max_tracking": 4},
    ],
)
def test_a_legacy_tracking_field_is_refused_rather_than_ignored(
    legacy: dict[str, object],
) -> None:
    """The legacy names had no faithful one-to-one modern meaning; accepting one
    and dropping it would run a tracker other than the one asked for."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SleapParams.model_validate({"model_paths": ["m"], **legacy})


def test_centroid_is_not_a_feature() -> None:
    """SLEAP 1.6 recognized ``centroids`` and silently scored ``centroid`` by
    keypoint OKS. The closed set has no ``centroid`` to mis-map."""
    with pytest.raises(ValidationError, match="centroids"):
        SleapParams.model_validate({"model_paths": ["m"], "features": "centroid"})
