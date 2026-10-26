"""Stand in for Lightning Pose in the tests of its tracker and frame adapter.

``run_litpose`` calls Lightning Pose through one module-level seam in
``litpose/dataset_runs.py``, its one inference phase. :func:`install_fake_litpose`
replaces it with a :class:`FakeLitpose`, whose prediction writes a small
DeepLabCut CSV, the layout that Lightning Pose exports. A test runs the whole
tracker protocol without a model or a GPU. :func:`write_litpose_model`
writes the model directory that a run resolves.

:func:`install_fake_litpose_frames` stands in for the Lightning Pose interpreter
that runs the two snippets of a Lightning Pose frame adapter: the version probe
and the single-frame prediction. Its :class:`FakeLitposeFrames` answers each
request image by position, in reverse request order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import pytest
from pydantic import TypeAdapter

from mosaic.tracking.litpose.frames import LitposeFrameAnswer, LitposeFrameResult
from mosaic.tracking.litpose.run import LitposePredictResult

from tests.helpers.tracks import write_dlc_csv

_BODYPARTS: tuple[str, ...] = ("nose", "tail")
"""The keypoints that :class:`FakeLitpose` reports.

The model directory of :func:`write_litpose_model` names them too.
"""

_REQUEST: Final = TypeAdapter(list[tuple[int, str]])
"""The request of the prediction snippet: ``[position, path]`` per image."""


@dataclass
class FakeLitpose:
    """Stand in for the Lightning Pose inference phase, recording each call."""

    predicted: list[Path] = field(default_factory=list)
    frames: int = 6
    written: list[npt.NDArray[np.float64]] = field(default_factory=list)
    """The values that each prediction wrote, in call order.

    Each entry contains the ``[x, y, likelihood]`` values, shaped
    ``(frame, bodypart, 3)``.
    """

    def predict(
        self, video_path: Path, out_csv: Path, **_kwargs: object
    ) -> LitposePredictResult:
        self.predicted.append(Path(video_path))
        out = Path(out_csv)
        self.written.append(
            write_dlc_csv(
                out, _BODYPARTS, n_frames=self.frames, scorer="heatmap_tracker"
            )
        )
        return LitposePredictResult(csv_path=out, stdout="", stderr="")


def install_fake_litpose(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLitpose | None = None
) -> FakeLitpose:
    """Replace the tracker's inference seam with *fake*, or a new fake."""
    import mosaic.tracking.litpose.dataset_runs as dataset_runs

    installed = fake if fake is not None else FakeLitpose()
    monkeypatch.setattr(dataset_runs, "run_litpose_predict", installed.predict)
    return installed


def write_litpose_model(
    model_dir: Path,
    *,
    weights: bytes = b"weights",
    model_type: str = "heatmap",
    keypoint_names: Sequence[str] = _BODYPARTS,
) -> Path:
    """Write a minimal Lightning Pose model directory, and return it.

    The directory contains a ``config.yaml`` declaring *model_type* and naming
    *keypoint_names*, by default the two keypoints of :class:`FakeLitpose`, and
    one checkpoint with *weights*. The model's identity is a digest of both
    files' bytes. Two directories written with the same arguments are one model.

    The config has no ``data`` section when *keypoint_names* is empty, which is
    the config a pinned digest was taken over.
    """
    checkpoint = model_dir / "tb_logs" / "m" / "version_0" / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _ = checkpoint.write_bytes(weights)
    config = f"model:\n  model_type: {model_type}\n"
    if keypoint_names:
        config += f"data:\n  keypoint_names: [{', '.join(keypoint_names)}]\n"
    _ = (model_dir / "config.yaml").write_text(config)
    return model_dir


type LitposePointsFor = Callable[[Path], Sequence[tuple[float, float, float]]]
"""Return ``(x, y, confidence)`` of each keypoint that the fake predicts in an image."""

type LitposeRewrite = Callable[[list[LitposeFrameResult]], list[LitposeFrameResult]]
"""Return the results that the fake writes, given the results it predicted."""


@dataclass(frozen=True, slots=True)
class LitposeSnippetRun:
    """One run of a Lightning Pose snippet that :class:`FakeLitposeFrames` answered.

    Attributes:
        source: The snippet's source.
        argv: The arguments after the source. The version probe takes none.
        env: The environment of the snippet's process.
        on_activity: The callback of both output streams.
        on_output: The callback of standard output alone.
    """

    source: str
    argv: tuple[str, ...]
    env: dict[str, str] | None
    on_activity: Callable[[str], None] | None
    on_output: Callable[[str], None] | None

    @property
    def request(self) -> list[tuple[int, str]]:
        """The ``[position, path]`` pairs of the prediction's request file."""
        return _REQUEST.validate_json(Path(self.argv[1]).read_bytes())


@dataclass
class FakeLitposeFrames:
    """Stand in for the Lightning Pose interpreter of a Lightning Pose frame adapter.

    The version probe prints ``version``. The prediction snippet reads its
    request, predicts ``points(path)`` in each image, prints one line on its
    ``on_activity`` and writes one result per image, in reverse request order,
    with ``keypoint_names``.

    Attributes:
        version: The Lightning Pose version that both snippets print.
        keypoint_names: The keypoint names of the model's config.
        points: The keypoints in an image. ``None`` predicts keypoint ``k`` at
            ``(10 + k, 20 + k)`` with confidence 0.5.
        rewrite: Rewrites the results before they are written. ``None``
            writes them as predicted.
        guard_exits: Whether the prediction snippet's version guard exits,
            printing ``version``, with the code that its argv names.
        returncode: The exit status of a prediction that fails, 0 for none.
        during: Called with each prediction run before it answers.
        runs: Every snippet run, in order.
    """

    version: str = "2.3.1"
    keypoint_names: tuple[str, ...] = _BODYPARTS
    points: LitposePointsFor | None = None
    rewrite: LitposeRewrite | None = None
    guard_exits: bool = False
    returncode: int = 0
    during: Callable[[LitposeSnippetRun], None] | None = None
    runs: list[LitposeSnippetRun] = field(default_factory=list[LitposeSnippetRun])

    @property
    def probes(self) -> list[LitposeSnippetRun]:
        """Every run of the version probe, in order."""
        return [run for run in self.runs if not run.argv]

    @property
    def predictions(self) -> list[LitposeSnippetRun]:
        """Every run of the prediction snippet, in order."""
        return [run for run in self.runs if run.argv]

    def invocation(self, env: object, *, executable: str) -> list[str]:
        """Return the fake interpreter's argv prefix."""
        _ = env
        return [f"/fake/litpose/bin/{executable}"]

    def run(
        self,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        on_activity: Callable[[str], None] | None = None,
        on_output: Callable[[str], None] | None = None,
        **_kwargs: object,
    ) -> tuple[str, str, int]:
        """Record the run, and answer as the snippet that *cmd* runs."""
        run = LitposeSnippetRun(
            source=cmd[2],
            argv=tuple(cmd[3:]),
            env=env,
            on_activity=on_activity,
            on_output=on_output,
        )
        self.runs.append(run)
        if not run.argv:
            return (f"{self.version}\n", "", 0)
        return self._predict(run)

    def _predict(self, run: LitposeSnippetRun) -> tuple[str, str, int]:
        answer, too_old = run.argv[2], run.argv[4]
        if self.guard_exits:
            return (f"{self.version}\n", "", int(too_old))
        if self.during is not None:
            self.during(run)
        if self.returncode:
            return ("", "the snippet failed", self.returncode)
        request = run.request
        results = [
            self._result(position, Path(path)) for position, path in reversed(request)
        ]
        if self.rewrite is not None:
            results = self.rewrite(results)
        if run.on_activity is not None:
            run.on_activity(f"predicted {len(request)} images")
        written = LitposeFrameAnswer(
            keypoint_names=list(self.keypoint_names), results=results
        )
        _ = Path(answer).write_text(written.model_dump_json())
        return ("", "", 0)

    def _result(self, position: int, path: Path) -> LitposeFrameResult:
        points = (
            self.points(path)
            if self.points is not None
            else [(10.0 + k, 20.0 + k, 0.5) for k in range(len(self.keypoint_names))]
        )
        return LitposeFrameResult(
            position=position,
            keypoints=[(x, y) for x, y, _confidence in points],
            confidence=[confidence for _x, _y, confidence in points],
        )


def install_fake_litpose_frames(
    monkeypatch: pytest.MonkeyPatch, fake: FakeLitposeFrames | None = None
) -> FakeLitposeFrames:
    """Replace the interpreter of the Lightning Pose frame snippets with *fake*.

    Without *fake*, a new :class:`FakeLitposeFrames` is installed and returned.
    """
    import mosaic.tracking.litpose.frames as litpose_frames

    installed = fake if fake is not None else FakeLitposeFrames()
    monkeypatch.setattr(litpose_frames, "tool_invocation", installed.invocation)
    monkeypatch.setattr(litpose_frames, "run_supervised", installed.run)
    return installed
