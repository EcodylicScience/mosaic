"""Stand in for SLEAP in the tests of the SLEAP tracker, frame adapter and preparation.

``run_sleap`` calls SLEAP through two module-level seams in
``sleap/dataset_runs.py``: ``sleap-nn track`` and ``sleap-convert``.
:func:`install_fake_sleap` replaces both with a :class:`FakeSleap`, whose
inference writes a ``.slp`` and whose export writes a small, converter-readable
analysis ``.h5``. A test runs the whole tracker protocol without a model.
:func:`write_sleap_model` writes the model directory that a run names.
:func:`record_slp_writes` stands in for the labels writer of a sleap
preparation.

:func:`install_fake_sleap_frames` stands in for the SLEAP environment's probe and
for the three calls of a SLEAP frame adapter: the frames-only ``.slp`` writer,
``sleap-nn track`` and the readback of its predictions. Its
:class:`FakeSleapFrames` predicts on the images of the frames-only input and
reports them in reverse order, as sleap-io's grouping by image shape can
reorder them. Like sleap-nn 0.2.0 and 0.3.1, it reports every image of the input
as a frame, with no instances where it found none, and lists every image of the
input as an image of its videos.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from mosaic.core.annotations.model import AnnotationSet
from mosaic.tracking.sleap.frames import (
    SleapFrameReadback,
    SleapInstanceReadback,
    SleapReadback,
)
from mosaic.tracking.sleap.probe import SleapProbeResponse
from mosaic.tracking.sleap.run import SleapConvertResult, SleapTrackResult

from tests.helpers.tracks import write_sleap_analysis_h5


@dataclass
class FakeSleap:
    """Recording stand-ins for the two SLEAP phases."""

    tracked: list[Path] = field(default_factory=list)
    track_settings: list[dict[str, object]] = field(default_factory=list)
    converted: list[Path] = field(default_factory=list)
    frames: int = 6

    def track(
        self, video_path: Path, output_slp: Path, **kwargs: object
    ) -> SleapTrackResult:
        self.tracked.append(Path(video_path))
        self.track_settings.append(kwargs)
        out = Path(output_slp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"slp")
        return SleapTrackResult(slp_path=out, stdout="", stderr="")

    def convert(
        self, slp_path: Path, output_h5: Path, **_kwargs: object
    ) -> SleapConvertResult:
        self.converted.append(Path(slp_path))
        out = Path(output_h5)
        tracks = np.random.default_rng(0).random((self.frames, 1, 1, 2))
        write_sleap_analysis_h5(out, tracks)
        return SleapConvertResult(analysis_h5_path=out, stdout="", stderr="")


def install_fake_sleap(
    monkeypatch: pytest.MonkeyPatch, fake: FakeSleap | None = None
) -> FakeSleap:
    """Replace the tracker's two seams with *fake*, or a new :class:`FakeSleap`."""
    import mosaic.tracking.sleap.dataset_runs as dataset_runs

    installed = fake if fake is not None else FakeSleap()
    monkeypatch.setattr(dataset_runs, "run_sleap_track", installed.track)
    monkeypatch.setattr(dataset_runs, "run_sleap_convert", installed.convert)
    return installed


def write_sleap_model(
    directory: Path, weights: bytes = b"weights", *, head: str = ""
) -> Path:
    """Write a SLEAP model directory at *directory*, and return it.

    The directory contains ``best.ckpt`` with *weights*, whose digest is the model's
    identity. Two directories with the same *weights* are one model. When *head* is
    given, a ``training_config.yaml`` selecting it under ``head_configs`` is written
    beside the checkpoint, as SLEAP's trainer leaves one. The config is provenance,
    read for the model type, and a model does not need one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _ = (directory / "best.ckpt").write_bytes(weights)
    if head:
        _ = (directory / "training_config.yaml").write_text(
            f"head_configs:\n  {head}: {{}}\n"
        )
    return directory


SlpWrite = tuple[Path, AnnotationSet, object]
"""One ``write_slp`` call: the file, the set and the ``images_dir`` it was given."""


def record_slp_writes(monkeypatch: pytest.MonkeyPatch) -> list[SlpWrite]:
    """Stand in for ``write_slp``, which runs in the SLEAP environment.

    Each call of a sleap preparation is recorded in the returned list and writes
    a placeholder file.
    """
    calls: list[SlpWrite] = []

    def fake_write_slp(
        annotations: AnnotationSet, out_path: Path, **kwargs: object
    ) -> Path:
        calls.append((Path(out_path), annotations, kwargs.get("images_dir")))
        _ = Path(out_path).write_bytes(b"slp")
        return Path(out_path)

    monkeypatch.setattr("mosaic.tracking.sleap.labels.write_slp", fake_write_slp)
    return calls


def sleap_instance(
    points: Sequence[tuple[float, float, float]],
    *,
    score: float = 0.9,
    track: str | None = None,
) -> SleapInstanceReadback:
    """Return an instance of the readback, with ``(x, y, score)`` per node."""
    return SleapInstanceReadback(
        score=score, track=track, points=[(x, y, s) for x, y, s in points]
    )


type SleapInstancesFor = Callable[[Path], list[SleapInstanceReadback]]
"""Return the instances that the fake predicts in the image at a resolved path."""

type SleapReportedAs = Callable[[Path], str | None]
"""Return the path that the fake reports for an image, or ``None`` to omit it."""


@dataclass(frozen=True, slots=True)
class SleapInput:
    """One frames-only ``.slp`` that a SLEAP frame adapter had written.

    Attributes:
        annotations: The set that was written.
        path: Where it was written.
    """

    annotations: AnnotationSet
    path: Path

    @property
    def images(self) -> list[Path]:
        """The image of each frame, resolved against the set's image root."""
        return [self.annotations.resolve(frame) for frame in self.annotations.frames]


@dataclass(frozen=True, slots=True)
class SleapTrackCall:
    """One ``sleap-nn track`` call of a SLEAP frame adapter.

    Attributes:
        input_slp: The ``.slp`` predicted on.
        output_slp: Where the predictions were written.
        settings: Every other keyword argument, by name.
        on_activity: The callback of both output streams.
        on_output: The callback of standard output alone.
    """

    input_slp: Path
    output_slp: Path
    settings: dict[str, object]
    on_activity: Callable[[str], None] | None
    on_output: Callable[[str], None] | None


@dataclass
class FakeSleapFrames:
    """Stand in for a SLEAP environment that predicts on extracted images.

    The fake ``sleap-nn track`` predicts ``instances(image)`` in each image of
    its input, and prints one line on its ``on_activity``. The readback lists
    each image under ``reported(image)`` as an image of the file's videos, and
    reports each image as a frame, in reverse input order.

    Attributes:
        nodes: The node names of the model's skeleton.
        edges: Its edges, as the readback reports them.
        symmetries: Its mirror pairs, as the readback reports them.
        instances: The instances in an image. ``None`` predicts one instance
            per image, node ``k`` at ``(10 + k, 20 + k)`` with score 0.5.
        reported: The path reported for an image, as a frame and as an image
            of the videos. ``None`` reports its path.
        empty_frames: Whether an image without an instance is reported as a
            frame with no instances. ``False`` omits its frame.
        version: The sleap-nn version that the probe reports.
        has_sleap_nn: Whether the probe finds sleap-nn.
        during: Called with each track call before it returns.
        probed: The labels path of each probe, in order.
        written: Every frames-only input, in order.
        tracked: Every track call, in order.
        read: Every predictions file read back, in order.
        answers: Every readback returned, in order.
    """

    nodes: tuple[str, ...] = ("head", "tail")
    edges: tuple[tuple[int, int], ...] = ((0, 1),)
    symmetries: tuple[tuple[int, int], ...] = ()
    instances: SleapInstancesFor | None = None
    reported: SleapReportedAs | None = None
    empty_frames: bool = True
    version: str = "0.3.1"
    has_sleap_nn: bool = True
    during: Callable[[SleapTrackCall], None] | None = None
    probed: list[Path | str | None] = field(default_factory=list[Path | str | None])
    written: list[SleapInput] = field(default_factory=list[SleapInput])
    tracked: list[SleapTrackCall] = field(default_factory=list[SleapTrackCall])
    read: list[Path] = field(default_factory=list[Path])
    answers: list[SleapReadback] = field(default_factory=list[SleapReadback])
    _images_of: dict[Path, list[Path]] = field(
        default_factory=dict[Path, list[Path]], init=False, repr=False
    )

    def probe(
        self, labels_path: Path | str | None = None, **_kwargs: object
    ) -> SleapProbeResponse:
        """Report sleap-nn's version, or that it is missing."""
        self.probed.append(labels_path)
        if not self.has_sleap_nn:
            return SleapProbeResponse(sleap_nn_import_error="ModuleNotFoundError")
        return SleapProbeResponse(has_sleap_nn=True, sleap_nn_version=self.version)

    def write_slp(
        self, annotations: AnnotationSet, out_path: Path, **_kwargs: object
    ) -> Path:
        """Record the frames-only input, and write a placeholder file."""
        path = Path(out_path)
        written = SleapInput(annotations=annotations, path=path)
        self.written.append(written)
        self._images_of[path] = written.images
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_bytes(b"frames")
        return path

    def track(
        self,
        video_path: Path,
        output_slp: Path,
        *,
        on_activity: Callable[[str], None] | None = None,
        on_output: Callable[[str], None] | None = None,
        **kwargs: object,
    ) -> SleapTrackResult:
        """Record the call, predict on the input's images, and print one line."""
        call = SleapTrackCall(
            input_slp=Path(video_path),
            output_slp=Path(output_slp),
            settings=dict(kwargs),
            on_activity=on_activity,
            on_output=on_output,
        )
        self.tracked.append(call)
        images = self._images_of[call.input_slp]
        self._images_of[call.output_slp] = images
        _ = call.output_slp.write_bytes(b"predicted")
        if self.during is not None:
            self.during(call)
        if on_activity is not None:
            on_activity(f"predicted {len(images)} frames")
        return SleapTrackResult(slp_path=call.output_slp, stdout="", stderr="")

    def readback(
        self, predicted: Path, answer: Path, **_kwargs: object
    ) -> SleapReadback:
        """Report the images of the predicted file and a frame for each image."""
        _ = answer
        self.read.append(Path(predicted))
        images: list[str] = []
        frames: list[SleapFrameReadback] = []
        for image in reversed(self._images_of[Path(predicted)]):
            reported = str(image) if self.reported is None else self.reported(image)
            if reported is None:
                continue
            images.append(reported)
            instances = self._instances_in(image)
            if instances or self.empty_frames:
                frames.append(SleapFrameReadback(image=reported, instances=instances))
        found = SleapReadback(
            nodes=list(self.nodes),
            edges=list(self.edges),
            symmetries=list(self.symmetries),
            images=images,
            frames=frames,
        )
        self.answers.append(found)
        return found

    def _instances_in(self, image: Path) -> list[SleapInstanceReadback]:
        if self.instances is not None:
            return self.instances(image)
        points = [(10.0 + k, 20.0 + k, 0.5) for k in range(len(self.nodes))]
        return [sleap_instance(points)]


def install_fake_sleap_frames(
    monkeypatch: pytest.MonkeyPatch, fake: FakeSleapFrames | None = None
) -> FakeSleapFrames:
    """Replace the SLEAP probe and the frame adapter's seams with *fake*.

    Without *fake*, a new :class:`FakeSleapFrames` is installed and returned.
    """
    import mosaic.tracking.sleap.frames as sleap_frames
    import mosaic.tracking.sleap.probe as sleap_probe

    installed = fake if fake is not None else FakeSleapFrames()
    monkeypatch.setattr(sleap_probe, "probe_sleap", installed.probe)
    monkeypatch.setattr(sleap_frames, "write_slp", installed.write_slp)
    monkeypatch.setattr(sleap_frames, "run_sleap_track", installed.track)
    monkeypatch.setattr(sleap_frames, "read_sleap_predictions", installed.readback)
    return installed
