"""Stand in for SLEAP in the tests of the SLEAP tracker, frame adapter and preparation.

``run_sleap`` calls SLEAP through module-level seams in ``sleap/dataset_runs.py``:
``sleap-nn track``, ``sleap-nn predict``, ``sleap-convert`` and the readback of
tracked instances. :func:`install_fake_sleap` replaces them with a
:class:`FakeSleap`, whose inference writes a ``.slp`` and whose export writes a
small, converter-readable analysis ``.h5``. A test runs the whole tracker
protocol without a model. Its ``predict`` stands in for ``run_sleap_predict``,
and writes a ``.slp`` or reports ``No detections to embed``. Its ``readback``
writes the ``.npz`` of an embedding run, one unit vector per frame.
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

:func:`write_appearance_readback` writes the ``.npz`` that the readback of
tracked instances publishes, from records that a test gives it.
:func:`write_stub_sleap_io` writes a stand-in ``sleap_io`` package whose
``load_slp`` reads a file of :func:`write_stub_slp` and whose ``save_slp`` writes
one that :func:`read_stub_labels` reads. A test then runs a snippet's source
against labels it describes.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pytest
import yaml
from pydantic import TypeAdapter

from mosaic.core.annotations.model import AnnotationSet
from mosaic.tracking.sleap.frames import (
    SleapFrameReadback,
    SleapInstanceReadback,
    SleapReadback,
)
from mosaic.tracking.sleap.probe import SleapProbeResponse
from mosaic.tracking.sleap.run import (
    SleapConvertResult,
    SleapNoDetections,
    SleapTrackResult,
)

from tests.helpers.tracks import write_sleap_analysis_h5


@dataclass
class FakeSleap:
    """Recording stand-ins for the SLEAP phases of a tracker run.

    The export holds one track, :data:`_FAKE_TRACK`, in every frame of
    :attr:`frames`, with every keypoint NaN in :attr:`nan_frames`. The readback
    of an embedding run records that track in every frame, with a unit vector
    of :attr:`dim` components, ``e_(frame mod dim)``, except in
    :attr:`unembedded_frames`, where the instance has no vector.
    """

    tracked: list[Path] = field(default_factory=list)
    track_settings: list[dict[str, object]] = field(default_factory=list)
    converted: list[Path] = field(default_factory=list)
    predicted: list[Path] = field(default_factory=list)
    predict_settings: list[dict[str, object]] = field(default_factory=list)
    read_back: list[Path] = field(default_factory=list[Path])
    detects: bool = True
    frames: int = 6
    nan_frames: tuple[int, ...] = ()
    unembedded_frames: tuple[int, ...] = ()
    dim: int = 128
    version: str = "0.3.5"
    probed: int = 0

    def probe(self, *_args: object, **_kwargs: object) -> SleapProbeResponse:
        """Count the call, and report sleap-nn at :attr:`version`."""
        self.probed += 1
        return SleapProbeResponse(has_sleap_nn=True, sleap_nn_version=self.version)

    def track(
        self, video_path: Path, output_slp: Path, **kwargs: object
    ) -> SleapTrackResult:
        self.tracked.append(Path(video_path))
        self.track_settings.append(kwargs)
        out = Path(output_slp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"slp")
        return SleapTrackResult(slp_path=out, stdout="", stderr="")

    def predict(
        self, video_path: Path, output_slp: Path, **kwargs: object
    ) -> SleapNoDetections | None:
        """Record the call, and write a ``.slp`` unless :attr:`detects` is false.

        Returns sleap-nn's report when the video had no detection, else
        ``None``, as ``run_sleap_predict`` does.
        """
        self.predicted.append(Path(video_path))
        self.predict_settings.append(kwargs)
        if not self.detects:
            return SleapNoDetections(
                message="No detections to embed",
                argv=("sleap-nn", "predict", "-i", str(video_path)),
            )
        out = Path(output_slp)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"slp")
        return None

    def convert(
        self, slp_path: Path, output_h5: Path, **_kwargs: object
    ) -> SleapConvertResult:
        self.converted.append(Path(slp_path))
        out = Path(output_h5)
        tracks = np.random.default_rng(0).random((self.frames, 1, 1, 2))
        tracks[list(self.nan_frames)] = np.nan
        write_sleap_analysis_h5(out, tracks, track_names=(_FAKE_TRACK,))
        return SleapConvertResult(analysis_h5_path=out, stdout="", stderr="")

    def readback(self, predicted: Path, out: Path, **_kwargs: object) -> Path:
        """Record the call, and write the readback of every frame to *out*."""
        self.read_back.append(Path(predicted))
        records = [
            ReadbackRecord(
                frame_idx=frame,
                track_name=_FAKE_TRACK,
                vector=(
                    None
                    if frame in self.unembedded_frames
                    else tuple(
                        float(component == frame % self.dim)
                        for component in range(self.dim)
                    )
                ),
            )
            for frame in range(self.frames)
        ]
        return write_appearance_readback(out, records, dim=self.dim)


_FAKE_TRACK: Final = "track_0"
"""The name of the one track that :class:`FakeSleap` exports and reads back."""


def install_fake_sleap(
    monkeypatch: pytest.MonkeyPatch, fake: FakeSleap | None = None
) -> FakeSleap:
    """Replace the tracker's SLEAP seams and the SLEAP probe with *fake*.

    The seams are ``sleap-nn track``, ``sleap-nn predict``, ``sleap-convert``
    and the readback of tracked instances. Without *fake*, a new
    :class:`FakeSleap` is installed and returned.
    """
    import mosaic.tracking.sleap.dataset_runs as dataset_runs
    import mosaic.tracking.sleap.probe as sleap_probe

    installed = fake if fake is not None else FakeSleap()
    monkeypatch.setattr(sleap_probe, "probe_sleap", installed.probe)
    monkeypatch.setattr(dataset_runs, "run_sleap_track", installed.track)
    monkeypatch.setattr(dataset_runs, "run_sleap_predict", installed.predict)
    monkeypatch.setattr(dataset_runs, "run_sleap_convert", installed.convert)
    monkeypatch.setattr(dataset_runs, "read_back_embeddings", installed.readback)
    return installed


_MERGED_HEADS: Final = (
    "single_instance",
    "centroid",
    "centered_instance",
    "bottomup",
    "multi_class_bottomup",
    "multi_class_topdown",
    "bottomup_segmentation",
    "centered_instance_segmentation",
    "semantic_segmentation",
    "embedding",
    "bbox_centernet",
    "bbox_ssd",
)
"""Every head that sleap-nn 0.3.5 lists under ``model_config.head_configs``, in order."""

_MERGED_NODES: Final = ("head", "centroid", "tail")
"""The skeleton of a merged config. Its middle node shares a head's name."""


def write_sleap_model(
    directory: Path,
    weights: bytes = b"weights",
    *,
    head: str = "",
    merged: bool = False,
    embedding_dim: object = 128,
) -> Path:
    """Write a SLEAP model directory at *directory*, and return it.

    The directory contains ``best.ckpt`` with *weights*, whose digest is the model's
    identity. Two directories with the same *weights* are one model. When *head* is
    given, a ``training_config.yaml`` selecting it under ``head_configs`` is written
    beside the checkpoint, as SLEAP's trainer leaves one. The config is provenance,
    read for the model type, and a model does not need one.

    With *merged*, the config is the merged one that ``sleap-nn train`` saves:
    ``model_config.head_configs`` lists every head of :data:`_MERGED_HEADS`, null
    but *head*, and ``data_config.skeletons`` names the nodes of
    :data:`_MERGED_NODES`. The section of *head* names those nodes and anchors on
    the ``centroid`` node, so the document's scalar values name a head that was not
    trained. A merged config without *head* configures no head. A merged
    ``embedding`` config declares *embedding_dim* as the vector length, and
    leaves the key out when it is ``None``.
    """
    directory.mkdir(parents=True, exist_ok=True)
    _ = (directory / "best.ckpt").write_bytes(weights)
    if merged:
        config = yaml.safe_dump(
            _merged_training_config(head, embedding_dim), sort_keys=False
        )
        _ = (directory / "training_config.yaml").write_text(config)
    elif head:
        _ = (directory / "training_config.yaml").write_text(
            f"head_configs:\n  {head}: {{}}\n"
        )
    return directory


def write_sleap_model_set(directory: Path, *heads: str) -> list[str]:
    """Write one merged model per head under *directory*, and return their paths.

    Each model has its own weights and the merged config of its head, in the
    order of *heads*. The list is a sleap op's ``model_paths``.
    """
    return [
        str(
            write_sleap_model(
                directory / f"{index}-{head}",
                f"{index}-{head}".encode(),
                head=head,
                merged=True,
            )
        )
        for index, head in enumerate(heads)
    ]


EMBEDDING_TRACKER_SETTINGS: Final[dict[str, object]] = {
    "features": "embeddings",
    "scoring_method": "cosine_sim",
    "candidates_method": "local_queues",
}
"""The tracker settings that a sleap run with an embedding model requires."""


def _merged_training_config(head: str, embedding_dim: object) -> dict[str, object]:
    """Return the merged training config of a sleap-nn model of *head*."""
    nodes = list(_MERGED_NODES)
    confmaps: dict[str, object] = {
        "part_names": nodes,
        "anchor_part": "centroid",
        "sigma": 5.0,
        "output_stride": 2,
    }
    if head == "embedding" and embedding_dim is not None:
        confmaps["embedding_dim"] = embedding_dim
    section = "embedding" if head == "embedding" else "confmaps"
    heads = {
        name: {section: confmaps} if name == head else None for name in _MERGED_HEADS
    }
    skeleton = {
        "nodes": [{"name": node} for node in nodes],
        "edges": [
            {"source": {"name": source}, "destination": {"name": target}}
            for source, target in zip(nodes, nodes[1:])
        ],
        "symmetries": [],
        "name": "skeleton",
    }
    return {
        "data_config": {"skeletons": [skeleton]},
        "model_config": {
            "backbone_config": {
                "unet": {"filters": 32},
                "convnext": None,
                "swint": None,
            },
            "head_configs": heads,
        },
        "trainer_config": {"max_epochs": 1},
        "sleap_nn_version": "0.3.5",
    }


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


@dataclass(frozen=True, slots=True)
class ReadbackRecord:
    """One tracked instance of a readback ``.npz``.

    Attributes:
        frame_idx: The instance's frame.
        track_name: The name of its track.
        is_user: Whether it is a user instance rather than a predicted one.
        vector: Its appearance vector, or ``None`` when it has none.
    """

    frame_idx: int
    track_name: str
    is_user: bool = False
    vector: tuple[float, ...] | None = None


def write_appearance_readback(
    path: Path, records: Sequence[ReadbackRecord], *, dim: int
) -> Path:
    """Write *records* to *path* as the readback of tracked instances writes them.

    A record without a vector has a row of *dim* zeros. A *dim* of 0 writes
    ``vectors`` of shape ``(n, 0)``, as a predictions file without embeddings
    does.

    Returns:
        *path*.
    """
    vectors = np.zeros((len(records), dim), dtype=np.float32)
    for row, record in enumerate(records):
        if record.vector is not None:
            vectors[row] = record.vector
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(
            handle,
            frame_idx=np.asarray([r.frame_idx for r in records], dtype=np.int64),
            track_name=np.asarray([r.track_name for r in records], dtype=np.str_),
            is_user=np.asarray([r.is_user for r in records], dtype=np.bool_),
            has_vector=np.asarray(
                [r.vector is not None for r in records], dtype=np.bool_
            ),
            vectors=vectors,
        )
    return path


@dataclass(frozen=True, slots=True)
class StubInstance:
    """One instance of a labeled frame in a file of :func:`write_stub_slp`.

    Attributes:
        track: The name of its track, or ``None`` for an instance without one.
        predicted: Whether it is a predicted instance rather than a user one.
        embedding: Its identity embedding, or ``None`` when it has none.
        x: The x coordinate of each of its points, which tells apart the
            instances of one slot. Every y coordinate is zero.
    """

    track: str | None
    predicted: bool = True
    embedding: tuple[float, ...] | None = None
    x: float = 0.0


StubFrame = tuple[int, Sequence[StubInstance]]
"""One labeled frame of a stub file: its frame index and its instances, in order."""

STUB_NODES: Final = ("head", "tail")
"""The skeleton of every stub file. Each instance has one point per node."""


def write_stub_slp(
    path: Path, frames: Sequence[StubFrame], *, video: str = "video.mp4"
) -> Path:
    """Write the labeled *frames* where the stub ``sleap_io.load_slp`` reads them.

    The file names one video, *video*, which holds every frame.

    Returns:
        *path*.
    """
    document = {
        "video": video,
        "frames": [
            {
                "frame_idx": frame_idx,
                "instances": [
                    {
                        "track": instance.track,
                        "predicted": instance.predicted,
                        "x": instance.x,
                        "embedding": (
                            None
                            if instance.embedding is None
                            else list(instance.embedding)
                        ),
                    }
                    for instance in instances
                ],
            }
            for frame_idx, instances in frames
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(json.dumps(document), encoding="utf-8")
    return path


class StubSavedInstance(TypedDict):
    """One instance of a file that the stub ``sleap_io.save_slp`` wrote."""

    track: str | None
    predicted: bool
    x: float


class StubSavedFrame(TypedDict):
    """One labeled frame of a saved stub file, its video given by position."""

    video: int
    frame_idx: int
    instances: list[StubSavedInstance]


class StubSavedLabels(TypedDict):
    """A file that the stub ``sleap_io.save_slp`` wrote.

    Attributes:
        embed: The ``embed`` argument of the save.
        videos: The file name of each video, in order.
        tracks: The name of each track, in order.
        frames: The labeled frames, in order.
    """

    embed: bool
    videos: list[str]
    tracks: list[str]
    frames: list[StubSavedFrame]


_SAVED_LABELS: Final = TypeAdapter(StubSavedLabels)


def read_stub_labels(path: Path) -> StubSavedLabels:
    """Read the file that the stub ``sleap_io.save_slp`` wrote at *path*."""
    return _SAVED_LABELS.validate_json(path.read_bytes())


_STUB_SLEAP_IO: Final = """
import json
import time

import numpy as np


class Track:
    def __init__(self, name):
        self.name = name


class Embedding:
    def __init__(self, vector):
        self.vector = np.asarray(vector, dtype=np.float32)


class Skeleton:
    def __init__(self, nodes):
        self.nodes = list(nodes)


class Video:
    def __init__(self, filename):
        self.filename = filename

    @classmethod
    def from_filename(cls, filename, keep_open=True):
        return cls(filename)


class Instance:
    def __init__(self, points, skeleton, track=None, identity_embedding=None):
        self.points = np.asarray(points, dtype=np.float64)
        self.skeleton = skeleton
        self.track = track
        self.identity_embedding = identity_embedding

    @classmethod
    def from_numpy(cls, points_data, skeleton, track=None):
        return cls(points_data, skeleton, track=track)

    def numpy(self):
        return self.points


class PredictedInstance(Instance):
    pass


class LabeledFrame:
    def __init__(self, video, frame_idx, instances):
        self.video = video
        self.frame_idx = frame_idx
        self.instances = instances


class Labels:
    def __init__(self, labeled_frames, videos=None, skeletons=None, tracks=None):
        self.labeled_frames = labeled_frames
        self.videos = list(videos or [])
        self.skeletons = list(skeletons or [])
        self.tracks = list(tracks or [])


def _instance(record, skeleton, tracks):
    track = None
    if record["track"] is not None:
        track = tracks.setdefault(record["track"], Track(record["track"]))
    embedding = None
    if record["embedding"] is not None:
        embedding = Embedding(record["embedding"])
    kind = PredictedInstance if record["predicted"] else Instance
    points = np.zeros((len(skeleton.nodes), 2))
    points[:, 0] = record["x"]
    return kind(points, skeleton, track=track, identity_embedding=embedding)


def load_slp(filename):
    with open(filename, encoding="utf-8") as handle:
        document = json.load(handle)
    video = Video(document["video"])
    skeleton = Skeleton(NODES)
    tracks = {}
    frames = [
        LabeledFrame(
            video,
            frame["frame_idx"],
            [_instance(record, skeleton, tracks) for record in frame["instances"]],
        )
        for frame in document["frames"]
    ]
    return Labels(frames, [video], [skeleton], list(tracks.values()))


def save_slp(labels, filename, embed=False):
    time.sleep(SAVE_SECONDS)
    positions = {id(video): position for position, video in enumerate(labels.videos)}
    document = {
        "embed": embed,
        "videos": [video.filename for video in labels.videos],
        "tracks": [track.name for track in labels.tracks],
        "frames": [
            {
                "video": positions[id(frame.video)],
                "frame_idx": frame.frame_idx,
                "instances": [
                    {
                        "track": None if each.track is None else each.track.name,
                        "predicted": isinstance(each, PredictedInstance),
                        "x": float(each.points[0, 0]),
                    }
                    for each in frame.instances
                ],
            }
            for frame in labels.labeled_frames
        ],
    }
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
"""
"""The stand-in ``sleap_io``: the classes, ``load_slp`` and ``save_slp`` a snippet uses.

``PredictedInstance`` subclasses ``Instance``, as in sleap-io. ``save_slp`` writes
what :func:`read_stub_labels` reads, after sleeping ``SAVE_SECONDS``. ``NODES``
is :data:`STUB_NODES`.
"""


def write_stub_sleap_io(directory: Path, *, save_seconds: float = 0.0) -> Path:
    """Write a stand-in ``sleap_io`` package under *directory*.

    Its ``load_slp`` reads a file of :func:`write_stub_slp`. Put *directory* on
    a snippet's module search path, for instance as the ``imports`` of
    :func:`~tests.helpers.decode_probe.install_fake_tool_python`.

    Args:
        directory: Where the package is written.
        save_seconds: How long each ``save_slp`` sleeps before it writes, as a
            save that embeds frames takes.

    Returns:
        *directory*.
    """
    package = directory / "sleap_io"
    package.mkdir(parents=True, exist_ok=True)
    header = f"NODES = {STUB_NODES!r}\nSAVE_SECONDS = {save_seconds!r}\n"
    source = header + _STUB_SLEAP_IO
    _ = (package / "__init__.py").write_text(source, encoding="utf-8")
    return directory
