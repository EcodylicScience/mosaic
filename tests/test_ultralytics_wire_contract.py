"""What mosaic sends against what the runner accepts, checked side by side.

Every other test on this boundary fakes one half of it. The marker suite
replaces both seams and asserts on the ``TrackRequest`` mosaic *built*; the
runner has no suite of its own, because CI never builds the Ultralytics
environment. So a field mosaic populates with one meaning and the runner reads
with another, a renamed argument, a required value never sent -- each of those
passes the whole suite green and fails on a user's machine, at the one place
mosaic cannot see. This file is what stands between the two.

It costs nothing to run. Every ``ultralytics`` import in the runner program is
deferred into a function body, so the module imports here on the strength of
numpy, pandas, pydantic and mosaic-media alone -- all of which mosaic's own
environment holds -- and no AGPL-licensed code is loaded by any of this.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from mosaic_media import MediaFacts
from pydantic import BaseModel

import mosaic.tracking.common.ultralytics_env as ultralytics_env
import mosaic.tracking.ops.infer as ops_infer
import mosaic.tracking.pose_training.ultralytics_infer as ultralytics_infer
import mosaic.tracking.ultralytics_track.dataset_runs as dataset_runs
import mosaic.tracking.ultralytics_track.run as ultralytics_run
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.media.read_target import verified_read_facts
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.track_library.ultralytics_tracks import raw_columns
from mosaic.tracking.external import runner as runner_package
from mosaic.tracking.external.runner.ultralytics_protocol import (
    InferPointsRequest,
    InferPoseRequest,
    ProbeResponse,
    TrackRequest,
    rows_from_result,
)
from mosaic.tracking.ultralytics_track.tracker_defaults import TRACKER_NAMES

from tests.helpers import write_media_index
from tests.test_ultralytics_rows import FakeDetections, FakeResult

# Selected by CI's `tracking` job with `-m tracker` rather than by a filename
# list in the workflow, so a new file here is covered the day it lands.
pytestmark = pytest.mark.tracker

_N_KEYPOINTS = 2


# --- the runner program, imported rather than spawned ----------------------


@pytest.fixture(scope="module")
def runner_module() -> Iterator[ModuleType]:
    """The runner program, imported into this process, and then unimported.

    Its directory goes on ``sys.path`` because the program resolves
    ``ultralytics_protocol`` as a bare top-level module -- what a script gets for
    free from its own directory, and what it will get when it is spawned. The
    insertion is safe: the directory holds those two modules and nothing that
    could shadow a mosaic import.

    Both are undone afterwards. Left in place they outlive this file: the search
    path keeps answering ``ultralytics_protocol`` for the rest of the session,
    and ``sys.modules`` holds a *second* copy of that module beside
    ``mosaic.tracking.external.runner.ultralytics_protocol`` -- same file, two
    classes. Nothing today compares a :class:`TrackRequest` by identity, so the
    leak is currently harmless and would stop being so quietly.
    """
    directory = str(Path(runner_package.__file__).parent)
    inserted = directory not in sys.path
    if inserted:
        sys.path.insert(0, directory)
    try:
        yield importlib.import_module("ultralytics_runner")
    finally:
        if inserted:
            sys.path.remove(directory)
        for name in ("ultralytics_runner", "ultralytics_protocol"):
            _ = sys.modules.pop(name, None)


# --- what each side says about the request ---------------------------------


def _module_source(module: ModuleType) -> ast.Module:
    return ast.parse(Path(str(module.__file__)).read_text())


def _keywords_of_call(module: ModuleType, name: str) -> set[str]:
    """Every keyword *module* passes when it constructs *name*.

    Read out of the syntax rather than restated here, so the assertion moves
    with the code instead of pinning a list somebody has to remember to edit.
    """
    sent: set[str] = set()
    for node in ast.walk(_module_source(module)):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        if isinstance(called, ast.Name) and called.id == name:
            sent.update(
                keyword.arg for keyword in node.keywords if keyword.arg is not None
            )
    return sent


def _attributes_read_from(module: ModuleType, annotation: str) -> set[str]:
    """Every attribute *module* reads off a parameter annotated *annotation*.

    Scoped to the functions that take one, rather than to every local called
    ``request``: the runner's probe path takes a ``ProbeRequest`` under that same
    name, and a ``tracker`` read there is not a field of the track request.
    """
    read: set[str] = set()
    for node in ast.walk(_module_source(module)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = {
            argument.arg
            for argument in node.args.args
            if isinstance(argument.annotation, ast.Name)
            and argument.annotation.id == annotation
        }
        if not names:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id in names
            ):
                read.add(inner.attr)
    return read


def _attributes_read_off(modules: Sequence[ModuleType], local: str) -> set[str]:
    """Every attribute *modules* read off a local named *local*."""
    read: set[str] = set()
    for module in modules:
        for node in ast.walk(_module_source(module)):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == local
            ):
                read.add(node.attr)
    return read


def test_mosaic_sends_exactly_the_track_fields_the_runner_reads(
    runner_module: ModuleType,
) -> None:
    """Neither side carries a field the other has no use for.

    A field mosaic stops sending is a validation error on the first real run; a
    field the runner stops reading is worse, because it is silent -- a knob that
    reads as configured and does nothing.
    """
    sent = _keywords_of_call(dataset_runs, "TrackRequest")
    read = _attributes_read_from(runner_module, "TrackRequest")
    declared = set(TrackRequest.model_fields)

    # The extraction has to have found something before its verdict means
    # anything: an empty pair of sets agrees with itself.
    assert len(declared) > 15
    assert sent == declared
    assert read == declared


def test_the_runner_reports_exactly_the_probe_fields_mosaic_reads(
    runner_module: ModuleType,
) -> None:
    """Every reported field is read, and every read field is reported.

    Both branches of the probe are checked, because the one that answers for an
    environment with no Ultralytics fills the same model from different values --
    a field only the other branch sets would raise there and nowhere else.
    """
    reported = _keywords_of_call(runner_module, "ProbeResponse")
    read = _attributes_read_off(
        [dataset_runs, ultralytics_run, ultralytics_env, ultralytics_infer, ops_infer],
        "probe",
    )
    declared = set(ProbeResponse.model_fields)

    assert len(declared) > 5
    assert reported == declared
    assert read == declared


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("InferPoseRequest", InferPoseRequest),
        ("InferPointsRequest", InferPointsRequest),
    ],
)
def test_mosaic_sends_exactly_the_inference_fields_the_runner_reads(
    runner_module: ModuleType, name: str, model: type[BaseModel]
) -> None:
    """Neither side of an inference request carries a field the other ignores.

    The same check the track request gets, and it matters more here, because two
    requests share a base: a field read only off the base, or sent only on one
    subclass, is exactly the asymmetry a single-request contract cannot express.

    Reads are collected off the base annotation as well as the subclass, since the
    runner's shared loop takes an ``InferRequestBase`` and only the two entry
    points take the narrower type.
    """
    sent = _keywords_of_call(ops_infer, name)
    read = _attributes_read_from(runner_module, name) | _attributes_read_from(
        runner_module, "InferRequestBase"
    )
    declared = set(model.model_fields)

    # The extraction has to have found something before its verdict means
    # anything: an empty pair of sets agrees with itself.
    assert len(declared) > 15
    assert sent == declared
    assert read == declared


# --- the column contract ---------------------------------------------------


@pytest.mark.parametrize("n_keypoints", [1, _N_KEYPOINTS, 17])
def test_the_columns_mosaic_names_match_the_block_the_runner_writes(
    n_keypoints: int,
) -> None:
    """One name per value, in the order the extraction fills them.

    Mosaic owns the names and sends them; the runner owns the layout. ``n = 1``
    is the arm most likely to drift: a detect model makes the probe report one
    keypoint and leaves ``result.keypoints`` ``None``, so the box centre and the
    detection confidence are written as the single keypoint.
    """
    columns = raw_columns(n_keypoints)
    assert len(columns) == 8 + 3 * n_keypoints

    box = (10.0, 20.0, 30.0, 60.0)
    keypoints = (
        None
        if n_keypoints == 1
        else FakeDetections(
            np.array([[[1.0 + k, 2.0 + k, 0.5] for k in range(n_keypoints)]], float)
        )
    )
    result = FakeResult(
        boxes=FakeDetections(np.array([[*box, 7.0, 0.25, 3.0]], dtype=float)),
        keypoints=keypoints,
    )
    block = rows_from_result(result, 5, n_keypoints=n_keypoints)
    assert block is not None
    row = pd.DataFrame(block, columns=list(columns)).iloc[0]

    assert row["frame"] == 5.0
    assert row["track_id"] == 7.0
    assert (row["x1"], row["y1"], row["x2"], row["y2"]) == box
    assert row["conf"] == 0.25
    assert row["cls"] == 3.0
    if n_keypoints == 1:
        assert (row["kpx0"], row["kpy0"]) == (20.0, 40.0)  # the box centre
        assert row["kpp0"] == 0.25  # its detection confidence
    else:
        assert (row["kpx0"], row["kpy0"], row["kpp0"]) == (1.0, 2.0, 0.5)


# --- the whole exchange, over a supervisor that answers as the runner does ---


@pytest.fixture
def ds(tmp_path: Path) -> Dataset:
    manifest = new_dataset_manifest("t", base_dir=tmp_path)
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    write_media_index(dataset, ["vid1"])
    return dataset


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "yolo" / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"weights")
    return path


class RunnerStandIn:
    """A supervised process that answers exactly as the runner program would.

    It parses the argv mosaic built with the runner's *own* parser, validates the
    request file with the runner's *own* models, and writes its answers through
    the runner's own writer. Nothing here restates what either side does, so a
    renamed flag, a moved field or a changed response shape fails here rather
    than on a machine with the environment built.
    """

    def __init__(self, module: ModuleType, n_keypoints: int) -> None:
        self.module: ModuleType = module
        self.n_keypoints: int = n_keypoints
        self.commands: list[str] = []
        self.payloads: list[str] = []
        """Each request as it was written, kept so a test can read it back.

        Whether the runner *accepts* one is settled below, by validating it with
        the runner's own model; re-reading a payload here with mosaic's model is
        for inspecting field values, which the two agree on by then.
        """
        self.reconstructed_facts: list[MediaFacts] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        cancel_check: Callable[[], bool] | None = None,
        timeout: float | None = None,
        idle_timeout: float | None = None,
        poll_interval: float = 0.5,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[str, str, int]:
        tokens = [str(token) for token in argv]
        assert Path(tokens[1]).is_file(), "mosaic spawns a program that is there"
        parsed = self.module._parser().parse_args(tokens[2:])
        command = str(parsed.command)
        self.commands.append(command)
        payload = Path(str(parsed.request)).read_text()
        self.payloads.append(payload)
        answer = self._probe(payload) if command == "probe" else self._track(payload)
        Path(str(parsed.out)).write_text(answer)
        return ("", "", 0)

    def _probe(self, payload: str) -> str:
        # Validated with the runner's own model: this is where a field mosaic
        # renamed, dropped or retyped stops the test.
        _ = self.module.ProbeRequest.model_validate_json(payload)
        return self.module.ProbeResponse(
            has_ultralytics=True,
            has_lap=True,
            has_locate=False,
            ultralytics_version="8.4.63",
            tracker_names=list(TRACKER_NAMES),
            model_task="pose",
            n_keypoints=self.n_keypoints,
            model_load_error="",
            installed_tracker_table={"track_buffer": 30},
        ).model_dump_json()

    def _track(self, payload: str) -> str:
        request = self.module.TrackRequest.model_validate_json(payload)
        # The facts mosaic flattened, rebuilt the way the runner rebuilds them
        # before handing them to its reader.
        self.reconstructed_facts.append(self.module._media_facts(request.media_facts))
        columns = list(request.columns)
        rows = [
            [float(frame), 1.0, 10.0, 20.0, 15.0, 25.0, 0.9, 0.0]
            + [0.0] * (len(columns) - 8)
            for frame in range(4)
        ]
        table = pd.DataFrame(np.array(rows, dtype=float), columns=columns)
        table = table.astype({"frame": "int64", "track_id": "int64", "cls": "int64"})
        self.module._publish_parquet(table, request.output_parquet)
        return self.module.TrackResponse(n_frames=4, n_ids=1).model_dump_json()


@pytest.fixture
def stand_in(
    runner_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> Iterator[RunnerStandIn]:
    fake = RunnerStandIn(runner_module, _N_KEYPOINTS)
    monkeypatch.setattr(ultralytics_env, "run_supervised", fake)
    yield fake


def test_a_run_speaks_a_protocol_the_runner_program_accepts(
    ds: Dataset, model: Path, stand_in: RunnerStandIn
) -> None:
    """One run, end to end, with only the process boundary replaced.

    Production builds the argv, the request and the tracker configuration; the
    runner's own parser and models are what accept them; and what mosaic reads
    back is what the runner's own models serialize. The tracks row at the end is
    the proof that the columns mosaic sent are the ones its converter reads.
    """
    run_id = dataset_runs.run_ultralytics(
        ds, model_path=str(model), ultralytics_bin="/x/bin/yolo"
    )

    assert stand_in.commands == ["probe", "track"]
    request = TrackRequest.model_validate_json(stand_in.payloads[1])
    assert list(request.columns) == list(raw_columns(_N_KEYPOINTS))
    assert Path(request.output_parquet).is_file()
    # The `project` argument Ultralytics computes its run directory from, which
    # is the run root and not the entry directory beside it.
    assert Path(request.project_dir) == dataset_runs.ultralytics_run_root(ds, run_id)

    tracks = read_tracks_index(ds)
    assert len(tracks) == 1
    assert int(tracks.iloc[0]["n_rows"]) == 4
    assert str(tracks.iloc[0]["producer_run_id"]) == run_id


def test_gated_media_facts_survive_the_crossing(
    ds: Dataset,
    model: Path,
    stand_in: RunnerStandIn,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    write_cfr_mp4: Callable[..., None],
) -> None:
    """Measured, gated, flattened, serialized, and equal on the far side.

    The facts are what keeps a raw stream reading with its true frame count
    instead of the garbage one its header declares, so a field lost in the
    crossing is a misindexed track under a valid identifier. Compared against
    the same gate mosaic ran, on the same file.
    """
    video = tmp_path / "clip.mp4"
    write_cfr_mp4(video, frames=6, size=(64, 48))

    def resolved(_ds: Dataset, _item: object, *, kind: str) -> Path:
        return video

    monkeypatch.setattr(dataset_runs, "resolve_tool_input", resolved)
    _ = dataset_runs.run_ultralytics(
        ds, model_path=str(model), ultralytics_bin="/x/bin/yolo"
    )

    measured = verified_read_facts(video, None, "analysis")[0]
    assert stand_in.reconstructed_facts == [measured]
    # Flattened rather than pickled: what crossed is JSON a reader can inspect.
    request = TrackRequest.model_validate_json(stand_in.payloads[1])
    assert request.media_facts == dataclasses.asdict(measured)
