"""Stand-in features that declare which source roots they consume.

The provenance walk, delete sets, media rearrangement and source staleness are all
decided by a feature's ``consumed_roots``. Their tests run a feature that does
nothing but declare one. :class:`LossyFeature` loses the entries it is told to and
:class:`ScopedFeature` also declares ``scope_dependent``, for the tests of how a run is
measured against the entries it was asked for. :class:`PairClusterFeature` writes pair
rows for the tests of scoring them. :class:`FittingFeature` fits once per run root and
saves its state, for the tests of the claim a fit holds, and
:class:`ScopedFittingFeature` fits over its scope, for the tests of a fit whose recorded
entries change. :class:`CenteringFeature` fits a mean and applies it, in worker
processes too, for the tests of a fit that a linked dataset holds.
:class:`ParallelFeature` declares ``parallelizable``, and a subclass that sets
``kill_on`` ends the process computing those sequences, for the tests of a run that
computes several entries at once. :class:`LabelsClassifier` reads ground-truth labels
and records the file of each entry it is handed, for the tests of labels that a linked
dataset holds. :class:`HoldsOnSequences` stops in the entries it names until it is
killed, for the tests of a cancel that reaches a run's workers.
:class:`HoldsUntilReleased` stops in the entries it names until the test releases them,
for the tests of how a run ends with entries in flight.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field

from mosaic.core.json_value import JsonValue
from mosaic.core.params import Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.types import (
    DependencyLookup,
    GroundTruthLabelsSource,
    Inputs,
    InputStream,
    TrackInput,
)
from mosaic.core.pipeline.feature_levels import EmitsLevel


class _P(Params):
    pass


class CropLike:
    """A per-frame feature that opens video: it declares ``media_raw``.

    The four protocol methods carry the protocol's own parameter names and types
    rather than ``object`` stand-ins. A structural protocol matches on parameter
    *names*, so a stub spelling them differently is not a ``Feature``, and every
    module passing one to ``run_feature`` inherited that error.
    """

    name = "prov-crop"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ("media_raw",)

    def __init__(
        self,
        inputs: Inputs | None = None,
        params: dict[str, object] | _P | None = None,
    ) -> None:
        self.inputs = inputs if inputs is not None else Inputs(("tracks",))
        self.params = params if isinstance(params, _P) else _P.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


class PlainFeature(CropLike):
    """The ordinary shape: forty of forty-two features declare no source root."""

    name = "prov-plain"
    consumed_roots: tuple[str, ...] = ()


class LossyFeature(PlainFeature):
    """A scope-free per-entry feature that fails on the sequences it is told to.

    Its apply raises for the sequences named in *fail_on*, so a test can make a
    run lose an entry inside the scope it was asked for.
    """

    name = "prov-lossy"

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.fail_on = frozenset(fail_on)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) in self.fail_on:
            raise RuntimeError("a lost entry")
        return df


class ScopedFeature(LossyFeature):
    """A :class:`LossyFeature` whose scope is hashed into its run id."""

    name = "prov-scoped"
    scope_dependent = True


class ParallelFeature(PlainFeature):
    """A per-entry feature that may compute several entries at once.

    A subclass that names sequences in ``kill_on`` ends the process computing one
    of them with SIGKILL, as a host does a process it runs out of memory for. The
    names are a class attribute because a process worker rebuilds the feature from
    its class and parameters, and an instance attribute would not reach it.
    """

    name = "prov-parallel"
    parallelizable = True
    accepts_overlap = False
    emits: EmitsLevel = "as-input"
    kill_on: tuple[str, ...] = ()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) in self.kill_on:
            os.kill(os.getpid(), signal.SIGKILL)
        return df


HOLD_DIRECTORY_VARIABLE = "MOSAIC_TEST_HOLD_DIRECTORY"
"""The environment variable naming the directory a :class:`HoldsOnSequences` marks in."""

_HOLD_SECONDS = 120.0


class HoldsOnSequences(ParallelFeature):
    """Sleeps in each entry named in ``hold_on`` until its process is killed.

    On entering such an entry it writes an empty file named for the sequence into the
    directory that ``HOLD_DIRECTORY_VARIABLE`` names, so a test sees which workers
    are busy. The names are a class attribute for the reason ``kill_on`` is.
    """

    name = "prov-parallel-holds"
    hold_on: tuple[str, ...] = ()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        sequence = str(df["sequence"].iloc[0]) if not df.empty else ""
        if sequence in self.hold_on:
            marks = Path(os.environ[HOLD_DIRECTORY_VARIABLE])
            (marks / sequence).touch()
            time.sleep(_HOLD_SECONDS)
        return df


_RELEASE_POLL_SECONDS = 0.02


class HoldsUntilReleased(ParallelFeature):
    """Waits in each entry named in ``hold_on`` until the test releases it.

    Each entry it applies leaves an empty file ``applied-<sequence>`` in the
    directory that ``HOLD_DIRECTORY_VARIABLE`` names, so a test reads what a run
    computed in any worker. On entering an entry named in ``hold_on`` it writes its
    process id into a file named for the sequence there, then waits for a file
    named ``release`` and returns once it exists. It fails after ``_HOLD_SECONDS``
    without one. It holds in serial, thread and process mode alike, and the names
    are a class attribute for the reason ``kill_on`` is.
    """

    name = "prov-parallel-released"
    hold_on: tuple[str, ...] = ()

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        sequence = str(df["sequence"].iloc[0]) if not df.empty else ""
        marks = Path(os.environ[HOLD_DIRECTORY_VARIABLE])
        (marks / f"applied-{sequence}").touch()
        if sequence in self.hold_on:
            process = str(os.getpid())
            atomic_write(marks / sequence, lambda temp: temp.write_text(process))
            release = marks / "release"
            deadline = time.monotonic() + _HOLD_SECONDS
            while not release.exists():
                if time.monotonic() >= deadline:
                    message = f"the test never released {sequence}"
                    raise TimeoutError(message)
                time.sleep(_RELEASE_POLL_SECONDS)
        return super().apply(df)


class PairClusterFeature(PlainFeature):
    """Both perspectives of the pair ``(0, 1)`` on every frame, with a ``cluster``.

    Perspective 0 is focal on id 0 and perspective 1 on id 1, as the pair
    features write them. ``cluster`` is ``frame % 2`` on both rows of a frame.
    """

    name = "prov-pair-cluster"
    emits: EmitsLevel = "pair"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = np.unique(df["frame"].to_numpy())
        n_frames = len(frames)
        return pd.DataFrame(
            {
                "frame": np.repeat(frames, 2),
                "id1": np.tile([0, 1], n_frames),
                "id2": np.tile([1, 0], n_frames),
                "perspective": np.tile([0, 1], n_frames),
                "group": str(df["group"].iloc[0]),
                "sequence": str(df["sequence"].iloc[0]),
                "cluster": np.repeat(frames % 2, 2),
            }
        )


_GATE_SECONDS = 30.0
"""How long a gated stand-in waits to be let through before its test fails."""


@dataclass
class FitControl:
    """What a test sees of the :class:`FittingFeature` runs sharing it, and steers.

    Attributes:
        fits: How many times any of them fitted.
        fitting: Set once a fit has begun.
        finish_fit: A fit waits for this before it returns, when given.
        applying: Set once an apply has begun.
        finish_apply: An apply waits for this before it returns, when given.
        fit_error: Raised by every fit, when given.
        peer_saves_first: The first ``load_state`` writes the state, as a peer
            finishing its fit at that moment would, and still answers ``False``.
        loads: How many times any of them loaded state.
        lock: Guards the two counts across the threads of one test.
    """

    fits: int = 0
    fitting: threading.Event = field(default_factory=threading.Event)
    finish_fit: threading.Event | None = None
    applying: threading.Event = field(default_factory=threading.Event)
    finish_apply: threading.Event | None = None
    fit_error: BaseException | None = None
    peer_saves_first: bool = False
    loads: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


def _wait(gate: threading.Event | None) -> None:
    if gate is not None and not gate.wait(_GATE_SECONDS):
        raise TimeoutError("the test never let the stand-in feature through")


class FittingFeature(PlainFeature):
    """A scope-free feature whose state is one ``state.json`` that its fit makes.

    Instances sharing one :class:`FitControl` share one identity and so one run
    root, which is how a test runs two executions of one fit at once.
    """

    name = "prov-fitting"
    accepts_overlap = False
    emits: EmitsLevel = "as-input"

    def __init__(self, control: FitControl) -> None:
        super().__init__()
        self.control = control

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        state = run_root / "state.json"
        with self.control.lock:
            self.control.loads += 1
            first = self.control.loads == 1
        if first and self.control.peer_saves_first:
            self.save_state(run_root)
            return False
        return state.exists()

    def fit(self, inputs: InputStream) -> None:
        with self.control.lock:
            self.control.fits += 1
        self.control.fitting.set()
        _wait(self.control.finish_fit)
        if self.control.fit_error is not None:
            raise self.control.fit_error

    def save_state(self, run_root: Path) -> None:
        atomic_write(run_root / "state.json", lambda temp: temp.write_text("{}"))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.control.applying.set()
        _wait(self.control.finish_apply)
        return df


class ScopedFittingFeature(FittingFeature):
    """A :class:`FittingFeature` over its scope, recording its entries in its fit."""

    name = "prov-fitting-scoped"
    scope_dependent = True


class CenteringFeature(PlainFeature):
    """A scope-free feature that fits the mean of ``X`` and subtracts it from ``X``.

    Its state is one ``centering.json`` in the run root. Its apply raises while
    no state is loaded, so a worker process that fails to load the state fails
    the entry rather than writing a table. It is ``parallelizable`` and rebuilds
    from its class and parameters, as a process worker rebuilds a feature.
    """

    name = "prov-centering"
    parallelizable = True
    accepts_overlap = False
    emits: EmitsLevel = "as-input"
    state_file = "centering.json"

    def __init__(
        self,
        inputs: Inputs | None = None,
        params: dict[str, object] | _P | None = None,
    ) -> None:
        super().__init__(inputs, params)
        self.mean: float | None = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        state = run_root / self.state_file
        if not state.is_file():
            return False
        self.mean = float(json.loads(state.read_text(encoding="utf-8"))["mean"])
        return True

    def fit(self, inputs: InputStream) -> None:
        tables = [df["X"] for _key, df in inputs()]
        self.mean = float(pd.concat(tables).mean())

    def save_state(self, run_root: Path) -> None:
        document = json.dumps({"mean": self.mean})
        atomic_write(run_root / self.state_file, lambda temp: temp.write_text(document))

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean is None:
            raise RuntimeError("no fitted mean is loaded")
        return df.assign(X=df["X"] - self.mean)


class LabelsClassifier:
    """A scope-free classifier stand-in over tracks that reads ground-truth labels.

    Each run records the labels file of every entry it is handed in
    :attr:`read`, and passes its tracks through. :func:`labels_classifier_registered`
    registers it by its slug, so a recipe can wire its ``labels`` field to a
    conversion step.
    """

    name = "prov-labels-classifier"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()
    read: ClassVar[dict[tuple[str, str], Path]] = {}
    """The labels file of each entry that the runs read."""

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        labels: GroundTruthLabelsSource = Field(default_factory=GroundTruthLabelsSource)

    def __init__(
        self,
        inputs: LabelsClassifier.Inputs | None = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs: LabelsClassifier.Inputs = inputs or self.Inputs(("tracks",))
        self.params: LabelsClassifier.Params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        LabelsClassifier.read.update(dependency_lookups["labels"])
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


LABELS_CLASSIFIER_STEP: dict[str, JsonValue] = {
    "id": "classifier",
    "type": "feature",
    "feature": LabelsClassifier.name,
    "inputs": ["tracks"],
    "tracks": {"step": "tracks"},
    "params": {"labels": {"step": "labels"}},
}
"""A :class:`LabelsClassifier` step over the ``tracks`` and ``labels`` steps."""


@contextmanager
def labels_classifier_registered() -> Generator[type[LabelsClassifier]]:
    """Register :class:`LabelsClassifier` for the block, with nothing read yet.

    The declaration catalog is memoized, so it is cleared on the way in and out.
    """
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.graph.resolve import builtin_declaration_catalog

    FEATURES[LabelsClassifier.__name__] = LabelsClassifier
    builtin_declaration_catalog.cache_clear()
    LabelsClassifier.read.clear()
    try:
        yield LabelsClassifier
    finally:
        del FEATURES[LabelsClassifier.__name__]
        builtin_declaration_catalog.cache_clear()
        LabelsClassifier.read.clear()
