"""Stand-in features that declare which source roots they consume.

The provenance walk, delete sets, media rearrangement and source staleness are
all decided by a feature's ``consumed_roots``, so their tests run a feature that
does nothing but declare one. :class:`LossyFeature` loses the entries it is told
to and :class:`ScopedFeature` also declares ``scope_dependent``, for the tests of
how a run is measured against the entries it was asked for.
:class:`PairClusterFeature` writes pair rows for the tests of scoring them.
:class:`FittingFeature` fits once per run root and saves its state, for the tests
of the claim a fit holds.
:class:`ParallelFeature` declares ``parallelizable``, and a subclass that sets
``kill_on`` ends the process computing those sequences, for the tests of a run
that computes several entries at once.
"""

from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.params import Params
from mosaic.core.pipeline._utils import atomic_write
from mosaic.core.pipeline.types import DependencyLookup, Inputs, InputStream
from mosaic.core.pipeline.types.feature import EmitsLevel


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


class PairClusterFeature(PlainFeature):
    """Both perspectives of the pair ``(0, 1)`` on every frame, with a ``cluster``.

    Perspective 0 is focal on id 0 and perspective 1 on id 1, as the pair
    features write them. ``cluster`` is ``frame % 2`` on both rows of a frame.
    """

    name = "prov-pair-cluster"

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
