"""Stand-in features that declare which source roots they consume.

The provenance walk, delete sets, media rearrangement and source staleness are
all decided by a feature's ``consumed_roots``, so their tests run a feature that
does nothing but declare one. :class:`ScopedFeature` declares ``scope_dependent``
instead, for the tests of how a run whose scope is in its id is measured, and
:class:`PairClusterFeature` writes pair rows for the tests of scoring them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.params import Params
from mosaic.core.pipeline.types import DependencyLookup, Inputs, InputStream


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


class ScopedFeature(PlainFeature):
    """A per-entry feature whose scope is hashed into its run id.

    Its apply raises for the sequences named in *fail_on*, so a test can make a
    run lose an entry inside the scope it declared.
    """

    name = "prov-scoped"
    scope_dependent = True

    def __init__(self, fail_on: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.fail_on = frozenset(fail_on)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if not df.empty and str(df["sequence"].iloc[0]) in self.fail_on:
            raise RuntimeError("a lost entry")
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
