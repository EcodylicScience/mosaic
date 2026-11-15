"""Feature definition sources, and a dataset with tracks that defines them.

Each source is the text of one definition file, whose class ``name`` is the
definition name of the same constant (``MINIMAL_NAME`` for ``MINIMAL_SOURCE``):

- ``MINIMAL_SOURCE``: a per-sequence feature, each row's distance from the origin.
- ``CANARY_SOURCE``: the same feature, whose top level appends a line holding
  ``os.getpid()`` and ``sys.argv`` to the file that ``CANARY_VARIABLE`` names, once
  per import.
- ``GLOBAL_SOURCE``: a feature that fits over its scope and whose ``save_state``
  pickles an instance of a class of its own file.
- ``POSE_SOURCE``: a per-sequence feature whose params hold a ``PoseConfig``
  default.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

from mosaic.core.dataset import Dataset
from tests.helpers.datasets import make_dataset
from tests.helpers.tracks import add_track_sequences

__all__ = [
    "CANARY_NAME",
    "CANARY_SOURCE",
    "CANARY_VARIABLE",
    "GLOBAL_NAME",
    "GLOBAL_SOURCE",
    "MINIMAL_NAME",
    "MINIMAL_SOURCE",
    "POSE_NAME",
    "POSE_SOURCE",
    "defined_dataset",
]

MINIMAL_NAME: Final = "origin-distance"
CANARY_NAME: Final = "canary-distance"
GLOBAL_NAME: Final = "mean-distance"
POSE_NAME: Final = "pose-count"

CANARY_VARIABLE: Final = "MOSAIC_TEST_DEFINITION_CANARY"
"""The environment variable naming the file that ``CANARY_SOURCE`` appends to."""

_PER_SEQUENCE_BODY: Final = '''
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.params import Params as ParamsBase
from mosaic.core.pipeline.types import DependencyLookup, InputStream, TrackInputs

_KEYS = ("frame", "time", "id", "group", "sequence")


class OriginDistance:
    """Each row's distance from the origin."""

    name = "{name}"
    version = "1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    emits = "individual"
    consumed_roots = ()

    class Inputs(TrackInputs):
        pass

    class Params(ParamsBase):
        scale: float = 1.0

    def __init__(self, inputs=None, params=None):
        self.inputs = inputs if inputs is not None else self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[[key for key in _KEYS if key in df.columns]].copy()
        out["distance"] = np.hypot(df["X"], df["Y"]) * self.params.scale
        return out
'''

MINIMAL_SOURCE: Final = (
    '"""Each row\'s distance from the origin."""\n'
    + _PER_SEQUENCE_BODY.format(name=MINIMAL_NAME)
)

CANARY_SOURCE: Final = (
    '"""Each row\'s distance from the origin, noting each import."""\n'
    "\n"
    "import json\n"
    "import os\n"
    "import sys\n"
    f"_canary = os.environ.get({CANARY_VARIABLE!r})\n"
    "if _canary:\n"
    "    with open(_canary, 'a', encoding='utf-8') as _handle:\n"
    "        _handle.write(json.dumps({'pid': os.getpid(), 'argv': sys.argv}) + '\\n')\n"
    + _PER_SEQUENCE_BODY.format(name=CANARY_NAME)
)

GLOBAL_SOURCE: Final = f'''"""The mean distance from the origin over the scope."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from mosaic.core.params import Params as ParamsBase
from mosaic.core.pipeline.types import DependencyLookup, InputStream, TrackInputs

_KEYS = ("frame", "time", "id", "group", "sequence")
_STATE = "mean.pkl"


class Mean:
    """The fitted mean distance."""

    def __init__(self, value: float) -> None:
        self.value = value


class MeanDistance:
    """Each row's distance from the origin less the mean over the scope."""

    name = "{GLOBAL_NAME}"
    version = "1"
    parallelizable = False
    scope_dependent = True
    accepts_overlap = False
    emits = "individual"
    consumed_roots = ()
    fits = True

    class Inputs(TrackInputs):
        pass

    class Params(ParamsBase):
        pass

    def __init__(self, inputs=None, params=None):
        self.inputs = inputs if inputs is not None else self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)
        self.mean = None

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        path = run_root / _STATE
        if not path.exists():
            return False
        with path.open("rb") as handle:
            self.mean = pickle.load(handle)
        return True

    def fit(self, inputs: InputStream) -> None:
        distances = [
            np.hypot(df["X"], df["Y"]).to_numpy() for _key, df in inputs()
        ]
        self.mean = Mean(float(np.concatenate(distances).mean()))

    def save_state(self, run_root: Path) -> None:
        with (run_root / _STATE).open("wb") as handle:
            pickle.dump(self.mean, handle)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[[key for key in _KEYS if key in df.columns]].copy()
        out["centered"] = np.hypot(df["X"], df["Y"]) - self.mean.value
        return out
'''

POSE_SOURCE: Final = f'''"""The number of keypoints a pose holds."""

from pathlib import Path

import pandas as pd
from pydantic import Field

from mosaic.core.params import Params as ParamsBase
from mosaic.core.pipeline.types import DependencyLookup, InputStream, TrackInputs
from mosaic.core.pipeline.types.data_config import PoseConfig

_KEYS = ("frame", "time", "id", "group", "sequence")


class PoseCount:
    """The number of keypoints the pose config declares, on every row."""

    name = "{POSE_NAME}"
    version = "1"
    parallelizable = True
    scope_dependent = False
    accepts_overlap = False
    emits = "individual"
    consumed_roots = ()

    class Inputs(TrackInputs):
        pass

    class Params(ParamsBase):
        pose: PoseConfig = Field(default_factory=PoseConfig)
        keypoints: list[int | str] = [0]

    def __init__(self, inputs=None, params=None):
        self.inputs = inputs if inputs is not None else self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, DependencyLookup],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        return None

    def save_state(self, run_root: Path) -> None:
        return None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[[key for key in _KEYS if key in df.columns]].copy()
        out["keypoints"] = len(self.params.keypoints)
        return out
'''


def defined_dataset(tmp_path: Path, sources: Mapping[str, str]) -> Dataset:
    """A dataset under *tmp_path* with two tracked sequences that defines *sources*.

    Args:
        tmp_path: The directory the dataset is made in, as ``tmp_path / "dataset"``.
        sources: Each definition name and the source defined under it, defined in
            order.
    """
    from mosaic.core.pipeline.definitions.define import define_feature

    dataset = make_dataset(tmp_path / "dataset")
    add_track_sequences(dataset, ("g", "s1"), ("g", "s2"))
    for name, source in sources.items():
        _ = define_feature(dataset, name, source)
    return dataset
