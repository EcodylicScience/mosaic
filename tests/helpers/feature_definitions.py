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
- ``APPLYING_SOURCE``: ``GLOBAL_SOURCE``'s feature with the param ``model``.
  ``load_state`` reads the state from the root of the fit that ``model`` names,
  and ``fit`` appends a line to the file that ``CANARY_VARIABLE`` names.
  ``APPLYING_SECOND_SOURCE`` is a second version of it.
- ``MISREADING_SOURCE``: ``APPLYING_SOURCE`` whose ``load_state`` reads the state
  from the run root alone, whatever ``model`` names.
- ``PARALLEL_APPLYING_SOURCE``: ``APPLYING_SOURCE`` that may run in process
  workers. Each ``apply`` appends ``{"applied": <pid>}`` to the canary file.
- ``NARROWING_SOURCE``: ``MINIMAL_SOURCE`` whose ``__init__`` sets
  ``scope_dependent`` under the param ``wide``, which its sidecar cannot declare, so
  loading it with ``wide`` set fails the identity check.

``plan_refused`` plans a recipe the dataset cannot run, refusing and not.
"""

from __future__ import annotations

import sys
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.definitions import loader
from mosaic.core.pipeline.definitions.scope import clear_definition_caches
from mosaic.core.pipeline.graph import (
    BoundRef,
    Plan,
    Recipe,
    RecipeRefused,
    plan_pipeline,
)
from mosaic.core.pipeline.graph.resolve import StepSpec
from mosaic.core.pipeline.types import Feature, Result
from tests.helpers.datasets import make_dataset
from tests.helpers.tracks import add_track_sequences

__all__ = [
    "APPLYING_NAME",
    "APPLYING_SECOND_SOURCE",
    "APPLYING_SOURCE",
    "APPLYING_STORAGE",
    "CANARY_NAME",
    "CANARY_SOURCE",
    "CANARY_VARIABLE",
    "GLOBAL_NAME",
    "GLOBAL_SOURCE",
    "MINIMAL_NAME",
    "MINIMAL_SOURCE",
    "MISREADING_NAME",
    "MISREADING_SOURCE",
    "NARROWING_NAME",
    "NARROWING_SOURCE",
    "PARALLEL_APPLYING_NAME",
    "PARALLEL_APPLYING_SOURCE",
    "POSE_NAME",
    "POSE_SOURCE",
    "built_by_class",
    "defined_dataset",
    "definitions_forgotten",
    "fit_over",
    "fit_result",
    "forget_imported_definitions",
    "imported_definitions",
    "noting_definition_loads",
    "plan_refused",
]

MINIMAL_NAME: Final = "origin-distance"
CANARY_NAME: Final = "canary-distance"
GLOBAL_NAME: Final = "mean-distance"
POSE_NAME: Final = "pose-count"
NARROWING_NAME: Final = "narrow-distance"
APPLYING_NAME: Final = "centered-distance"
APPLYING_STORAGE: Final = f"def.{APPLYING_NAME}__from__tracks"
"""The storage of the runs of ``APPLYING_NAME`` over the tracks."""
MISREADING_NAME: Final = "misreading-distance"
PARALLEL_APPLYING_NAME: Final = "parallel-centered-distance"

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

NARROWING_SOURCE: Final = (
    MINIMAL_SOURCE.replace(f'name = "{MINIMAL_NAME}"', f'name = "{NARROWING_NAME}"')
    .replace("scale: float = 1.0", "scale: float = 1.0\n        wide: bool = False")
    .replace(
        "        self.params = self.Params.from_overrides(params)\n",
        "        self.params = self.Params.from_overrides(params)\n"
        "        if self.params.wide:\n"
        "            self.scope_dependent = True\n",
    )
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


def _replaced(source: str, old: str, new: str) -> str:
    """*source* with its one occurrence of *old* replaced by *new*."""
    assert source.count(old) == 1, old
    return source.replace(old, new)


_FIT_START: Final = "    def fit(self, inputs: InputStream) -> None:\n"
_STATE_PATH: Final = "        path = run_root / _STATE\n"

APPLYING_SOURCE: Final = reduce(
    lambda source, edit: _replaced(source, *edit),
    (
        (f'name = "{GLOBAL_NAME}"', f'name = "{APPLYING_NAME}"'),
        ("import pickle\n", "import json\nimport os\nimport pickle\n"),
        (
            "from pathlib import Path\n",
            "from pathlib import Path\nfrom typing import Annotated\n",
        ),
        (
            "from mosaic.core.params import Params as ParamsBase\n"
            "from mosaic.core.pipeline.types import DependencyLookup, InputStream, TrackInputs\n",
            "from mosaic.core.params import Declared\n"
            "from mosaic.core.params import Params as ParamsBase\n"
            "from mosaic.core.pipeline.types import (\n"
            "    DefinitionModel,\n"
            "    DependencyLookup,\n"
            "    InputStream,\n"
            "    TrackInputs,\n"
            ")\n",
        ),
        (
            "    class Params(ParamsBase):\n        pass\n",
            "    class Params(ParamsBase):\n"
            "        model: Annotated[\n"
            '            DefinitionModel | None, Declared("The fit to apply.")\n'
            "        ] = None\n",
        ),
        (
            _STATE_PATH,
            '        path = artifact_paths.get("model", run_root) / _STATE\n',
        ),
        (
            _FIT_START,
            _FIT_START + f"        _canary = os.environ.get({CANARY_VARIABLE!r})\n"
            "        if _canary:\n"
            "            with open(_canary, 'a', encoding='utf-8') as _handle:\n"
            "                _handle.write(json.dumps({'pid': os.getpid()}) + '\\n')\n",
        ),
    ),
    GLOBAL_SOURCE,
)

APPLYING_SECOND_SOURCE: Final = (
    APPLYING_SOURCE + "# A second version of the same name.\n"
)

MISREADING_SOURCE: Final = _replaced(
    _replaced(
        APPLYING_SOURCE,
        'path = artifact_paths.get("model", run_root) / _STATE',
        "path = run_root / _STATE",
    ),
    f'name = "{APPLYING_NAME}"',
    f'name = "{MISREADING_NAME}"',
)

_APPLY_START: Final = "    def apply(self, df: pd.DataFrame) -> pd.DataFrame:\n"

PARALLEL_APPLYING_SOURCE: Final = reduce(
    lambda source, edit: _replaced(source, *edit),
    (
        (f'name = "{APPLYING_NAME}"', f'name = "{PARALLEL_APPLYING_NAME}"'),
        ("parallelizable = False", "parallelizable = True"),
        (
            _APPLY_START,
            _APPLY_START + f"        _canary = os.environ.get({CANARY_VARIABLE!r})\n"
            "        if _canary:\n"
            "            with open(_canary, 'a', encoding='utf-8') as _handle:\n"
            "                _handle.write(json.dumps({'applied': os.getpid()}) + '\\n')\n",
        ),
    ),
    APPLYING_SOURCE,
)


def built_by_class(cls: type[Feature], params: Mapping[str, object] | None) -> Feature:
    """An instance of *cls*, built by calling the class with default inputs.

    A worker builds a feature the same way, by calling its class directly
    rather than through ``build_feature``, so the instance holds what its
    constructor gave it. The ``Feature`` protocol declares no constructor, so
    the call goes through ``type.__call__`` and the result is narrowed by
    ``isinstance``.
    """
    built: object = type.__call__(cls, None, None if params is None else dict(params))
    assert isinstance(built, cls)
    return built


def fit_result(ds: Dataset, slug: str, entry: tuple[str, str]) -> Result[str]:
    """Run the definition *slug* over the one entry *entry*.

    The run is a fit of a definition that fits, and tracks no attempt.

    Args:
        ds: The dataset that defines *slug*.
        slug: The ``def:`` slug, bare or pinned to a digest.
        entry: The ``(group, sequence)`` the run covers.

    Returns:
        The result of the run.
    """
    from mosaic.core.pipeline.definitions.define import load_feature_definition
    from mosaic.core.pipeline.run import run_feature
    from mosaic.core.scope import Scope

    feature = load_feature_definition(ds, slug)()
    return run_feature(ds, feature, scope=Scope(entries=[entry]), track=False)


def fit_over(ds: Dataset, slug: str, entry: tuple[str, str]) -> str:
    """The run id of the definition *slug* run over the one entry *entry*.

    Args:
        ds: The dataset that defines *slug*.
        slug: The ``def:`` slug, bare or pinned to a digest.
        entry: The ``(group, sequence)`` the run covers.
    """
    run_id = fit_result(ds, slug, entry).run_id
    assert run_id is not None
    return run_id


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


def imported_definitions() -> list[str]:
    """The names of the definition modules this process has imported."""
    return [name for name in sys.modules if name.startswith("mosaic_definitions.")]


def forget_imported_definitions() -> None:
    """Remove every definition module from ``sys.modules``."""
    for name in imported_definitions():
        del sys.modules[name]


def plan_refused(
    ds: Dataset,
    recipe: Recipe,
    *,
    bind: Mapping[str, BoundRef] | None = None,
    pins: Mapping[str, str] | None = None,
    applies_request_id: str | None = None,
) -> Plan:
    """The plan of *recipe*, after checking that a refusing plan raises its problems.

    The plan is made twice: without refusing, and refusing, which must raise
    ``RecipeRefused`` with the same problems. The definition modules imported
    before are forgotten first, and neither plan may import one.

    Returns:
        The plan made without refusing.
    """
    forget_imported_definitions()
    plan = plan_pipeline(
        ds,
        recipe,
        bind=bind,
        pins=pins,
        applies_request_id=applies_request_id,
        refuse=False,
    )
    with pytest.raises(RecipeRefused) as refused:
        _ = plan_pipeline(
            ds, recipe, bind=bind, pins=pins, applies_request_id=applies_request_id
        )
    assert not imported_definitions()
    assert refused.value.plan.problems == plan.problems
    return plan


@contextmanager
def definitions_forgotten() -> Generator[None]:
    """Remove the definition modules imported inside, and the caches filled inside.

    Two datasets that define one source share a module name. A module left in
    ``sys.modules`` answers a later test's load without reading its files.
    """
    before = set(imported_definitions())
    clear_definition_caches()
    try:
        yield
    finally:
        clear_definition_caches()
        for name in set(imported_definitions()) - before:
            _ = sys.modules.pop(name)


def noting_definition_loads(
    monkeypatch: pytest.MonkeyPatch, builders: ModuleType
) -> list[tuple[str, str]]:
    """Note each step feature that *builders* builds and each definition loaded.

    Replaces ``builders.build_step_feature`` and ``loader.load_definition`` for the
    test. The events, in order, are ``("build", <the step's feature slug>)`` and
    ``("load", "def:<name>@<digest>")``.
    """
    events: list[tuple[str, str]] = []
    build = builders.build_step_feature
    load = loader.load_definition

    def noting_build(spec: StepSpec) -> Feature:
        events.append(("build", spec.feature))
        return build(spec)

    def noting_load(definitions_dir: Path, name: str, digest: str) -> type[Feature]:
        events.append(("load", f"def:{name}@{digest}"))
        return load(definitions_dir, name, digest)

    monkeypatch.setattr(builders, "build_step_feature", noting_build)
    monkeypatch.setattr(loader, "load_definition", noting_load)
    return events
