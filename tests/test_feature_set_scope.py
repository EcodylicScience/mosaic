"""A feature that re-reads tracks itself is told which variant the run resolved.

``run_feature`` builds a :class:`Scope` carrying the tracks variant the run was
computed from, and hands the feature a manifest built from it. The three media
features -- ``overlay``, ``egocentric-crop`` and ``interaction-crop-pipeline`` --
do not only consume that manifest: they call back into ``play_video`` /
``load_tracks``, which resolve a variant of their own.

``set_scope`` is the hook that closes the gap, and each of those three defines
one. Nothing called it. So on any dataset where a single entry carries two
recipes -- the ordinary state once a tracker has run beside a conversion --
``select_variant_rows`` refused, and ``run_feature(..., tracks_run_id=...)`` was
silently ineffective for precisely the features whose job is to draw the result.
"""

from pathlib import Path
from typing import ClassVar

import pandas as pd

from mosaic.core.dataset import Dataset, new_dataset_manifest, open_dataset
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.types.feature import EmitsLevel
from mosaic.core.pipeline.types import (
    Inputs,
    InputRequire,
    InputStream,
    Params,
    TrackInput,
)
from tests.helpers.tracks import add_tracks_variant

CONVERTED = "convert-demo.0.1-1111111111"
TRACKED = "trex.0.1-2222222222"


class _ScopeRecorder:
    """Records the scope it was handed, and nothing else."""

    name = "scope-recorder"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "as-input"
    consumed_roots: tuple[str, ...] = ()

    class Params(Params):
        pass

    class Inputs(Inputs[TrackInput]):
        _require: ClassVar[InputRequire] = "any"

    def __init__(self) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(None)
        self.seen: tuple[str, ...] | None = None

    def set_scope(self, scope: object) -> None:
        self.seen = tuple(getattr(scope, "tracks_variants", ()))

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return True

    def fit(self, inputs: InputStream) -> None:
        pass

    def save_state(self, run_root: Path) -> None:
        pass

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "value": df["X"]})


def _two_variant_dataset(base: Path) -> Dataset:
    """One entry carrying both a conversion and a tracker run.

    The shape every dataset takes the moment a tracker runs beside the tables it
    was compared against, and the one where guessing a variant is refused.
    """
    new_dataset_manifest(name="two-variants", base_dir=base)
    dataset = open_dataset(base)
    add_tracks_variant(dataset, CONVERTED, "seq_a", std_format="mosaic_v1")
    add_tracks_variant(dataset, TRACKED, "seq_a", std_format="mosaic_v1")
    return dataset


def test_set_scope_receives_the_resolved_tracks_variant(tmp_path: Path) -> None:
    """The variant the run resolved reaches the feature, rather than nothing."""
    dataset = _two_variant_dataset(tmp_path / "ds")

    feature = _ScopeRecorder()
    run_feature(dataset, feature, tracks_run_id=TRACKED)

    assert feature.seen == (TRACKED,), (
        "set_scope was not called with the resolved variant, so a feature that "
        "re-reads tracks cannot tell two recipes apart"
    )


def test_each_variant_is_reported_as_itself(tmp_path: Path) -> None:
    """Not a fixed value: whichever variant was asked for is the one delivered."""
    dataset = _two_variant_dataset(tmp_path / "ds")

    for variant in (CONVERTED, TRACKED):
        feature = _ScopeRecorder()
        run_feature(dataset, feature, tracks_run_id=variant)
        assert feature.seen == (variant,)
