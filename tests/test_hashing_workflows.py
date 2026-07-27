"""The five hashing workflows, as executable assertions.

Transcribed from the workflow walkthroughs (H1-H5), which specify what each one
must assert and note that two synthetic sequences of a few hundred frames cover
all five. The fixture is ``scenario_dataset`` in ``conftest.py``.

These are the milestone gates: H5 gates M1, H1 and H2 gate M2, H3 gates M3 and
M4, H4 gates M4. Assertions describing target state are marked
``xfail(strict=True)`` naming the milestone that closes them, so the suite
doubles as a progress meter -- when a milestone lands, its scenarios report XPASS
and fail until the markers come off.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from mosaic.core.pipeline._utils import Scope
from mosaic.core.pipeline.run import compute_run_id, run_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.types import (
    InputRequire,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
)

from .conftest import add_track_sequences, track_sequences

# --- Mock features -----------------------------------------------------------
#
# Deliberately trivial: these scenarios are about identity and blast radius, not
# about what a feature computes. Two shapes are enough -- one that computes each
# sequence from itself, and one that fits across the scope.


class _FeatureBase:
    """The protocol members these scenarios do not care about.

    ``Inputs`` is deliberately absent: each scenario feature narrows it to a
    different item type, and a nested class cannot be narrowed in a subclass.
    """

    version = "0.1"
    parallelizable = True
    scope_dependent = False

    class Params(Params):
        pass

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
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


class _PerFrame(_FeatureBase):
    """Scope-free: computes sequence S from S alone."""

    name = "scenario-per-frame"

    class Inputs(Inputs[TrackInput]):
        _require: ClassVar[InputRequire] = "any"

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self.inputs = self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)


class _GlobalFit(_PerFrame):
    """Scope-dependent: the scope is the training set, so it enters identity."""

    name = "scenario-global-fit"
    version = "0.1"
    scope_dependent = True

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        return (run_root / "state.txt").exists()

    def save_state(self, run_root: Path) -> None:
        (run_root / "state.txt").write_text("fitted")


class _FromUpstream(_FeatureBase):
    """Consumes another feature's result, so its inputs carry an upstream identity."""

    name = "scenario-downstream"

    class Inputs(Inputs[Result]):
        _require: ClassVar[InputRequire] = "any"

    def __init__(self, upstream_run_id: str) -> None:
        self.inputs = self.Inputs((Result(feature="upstream", run_id=upstream_run_id),))
        self.params = self.Params.from_overrides(None)


def _run_dir(dataset: Dataset, feature_dir: str, result: Result) -> Path:
    """The run root a completed run wrote to.

    ``Result.run_id`` is optional on the type because a result can be
    constructed before a run happens; here it always exists.
    """
    run_id = result.run_id
    assert run_id is not None, "a completed run always carries its identifier"
    return dataset.get_root("features") / feature_dir / run_id


def _outputs(dataset: Dataset, feature_dir: str, result: Result) -> set[str]:
    return {p.name for p in _run_dir(dataset, feature_dir, result).glob("*.parquet")}


# --- H1: cold path -- where everything lands the first time -------------------


def test_h1_cold_run_lands_where_expected(scenario_dataset: Dataset) -> None:
    """After a cold run, the expected paths exist and hold one file per sequence."""
    result = run_feature(scenario_dataset, _PerFrame())
    written = _outputs(scenario_dataset, "scenario-per-frame__from__tracks", result)
    assert written == {"seq_a.parquet", "seq_b.parquet"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M5: tracker intermediates still default under tracks_raw/trex and inference "
        "writes a top-level predictions/ root. Closes when the _tracking root lands "
        "(implementation item 8.1)."
    ),
)
def test_h1_tracking_intermediates_are_separated_from_results(
    scenario_dataset: Dataset,
) -> None:
    """``tracks/`` means standardized results; every intermediate goes elsewhere."""
    roots = {p.name for p in Path(scenario_dataset.get_root("tracks")).iterdir()}
    assert "trex" not in roots, "a tracker intermediate is living inside tracks/"
    assert (scenario_dataset.get_root("_tracking")).exists()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M3: derived media is named positionally and media/ is not organized by "
        "artifact kind. Closes with implementation items 7.1 and 7.3."
    ),
)
def test_h1_derived_media_is_organized_by_kind(
    scenario_dataset: Dataset,
) -> None:
    """``media/`` carries no sequence semantics; every child of it is a kind."""
    media = scenario_dataset.get_root("media")
    assert (media / "transcode").is_dir()
    assert track_sequences(scenario_dataset)
    assert not any(
        (media / name).exists() for name in track_sequences(scenario_dataset)
    )


# --- H2: recipe change -- same input, different how ---------------------------


def test_h2_upstream_identity_changes_the_run_id_not_the_directory() -> None:
    """Two upstream variants: two identifiers, one storage directory.

    The invariant the whole scenario rests on. If the upstream identity reached
    the storage suffix, ``features/<name>/`` would fragment into one directory
    and one index per upstream variant, and every tool that enumerates a
    feature's history would see partitioned universes.
    """
    first = _FromUpstream("0.1-aaaaaaaaaa")
    second = _FromUpstream("0.1-bbbbbbbbbb")

    first_id, _ = compute_run_id(first, None, None, Scope())
    second_id, _ = compute_run_id(second, None, None, Scope())

    assert first_id != second_id, "the resolved upstream identity must reach the run_id"
    assert first.inputs.storage_suffix() == second.inputs.storage_suffix(), (
        "the upstream identity must not reach the storage directory name"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M2: tracks are not variant-addressed, so two tracker settings cannot "
        "coexist -- all five writers target one flat path behind an exists() skip "
        "and the second is discarded. Closes with Stage 3."
    ),
)
def test_h2_two_tracks_variants_coexist(scenario_dataset: Dataset) -> None:
    """Two settings, two variants, two feature runs, one storage directory."""
    tracks = scenario_dataset.get_root("tracks")
    variants = [p for p in tracks.iterdir() if p.is_dir()]
    assert len(variants) >= 2, (
        "tracks/ holds one flat namespace, not variant directories"
    )


# --- H3: source change -- the input moved -------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M3/M4: no composition hash and no consumed-root record exist, so the blast "
        "radius cannot be enumerated. Closes with Stage 5 and item 6.2."
    ),
)
def test_h3_case1_membership_change_invalidates_tracks_but_not_derivatives(
    scenario_dataset: Dataset,
) -> None:
    """Reorder or reassign: tracks and features go, transcodes and frames stay.

    Tracking consumed the composition -- frames are concatenated into one
    sequence-global index, so order genuinely is part of its input. Derivatives
    are named by source uid, so nothing moves and nothing re-encodes.
    """
    raise NotImplementedError("requires the composition hash and consumed-root record")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M3/M4: same prerequisites as case 1. This is the case that ruled out a "
        "single per-sequence hash, so it is the one most worth having as a test."
    ),
)
def test_h3_case2_a_new_source_invalidates_nothing(
    scenario_dataset: Dataset,
) -> None:
    """A track-only sequence gains its first video: the delete set is empty.

    Nothing that consumes video had ever been runnable here, so the change adds
    capability rather than invalidating anything. A single hash over all sources
    would have thrown away a feature chain months deep to no purpose.
    """
    raise NotImplementedError("requires per-root composition hashes")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M3/M4: the tracks row now records `producer` and `consumed_source_roots` "
        "(item 2.4, Stage 2), so which producer made a table and which roots it "
        "read are both answerable. What is still missing is the composition hash "
        "that says a root *changed*, and the reverse-dependency walk that turns "
        "that into a blast radius -- Stage 5 and item 6.2."
    ),
)
def test_h3_case3_only_the_branch_whose_source_changed_is_invalidated(
    scenario_dataset: Dataset,
) -> None:
    """Two producers, one sequence: a tracks_raw change spares the tracked variant.

    Both variants are ``tracks/<something>/A.parquet`` with identical columns,
    deliberately, so a downstream feature never has to ask which produced them.
    The index row is where the answer is kept for the one job that needs it.
    """
    raise NotImplementedError("requires the consumed-root record")


# --- H4: human input -- what cannot be recomputed -----------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M4: interactive correction and the promotion gesture do not exist "
        "(implementation item 8.6, itself gated on open item O1)."
    ),
)
def test_h4_a_promoted_correction_lands_in_a_source_root_with_lineage(
    scenario_dataset: Dataset,
) -> None:
    """A correction cannot be recomputed, so it is source, not a derived variant."""
    raise NotImplementedError("requires the promotion gesture")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M4: the delete set does not exist yet, so its carve-outs cannot be "
        "asserted. Closes with items 6.2 and 6.4."
    ),
)
def test_h4_annotated_frames_and_labels_are_never_in_a_delete_set(
    scenario_dataset: Dataset,
) -> None:
    """An annotated frame stays valid across a rearrangement; labels are remapped.

    An annotated frame teaches a model what a subject looks like, which is true
    regardless of which sequence the image came from. Converted labels shift
    with the cumulative frame offsets, and both offsets are derivable from the
    per-video frame counts already in the media index, so a remap is exact.
    """
    raise NotImplementedError("requires the scoped delete set")


# --- H5: scope -- the dataset grew --------------------------------------------


def test_h5_widening_the_scope_does_not_move_a_per_frame_identifier(
    scenario_dataset: Dataset,
) -> None:
    """Two sequences become three: same identifier, and only the new one computes.

    Scope-freeness is not an optimization. It is the difference between one
    sequence computing and two hundred.
    """
    feature_dir = "scenario-per-frame__from__tracks"
    narrow = run_feature(scenario_dataset, _PerFrame())
    assert _outputs(scenario_dataset, feature_dir, narrow) == {
        "seq_a.parquet",
        "seq_b.parquet",
    }

    narrow_dir = _run_dir(scenario_dataset, feature_dir, narrow)
    before = {
        name: (narrow_dir / name).stat().st_mtime_ns
        for name in ("seq_a.parquet", "seq_b.parquet")
    }

    add_track_sequences(scenario_dataset, "seq_c")
    wide = run_feature(scenario_dataset, _PerFrame())

    assert wide.run_id == narrow.run_id, (
        "the identifier never contained a sequence name"
    )
    assert _outputs(scenario_dataset, feature_dir, wide) == {
        "seq_a.parquet",
        "seq_b.parquet",
        "seq_c.parquet",
    }
    wide_dir = _run_dir(scenario_dataset, feature_dir, wide)
    after = {name: (wide_dir / name).stat().st_mtime_ns for name in before}
    assert after == before, "an already-computed sequence was recomputed"


def test_h5_widening_the_scope_moves_a_global_fit_identifier(
    scenario_dataset: Dataset,
) -> None:
    """A fit over three sequences is a different artifact from a fit over two."""
    narrow = run_feature(scenario_dataset, _GlobalFit())

    add_track_sequences(scenario_dataset, "seq_c")
    wide = run_feature(scenario_dataset, _GlobalFit())

    assert wide.run_id != narrow.run_id, "the fit scope must reach the identifier"
    feature_dir = "scenario-global-fit__from__tracks"
    narrow_state = _run_dir(scenario_dataset, feature_dir, narrow) / "state.txt"
    assert narrow_state.exists(), (
        "the narrower fit must remain on disk and valid for its own scope"
    )
    assert (_run_dir(scenario_dataset, feature_dir, wide) / "state.txt").exists()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "M3: the scope term carries bare (group, sequence) pairs; composition "
        "hashes do not exist yet. Closes with implementation item 4.4."
    ),
)
def test_h5_scope_term_carries_composition_hashes() -> None:
    """The scope term is a sorted list of ``(group, sequence, composition hash)``.

    A list, never a set of bare hashes: cardinality and distinctness have to
    survive the hash.
    """
    raise NotImplementedError("requires per-root composition hashes")
