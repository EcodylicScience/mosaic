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
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.run import compute_run_id, run_feature
from mosaic.core.pipeline.tracks_index import read_tracks_index
from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.types import (
    InputRequire,
    Inputs,
    InputStream,
    Params,
    Result,
    TrackInput,
)

from .conftest import add_track_sequences, add_tracks_variant, track_sequences

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
    """``tracks/`` means standardized results; every intermediate goes elsewhere.

    Asserted through ``has_root`` rather than ``get_root``, which raises
    ``KeyError`` on an unset root -- so this used to die before reaching an
    assertion at all, and a strict xfail only proves a test fails, not that it
    fails for the reason its marker claims.
    """
    roots = {p.name for p in Path(scenario_dataset.get_root("tracks")).iterdir()}
    assert "trex" not in roots, "a tracker intermediate is living inside tracks/"
    assert scenario_dataset.has_root("_tracking"), "the _tracking root does not exist"
    assert scenario_dataset.get_root("_tracking").exists()


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


def test_h2_tracks_identity_changes_the_run_id_not_the_directory() -> None:
    """The same invariant for the other input kind, which had no test at all.

    ``storage_suffix()`` reads the ``"tracks"`` literal out of ``Inputs.root``,
    and item 3.3 leaves that literal in place precisely so this holds. Had the
    resolved variant been written into ``Inputs`` instead -- the obvious reading
    of "resolve the literal" -- ``features/<name>__from__tracks/`` would have
    become one directory per tracks recipe, silently, with every test still
    green: nothing else asserts the suffix for a tracks input.
    """
    feature = _PerFrame()
    one = Scope(tracks_variants=("convert-trex_npz.0.1-aaaaaaaaaa",))
    other = Scope(tracks_variants=("trex.0.1-bbbbbbbbbb",))

    first_id, _ = compute_run_id(feature, None, None, one)
    second_id, _ = compute_run_id(feature, None, None, other)
    unresolved_id, _ = compute_run_id(feature, None, None, Scope())

    assert first_id != second_id, "the resolved tracks identity must reach the run_id"
    assert unresolved_id not in {first_id, second_id}, (
        "a dataset naming no variant must not collide with one that does"
    )
    assert feature.inputs.storage_suffix() == "tracks", (
        "the tracks identity must not reach the storage directory name"
    )


def test_h2_an_unlabelled_tracks_index_leaves_the_identifier_alone() -> None:
    """The archived-analysis guarantee, at the level of the hash function.

    A dataset converted before tracks carried identities resolves to no variants
    at all, and an absent term digests differently from an empty one -- so its
    identifiers are the ones it already has on disk.
    """
    feature = _PerFrame()
    before, _ = compute_run_id(feature, None, None, Scope(entries={("", "seq_a")}))
    after, _ = compute_run_id(
        feature, None, None, Scope(entries={("", "seq_a")}, tracks_variants=())
    )
    assert before == after


def test_h2_two_tracks_variants_coexist(tmp_path: Path) -> None:
    """Two settings, two variants, two feature runs, one storage directory.

    The M2 gate. Before Stage 3 the second producer's table was discarded behind
    an ``exists() and not overwrite`` skip and its row overwrote the first's, so
    none of this was observable: one directory, one row, one identifier, and the
    numbers in the parquet belonging to whichever ran first.
    """
    manifest = new_dataset_manifest(name="two-variants", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    first_variant = "convert-trex_npz.0.1-aaaaaaaaaa"
    second_variant = "trex.0.1-bbbbbbbbbb"
    add_tracks_variant(dataset, first_variant, "seq_a")
    add_tracks_variant(dataset, second_variant, "seq_a")

    tracks = dataset.get_root("tracks")
    assert sorted(p.name for p in tracks.iterdir() if p.is_dir()) == [
        first_variant,
        second_variant,
    ]
    assert len(read_tracks_index(dataset)) == 2, "both rows must survive"

    first = run_feature(dataset, _PerFrame(), tracks_run_id=first_variant)
    second = run_feature(dataset, _PerFrame(), tracks_run_id=second_variant)

    assert first.run_id != second.run_id, "two variants must be two identifiers"

    # ...inside ONE storage directory and ONE index. If the tracks identity had
    # reached the storage suffix instead, every tool that enumerates a feature's
    # history would see one partitioned universe per tracks recipe.
    storage = dataset.get_root("features") / "scenario-per-frame__from__tracks"
    assert storage.is_dir()
    assert sorted(p.name for p in storage.iterdir() if p.is_dir()) == sorted(
        [str(first.run_id), str(second.run_id)]
    )
    index = feature_index(feature_index_path(dataset, storage.name))
    assert set(index.read()["run_id"]) == {first.run_id, second.run_id}


def test_h2_an_unpinned_run_over_two_variants_refuses_rather_than_guessing(
    tmp_path: Path,
) -> None:
    """The other half of the gate: no silent default between two recipes."""
    manifest = new_dataset_manifest(name="ambiguous", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    add_tracks_variant(dataset, "convert-trex_npz.0.1-aaaaaaaaaa", "seq_a")
    add_tracks_variant(dataset, "trex.0.1-bbbbbbbbbb", "seq_a")

    with pytest.raises(ValueError, match="tracks_run_id"):
        _ = run_feature(dataset, _PerFrame())


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
