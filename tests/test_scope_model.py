"""What a caller may ask a run to cover, and what that request resolves to.

The selector is a value with no dataset behind it. Resolving one enumerates the
entries it names against the media index. A feature run takes the same value,
resolves it against the indexes its inputs read, and records it beside the
entries it resolved to.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.speed_angvel import SpeedAngvel
from mosaic.core.dataset import Dataset
from mosaic.core.params import Params
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.index import feature_index, feature_index_path
from mosaic.core.pipeline.run import resolve_feature_identity, run_feature
from mosaic.core.pipeline.types import Inputs, InputStream, Result, TrackInput
from mosaic.core.pipeline.types.feature import EmitsLevel
from mosaic.core.scope import (
    SCOPE_PARAM_KEYS,
    Scope,
    camera_grain_refusal,
    entries_exclude_pair_refusal,
)


class TestExclusivity:
    def test_entries_alone_is_accepted(self) -> None:
        scope = Scope(entries=[("A", "one")])
        assert scope.entries == [("A", "one")]

    def test_groups_alone_is_accepted(self) -> None:
        assert Scope(groups=["A"]).groups == ["A"]

    def test_sequences_alone_is_accepted(self) -> None:
        assert Scope(sequences=["one"]).sequences == ["one"]

    def test_groups_and_sequences_together_are_accepted(self) -> None:
        scope = Scope(groups=["A"], sequences=["one"])
        assert (scope.groups, scope.sequences) == (["A"], ["one"])

    def test_entries_with_groups_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="entries"):
            _ = Scope(entries=[("A", "one")], groups=["A"])

    def test_entries_with_sequences_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="entries"):
            _ = Scope(entries=[("A", "one")], sequences=["one"])


class TestOneRefusalForBothVocabularies:
    """The sentence another command line refuses in is this model's own.

    mosaic-queue names the same selector as ``--entries`` / ``--groups`` /
    ``--sequences``. It held a copy of this rule, which had already drifted:
    it tested truthiness where the model tests presence, so an empty
    ``entries`` list beside a group passed there and was refused here.
    """

    def test_the_model_raises_the_shared_sentence(self) -> None:
        expected = entries_exclude_pair_refusal([("A", "one")], ["A"], None)
        with pytest.raises(ValidationError) as caught:
            _ = Scope(entries=[("A", "one")], groups=["A"])
        assert expected in str(caught.value)

    def test_a_prefix_spells_the_names_as_flags(self) -> None:
        refusal = entries_exclude_pair_refusal(
            [("A", "one")], ["A"], ["one"], prefix="--"
        )
        assert "--entries" in refusal
        assert "--groups and --sequences" in refusal

    def test_an_empty_selector_is_given_rather_than_absent(self) -> None:
        """Naming no entry is a statement, and it excludes the pair too."""
        assert entries_exclude_pair_refusal([], ["A"], None) != ""
        assert entries_exclude_pair_refusal([], None, None) == ""
        assert entries_exclude_pair_refusal(None, ["A"], ["one"]) == ""

    def test_only_the_selectors_given_are_named(self) -> None:
        """A selector nobody gave is left out of the sentence.

        Naming all three regardless tells a caller to remove a flag they never
        typed.
        """
        one = entries_exclude_pair_refusal([("A", "one")], ["A"], None)
        assert "groups" in one
        assert "sequences" not in one


class TestScopeParamKeys:
    def test_every_selector_is_covered(self) -> None:
        """The names no params model may declare, read by two command lines.

        A fourth selector added to the model widens this set, and both
        refusals cover it without being edited.
        """
        assert SCOPE_PARAM_KEYS == {"entries", "groups", "sequences"}


class TestUnset:
    def test_nothing_set_is_unset(self) -> None:
        assert Scope().is_unset

    def test_an_empty_entry_list_is_not_unset(self) -> None:
        """Naming no entry explicitly is a statement, not an omission."""
        assert not Scope(entries=[]).is_unset

    def test_a_named_selector_is_not_unset(self) -> None:
        assert not Scope(groups=["A"]).is_unset


class TestCameraGrain:
    def test_pairs_do_not_address_cameras(self) -> None:
        assert not Scope(entries=[("A", "one")]).addresses_cameras

    def test_triples_address_cameras(self) -> None:
        assert Scope(entries=[("A", "one", "cam0")]).addresses_cameras

    def test_an_unset_scope_does_not_address_cameras(self) -> None:
        assert not Scope().addresses_cameras

    def test_mixing_grains_is_refused(self) -> None:
        """One scope names entries or camera-entries, never both grains."""
        with pytest.raises(ValidationError):
            _ = Scope.model_validate({"entries": [("A", "one"), ("A", "two", "cam0")]})


class TestCameraGrainRefusal:
    """Refuse a camera-addressed selector at every wire, in one shared sentence.

    Sixteen of seventeen ops leave a camera narrowing unread. A grain accepted
    on a wire covers every camera of the entry under a selector that names one,
    and it reports success.
    """

    def test_an_absent_selector_is_not_refused(self) -> None:
        """A caller that named no scope at all passes ``None``."""
        assert camera_grain_refusal(None) == ""

    def test_a_pair_selector_is_not_refused(self) -> None:
        assert camera_grain_refusal(Scope(entries=[("A", "one")])) == ""

    def test_an_unset_selector_is_not_refused(self) -> None:
        assert camera_grain_refusal(Scope()) == ""

    def test_an_empty_entry_list_is_not_refused(self) -> None:
        assert camera_grain_refusal(Scope(entries=[])) == ""

    def test_a_camera_addressed_selector_is_refused(self) -> None:
        refusal = camera_grain_refusal(Scope(entries=[("A", "one", "cam0")]))
        assert "camera" in refusal
        assert "cam0" in refusal, "the refusal names what was asked for"

    def test_the_refusal_names_every_camera_in_order(self) -> None:
        refusal = camera_grain_refusal(
            Scope(entries=[("A", "one", "cam1"), ("A", "two", "cam0")])
        )
        assert "cam0, cam1" in refusal

    def test_the_sentence_names_no_command_line(self) -> None:
        """A RunSpec is built in Python by mosaic-api, not only from flags."""
        refusal = camera_grain_refusal(Scope(entries=[("A", "one", "cam0")]))
        assert "--" not in refusal
        assert "command line" not in refusal


class TestDuplicates:
    def test_a_repeated_entry_collapses_keeping_order(self) -> None:
        scope = Scope(entries=[("B", "two"), ("A", "one"), ("B", "two")])
        assert scope.entries == [("B", "two"), ("A", "one")]

    def test_a_repeated_camera_entry_collapses(self) -> None:
        scope = Scope(entries=[("A", "one", "cam0"), ("A", "one", "cam0")])
        assert scope.entries == [("A", "one", "cam0")]

    def test_two_cameras_of_one_entry_are_not_duplicates(self) -> None:
        scope = Scope(entries=[("A", "one", "cam0"), ("A", "one", "cam1")])
        assert len(scope.entries or []) == 2


class TestExtraKeys:
    def test_an_unknown_key_is_refused(self) -> None:
        """StrictModel forbids extras: a misspelled selector is an error."""
        with pytest.raises(ValidationError):
            _ = Scope.model_validate({"group": ["A"]})


class TestFrozen:
    """A constructed Scope is a value, fixed once construction completes."""

    def test_entries_cannot_be_reassigned(self) -> None:
        scope = Scope(entries=[("A", "one")])
        with pytest.raises(ValidationError):
            scope.entries = [("B", "two")]

    def test_groups_cannot_be_reassigned(self) -> None:
        scope = Scope(groups=["A"])
        with pytest.raises(ValidationError):
            scope.groups = ["B"]

    def test_sequences_cannot_be_reassigned(self) -> None:
        scope = Scope(sequences=["one"])
        with pytest.raises(ValidationError):
            scope.sequences = ["two"]


class TestResolvedScope:
    def test_it_records_the_selector_it_came_from(self) -> None:
        selector = Scope(groups=["A"])
        resolved = ResolvedScope(entries={("A", "one")}, selector=selector)
        assert resolved.selector == selector

    def test_an_unset_selector_is_the_default(self) -> None:
        assert ResolvedScope().selector.is_unset

    def test_empty_and_unscoped_are_distinguishable(self) -> None:
        """Naming a group that holds nothing is not naming no group."""
        empty = ResolvedScope(entries=set(), selector=Scope(groups=["absent"]))
        unscoped = ResolvedScope(entries=set(), selector=Scope())
        assert not empty.selector.is_unset
        assert unscoped.selector.is_unset


class TestOpEntries:
    """The entry list both commands hand an op."""

    def test_an_unset_selector_gives_none(self) -> None:
        """An op reads ``None`` as every indexed entry."""
        assert ResolvedScope(entries=set(), selector=Scope()).op_entries is None

    def test_an_empty_resolution_gives_an_empty_list(self) -> None:
        """A group that holds nothing runs nothing, and says so as ``[]``."""
        resolved = ResolvedScope(entries=set(), selector=Scope(groups=["absent"]))
        assert resolved.op_entries == []

    def test_the_entries_come_back_sorted(self) -> None:
        """A set has no order, and an op's entry list is compared and recorded."""
        resolved = ResolvedScope(
            entries={("B", "one"), ("A", "two"), ("A", "one")},
            selector=Scope(groups=["A", "B"]),
        )
        assert resolved.op_entries == [("A", "one"), ("A", "two"), ("B", "one")]


class TestResolveScope:
    """A selector resolves against the media index into the entries it names."""

    def test_an_unset_scope_reads_no_index(
        self, dataset_without_index: Dataset
    ) -> None:
        """A dataset with no media index still resolves an unscoped run."""
        resolved = dataset_without_index.resolve_scope(Scope())
        assert resolved.entries == set()
        assert resolved.selector.is_unset

    def test_an_entries_only_scope_reads_no_index(
        self, dataset_without_index: Dataset
    ) -> None:
        """An explicit entry list is already the enumeration."""
        resolved = dataset_without_index.resolve_scope(Scope(entries=[("A", "one")]))
        assert resolved.entries == {("A", "one")}

    def test_groups_enumerate_against_the_index(
        self, three_entry_dataset: Dataset
    ) -> None:
        resolved = three_entry_dataset.resolve_scope(Scope(groups=["A"]))
        assert resolved.entries == {("A", "one"), ("A", "two")}

    def test_a_sequence_name_repeated_across_groups_yields_both(
        self, three_entry_dataset: Dataset
    ) -> None:
        resolved = three_entry_dataset.resolve_scope(Scope(sequences=["one"]))
        assert resolved.entries == {("A", "one"), ("B", "one")}

    def test_naming_an_absent_group_resolves_to_nothing(
        self, three_entry_dataset: Dataset
    ) -> None:
        """Empty is not unscoped: a group holding nothing runs nothing."""
        resolved = three_entry_dataset.resolve_scope(Scope(groups=["absent"]))
        assert resolved.entries == set()
        assert not resolved.selector.is_unset

    def test_camera_entries_resolve_to_their_pairs(
        self, two_camera_dataset: Dataset
    ) -> None:
        """The entry set is pairs, and the camera is recorded on the selector.

        The dataset indexes both cameras of ``(A, one)``. Naming one of them
        resolves to the pair either way.
        """
        resolved = two_camera_dataset.resolve_scope(
            Scope(entries=[("A", "one", "cam0")])
        )
        assert resolved.entries == {("A", "one")}
        assert resolved.selector.addresses_cameras


class _ScopeCapture:
    """A do-nothing feature that keeps the ``ResolvedScope`` the run hands it.

    ``run_feature`` returns the entries it wrote and does not return what it
    resolved from. The ``set_scope`` callback is where a test reads the selector
    a run was given.
    """

    name = "scope-capture"
    version = "0.1"
    parallelizable = False
    scope_dependent = False
    accepts_overlap = False
    emits: EmitsLevel = "individual"
    consumed_roots: tuple[str, ...] = ()

    class Inputs(Inputs[TrackInput]):
        pass

    class Params(Params):
        pass

    def __init__(self) -> None:
        self._inputs = self.Inputs(("tracks",))
        self._params = self.Params.from_overrides(None)
        self.seen: ResolvedScope | None = None

    @property
    def inputs(self) -> _ScopeCapture.Inputs:
        return self._inputs

    @property
    def params(self) -> _ScopeCapture.Params:
        return self._params

    def set_scope(self, scope: ResolvedScope) -> None:
        self.seen = scope

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


class TestFeatureScope:
    """A feature run takes the selector, and covers the dataset without one."""

    def _covered(self, dataset: Dataset, result: Result[str]) -> set[tuple[str, str]]:
        """The entries *result* left index rows for."""
        index = feature_index(feature_index_path(dataset, result.feature))
        rows = index.read(run_id=result.run_id)
        return {
            (str(group), str(sequence))
            for group, sequence in zip(rows["group"], rows["sequence"], strict=True)
        }

    def test_an_entry_selector_narrows_the_run(self, scenario_dataset: Dataset) -> None:
        result = run_feature(
            scenario_dataset, SpeedAngvel(), scope=Scope(entries=[("", "seq_a")])
        )
        assert self._covered(scenario_dataset, result) == {("", "seq_a")}

    def test_an_omitted_scope_covers_the_dataset(
        self, scenario_dataset: Dataset
    ) -> None:
        result = run_feature(scenario_dataset, SpeedAngvel())
        assert self._covered(scenario_dataset, result) == {
            ("", "seq_a"),
            ("", "seq_b"),
        }

    def test_a_sequence_selector_narrows_the_run(
        self, scenario_dataset: Dataset
    ) -> None:
        result = run_feature(
            scenario_dataset, SpeedAngvel(), scope=Scope(sequences=["seq_b"])
        )
        assert self._covered(scenario_dataset, result) == {("", "seq_b")}

    def test_the_run_records_the_selector_it_was_given(
        self, scenario_dataset: Dataset
    ) -> None:
        """The resolution alone cannot say what was asked for."""
        feature = _ScopeCapture()
        scope = Scope(entries=[("", "seq_a")])
        _ = run_feature(scenario_dataset, feature, scope=scope)
        assert feature.seen is not None
        assert feature.seen.selector == scope
        assert feature.seen.entries == {("", "seq_a")}

    def test_an_omitted_scope_reaches_the_run_unset(
        self, scenario_dataset: Dataset
    ) -> None:
        feature = _ScopeCapture()
        _ = run_feature(scenario_dataset, feature)
        assert feature.seen is not None
        assert feature.seen.selector.is_unset

    def test_a_named_selector_that_resolves_to_nothing_stays_named(
        self, scenario_dataset: Dataset
    ) -> None:
        """An empty resolution under a named group is not an unscoped run."""
        feature = _ScopeCapture()
        _ = run_feature(scenario_dataset, feature, scope=Scope(groups=["absent"]))
        assert feature.seen is not None
        assert feature.seen.entries == set()
        assert not feature.seen.selector.is_unset

    def test_an_empty_selector_is_refused(self, scenario_dataset: Dataset) -> None:
        """Naming nothing reads two ways across the input kinds, and raises."""
        with pytest.raises(ValueError, match="selects nothing"):
            _ = run_feature(scenario_dataset, SpeedAngvel(), scope=Scope(entries=[]))

    def test_a_predicted_identity_records_the_selector(
        self, scenario_dataset: Dataset
    ) -> None:
        scope = Scope(entries=[("", "seq_a")])
        _, resolved = resolve_feature_identity(
            scenario_dataset, SpeedAngvel(), scope=scope
        )
        assert resolved.selector == scope
        assert resolved.entries == {("", "seq_a")}

    def test_prediction_refuses_a_selector_it_cannot_enumerate(
        self, scenario_dataset: Dataset
    ) -> None:
        """A group name needs an index that a cold step has yet to write."""
        with pytest.raises(ValueError, match="Enumerate them"):
            _ = resolve_feature_identity(
                scenario_dataset, SpeedAngvel(), scope=Scope(groups=[""])
            )

    def test_the_exclusion_rule_reaches_a_feature_run(self) -> None:
        """A caller cannot hand a feature run a selector the model refuses."""
        with pytest.raises(ValidationError, match="cannot be combined"):
            _ = Scope(entries=[("", "seq_a")], groups=[""])
