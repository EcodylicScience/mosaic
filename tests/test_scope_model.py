"""The scope selector: what a caller may ask a run to cover, and what it resolves to.

The selector is a value with no dataset behind it. Resolving one enumerates the
entries it names against the media index.
"""

import pytest
from pydantic import ValidationError

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.scope import Scope


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
