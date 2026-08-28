"""The one place a scope refusal is raised, and what it tells the caller.

Two of the four cases read the selector rather than the resolved entries. An
unset selector and a selector naming an absent group both resolve to zero
entries and mean opposite things. Unset covers every indexed entry, and empty
covers none.
"""

import inspect

import pytest

from mosaic.core.pipeline import ops as ops_module
from mosaic.core.pipeline.ops import ScopeRefused, check_scope_takes, run_op
from mosaic.core.scope import Scope
from tests.helpers import names_called_by, resolved_scope


class TestNone:
    def test_an_unset_selector_passes(self) -> None:
        check_scope_takes("train-pose", "none", resolved_scope(Scope()))

    def test_a_set_selector_is_refused(self) -> None:
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes(
                "train-pose", "none", resolved_scope(Scope(groups=["A"]), ("A", "one"))
            )
        message = str(caught.value)
        assert "train-pose" in message
        assert "Scope()" in message

    def test_a_set_selector_resolving_to_nothing_is_refused(self) -> None:
        """Read from the selector, which is set here while the entries are not."""
        with pytest.raises(ScopeRefused):
            check_scope_takes(
                "train-pose", "none", resolved_scope(Scope(groups=["absent"]))
            )


class TestAny:
    def test_an_unset_selector_passes(self) -> None:
        check_scope_takes("trex", "any", resolved_scope(Scope()))

    def test_a_resolution_of_zero_passes(self) -> None:
        """Naming a group that holds nothing runs nothing, which is not an error."""
        check_scope_takes("trex", "any", resolved_scope(Scope(groups=["absent"])))

    def test_many_entries_pass(self) -> None:
        check_scope_takes(
            "trex",
            "any",
            resolved_scope(Scope(groups=["A"]), ("A", "one"), ("A", "two")),
        )


class TestAtLeastOne:
    def test_an_unset_selector_is_refused(self) -> None:
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes("transcode", "at-least-one", resolved_scope(Scope()))
        message = str(caught.value)
        assert "transcode" in message
        assert "every video" in message
        assert "Scope(entries=" in message

    def test_an_unset_selector_is_refused_even_covering_everything(self) -> None:
        """Decided from the selector, never from how much it resolves to.

        ``Dataset.resolve_scope`` hands an unset selector an empty entry set, so
        a count-only check refuses this today for the wrong reason. Three
        feature-side producers already build a ``ResolvedScope`` holding entries
        under a defaulted unset selector, and a count-only check admits the
        moment one of those shapes reaches here.
        """
        covering_everything = resolved_scope(
            Scope(), ("A", "one"), ("A", "two"), ("B", "one")
        )
        with pytest.raises(ScopeRefused, match="every video"):
            check_scope_takes("transcode", "at-least-one", covering_everything)

    def test_a_resolution_of_zero_is_refused_and_names_the_selector(self) -> None:
        scope = Scope(groups=["absent"])
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes("transcode", "at-least-one", resolved_scope(scope))
        assert "absent" in str(caught.value)

    def test_one_entry_passes(self) -> None:
        named = Scope(entries=[("A", "one")])
        check_scope_takes(
            "transcode", "at-least-one", resolved_scope(named, ("A", "one"))
        )


class TestExactlyOne:
    def test_one_entry_passes(self) -> None:
        named = Scope(entries=[("A", "one")])
        check_scope_takes(
            "export-store", "exactly-one", resolved_scope(named, ("A", "one"))
        )

    def test_more_than_one_is_refused_and_lists_them(self) -> None:
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes(
                "export-store",
                "exactly-one",
                resolved_scope(Scope(groups=["A"]), ("A", "one"), ("A", "two")),
            )
        message = str(caught.value)
        assert "2" in message
        assert "A" in message and "one" in message and "two" in message

    def test_an_unset_selector_is_refused(self) -> None:
        with pytest.raises(ScopeRefused, match="every imgstore"):
            check_scope_takes("export-store", "exactly-one", resolved_scope(Scope()))

    def test_an_unset_selector_is_refused_even_covering_everything(self) -> None:
        """The companion to ``at-least-one``'s, for the same reason."""
        covering_everything = resolved_scope(Scope(), ("A", "one"), ("B", "one"))
        with pytest.raises(ScopeRefused, match="every imgstore"):
            check_scope_takes("export-store", "exactly-one", covering_everything)

    def test_the_unscoped_refusal_offers_only_the_single_entry_form(self) -> None:
        """A group selector resolves to several, which the count branch refuses."""
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes("export-store", "exactly-one", resolved_scope(Scope()))
        assert "Scope(groups=" not in str(caught.value)

    def test_a_resolution_of_zero_is_refused(self) -> None:
        with pytest.raises(ScopeRefused, match="matches no entry"):
            check_scope_takes(
                "export-store", "exactly-one", resolved_scope(Scope(groups=["absent"]))
            )


class TestEachRefusalSaysItsOwnReason:
    """Five refusals, five messages, and none of them borrows another's."""

    def test_the_unscoped_refusal_does_not_report_an_empty_match(self) -> None:
        """Read against a selector that resolved to entries, which bites."""
        covering_everything = resolved_scope(Scope(), ("A", "one"), ("B", "one"))
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes("transcode", "at-least-one", covering_everything)
        assert "matches no entry" not in str(caught.value)

    def test_the_empty_refusal_does_not_report_an_unscoped_run(self) -> None:
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes(
                "transcode", "at-least-one", resolved_scope(Scope(groups=["absent"]))
            )
        message = str(caught.value)
        assert "matches no entry" in message
        assert "every video" not in message

    def test_an_empty_entry_list_reports_neither_of_the_other_two(self) -> None:
        """Naming zero entries read no index, and there are no names to check."""
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes(
                "transcode", "at-least-one", resolved_scope(Scope(entries=[]))
            )
        message = str(caught.value)
        assert "empty entry list" in message
        assert "matches no entry" not in message
        assert "every video" not in message

    def test_the_too_many_refusal_reports_the_count_and_the_entries(self) -> None:
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes(
                "export-store",
                "exactly-one",
                resolved_scope(Scope(groups=["A"]), ("A", "one"), ("A", "two")),
            )
        message = str(caught.value)
        assert "resolves 2" in message
        assert "matches no entry" not in message

    def test_the_scope_free_refusal_names_no_arity(self) -> None:
        with pytest.raises(ScopeRefused) as caught:
            check_scope_takes("train-pose", "none", resolved_scope(Scope(groups=["A"])))
        message = str(caught.value)
        assert "matches no entry" not in message
        assert "every video" not in message


class TestRunOpAcceptsAScope:
    """``run_op`` takes a scope and resolves it, and refuses nothing itself.

    ``transcode`` and ``export-store`` still declare their own entry fields and
    read those rather than this argument, so their callers leave it unset.
    ``check_scope_takes`` refuses an unset selector for ``"at-least-one"`` and
    ``"exactly-one"``, which those two declare. Calling the checker here would
    therefore refuse every ``transcode`` and ``export-store`` run in mosaic and
    in its consumers.

    The refusal is applied where a caller does name a scope. ``mosaic run
    --kind`` reaches it through ``split_op_scope``; see
    ``tests/test_tracker_scope.py``. It moves here once those two ops read the
    argument.
    """

    def test_run_op_takes_a_scope_keyword(self) -> None:
        parameter = inspect.signature(run_op).parameters.get("scope")
        assert parameter is not None
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is None

    def test_run_op_resolves_the_scope_and_checks_nothing(self) -> None:
        """The seam is the whole product here, and the refusal is not wired."""
        called = names_called_by(ops_module, "run_op")
        assert "resolve_scope" in called
        assert "check_scope_takes" not in called
