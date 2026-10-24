"""The shared fold helpers: dealing sequences into folds, and choosing a fold's model.

A classifier that cross-validates by sequence calls `cv_fold_assignment` once, at its
fit, and `out_of_fold_choice` for every entry it scores. The assignment is the
contract between a run and everything that reads it, so it is pinned here by its
properties: a partition, the same whatever order and process deal it, and a
function of the seed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import mosaic
import pandas as pd
import pytest

from mosaic.behavior.feature_library.helpers import (
    cv_block_assignment,
    cv_block_bounds,
    cv_fold_assignment,
    out_of_fold_choice,
    scored_entry,
)
from mosaic.behavior.feature_library.helpers import (
    _cv_block_key,  # pyright: ignore[reportPrivateUsage]
    _cv_entry_key,  # pyright: ignore[reportPrivateUsage]
)

ENTRIES = [(f"g{i % 3}", f"seq{i:02d}") for i in range(23)]


class TestFoldAssignment:
    def test_the_folds_partition_the_entries_and_differ_by_at_most_one(self) -> None:
        folds = cv_fold_assignment(ENTRIES, 5, seed=0)

        assert set(folds) == set(ENTRIES)
        sizes = [list(folds.values()).count(fold) for fold in range(5)]
        assert sorted(sizes) == [4, 4, 5, 5, 5]
        assert set(folds.values()) == set(range(5))

    def test_the_order_of_the_entries_changes_nothing(self) -> None:
        shuffled = ENTRIES[7:] + ENTRIES[:7][::-1]

        assert cv_fold_assignment(shuffled, 4, seed=3) == cv_fold_assignment(
            ENTRIES, 4, seed=3
        )

    def test_the_seed_changes_the_folds(self) -> None:
        assert cv_fold_assignment(ENTRIES, 5, seed=0) != cv_fold_assignment(
            ENTRIES, 5, seed=1
        )

    def test_a_repeated_entry_is_dealt_once(self) -> None:
        folds = cv_fold_assignment(ENTRIES + ENTRIES[:5], 5, seed=0)

        assert folds == cv_fold_assignment(ENTRIES, 5, seed=0)

    def test_the_process_that_deals_does_not_matter(self) -> None:
        """Python's `hash()` is salted per process, and the folds must not be."""
        program = (
            "import json; "
            "from mosaic.behavior.feature_library.helpers import cv_fold_assignment; "
            f"entries = {ENTRIES!r}; "
            "folds = cv_fold_assignment(entries, 5, seed=2); "
            "print(json.dumps([[g, s, folds[(g, s)]] for g, s in entries]))"
        )
        source = str(Path(mosaic.__file__).parent.parent)
        dealt: list[object] = []
        for hash_seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": source}
            done = subprocess.run(
                [sys.executable, "-c", program],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            dealt.append(json.loads(done.stdout))

        assert dealt[0] == dealt[1]
        expected = cv_fold_assignment(ENTRIES, 5, seed=2)
        assert dealt[0] == [[g, s, expected[(g, s)]] for g, s in ENTRIES]

    def test_the_key_of_a_split_name_is_not_that_of_another_split(self) -> None:
        """("a", "bc") and ("ab", "c") spell "abc" twice, and are two entries."""
        assert _cv_entry_key(0, "a", "bc") != _cv_entry_key(0, "ab", "c")
        assert _cv_entry_key(0, "", "x") != _cv_entry_key(0, "x", "")
        assert _cv_entry_key(0, "a", "b") != _cv_entry_key(1, "a", "b")

    def test_a_blank_group_is_an_entry_like_any_other(self) -> None:
        folds = cv_fold_assignment([("", "a"), ("", "b"), ("x", "a")], 3, seed=0)

        assert sorted(folds.values()) == [0, 1, 2]

    def test_one_fold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            cv_fold_assignment(ENTRIES, 1, seed=0)

    def test_more_folds_than_entries_is_refused_with_the_count(self) -> None:
        with pytest.raises(ValueError, match=r"23 .*train.* entries"):
            cv_fold_assignment(ENTRIES, 24, seed=0)


class TestOutOfFoldChoice:
    FOLDS = {("g", "a"): 0, ("g", "b"): 1, ("", "12"): 2}

    @staticmethod
    def _entry(group: object, sequence: object, n: int = 3) -> pd.DataFrame:
        return pd.DataFrame(
            {"frame": range(n), "group": [group] * n, "sequence": [sequence] * n}
        )

    def test_an_entry_in_a_fold_is_scored_by_that_fold(self) -> None:
        assert out_of_fold_choice(self._entry("g", "b"), self.FOLDS, "xgboost") == 1

    def test_the_identity_is_read_as_text(self) -> None:
        assert out_of_fold_choice(self._entry("", 12), self.FOLDS, "xgboost") == 2

    def test_an_entry_in_no_fold_has_no_choice(self) -> None:
        assert (
            out_of_fold_choice(self._entry("test", "z"), self.FOLDS, "xgboost") is None
        )

    @pytest.mark.parametrize("missing", ["group", "sequence"])
    def test_a_table_without_an_identity_column_is_refused_naming_the_feature(
        self, missing: str
    ) -> None:
        frame = self._entry("g", "a").drop(columns=[missing])

        with pytest.raises(ValueError, match=rf"^\[xgboost\] .*{missing}"):
            out_of_fold_choice(frame, self.FOLDS, "xgboost")

    def test_a_table_of_two_entries_is_refused(self) -> None:
        frame = pd.concat([self._entry("g", "a"), self._entry("g", "b")])

        with pytest.raises(ValueError, match=r"^\[xgboost\] .*more than one entry"):
            out_of_fold_choice(frame, self.FOLDS, "xgboost")


class TestBlockBounds:
    def test_the_blocks_tile_the_range_and_differ_by_at_most_one_frame(self) -> None:
        blocks = cv_block_bounds(10, 107, 4)

        assert [index for index, _, _ in blocks] == [0, 1, 2, 3]
        assert blocks[0][1] == 10
        assert blocks[-1][2] == 107
        for (_, _, stop), (_, start, _) in zip(blocks, blocks[1:]):
            assert stop == start
        lengths = [stop - start for _, start, stop in blocks]
        assert max(lengths) - min(lengths) <= 1
        assert sum(lengths) == 97

    def test_an_even_range_is_cut_at_whole_multiples(self) -> None:
        assert cv_block_bounds(0, 60, 4) == [
            (0, 0, 15),
            (1, 15, 30),
            (2, 30, 45),
            (3, 45, 60),
        ]

    def test_a_range_shorter_than_the_blocks_leaves_the_empty_ones_out(self) -> None:
        blocks = cv_block_bounds(5, 8, 5)

        assert all(stop > start for _, start, stop in blocks)
        assert sum(stop - start for _, start, stop in blocks) == 3
        assert len(blocks) == 3
        assert [index for index, _, _ in blocks] != [0, 1, 2]

    def test_an_empty_range_has_no_block(self) -> None:
        assert cv_block_bounds(7, 7, 3) == []

    def test_the_cut_depends_on_the_range_alone(self) -> None:
        assert cv_block_bounds(3, 50, 6) == cv_block_bounds(3, 50, 6)


BLOCKS = [(g, s, b) for g, s in ENTRIES[:4] for b in range(3)]


class TestBlockAssignment:
    def test_every_block_has_a_fold_and_the_counts_differ_by_at_most_one(self) -> None:
        folds = cv_block_assignment(BLOCKS, 5, seed=0)

        assert set(folds) == set(BLOCKS)
        sizes = [list(folds.values()).count(fold) for fold in range(5)]
        assert max(sizes) - min(sizes) <= 1
        assert set(folds.values()) == set(range(5))

    def test_one_entry_still_fills_the_folds(self) -> None:
        folds = cv_block_assignment([("g", "only", b) for b in range(4)], 4, seed=2)

        assert sorted(folds.values()) == [0, 1, 2, 3]

    def test_the_order_of_the_blocks_changes_nothing(self) -> None:
        assert cv_block_assignment(BLOCKS[::-1], 4, seed=3) == cv_block_assignment(
            BLOCKS, 4, seed=3
        )

    def test_the_seed_changes_the_folds(self) -> None:
        assert cv_block_assignment(BLOCKS, 4, seed=0) != cv_block_assignment(
            BLOCKS, 4, seed=1
        )

    def test_a_repeated_block_is_dealt_once(self) -> None:
        assert cv_block_assignment(BLOCKS + BLOCKS[:3], 4, seed=0) == (
            cv_block_assignment(BLOCKS, 4, seed=0)
        )

    def test_the_fold_of_a_block_is_its_rank_by_sha256_of_the_json_key_mod_folds(
        self,
    ) -> None:
        import hashlib

        ranked = sorted(
            BLOCKS,
            key=lambda block: hashlib.sha256(
                json.dumps([7, *block], separators=(",", ":")).encode()
            ).hexdigest(),
        )

        folds = cv_block_assignment(BLOCKS, 5, seed=7)

        assert folds == {block: rank % 5 for rank, block in enumerate(ranked)}

    def test_the_process_that_deals_does_not_matter(self) -> None:
        program = (
            "import json; "
            "from mosaic.behavior.feature_library.helpers import cv_block_assignment; "
            f"blocks = {BLOCKS!r}; "
            "folds = cv_block_assignment(blocks, 4, seed=2); "
            "print(json.dumps([folds[b] for b in blocks]))"
        )
        source = str(Path(mosaic.__file__).parent.parent)
        dealt: list[object] = []
        for hash_seed in ("1", "2"):
            env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": source}
            done = subprocess.run(
                [sys.executable, "-c", program],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            dealt.append(json.loads(done.stdout))

        assert dealt[0] == dealt[1]
        expected = cv_block_assignment(BLOCKS, 4, seed=2)
        assert dealt[0] == [expected[b] for b in BLOCKS]

    def test_the_key_of_a_split_name_is_not_that_of_another_split(self) -> None:
        assert _cv_block_key(0, "a", "bc", 0) != _cv_block_key(0, "ab", "c", 0)
        assert _cv_block_key(0, "a", "b", 0) != _cv_block_key(0, "a", "b", 1)
        assert _cv_block_key(0, "a", "b", 0) != _cv_block_key(1, "a", "b", 0)

    def test_one_fold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            cv_block_assignment(BLOCKS, 1, seed=0)

    def test_more_folds_than_blocks_is_refused_with_the_count(self) -> None:
        with pytest.raises(ValueError, match=r"cannot deal 12 blocks into 13 folds"):
            cv_block_assignment(BLOCKS, 13, seed=0)


class TestScoredEntry:
    def test_the_entry_is_read_as_text(self) -> None:
        frame = pd.DataFrame({"frame": [0, 1], "group": ["", ""], "sequence": [12, 12]})

        assert scored_entry(frame, "xgboost") == ("", "12")

    def test_a_table_of_two_entries_is_refused(self) -> None:
        frame = pd.DataFrame({"group": ["g", "g"], "sequence": ["a", "b"]})

        with pytest.raises(ValueError, match=r"^\[xgboost\] .*more than one entry"):
            scored_entry(frame, "xgboost")
