"""A claim over a directory, whole or through a filter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mosaic.core.pipeline.scan_claim import AnyFilter, ScanClaim, Unclaimed


@dataclass(frozen=True)
class SuffixFilter:
    """Accepts a path by its suffix, as a source's patterns would."""

    suffix: str

    def accepts(self, resolved: Path) -> bool:
        return resolved.suffix == self.suffix


DATA = Path("/data")


class TestAFilteredDirectory:
    def test_it_claims_an_accepted_path_and_not_a_rejected_one(self) -> None:
        claim = ScanClaim.over_filtered(DATA, SuffixFilter(".csv"))

        assert claim.claims(DATA / "a.csv")
        assert claim.claims(DATA / "deep" / "b.csv")
        assert not claim.claims(DATA / "a.npz")

    def test_it_claims_nothing_outside_its_directory(self) -> None:
        claim = ScanClaim.over_filtered(DATA, SuffixFilter(".csv"))

        assert not claim.claims(Path("/other/a.csv"))

    def test_a_none_filter_claims_the_subtree(self) -> None:
        claim = ScanClaim.over_filtered(DATA, None)

        assert claim.claims(DATA / "a.npz")
        assert claim.claims(DATA / "deep" / "b.txt")
        assert claim == ScanClaim.over_directories([DATA])

    def test_a_filtered_claim_is_not_empty(self) -> None:
        assert ScanClaim.over_filtered(DATA, SuffixFilter(".csv"))


class TestUnion:
    def test_two_filters_over_one_directory_claim_what_either_accepts(self) -> None:
        claim = ScanClaim.over_filtered(
            DATA, SuffixFilter(".csv")
        ) | ScanClaim.over_filtered(DATA, SuffixFilter(".npz"))

        assert claim.claims(DATA / "a.csv")
        assert claim.claims(DATA / "a.npz")
        assert not claim.claims(DATA / "a.h5")
        assert isinstance(claim.directories[DATA], AnyFilter)

    def test_a_none_filter_absorbs_the_other(self) -> None:
        filtered = ScanClaim.over_filtered(DATA, SuffixFilter(".csv"))
        whole = ScanClaim.over_directories([DATA])

        for claim in (filtered | whole, whole | filtered):
            assert claim.directories == {DATA: None}
            assert claim.claims(DATA / "a.h5")

    def test_one_filter_joined_with_itself_stays_one(self) -> None:
        only = SuffixFilter(".csv")
        claim = ScanClaim.over_filtered(DATA, only) | ScanClaim.over_filtered(
            DATA, only
        )

        assert claim.directories == {DATA: only}

    def test_filters_over_two_directories_stay_apart(self) -> None:
        nested = DATA / "day2"
        claim = ScanClaim.over_filtered(
            DATA, SuffixFilter(".csv")
        ) | ScanClaim.over_filtered(nested, SuffixFilter(".npz"))

        assert claim.claims(nested / "a.csv")
        assert claim.claims(nested / "a.npz")
        assert not claim.claims(DATA / "a.npz")


class TestDescribe:
    def test_a_filtered_directory_says_so(self) -> None:
        claim = ScanClaim.over_filtered(
            DATA, SuffixFilter(".csv")
        ) | ScanClaim.over_directories([Path("/nas")])

        assert claim.describe() == "under /data (filtered), /nas"


def present(directory: Path, *names: str) -> Path:
    """Create *names* under *directory* and return it resolved."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()
    return directory.resolve()


class TestUnclaimed:
    def test_it_accepts_what_the_other_claim_leaves_out(self, tmp_path: Path) -> None:
        data = present(tmp_path / "data", "a.csv", "a.npz")
        others = Unclaimed(lambda: ScanClaim.over_filtered(data, SuffixFilter(".npz")))
        claim = ScanClaim.over_filtered(data, others)

        assert claim.claims(data / "a.csv")
        assert not claim.claims(data / "a.npz")

    def test_it_builds_the_other_claim_once_and_only_when_asked(
        self, tmp_path: Path
    ) -> None:
        data = present(tmp_path / "data", "a.csv", "b.csv")
        built: list[int] = []

        def build() -> ScanClaim:
            built.append(1)
            return ScanClaim()

        claim = ScanClaim.over_filtered(data, Unclaimed(build))
        assert not claim.claims(tmp_path.resolve() / "other" / "a.csv")
        assert built == []

        assert claim.claims(data / "a.csv")
        assert claim.claims(data / "b.csv")
        assert built == [1]

    def test_a_path_without_a_file_is_accepted_without_building(
        self, tmp_path: Path
    ) -> None:
        data = present(tmp_path / "data")
        built: list[int] = []

        def build() -> ScanClaim:
            built.append(1)
            return ScanClaim.over_filtered(data, None)

        claim = ScanClaim.over_filtered(data, Unclaimed(build))

        assert claim.claims(data / "gone.npz")
        assert built == []
