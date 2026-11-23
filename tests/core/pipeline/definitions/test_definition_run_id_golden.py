"""Golden run ids of feature definitions, and the rules that make them.

A definition's run id carries a ``_definition`` term, the SHA-256 of its source, so
an edit never meets a stale cache hit. The literal ids live in
``data/definition_run_id_golden.json``, a file of its own so that
``feature_run_id_golden.json`` holds built-ins only. The source is
``MINIMAL_SOURCE``, whose digest is the first line of the file: a change of the
source moves it, and every id with it.

Regenerating after a deliberate identity change::

    MOSAIC_UPDATE_GOLDEN=1 pytest \
        tests/core/pipeline/definitions/test_definition_run_id_golden.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.scope import definition_scope
from mosaic.core.pipeline.graph.resolve import build_feature
from mosaic.core.pipeline.run import compute_run_id
from tests.helpers import (
    APPLYING_NAME,
    APPLYING_SOURCE,
    APPLYING_STORAGE,
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    UPDATE_GOLDEN_ENV,
    defined_dataset,
    read_string_golden,
    regenerate_command,
    updating_golden,
    write_golden,
)

GOLDEN: Final = "definition_run_id_golden.json"
"""``case id -> run_id``, and ``source/<name>`` to the digest of each source."""

type Instance = Literal["declared", "loaded"]
_LOAD: Final[dict[Instance, bool]] = {"declared": False, "loaded": True}


@dataclass(frozen=True)
class Case:
    """One identity computation of a definition's feature.

    Attributes:
        case_id: Stable key into the golden file.
        name: The definition's name.
        params: Params overrides, or ``None`` for the defaults.
        scope: Resolved scope entries, hashed for a scope-dependent definition.
    """

    case_id: str
    name: str
    params: dict[str, object] | None = None
    scope: tuple[tuple[str, str], ...] = field(default_factory=tuple)


CASES: Final = (
    Case(case_id=f"def:{MINIMAL_NAME}/default", name=MINIMAL_NAME),
    Case(
        case_id=f"def:{MINIMAL_NAME}/params-scale",
        name=MINIMAL_NAME,
        params={"scale": 2.5},
    ),
    Case(
        case_id=f"def:{GLOBAL_NAME}/scope-a",
        name=GLOBAL_NAME,
        scope=(("g", "s1"),),
    ),
    Case(
        case_id=f"def:{GLOBAL_NAME}/scope-ab",
        name=GLOBAL_NAME,
        scope=(("g", "s1"), ("g", "s2")),
    ),
)
_SOURCES: Final = {MINIMAL_NAME: MINIMAL_SOURCE, GLOBAL_NAME: GLOBAL_SOURCE}


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Dataset:
    """A dataset that defines every source of the corpus once."""
    return defined_dataset(tmp_path_factory.mktemp("golden"), _SOURCES)


def _identifier(
    ds: Dataset,
    name: str,
    instance: Instance,
    params: dict[str, object] | None = None,
    scope: tuple[tuple[str, str], ...] = (),
    digest: str | None = None,
) -> str:
    """The run id of the definition *name* as its declared or its loaded instance."""
    slug = f"def:{name}" if digest is None else f"def:{name}@{digest}"
    with definition_scope(ds, load=_LOAD[instance]):
        feature = build_feature(slug, None, params)
    run_id, _ = compute_run_id(feature, None, None, ResolvedScope(entries=set(scope)))
    return run_id


def _fresh(ds: Dataset) -> dict[str, str]:
    """Every golden line: each case's run id and each source's digest."""
    lines = {
        case.case_id: _identifier(ds, case.name, "declared", case.params, case.scope)
        for case in CASES
    }
    lines.update({f"source/{name}": _digest(src) for name, src in _SOURCES.items()})
    return lines


def test_case_ids_are_unique() -> None:
    ids = [case.case_id for case in CASES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("instance", ["declared", "loaded"])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_run_id_matches_golden(
    case: Case, instance: Instance, dataset: Dataset
) -> None:
    """The declared and the loaded instance each have the golden run id."""
    if updating_golden():
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: regenerating")
    golden = read_string_golden(GOLDEN)
    if case.case_id not in golden:
        pytest.fail(
            f"No golden run id for '{case.case_id}'. If this case is new, run "
            f"`{regenerate_command(__name__)}` and review the diff."
        )
    assert (
        _identifier(dataset, case.name, instance, case.params, case.scope)
        == golden[case.case_id]
    ), (
        f"The run id of '{case.case_id}' changed. If this shift is intended, run "
        f"`{regenerate_command(__name__)}` and explain every moved line."
    )


@pytest.mark.parametrize("name", sorted(_SOURCES))
def test_the_sources_keep_their_digests(name: str) -> None:
    """A moved source digest explains every moved run id that follows it."""
    if updating_golden():
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: regenerating")
    assert read_string_golden(GOLDEN).get(f"source/{name}") == _digest(_SOURCES[name])


def test_regenerate_golden(dataset: Dataset) -> None:
    """Rewrite the golden file. Runs only under the update environment variable."""
    if not updating_golden():
        pytest.skip(f"set {UPDATE_GOLDEN_ENV}=1 to regenerate")
    write_golden(GOLDEN, _fresh(dataset))


def test_golden_file_has_no_stale_entries() -> None:
    if updating_golden():
        pytest.skip(f"{UPDATE_GOLDEN_ENV}=1: regenerating")
    expected = {case.case_id for case in CASES} | {f"source/{n}" for n in _SOURCES}
    assert set(read_string_golden(GOLDEN)) == expected


def test_the_run_id_does_not_depend_on_the_dataset(
    dataset: Dataset, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """One source and one set of params give one run id in every dataset."""
    other = defined_dataset(
        tmp_path_factory.mktemp("twin"), {MINIMAL_NAME: MINIMAL_SOURCE}
    )
    assert _identifier(dataset, MINIMAL_NAME, "declared") == _identifier(
        other, MINIMAL_NAME, "declared"
    )


@pytest.mark.parametrize("instance", ["declared", "loaded"])
def test_a_comment_in_the_source_is_another_run_with_the_same_version(
    instance: Instance, tmp_path: Path
) -> None:
    """Two sources differing by a comment keep one ``version`` and have two run ids."""
    ds = defined_dataset(tmp_path, {})
    first = define_feature(ds, MINIMAL_NAME, MINIMAL_SOURCE)
    second = define_feature(ds, MINIMAL_NAME, MINIMAL_SOURCE + "\n# a comment\n")
    assert first.digest != second.digest

    one = _identifier(ds, MINIMAL_NAME, instance, digest=first.digest)
    two = _identifier(ds, MINIMAL_NAME, instance, digest=second.digest)

    assert one != two
    assert one.split("-")[0] == two.split("-")[0] == "1"


# --- an apply of a fit ----------------------------------------------------


def _apply_run_id(
    ds: Dataset, instance: Instance, fit: str, scope: ResolvedScope
) -> str:
    """The run id of the definition applying the fit run *fit* over *scope*."""
    model = {"feature": APPLYING_STORAGE, "run_id": fit}
    with definition_scope(ds, load=_LOAD[instance]):
        feature = build_feature(f"def:{APPLYING_NAME}", None, {"model": model})
    run_id, _ = compute_run_id(feature, None, None, scope)
    return run_id


@pytest.mark.parametrize("instance", ["declared", "loaded"])
def test_an_apply_has_one_run_id_over_every_scope_that_reads_the_same_tracks(
    instance: Instance, tmp_path: Path
) -> None:
    """An apply's id holds the fit's run id, and not the entries it covers."""
    ds = defined_dataset(tmp_path, {APPLYING_NAME: APPLYING_SOURCE})
    variants = ("convert-trex.0.2-1111111111",)
    first = ResolvedScope(entries={("g", "s1")}, tracks_variants=variants)
    second = ResolvedScope(entries={("g", "s2")}, tracks_variants=variants)
    both = ResolvedScope(entries={("g", "s1"), ("g", "s2")}, tracks_variants=variants)
    elsewhere = ResolvedScope(
        entries={("g", "s1")}, tracks_variants=("convert-trex.0.2-2222222222",)
    )

    ids = {
        _apply_run_id(ds, instance, "1-abc", scope) for scope in (first, second, both)
    }

    assert len(ids) == 1
    assert _apply_run_id(ds, instance, "1-def", first) not in ids
    assert _apply_run_id(ds, instance, "1-abc", elsewhere) not in ids
