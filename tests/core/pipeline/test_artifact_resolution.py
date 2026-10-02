"""Which file an artifact reference resolves to, and when it refuses to guess.

A params field typed ``ArtifactSpec`` names an upstream *run root*, not a file.
The file is chosen by globbing the reference's ``pattern`` inside that root --
and a run root is not a directory of named state files. ``run_feature`` writes
one per-entry output parquet per sequence into it, beside whatever ``save_state``
wrote, so the pattern an unpinned ``ArtifactSpec`` derives (``*.<load kind>``)
matches those too and they sort first.

That used to be resolved by ``sorted(...)[0]``: a ``global-scaler`` step fitted
on one sequence's pass-through table instead of the template matrix, reported
success, and left nothing on disk saying which file it had read. These are the
tests for the two halves of the answer -- a named file resolves past its
siblings, and a pattern naming more than one refuses rather than picking.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mosaic.core.pipeline.index import resolve_artifact_file


def _run_root(tmp_path: Path, *names: str) -> Path:
    """A stand-in run root holding *names*, each a readable parquet."""
    root = tmp_path / "run"
    root.mkdir()
    for name in names:
        pd.DataFrame({"value": [1.0, 2.0]}).to_parquet(root / name, index=False)
    return root


def test_a_named_file_resolves_past_its_per_entry_siblings(tmp_path: Path) -> None:
    """A run root holding per-entry outputs, a provenance table and the artifact.

    What ``extract-templates`` leaves behind over two sequences, which is every
    way a first-sorted-match resolves to the wrong file at once.
    """
    root = _run_root(
        tmp_path,
        "alpha_seq.parquet",
        "beta_seq.parquet",
        "template_provenance.parquet",
        "templates.parquet",
    )

    resolved = resolve_artifact_file(
        "templates", "extract-templates", root, "templates.parquet"
    )

    assert resolved == root / "templates.parquet"


def test_a_pattern_matching_more_than_one_file_is_refused(tmp_path: Path) -> None:
    """Refused, not guessed.

    ``alpha_seq.parquet`` sorts first, so this is the case that returned a
    per-entry output under the name of the training set.
    """
    root = _run_root(
        tmp_path, "alpha_seq.parquet", "beta_seq.parquet", "templates.parquet"
    )

    with pytest.raises(ValueError) as caught:
        _ = resolve_artifact_file("templates", "extract-templates", root, "*.parquet")

    message = str(caught.value)
    assert "templates" in message
    assert "*.parquet" in message
    assert "alpha_seq.parquet" in message
    assert "beta_seq.parquet" in message
    assert str(root) in message


def test_the_refusal_caps_what_it_lists(tmp_path: Path) -> None:
    """A hundred-sequence run root must still produce a readable message."""
    root = _run_root(tmp_path, *(f"seq_{index:03d}.parquet" for index in range(9)))

    with pytest.raises(ValueError) as caught:
        _ = resolve_artifact_file("templates", "up", root, "*.parquet")

    assert "and 4 more" in str(caught.value)


def test_an_underscore_sibling_would_still_sort_ahead(tmp_path: Path) -> None:
    """Removing the per-entry outputs does not save an unpinned reference.

    ``_`` precedes ``s``, so ``template_provenance.parquet`` sorts ahead of
    ``templates.parquet`` -- which is why excluding entry-key filenames from the
    glob was never a fix.
    """
    root = _run_root(tmp_path, "template_provenance.parquet", "templates.parquet")

    with pytest.raises(ValueError):
        _ = resolve_artifact_file("templates", "extract-templates", root, "*.parquet")


def test_no_match_is_not_an_ambiguity(tmp_path: Path) -> None:
    """Absent and ambiguous are different questions.

    A consumer branches on whether its field resolved -- loading a cached model,
    then a referenced one, then fitting from templates -- so an artifact that
    resolves to nothing is a state features already handle.
    """
    root = _run_root(tmp_path, "templates.parquet")

    assert resolve_artifact_file("model", "up", root, "*.joblib") is None


def test_one_match_resolves_even_through_a_glob(tmp_path: Path) -> None:
    """The refusal is about ambiguity, not about wildcards as such."""
    root = _run_root(tmp_path, "templates.parquet")

    resolved = resolve_artifact_file("templates", "up", root, "*.parquet")

    assert resolved == root / "templates.parquet"
