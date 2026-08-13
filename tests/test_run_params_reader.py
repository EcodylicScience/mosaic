"""One reader for ``params.json``, and the divergence it must preserve.

Three parsers read this document before the ``inventory`` package: a typed model
in ``reconcile_features`` reading every key, and two raw ``json.loads`` walks --
in ``provenance`` and ``track_universe`` -- reading ``_resolved`` and nothing
else. Folding three readers into one is only safe if the one keeps every
behaviour the three had, and the two walks **deliberately disagree** on a case
that decides whether a legacy run is ever seen as another run's upstream.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from mosaic.core.pipeline.inventory.params import RunParams, read_run_params

# One document holding every case the two walks disagree about, plus the two
# that neither keeps. Written once so a change to either rule is a visible diff.
DIVERGENCE_DOC: dict[str, object] = {
    "_resolved": [
        # The unlabelled tracks variant: a real upstream whose name is "".
        {"where": "inputs[tracks]", "feature": "tracks", "run_id": ""},
        {"where": "inputs[tracks]", "feature": "tracks", "run_id": "v1"},
        {"where": "params[templates]", "feature": "extract-templates", "run_id": "r9"},
        # Never pinned at all, which is not the same as the unlabelled variant.
        {"where": "inputs[labels]", "feature": "labels", "run_id": None},
    ]
}


def _write(root: Path, document: object) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "params.json").write_text(json.dumps(document))
    return root


def test_consumed_variants_drops_the_unlabelled_variant() -> None:
    """The blast-radius rule: an unnamed variant is not a member of one."""
    params = RunParams.model_validate(DIVERGENCE_DOC)

    assert params.consumed_variants() == frozenset({"v1"})
    assert params.consumed_variants("inputs[labels]") == frozenset()


def test_consumed_run_ids_keeps_the_unlabelled_variant() -> None:
    """The leaf-of-chain rule, and the case the two walks disagree about.

    Dropped, a feature that consumed the unlabelled variant looks as though it
    consumed no tracks, so it never appears as another run's upstream and reads
    as a leaf of its chain forever -- on exactly the legacy datasets where a
    chain is most likely to already exist.
    """
    params = RunParams.model_validate(DIVERGENCE_DOC)

    assert params.consumed_run_ids() == frozenset({"", "v1", "r9"})


def test_a_null_run_id_is_kept_by_neither() -> None:
    """A JSON null records a reference never pinned, not the unlabelled variant."""
    params = RunParams.model_validate(DIVERGENCE_DOC)

    assert None not in params.consumed_run_ids()
    assert params.consumed_variants("inputs[labels]") == frozenset()


# --- absent is not unreadable is not empty -----------------------------------


def test_a_run_with_no_sidecar_reads_absent(tmp_path: Path) -> None:
    """The write sits inside a bare ``except`` that prints, so this is real."""
    read = read_run_params(tmp_path / "nothing-here")

    assert read.state == "absent"
    assert read.params is None
    assert read.finding == "params.json is missing"


def test_a_corrupt_sidecar_reads_unreadable_and_says_why(tmp_path: Path) -> None:
    """Distinct from absent, because the remedies are: one is a file to look at."""
    root = tmp_path / "run"
    root.mkdir()
    (root / "params.json").write_text("{not json at all")

    read = read_run_params(root)

    assert read.state == "unreadable"
    assert read.params is None
    assert read.finding.startswith("params.json is unreadable: ")


def test_a_readable_sidecar_reads_present(tmp_path: Path) -> None:
    read = read_run_params(_write(tmp_path / "run", DIVERGENCE_DOC))

    assert read.state == "present"
    assert read.params is not None
    assert read.finding == ""


# --- a block that cannot be read is dropped, never fatal ---------------------


def test_a_garbage_block_costs_that_block_and_not_the_document(
    tmp_path: Path,
) -> None:
    """The regression the fold could have introduced, and the reason for the
    shape check.

    The two raw walks read ``_resolved`` and never looked at the other keys, so
    they tolerated a document whose other keys were junk. Validating strictly
    would have failed the whole document and silently emptied both provenance
    walks on such a file -- a run reading as dependency-free, which is the
    reading that moves an artifact under a lineage it never had.
    """
    document: dict[str, object] = {
        "_frame_range": "not a range at all",
        "_overlap_frames": "seventeen",
        "_scope": ["not", "a", "block"],
        **DIVERGENCE_DOC,
    }

    read = read_run_params(_write(tmp_path / "run", document))

    assert read.state == "present"
    assert read.params is not None
    assert read.params.consumed_run_ids() == frozenset({"", "v1", "r9"})
    assert read.params.frames() == (None, None)
    assert read.params.overlap_frames == 0
    assert read.params.entry_scope() == set()


def test_a_malformed_resolved_entry_costs_that_entry(tmp_path: Path) -> None:
    """Per element, matching what both raw walks did: they skipped a bad entry."""
    document: dict[str, object] = {
        "_resolved": [
            "not a reference",
            {"where": "inputs[tracks]", "run_id": "v1"},
        ]
    }

    read = read_run_params(_write(tmp_path / "run", document))

    assert read.params is not None
    assert read.params.consumed_variants() == frozenset({"v1"})


def test_a_scope_block_with_one_bad_key_keeps_the_others(tmp_path: Path) -> None:
    document: dict[str, object] = {
        "_scope": {
            "entries": [["", "seq_a"]],
            "compositions": "not a mapping",
            "scope_dependent": True,
        }
    }

    read = read_run_params(_write(tmp_path / "run", document))

    assert read.params is not None
    assert read.params.entry_scope() == {("", "seq_a")}
    assert read.params.entry_compositions({("", "seq_a")}) == {}
    assert read.params.scope.scope_dependent is True


def test_an_absent_resolved_block_is_not_an_empty_one(tmp_path: Path) -> None:
    """``run_feature`` writes the key unconditionally, empty list included, so an
    absent one dates the file rather than saying the run resolved nothing."""
    without = read_run_params(_write(tmp_path / "old", {"_params": {}}))
    with_empty = read_run_params(_write(tmp_path / "new", {"_resolved": []}))

    assert without.params is not None
    assert with_empty.params is not None
    assert without.params.records_resolutions is False
    assert with_empty.params.records_resolutions is True


# --- and it is the only reader -----------------------------------------------


def _modules_naming_the_file() -> dict[str, list[int]]:
    """Every module holding the exact string literal ``"params.json"``.

    Matched in the parse tree against the literal, not by searching the text: a
    docstring that mentions the filename in a sentence is not a second reader,
    and half a dozen modules legitimately do. Anything that *opens* the document
    has to name it, so the literal is the honest anchor.
    """
    import mosaic

    source_root = Path(mosaic.__file__).parent
    found: dict[str, list[int]] = {}
    for source in sorted(source_root.rglob("*.py")):
        if "feature_library/external" in source.as_posix():
            continue
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Constant) and node.value == "params.json":
                found.setdefault(str(source.relative_to(source_root)), []).append(
                    node.lineno
                )
    return found


def test_nothing_else_opens_params_json() -> None:
    """The test that keeps this a consolidation rather than a fourth reader."""
    allowed = {
        # The reader.
        "core/pipeline/inventory/params.py",
        # The writers: run_feature saves the document, reconcile restamps it.
        "core/pipeline/run.py",
        "core/pipeline/reconcile_features.py",
        # A *different* document that happens to share the name: the tracks and
        # labels variant sidecar, whose schema has nothing in common with this
        # one. Named here so the exemption is a decision rather than an omission.
        "core/pipeline/reconcile_variants.py",
        "core/pipeline/tracks_identity.py",
        "core/pipeline/labels_identity.py",
    }
    matched = set(_modules_naming_the_file())

    assert sorted(matched - allowed) == [], (
        "these name params.json outside the one reader and its two writers: "
        f"{sorted(matched - allowed)}"
    )
    assert allowed - matched == set(), (
        "expected every allowlisted module to still name params.json, but these "
        f"did not: {sorted(allowed - matched)} -- a rename may have made this "
        "guard scan for something that no longer exists"
    )


def test_no_module_parses_a_params_document_itself() -> None:
    """A raw ``json.loads`` beside the literal is the defect this retired.

    Scoped to the modules that name the file, so an unrelated json read stays
    unremarkable; the variant-sidecar modules are exempt for the reason above.
    """
    import mosaic

    source_root = Path(mosaic.__file__).parent
    exempt = {
        "core/pipeline/inventory/params.py",
        "core/pipeline/reconcile_variants.py",
        "core/pipeline/tracks_identity.py",
        "core/pipeline/labels_identity.py",
    }
    offenders: list[str] = []
    for relative in _modules_naming_the_file():
        if relative in exempt:
            continue
        tree = ast.parse((source_root / relative).read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in {"loads", "load"}
                and isinstance(node.value, ast.Name)
                and node.value.id == "json"
            ):
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == [], f"these parse a params document themselves: {offenders}"
