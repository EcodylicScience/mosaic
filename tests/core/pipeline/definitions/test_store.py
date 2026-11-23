"""The name grammar, the slug grammar, the version listing and the error codes of definitions."""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import get_args

import pytest

from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DIGEST_PREFIX_MIN,
    NAME_MAX_LENGTH,
    NAME_PATTERN,
    SLUG_PATTERN,
    DefinitionCheck,
    DefinitionCorrupt,
    DefinitionDigestAmbiguous,
    DefinitionFinding,
    DefinitionInvalid,
    DefinitionMismatch,
    DefinitionModelMissing,
    DefinitionModelRefused,
    DefinitionNameInvalid,
    DefinitionNameTaken,
    DefinitionNotFound,
    DefinitionNotLoaded,
    DefinitionRecord,
    DefinitionToolkitModelMissing,
    FeatureDefinitionsNotAllowed,
    VersionStem,
    check_definition_name,
    definition_module_name,
    is_under_definitions_root,
    list_definition_names,
    list_versions,
    parse_definition_slug,
    read_current,
    resolve_digest,
)
from mosaic.core.pipeline.error_envelope import error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_version(
    directory: Path, day: str, digest: str, *, py: bool = True, js: bool = True
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if py:
        (directory / f"{day}_{digest}.py").write_text("")
    if js:
        (directory / f"{day}_{digest}.json").write_text("{}")


@pytest.mark.parametrize("name", ["speed-x", "a1", "a" * 64, "a-b-c", "x2-3"])
def test_the_name_grammar_accepts(name: str) -> None:
    check_definition_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "Speed",
        "a_b",
        "a.b",
        "a:b",
        "a@b",
        "-a",
        "a-",
        "a--b",
        "a" * 65,
        "",
        "1a",
        "a b",
    ],
)
def test_the_name_grammar_refuses(name: str) -> None:
    with pytest.raises(DefinitionNameInvalid) as caught:
        check_definition_name(name)
    detail = caught.value.error_detail()
    assert detail is not None
    assert detail["name"] == name
    assert "pattern" in detail


def test_the_root_names_the_directory() -> None:
    assert DEFINITIONS_ROOT == "feature_definitions"


def test_a_slug_without_a_digest_parses_to_the_name() -> None:
    assert parse_definition_slug("def:x") == ("x", None)


def test_a_slug_keeps_a_short_and_a_full_digest() -> None:
    short, full = "a" * 12, "0123456789abcdef" * 4
    assert parse_definition_slug(f"def:x@{short}") == ("x", short)
    assert parse_definition_slug(f"def:speed-x@{full}") == ("speed-x", full)


@pytest.mark.parametrize(
    "slug",
    [
        "def:",
        "def:x@" + "a" * 11,
        "def:x@" + "A" * 12,
        "def:x@" + "a" * 65,
        "def:x@",
        "def:X",
        "def:x@@" + "a" * 12,
        "speed",
        "def.x",
    ],
)
def test_a_malformed_slug_is_refused(slug: str) -> None:
    assert DIGEST_PREFIX_MIN == 12
    with pytest.raises(DefinitionNameInvalid):
        parse_definition_slug(slug)


def test_the_slug_pattern_states_the_name_grammar_and_the_digest_range() -> None:
    assert NAME_PATTERN in SLUG_PATTERN
    assert f"at most {NAME_MAX_LENGTH} characters" in SLUG_PATTERN
    assert f"{DIGEST_PREFIX_MIN} to 64" in SLUG_PATTERN
    with pytest.raises(DefinitionNameInvalid) as caught:
        parse_definition_slug("def:Bad")
    assert NAME_PATTERN in str(caught.value)
    assert f"at most {NAME_MAX_LENGTH} characters" in str(caught.value)


def test_the_finding_checks_are_the_declared_set() -> None:
    assert get_args(DefinitionCheck.__value__) == (
        "class",
        "attribute",
        "flag_in_init",
        "temporal_reach",
        "inputs",
        "params_base",
        "params_default",
        "params_type",
        "params_metadata",
        "params_alias",
        "params_config",
        "params_validator",
        "params_serializer",
        "params_post_init",
        "params_reference",
        "model_marker",
        "model_rebuild",
        "construction",
        "source",
        "registry",
    )


def test_the_module_name_ends_in_the_full_digest_and_swaps_hyphens() -> None:
    digest = _digest("source")
    assert definition_module_name("a-b", digest) == f"mosaic_definitions.a_b__{digest}"


def test_the_module_name_refuses_the_underscore_spelling_of_a_hyphenated_name() -> None:
    digest = _digest("source")
    with pytest.raises(DefinitionNameInvalid):
        definition_module_name("a_b", digest)


def test_a_version_stem_names_its_date_and_digest() -> None:
    digest = _digest("a")
    stem = VersionStem(date(2026, 11, 13), digest)
    assert stem.stem == f"2026-11-13_{digest}"
    assert VersionStem.parse(stem.stem) == stem
    assert VersionStem.parse("2026-11-13_short") is None
    assert VersionStem.parse(f"2026-13-45_{digest}") is None
    assert VersionStem.parse(f"2026-11-13_{digest.upper()}") is None


def test_list_versions_sorts_by_stem_and_skips_the_remains(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    first, second, third = _digest("1"), _digest("2"), _digest("3")
    _write_version(directory, "2026-11-12", second)
    _write_version(directory, "2026-11-10", third)
    _write_version(directory, "2026-11-10", first)
    _write_version(directory, "2026-11-01", _digest("lone-py"), js=False)
    _write_version(directory, "2026-11-01", _digest("lone-json"), py=False)
    (directory / f".2026-11-12_{second}-abc.tmp").write_text("")
    (directory / "current.lock").write_text("")
    (directory / "current").write_text(f"2026-11-12_{second}\n")
    (directory / "notes.txt").write_text("")
    (directory / f"2026-11-12_{second}.py.bak").write_text("")
    expected = sorted(
        [
            VersionStem(date(2026, 11, 12), second),
            VersionStem(date(2026, 11, 10), third),
            VersionStem(date(2026, 11, 10), first),
        ],
        key=lambda stem: stem.stem,
    )
    assert list_versions(directory) == expected
    assert [v.stem for v in list_versions(directory)] == sorted(
        v.stem for v in expected
    )


def test_list_versions_of_an_absent_directory_is_empty(tmp_path: Path) -> None:
    assert list_versions(tmp_path / "absent") == []


def test_read_current_of_an_absent_file_is_none(tmp_path: Path) -> None:
    assert read_current(tmp_path / "x") is None
    (tmp_path / "x").mkdir()
    assert read_current(tmp_path / "x") is None


def test_read_current_of_an_empty_file_is_none(tmp_path: Path) -> None:
    (tmp_path / "current").write_text("")
    assert read_current(tmp_path) is None
    (tmp_path / "current").write_text("\n")
    assert read_current(tmp_path) is None


def test_read_current_reads_the_stem(tmp_path: Path) -> None:
    digest = _digest("a")
    (tmp_path / "current").write_text(f"2026-11-13_{digest}\n")
    assert read_current(tmp_path) == VersionStem(date(2026, 11, 13), digest)


def test_read_current_of_text_that_is_no_stem_is_corrupt(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    directory.mkdir()
    (directory / "current").write_text("garbage\n")
    with pytest.raises(DefinitionCorrupt) as caught:
        read_current(directory)
    detail = caught.value.error_detail()
    assert detail is not None
    assert detail["name"] == "x"
    assert detail["path"] == str(directory / "current")


def test_list_definition_names_lists_names_whose_current_names_a_complete_pair(
    tmp_path: Path,
) -> None:
    digest = _digest("a")
    stem = f"2026-11-13_{digest}"
    _write_version(tmp_path / "b-def", "2026-11-13", digest)
    (tmp_path / "b-def" / "current").write_text(f"{stem}\n")
    _write_version(tmp_path / "a-def", "2026-11-13", digest)
    (tmp_path / "a-def" / "current").write_text(f"{stem}\n")
    _write_version(tmp_path / "no-current", "2026-11-13", digest)
    _write_version(tmp_path / "empty-current", "2026-11-13", digest)
    (tmp_path / "empty-current" / "current").write_text("")
    _write_version(tmp_path / "half", "2026-11-13", digest, js=False)
    (tmp_path / "half" / "current").write_text(f"{stem}\n")
    _write_version(tmp_path / "Bad_Name", "2026-11-13", digest)
    (tmp_path / "Bad_Name" / "current").write_text(f"{stem}\n")
    (tmp_path / "stray.txt").write_text("")
    assert list_definition_names(tmp_path) == ["a-def", "b-def"]


def test_list_definition_names_skips_a_name_whose_current_is_no_stem(
    tmp_path: Path,
) -> None:
    digest = _digest("a")
    _write_version(tmp_path / "good", "2026-11-13", digest)
    (tmp_path / "good" / "current").write_text(f"2026-11-13_{digest}\n")
    _write_version(tmp_path / "bad", "2026-11-13", digest)
    (tmp_path / "bad" / "current").write_text("garbage\n")
    assert list_definition_names(tmp_path) == ["good"]


def test_list_definition_names_of_an_absent_root_is_empty(tmp_path: Path) -> None:
    assert list_definition_names(tmp_path / "absent") == []


def test_a_full_digest_resolves_to_itself(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    digest = _digest("a")
    _write_version(directory, "2026-11-13", digest)
    assert resolve_digest(directory, digest) == digest


def test_a_unique_prefix_resolves_to_the_full_digest(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    first, second = _digest("a"), _digest("b")
    assert first[:12] != second[:12]
    _write_version(directory, "2026-11-13", first)
    _write_version(directory, "2026-11-13", second)
    assert resolve_digest(directory, first[:12]) == first
    assert resolve_digest(directory, first[:40]) == first


def test_an_unknown_digest_is_not_found(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    _write_version(directory, "2026-11-13", _digest("a"))
    for given in (_digest("zzz"), "f" * 12):
        with pytest.raises(DefinitionNotFound) as caught:
            resolve_digest(directory, given)
        detail = caught.value.error_detail()
        assert detail is not None
        assert detail["slug"] == f"def:x@{given}"


def test_a_digest_of_an_incomplete_version_is_not_found(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    digest = _digest("a")
    _write_version(directory, "2026-11-13", digest, js=False)
    with pytest.raises(DefinitionNotFound):
        resolve_digest(directory, digest)


def test_two_versions_sharing_a_prefix_are_ambiguous_and_both_are_named(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "x"
    first = "ab" * 6 + "0" * 52
    second = "ab" * 6 + "1" * 52
    _write_version(directory, "2026-11-12", first)
    _write_version(directory, "2026-11-13", second)
    with pytest.raises(DefinitionDigestAmbiguous) as caught:
        resolve_digest(directory, "ab" * 6)
    detail = caught.value.error_detail()
    assert detail is not None
    assert detail["slug"] == "def:x@" + "ab" * 6
    assert detail["matches"] == [first, second]
    assert first in str(caught.value) and second in str(caught.value)
    assert resolve_digest(directory, "ab" * 6 + "0") == first


def test_a_prefix_shorter_than_the_minimum_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "x"
    digest = _digest("a")
    _write_version(directory, "2026-11-13", digest)
    with pytest.raises(DefinitionNameInvalid):
        resolve_digest(directory, digest[:11])


def test_a_record_holds_the_facts_of_a_define() -> None:
    record = DefinitionRecord(
        name="x", digest=_digest("a"), defined_on=date(2026, 11, 13), reused=False
    )
    assert record.previous_digest is None
    assert record.model_dump(mode="json")["defined_on"] == "2026-11-13"
    with pytest.raises(ValueError):
        DefinitionRecord.model_validate({**record.model_dump(), "extra": 1})


FINDING = DefinitionFinding(check="class", field=None, message="no class named 'x'")
DIGEST = _digest("a")


@pytest.mark.parametrize(
    ("error", "kind", "code", "detail"),
    [
        (
            DefinitionNameInvalid("A", pattern="p"),
            "invalid",
            "definition_name_invalid",
            {"name": "A", "pattern": "p"},
        ),
        (DefinitionNameTaken("x"), "conflict", "definition_name_taken", {"name": "x"}),
        (
            DefinitionInvalid("x", [FINDING]),
            "invalid",
            "definition_invalid",
            {
                "name": "x",
                "findings": [
                    {"check": "class", "field": None, "message": "no class named 'x'"}
                ],
            },
        ),
        (
            DefinitionCorrupt("x", DIGEST, Path("/a/b.py")),
            "conflict",
            "definition_corrupt",
            {"name": "x", "digest": DIGEST, "path": "/a/b.py"},
        ),
        (
            DefinitionNotFound("def:x"),
            "not_found",
            "definition_not_found",
            {"slug": "def:x"},
        ),
        (
            DefinitionDigestAmbiguous("def:x@ab", ["1", "2"]),
            "conflict",
            "definition_digest_ambiguous",
            {"slug": "def:x@ab", "matches": ["1", "2"]},
        ),
        (
            DefinitionMismatch("x", DIGEST, ["version", "flags"]),
            "conflict",
            "definition_mismatch",
            {"name": "x", "digest": DIGEST, "fields": ["version", "flags"]},
        ),
        (
            DefinitionToolkitModelMissing("x", "mosaic.a.B"),
            "unavailable",
            "definition_toolkit_model_missing",
            {"name": "x", "ref": "mosaic.a.B"},
        ),
        (
            DefinitionNotLoaded("x", "apply"),
            "invalid",
            "definition_not_loaded",
            {"name": "x", "method": "apply"},
        ),
        (
            DefinitionModelMissing(
                "x",
                "def.x__from__tracks",
                None,
                "unpinned",
                "m",
                candidates=("r1", "r2"),
            ),
            "not_found",
            "definition_model_missing",
            {
                "name": "x",
                "feature": "def.x__from__tracks",
                "run_id": None,
                "reason": "unpinned",
                "candidates": ["r1", "r2"],
            },
        ),
        (
            DefinitionModelMissing("x", "def.x__from__tracks", None, "unpinned", "m"),
            "not_found",
            "definition_model_missing",
            {
                "name": "x",
                "feature": "def.x__from__tracks",
                "run_id": None,
                "reason": "unpinned",
                "candidates": [],
            },
        ),
        (
            DefinitionModelMissing("x", "def.x__from__tracks", "r1", "missing", "m"),
            "not_found",
            "definition_model_missing",
            {
                "name": "x",
                "feature": "def.x__from__tracks",
                "run_id": "r1",
                "reason": "missing",
            },
        ),
        (
            DefinitionModelMissing("x", "def.x__from__tracks", "r1", "unfitted", "m"),
            "not_found",
            "definition_model_missing",
            {
                "name": "x",
                "feature": "def.x__from__tracks",
                "run_id": "r1",
                "reason": "unfitted",
            },
        ),
        (
            DefinitionModelRefused("x", "f", "r1", "other_feature", "m"),
            "invalid",
            "definition_model_refused",
            {"name": "x", "feature": "f", "run_id": "r1", "reason": "other_feature"},
        ),
        (
            DefinitionModelRefused("x", "f", "r1", "is_apply", "m"),
            "invalid",
            "definition_model_refused",
            {"name": "x", "feature": "f", "run_id": "r1", "reason": "is_apply"},
        ),
        (
            DefinitionModelRefused("x", "f", "r1", "other_version", "m"),
            "invalid",
            "definition_model_refused",
            {"name": "x", "feature": "f", "run_id": "r1", "reason": "other_version"},
        ),
        (
            DefinitionModelRefused("x", "f", "r1", "not_loaded", "m"),
            "invalid",
            "definition_model_refused",
            {"name": "x", "feature": "f", "run_id": "r1", "reason": "not_loaded"},
        ),
        (
            FeatureDefinitionsNotAllowed("def:x"),
            "unavailable",
            "feature_definitions_not_allowed",
            {"slug": "def:x"},
        ),
    ],
)
def test_each_error_declares_its_kind_code_and_detail(
    error: ClassifiedError, kind: str, code: str, detail: dict[str, object]
) -> None:
    envelope = error_envelope(error)
    assert (envelope.kind, envelope.reason, envelope.detail) == (kind, code, detail)
    assert str(error)


def test_the_definitions_root_test_is_component_wise_not_a_prefix() -> None:
    assert is_under_definitions_root(("ds", DEFINITIONS_ROOT, "speed", "x.json"))
    assert not is_under_definitions_root(("ds", f"{DEFINITIONS_ROOT}_backup", "x"))
    assert not is_under_definitions_root(("ds", "tracks_raw", "x.json"))
