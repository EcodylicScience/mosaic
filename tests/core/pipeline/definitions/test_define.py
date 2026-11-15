"""Defining a feature: the check in a child process and the files it writes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
from datetime import date
from pathlib import Path
from typing import Final

import pytest

from mosaic.core.dataset import Dataset, open_dataset
from mosaic.core.pipeline.definitions import declare, define
from mosaic.core.pipeline.definitions.define import define_feature
from mosaic.core.pipeline.definitions.sidecar import read_sidecar, sidecar_json
from mosaic.core.pipeline.definitions.store import (
    DEFINITIONS_ROOT,
    DefinitionCheck,
    DefinitionCorrupt,
    DefinitionInvalid,
    DefinitionNameInvalid,
    DefinitionNameTaken,
    DefinitionRecord,
    VersionStem,
    list_versions,
    read_current,
    write_definition,
)
from mosaic.core.pipeline.graph.resolve import feature_class_for_slug
from tests.helpers import (
    CANARY_NAME,
    CANARY_SOURCE,
    CANARY_VARIABLE,
    GLOBAL_NAME,
    GLOBAL_SOURCE,
    MINIMAL_NAME,
    MINIMAL_SOURCE,
    POSE_NAME,
    POSE_SOURCE,
    defined_dataset,
    run_child,
)

SCALED_SOURCE: Final = MINIMAL_SOURCE.replace(
    "scale: float = 1.0", "scale: float = 2.0"
)
OTHER_NAME: Final = "other-distance"
OTHER_SOURCE: Final = MINIMAL_SOURCE.replace(
    f'name = "{MINIMAL_NAME}"', f'name = "{OTHER_NAME}"'
)


def _digest(source: str | bytes) -> str:
    data = source.encode("utf-8") if isinstance(source, str) else source
    return hashlib.sha256(data).hexdigest()


def _directory(dataset: Dataset, name: str) -> Path:
    return dataset.get_root(DEFINITIONS_ROOT) / name


def _version_file(dataset: Dataset, record: DefinitionRecord, suffix: str) -> Path:
    stem = VersionStem(record.defined_on, record.digest).stem
    return _directory(dataset, record.name) / f"{stem}.{suffix}"


def _stats(directory: Path) -> dict[str, tuple[int, int, int]]:
    return {
        path.name: (path.stat().st_ino, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.iterdir())
    }


def _findings(
    caught: pytest.ExceptionInfo[DefinitionInvalid],
) -> list[tuple[DefinitionCheck, str | None]]:
    return [(finding.check, finding.field) for finding in caught.value.findings]


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return defined_dataset(tmp_path, {})


def test_a_minimal_source_defines_its_pair_and_current(dataset: Dataset) -> None:
    record = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)

    assert record.name == MINIMAL_NAME
    assert record.digest == _digest(MINIMAL_SOURCE)
    assert (record.reused, record.previous_digest) == (False, None)
    assert _version_file(dataset, record, "py").read_text() == MINIMAL_SOURCE
    sidecar = read_sidecar(_version_file(dataset, record, "json"))
    assert (sidecar.name, sidecar.digest, sidecar.defined_on) == (
        MINIMAL_NAME,
        record.digest,
        record.defined_on,
    )
    assert read_current(_directory(dataset, MINIMAL_NAME)) == VersionStem(
        record.defined_on, record.digest
    )
    with pytest.raises(ValueError, match="digest"):
        _ = write_definition(
            dataset.get_root(DEFINITIONS_ROOT),
            MINIMAL_NAME,
            SCALED_SOURCE.encode("utf-8"),
            sidecar,
        )


def test_defining_again_reuses_and_a_changed_source_adds_a_version(
    dataset: Dataset,
) -> None:
    directory = _directory(dataset, MINIMAL_NAME)
    first = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)
    before = _stats(directory)

    again = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)

    assert _stats(directory) == before
    assert (again.reused, again.previous_digest) == (True, None)
    assert again.digest == first.digest

    second = define_feature(dataset, MINIMAL_NAME, SCALED_SOURCE)

    assert (second.reused, second.previous_digest) == (False, first.digest)
    assert [version.digest for version in list_versions(directory)] == sorted(
        [first.digest, second.digest]
    )
    assert read_current(directory) == VersionStem(second.defined_on, second.digest)

    restored = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)

    assert (restored.reused, restored.previous_digest) == (True, second.digest)
    assert read_current(directory) == VersionStem(first.defined_on, first.digest)


def test_a_lone_source_is_completed_under_its_own_stem(dataset: Dataset) -> None:
    directory = _directory(dataset, MINIMAL_NAME)
    directory.mkdir(parents=True)
    stem = VersionStem(date(2026, 1, 2), _digest(MINIMAL_SOURCE))
    (directory / f"{stem.stem}.py").write_text(MINIMAL_SOURCE)

    record = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)

    assert (record.defined_on, record.reused) == (stem.date, False)
    assert read_sidecar(directory / f"{stem.stem}.json").defined_on == stem.date
    assert list_versions(directory) == [stem]
    assert read_current(directory) == stem


def test_a_pair_whose_source_bytes_differ_from_its_digest_is_corrupt(
    dataset: Dataset,
) -> None:
    directory = _directory(dataset, MINIMAL_NAME)
    directory.mkdir(parents=True)
    stem = VersionStem(date(2026, 1, 2), _digest(MINIMAL_SOURCE))
    (directory / f"{stem.stem}.py").write_text(MINIMAL_SOURCE + "# edited\n")
    (directory / f"{stem.stem}.json").write_text("{}")

    with pytest.raises(DefinitionCorrupt) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)

    assert caught.value.path == directory / f"{stem.stem}.py"
    assert read_current(directory) is None


def test_a_builtin_name_is_taken(dataset: Dataset) -> None:
    with pytest.raises(DefinitionNameTaken):
        _ = define_feature(dataset, "speed-angvel", MINIMAL_SOURCE)


def test_a_name_outside_the_grammar_is_invalid(dataset: Dataset) -> None:
    with pytest.raises(DefinitionNameInvalid):
        _ = define_feature(dataset, "Speed", MINIMAL_SOURCE)


REGISTERING_SOURCE: Final = MINIMAL_SOURCE.replace(
    "class OriginDistance:",
    "from mosaic.behavior.feature_library.registry import register_feature\n\n\n"
    "@register_feature\n"
    "class OriginDistance:",
)


def test_a_source_that_registers_a_feature_is_refused_naming_the_registry(
    dataset: Dataset,
) -> None:
    from mosaic.behavior.feature_library import FEATURES

    before = set(FEATURES)

    with pytest.raises(DefinitionInvalid) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, REGISTERING_SOURCE)

    assert _findings(caught) == [("registry", "FEATURES")]
    assert "OriginDistance" in caught.value.findings[0].message
    assert set(FEATURES) == before
    assert not _directory(dataset, MINIMAL_NAME).exists()


def test_a_source_importing_the_feature_helpers_is_accepted(dataset: Dataset) -> None:
    source = MINIMAL_SOURCE.replace(
        "from mosaic.core.params",
        "from mosaic.behavior.feature_library.helpers import feature_columns\n"
        "from mosaic.core.params",
    )

    record = define_feature(dataset, MINIMAL_NAME, source)

    assert record.digest == _digest(source)


def test_a_source_that_runs_past_the_timeout_is_refused(
    dataset: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(define, "DECLARE_TIMEOUT_SECONDS", 2)
    source = "import time\n\ntime.sleep(600)\n" + MINIMAL_SOURCE

    with pytest.raises(DefinitionInvalid) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, source)

    assert _findings(caught) == [("source", None)]
    assert "2 seconds" in caught.value.findings[0].message


def test_what_a_source_prints_stays_out_of_its_sidecar(dataset: Dataset) -> None:
    source = 'import os\n\nprint("x")\nos.write(1, b"x")\n' + MINIMAL_SOURCE

    record = define_feature(dataset, MINIMAL_NAME, source)

    sidecar = read_sidecar(_version_file(dataset, record, "json"))
    assert (sidecar.name, sidecar.digest) == (MINIMAL_NAME, _digest(source))


def test_a_child_that_exits_without_a_result_is_refused_naming_its_status(
    dataset: Dataset,
) -> None:
    source = "import os\n\nos._exit(3)\n" + MINIMAL_SOURCE

    with pytest.raises(DefinitionInvalid) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, source)

    assert _findings(caught) == [("source", None)]
    assert "exit status 3" in caught.value.findings[0].message


def test_text_that_cannot_be_encoded_is_refused(dataset: Dataset) -> None:
    with pytest.raises(DefinitionInvalid) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE + "# \udc80\n")

    assert _findings(caught) == [("source", None)]
    assert not _directory(dataset, MINIMAL_NAME).exists()


def test_a_current_that_names_no_version_is_replaced(dataset: Dataset) -> None:
    directory = _directory(dataset, MINIMAL_NAME)
    directory.mkdir(parents=True)
    (directory / "current").write_text("garbage\n")

    record = define_feature(dataset, MINIMAL_NAME, MINIMAL_SOURCE)

    assert read_current(directory) == VersionStem(record.defined_on, record.digest)
    assert record.previous_digest is None


def test_a_registration_made_while_the_class_is_built_is_refused(
    dataset: Dataset,
) -> None:
    source = MINIMAL_SOURCE.replace(
        "self.params = self.Params.from_overrides(params)",
        "self.params = self.Params.from_overrides(params)\n"
        "        from mosaic.behavior.feature_library.registry import FEATURES\n"
        '        FEATURES.setdefault("InjectedDistance", type(self))',
    )

    with pytest.raises(DefinitionInvalid) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, source)

    assert _findings(caught) == [("registry", "FEATURES")]
    assert "InjectedDistance" in caught.value.findings[0].message


REFUSED_SOURCES: Final = [
    pytest.param(
        MINIMAL_SOURCE.replace(f'name = "{MINIMAL_NAME}"', 'name = "elsewhere"'),
        [("class", None)],
        id="no class",
    ),
    pytest.param(
        MINIMAL_SOURCE.replace('emits = "individual"', 'emits = "crowd"').replace(
            "consumed_roots = ()", "consumed_roots = ()\n    temporal_reach = len"
        ),
        [("attribute", "emits"), ("temporal_reach", "temporal_reach")],
        id="flags",
    ),
    pytest.param(
        MINIMAL_SOURCE.replace(
            "scale: float = 1.0",
            "scale: float = Field(default=1.0, alias='speed')\n        window: int",
        ).replace("import numpy", "from pydantic import Field\nimport numpy"),
        [("params_alias", "scale"), ("params_default", "window")],
        id="params",
    ),
    pytest.param(
        MINIMAL_SOURCE.replace(
            "self.params = self.Params.from_overrides(params)",
            "self.params = self.Params.from_overrides(params)\n"
            "        self.scope_dependent = False",
        ),
        [("flag_in_init", "scope_dependent")],
        id="flag in __init__",
    ),
    pytest.param(
        MINIMAL_SOURCE.replace(
            "self.inputs = inputs",
            "raise RuntimeError('no')\n        self.inputs = inputs",
        ),
        [("construction", None)],
        id="construction",
    ),
    pytest.param(
        MINIMAL_SOURCE + "\nraise RuntimeError('at import')\n",
        [("source", None)],
        id="raises at import",
    ),
    pytest.param(MINIMAL_SOURCE + "\ndef (\n", [("source", None)], id="syntax"),
    pytest.param(
        MINIMAL_SOURCE.encode("utf-8") + b"\n# \xff\n",
        [("source", None)],
        id="not UTF-8",
    ),
    pytest.param(
        MINIMAL_SOURCE + "#" * (1024 * 1024) + "\n",
        [("source", None)],
        id="over 1 MiB",
    ),
]


@pytest.mark.parametrize(("source", "expected"), REFUSED_SOURCES)
def test_each_refusal_arrives_from_the_child_and_leaves_the_parent_unchanged(
    dataset: Dataset,
    source: str | bytes,
    expected: list[tuple[DefinitionCheck, str | None]],
) -> None:
    from mosaic.behavior.feature_library import FEATURES
    from mosaic.core.pipeline.ops import OPS

    _ = feature_class_for_slug(MINIMAL_NAME)
    features, ops, modules = set(FEATURES), set(OPS), set(sys.modules)

    with pytest.raises(DefinitionInvalid) as caught:
        _ = define_feature(dataset, MINIMAL_NAME, source)

    assert _findings(caught) == expected
    assert (set(FEATURES), set(OPS)) == (features, ops)
    assert set(sys.modules) - modules == set()
    assert not _directory(dataset, MINIMAL_NAME).exists()


def test_the_check_runs_once_in_a_child_of_this_installation(
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    canary = tmp_path / "canary.jsonl"
    monkeypatch.setenv(CANARY_VARIABLE, str(canary))

    _ = define_feature(dataset, CANARY_NAME, CANARY_SOURCE)

    lines = [json.loads(line) for line in canary.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["pid"] != os.getpid()
    assert Path(lines[0]["argv"][0]) == Path(declare.__file__)
    assert CANARY_NAME in lines[0]["argv"]


def test_the_sidecar_is_the_same_bytes_under_any_hash_seed(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text(POSE_SOURCE)
    outputs: list[str] = []
    for seed in ("1", "2"):
        child = run_child(
            [sys.executable, "-m", declare.__name__, POSE_NAME, str(source)],
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert child.returncode == 0, child.stderr
        outputs.append(child.stdout)

    assert outputs[0] == outputs[1]
    record = define_feature(defined_dataset(tmp_path, {}), POSE_NAME, POSE_SOURCE)
    assert outputs[0].encode("utf-8") == sidecar_json(
        read_sidecar(
            tmp_path
            / "dataset"
            / DEFINITIONS_ROOT
            / POSE_NAME
            / (f"{VersionStem(record.defined_on, record.digest).stem}.json")
        )
    )


def test_a_dataset_opened_without_its_roots_creates_the_definitions_directory(
    dataset: Dataset,
) -> None:
    root = dataset.get_root(DEFINITIONS_ROOT)
    shutil.rmtree(root)
    reopened = open_dataset(dataset.manifest_path, ensure_roots=False)
    assert not root.exists()

    record = define_feature(reopened, MINIMAL_NAME, MINIMAL_SOURCE)

    assert _version_file(reopened, record, "py").is_file()
    assert _version_file(reopened, record, "json").is_file()


def _define_at_once(
    dataset: Dataset, definitions: list[tuple[str, str]]
) -> list[DefinitionRecord | BaseException]:
    results: list[DefinitionRecord | BaseException] = [
        RuntimeError("not run") for _ in definitions
    ]

    def run(index: int, name: str, source: str) -> None:
        try:
            results[index] = define_feature(dataset, name, source)
        except BaseException as exc:
            results[index] = exc

    threads = [
        threading.Thread(target=run, args=(index, name, source))
        for index, (name, source) in enumerate(definitions)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def test_two_names_defined_at_once_both_succeed(dataset: Dataset) -> None:
    results = _define_at_once(
        dataset, [(MINIMAL_NAME, MINIMAL_SOURCE), (OTHER_NAME, OTHER_SOURCE)]
    )

    assert all(isinstance(result, DefinitionRecord) for result in results), results
    for name in (MINIMAL_NAME, OTHER_NAME):
        assert read_current(_directory(dataset, name)) is not None


def test_two_sources_of_one_name_at_once_leave_two_pairs_and_one_current(
    dataset: Dataset,
) -> None:
    results = _define_at_once(
        dataset, [(MINIMAL_NAME, MINIMAL_SOURCE), (MINIMAL_NAME, SCALED_SOURCE)]
    )

    assert all(isinstance(result, DefinitionRecord) for result in results), results
    directory = _directory(dataset, MINIMAL_NAME)
    digests = sorted([_digest(MINIMAL_SOURCE), _digest(SCALED_SOURCE)])
    assert [version.digest for version in list_versions(directory)] == digests
    current = read_current(directory)
    assert current is not None and current.digest in digests


def test_every_helper_source_defines(tmp_path: Path) -> None:
    sources = {
        MINIMAL_NAME: MINIMAL_SOURCE,
        CANARY_NAME: CANARY_SOURCE,
        GLOBAL_NAME: GLOBAL_SOURCE,
        POSE_NAME: POSE_SOURCE,
    }

    dataset = defined_dataset(tmp_path, sources)

    for name, source in sources.items():
        current = read_current(_directory(dataset, name))
        assert current is not None and current.digest == _digest(source)
