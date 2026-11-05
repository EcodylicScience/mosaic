"""A request id is a ULID, and an id of any other form names no request file.

An id becomes a file name under ``requests/``, so an id shaped like a path would
otherwise read or write a file elsewhere in the dataset.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from mosaic.core.pipeline.graph.model import Request
from mosaic.core.pipeline.graph.recipe_store import pipelines_root
from mosaic.core.pipeline.graph.store import (
    REQUEST_ID_PATTERN,
    RequestIdMalformed,
    RequestUnreadable,
    computes_path,
    load_request,
    open_duplicates,
    request_ids,
    request_path,
    requests_root,
    save_computes,
    save_request,
)
from mosaic.core.scope import Scope
from mosaic.runlog import new_execution_id

REQUEST_ID = "01JC0000000000000000000000"

MALFORMED = (
    "../x",
    "a/b",
    "..",
    "",
    REQUEST_ID.lower(),
    REQUEST_ID[:-1],
    REQUEST_ID + "0",
    REQUEST_ID + "\n",
    "01JC000000000000000000000I",
    "01JC000000000000000000000L",
    "01JC000000000000000000000O",
    "01JC000000000000000000000U",
)


def _request(request_id: str = REQUEST_ID) -> Request:
    return Request(request_id=request_id, recipe_digest="d" * 64)


def _plant_outside(base: Path) -> Path:
    """A valid request file at ``pipelines/x.json``, beside ``requests/``."""
    saved = save_request(base, _request())
    planted = pipelines_root(base) / "x.json"
    _ = shutil.copyfile(saved, planted)
    return planted


def test_a_minted_id_matches_the_pattern() -> None:
    assert re.fullmatch(REQUEST_ID_PATTERN, new_execution_id())


@pytest.mark.parametrize("request_id", MALFORMED)
def test_a_malformed_id_names_no_request_file(request_id: str, tmp_path: Path) -> None:
    with pytest.raises(RequestIdMalformed) as raised:
        _ = request_path(tmp_path, request_id)

    assert raised.value.error_reason() == "request_id_malformed"
    assert raised.value.error_detail() == {"request_id": request_id}


@pytest.mark.parametrize("request_id", MALFORMED)
def test_a_malformed_id_names_no_computes_file(request_id: str, tmp_path: Path) -> None:
    with pytest.raises(RequestIdMalformed):
        _ = computes_path(tmp_path, request_id)


def test_a_path_shaped_id_does_not_read_a_file_outside_the_requests(
    tmp_path: Path,
) -> None:
    planted = _plant_outside(tmp_path)
    escaped = requests_root(tmp_path) / "../x.json"
    assert escaped.resolve() == planted.resolve()

    with pytest.raises(FileNotFoundError, match="a request id is a ULID"):
        _ = load_request(tmp_path, "../x")


def test_a_request_with_a_malformed_id_is_not_written(tmp_path: Path) -> None:
    with pytest.raises(RequestIdMalformed):
        _ = save_request(tmp_path, _request("../x"))

    assert not pipelines_root(tmp_path).exists()


def test_computed_runs_with_a_malformed_id_are_not_written(tmp_path: Path) -> None:
    with pytest.raises(RequestIdMalformed):
        _ = save_computes(tmp_path, "../x", [])

    assert not pipelines_root(tmp_path).exists()


def test_a_request_file_is_read_only_under_its_own_id(tmp_path: Path) -> None:
    """The id read from a file is joined into other paths, so it is its file's name."""
    other = "01JC0000000000000000000001"
    _ = request_path(tmp_path, other).parent.mkdir(parents=True)
    _ = request_path(tmp_path, other).write_text(_request("../x").model_dump_json())

    with pytest.raises(RequestUnreadable, match="is not its file name"):
        _ = load_request(tmp_path, other)


def test_a_request_file_under_another_id_is_no_open_duplicate(tmp_path: Path) -> None:
    """Skipped with a warning, as a damaged request file is, so submission goes on."""
    other = "01JC0000000000000000000001"
    _ = request_path(tmp_path, other).parent.mkdir(parents=True)
    _ = request_path(tmp_path, other).write_text(_request("../x").model_dump_json())

    with pytest.warns(UserWarning, match="is not its file name"):
        found = open_duplicates(tmp_path, "d" * 64, Scope(), [])

    assert found == ()


def test_a_file_whose_name_is_not_a_ulid_is_not_a_request(tmp_path: Path) -> None:
    _ = save_request(tmp_path, _request())
    _ = (requests_root(tmp_path) / "notes.json").write_text("{}")

    assert request_ids(tmp_path) == [REQUEST_ID]
