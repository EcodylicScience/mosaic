"""A ``~`` is expanded where a path enters mosaic, and nowhere after.

Every test here turns on one observation: an unexpanded tilde never raises.
``Path("~/x").resolve()`` is a perfectly good path -- it is ``$CWD/~/x`` -- so
the bug's only evidence is the directory named ``~`` it leaves behind, or a scan
that quietly finds nothing. Hence :func:`assert_no_literal_tilde` at the end of
the boundary tests, and hence the guard below: this is a rule that fails silently
when it is broken, so the suite has to go looking.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from mosaic.cli._context import load_dataset
from mosaic.cli._io import load_json_arg
from mosaic.core.dataset import Dataset, new_dataset_manifest, validate_root_inside
from mosaic.core.pipeline.scan_claim import ScanClaim
from mosaic.core.stored_paths import resolve_stored_path
from mosaic.user_paths import user_path

from tests.helpers import assert_no_literal_tilde, sandbox_home, write_media_index


# --- The helper's own contract ----------------------------------------------


def test_a_leading_tilde_becomes_the_home_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = sandbox_home(monkeypatch, tmp_path / "home")

    assert user_path("~/videos/a.mp4") == home / "videos" / "a.mp4"
    assert user_path("~") == home


def test_a_path_without_a_tilde_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Relative stays relative: expanding is not resolving.

    The distinction is load-bearing. ``MOSAIC_TREX_BIN=trex`` is a bare name for
    ``$PATH`` to find, and a helper that resolved would turn it into ``$CWD/trex``
    and break the launch it was meant to fix.
    """
    _ = sandbox_home(monkeypatch, tmp_path / "home")

    assert user_path("tracks/a.parquet") == Path("tracks/a.parquet")
    assert not user_path("tracks/a.parquet").is_absolute()
    assert user_path("/data/ds") == Path("/data/ds")
    # A tilde anywhere but the first component is an ordinary character.
    assert user_path("data/~odd/a.mp4") == Path("data/~odd/a.mp4")


def test_an_unresolvable_home_is_left_alone_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The whole reason this is a function and not a bare ``.expanduser()``.

    ``Path.expanduser`` raises ``RuntimeError`` naming no path when it cannot
    determine a home directory -- an unknown ``~user``, or ``~`` on an account
    with no home set. ``os.path.expanduser`` returns the input unchanged in
    exactly those cases, which is the tolerant behavior every call site had
    before this helper existed. Forty boundaries each able to raise a message
    that names nothing would be the worse trade.
    """
    _ = sandbox_home(monkeypatch, tmp_path / "home")

    # Pinned against the stdlib rather than asserted from memory.
    with pytest.raises(RuntimeError):
        _ = Path("~nosuchuser4a7f/x").expanduser()

    assert user_path("~nosuchuser4a7f/x") == Path("~nosuchuser4a7f/x")


def test_surrounding_whitespace_is_stripped() -> None:
    """A stored index cell may carry it, and three call sites stripped by hand."""
    assert user_path("  /data/ds  ") == Path("/data/ds")


# --- One spelling of the rule -----------------------------------------------


def test_expanduser_is_called_in_exactly_one_place() -> None:
    """Two spellings of one rule is how the rule drifts.

    Before this change ``src/`` held 23 bare ``.expanduser()`` calls and about
    200 bare ``Path(...)`` sites, and no way to tell which of the second group
    was a considered omission. Reintroducing the bare call is the specific
    regression that would make the boundary unreviewable again, so it is the
    thing pinned -- narrowly, by name, with a one-line repair.
    """
    package = Path(__file__).resolve().parent.parent / "src" / "mosaic"
    # The sandboxed keypoint-MoSeq runner has its own environment and is
    # deliberately outside the main package's rules.
    sandboxed = package / "behavior" / "feature_library" / "external"
    home = package / "user_paths.py"

    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        if path == home or sandboxed in path.parents:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "expanduser":
                offenders.append(f"{path.relative_to(package)}:{node.lineno}")

    assert not offenders, (
        "call mosaic.user_paths.user_path instead of .expanduser() directly, so "
        f"the boundary stays greppable: {offenders}"
    )


# --- The boundaries ---------------------------------------------------------


def test_a_dataset_is_created_under_the_home_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reported bug: fifteen roots under a directory named ``~``."""
    home = sandbox_home(monkeypatch, tmp_path / "home")

    written = new_dataset_manifest("Cage A 2026", "~/study")

    assert written == home / "study" / "dataset.yaml"
    assert written.exists()
    assert (home / "study" / "tracks").is_dir()
    assert_no_literal_tilde(tmp_path)


def test_a_dataset_is_opened_from_a_tilde_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = sandbox_home(monkeypatch, tmp_path / "home")
    _ = new_dataset_manifest("Cage A 2026", home / "study")

    ds = Dataset("~/study/dataset.yaml").load()

    assert ds.name == "Cage A 2026"
    assert ds.base_dir == home / "study"
    assert_no_literal_tilde(tmp_path)


def test_the_cli_opens_a_dataset_from_a_tilde_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One expansion in ``load_dataset`` serves every ``--manifest`` there is."""
    home = sandbox_home(monkeypatch, tmp_path / "home")
    _ = new_dataset_manifest("Cage A 2026", home / "study")

    ds = load_dataset(Path("~/study/dataset.yaml"))

    assert ds.base_dir == home / "study"


def test_a_tilde_root_is_refused_rather_than_created_inside_the_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``~/x`` is not absolute, so unexpanded it read as *relative to the dataset*.

    That is what let it pass the inside-the-dataset guard and be persisted as a
    root, to be recreated under the dataset on every load. Expanded it is an
    outside root, and outside roots are refused.
    """
    home = sandbox_home(monkeypatch, tmp_path / "home")
    base = home / "study"
    _ = new_dataset_manifest("Cage A 2026", base)

    with pytest.raises(ValueError, match="would resolve outside the dataset"):
        _ = validate_root_inside(base, "~/elsewhere", "tracks")

    with pytest.raises(ValueError, match="would resolve outside the dataset"):
        _ = new_dataset_manifest("other", base / "two", roots={"tracks": "~/elsewhere"})

    assert_no_literal_tilde(tmp_path)


def test_a_stored_cell_with_a_tilde_is_not_anchored_into_the_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same laundering one level down, on the path every op's params take."""
    home = sandbox_home(monkeypatch, tmp_path / "home")
    anchor = home / "study"
    anchor.mkdir(parents=True)

    assert resolve_stored_path("~/videos/a.mp4", anchor) == home / "videos" / "a.mp4"
    # Unchanged for the two cases mosaic itself writes.
    assert resolve_stored_path("tracks/a.parquet", anchor) == anchor / "tracks/a.parquet"
    assert resolve_stored_path(str(home / "a.mp4"), anchor) == home / "a.mp4"


def test_a_json_argument_file_is_read_from_a_tilde_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No shell expands ``@~/params.json``: the tilde is not word-initial."""
    home = sandbox_home(monkeypatch, tmp_path / "home")
    _ = (home / "params.json").write_text(json.dumps({"fps": 30}), encoding="utf-8")

    assert load_json_arg("@~/params.json") == {"fps": 30}


def test_a_scan_directory_with_a_tilde_is_searched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The silent class: an unexpanded directory does not exist, so it is skipped.

    ``_probe_dir_rows`` passes over a directory that is not there, so the scan
    reports finding nothing and writes an index that claims nothing, without an
    error anywhere.
    """
    home = sandbox_home(monkeypatch, tmp_path / "home")
    videos = home / "videos"
    videos.mkdir()
    _ = (videos / "a.mp4").write_bytes(b"")
    _ = new_dataset_manifest("Cage A 2026", home / "study")
    ds = Dataset(home / "study" / "dataset.yaml").load()

    claim = ScanClaim.over_directories([user_path("~/videos")])

    assert claim.claims((videos / "a.mp4").resolve())
    assert ds.base_dir == home / "study"
    assert_no_literal_tilde(tmp_path)


def test_a_generator_of_search_directories_still_claims_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``search_dirs`` is an Iterable and used to be consumed twice.

    The first read built the rows and the second built the claim, so a generator
    argument produced an empty claim -- and a scan that claims nothing preserves
    every existing row, which is an append wearing the word "replace".
    """
    home = sandbox_home(monkeypatch, tmp_path / "home")
    _ = new_dataset_manifest("Cage A 2026", home / "study")
    ds = Dataset(home / "study" / "dataset.yaml").load()
    write_media_index(ds, ["a"])
    media_root = ds.get_root(ds.resolve_media_root())
    assert len(ds.read_media_index()) == 1

    # The file is gone, so a scan that genuinely claims this directory must drop
    # its row. An empty claim preserves it and reports success either way, which
    # is why the assertion is on the row rather than on the return value.
    (media_root / "a.mp4").unlink()
    _ = ds.index_media(search_dirs=(d for d in [media_root]))

    assert ds.read_media_index() == []
    assert_no_literal_tilde(tmp_path)
