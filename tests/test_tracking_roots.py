"""The tracking-root registry and the backfill a pre-item-8.1 dataset needs.

Item 8.1 asked for the ``_tracking`` literal collapsed into one constant. These
assert the collapse held -- that no module spells the path itself -- and that
loading a manifest written before the relocation neither crashes nor silently
rewrites what is already on disk.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from mosaic.core.dataset import (
    Dataset,
    default_roots,
    legacy_tracking_roots,
    new_dataset_manifest,
)
from mosaic.core.pipeline.tracking_roots import (
    TRACKING_ROOT,
    TRACKING_ROOTS,
    is_under_tracking_root,
    tracking_root_default,
)

_REPO_SRC = Path(__file__).resolve().parents[1] / "src" / "mosaic"


# --- The registry ------------------------------------------------------------


def test_every_tracker_root_defaults_under_the_tracking_root() -> None:
    """The one thing the registry exists to make unspellable elsewhere."""
    for key, root in TRACKING_ROOTS.items():
        assert root.default_path == f"{TRACKING_ROOT}/{key}"
        assert default_roots[key] == root.default_path


def test_the_default_roots_and_the_registry_cannot_drift() -> None:
    """``default_roots`` is built from the table rather than beside it.

    A tracker added to the registry and forgotten in ``default_roots`` would
    self-create its root on first use and be invisible to every portability pass
    until then -- which is the state Lightning Pose was actually in before this
    landed.
    """
    assert set(TRACKING_ROOTS) <= set(default_roots)
    assert default_roots[TRACKING_ROOT] == TRACKING_ROOT


def test_an_unregistered_tracker_gets_no_root_rather_than_a_composed_one() -> None:
    """Composing a path for an unknown tool would put output nothing reclaims."""
    with pytest.raises(KeyError, match="unknown tracking root"):
        _ = tracking_root_default("deepsomething")


def test_the_tracking_root_test_is_component_wise_not_a_prefix() -> None:
    """A sibling named ``_tracking_backup`` is not under ``_tracking``.

    ``str.startswith`` gets this wrong in both directions: it fires on the
    sibling, and it misses a match below whichever directory a scan was handed.
    """
    assert is_under_tracking_root(("ds", "_tracking", "trex", "run", "seq"))
    assert not is_under_tracking_root(("ds", "_tracking_backup", "trex"))
    assert not is_under_tracking_root(("ds", "tracks_raw", "trex"))


def test_no_module_spells_the_tracker_root_path_itself() -> None:
    """Item 8.1's actual request: one edit, not six.

    Asserted over the source rather than over behaviour, because the defect it
    rejects is duplication and duplication is invisible to a passing test. The
    registry module is excluded -- it is where the string is allowed to live.
    """
    hits = subprocess.run(
        ["grep", "-rn", '"_tracking/', str(_REPO_SRC)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    stray = [line for line in hits if "tracking_roots.py" not in line]
    assert not stray, (
        "the tracker root path is spelled outside the registry:\n" + "\n".join(stray)
    )


# --- The backfill ------------------------------------------------------------


def _legacy_manifest(tmp_path: Path) -> Path:
    """A manifest as written before item 8.1: trex under tracks_raw, no _tracking."""
    base = tmp_path / "legacy"
    base.mkdir()
    payload = {
        "name": "legacy",
        "version": "0.1.0",
        "index_format": "group/sequence",
        "roots": {
            "media_raw": "media_raw",
            "tracks_raw": "tracks_raw",
            "tracks": "tracks",
            "trex": "tracks_raw/trex",
            "features": "features",
        },
    }
    manifest = base / "dataset.yaml"
    _ = manifest.write_text(yaml.safe_dump(payload))
    return manifest


def test_a_legacy_manifest_gains_the_tracking_root_it_never_had(tmp_path: Path) -> None:
    """``get_root("_tracking")`` must answer, or the sweeper crashes on it."""
    dataset = Dataset(manifest_path=_legacy_manifest(tmp_path)).load(ensure_roots=False)

    assert dataset.has_root(TRACKING_ROOT)
    assert dataset.get_root(TRACKING_ROOT).name == TRACKING_ROOT
    for key in TRACKING_ROOTS:
        assert key in dataset.roots


def test_a_legacy_tracker_root_is_left_where_it_is(tmp_path: Path) -> None:
    """Repointing it would orphan every run on disk and strand its index.

    The backfill fills absent keys and repoints nothing. This is the assertion
    that distinguishes the two, and it is the one a "helpful" migration would
    break.
    """
    dataset = Dataset(manifest_path=_legacy_manifest(tmp_path)).load(ensure_roots=False)

    assert dataset.roots["trex"] == "tracks_raw/trex"
    assert legacy_tracking_roots(dataset.roots) == {"trex": "tracks_raw/trex"}


def test_a_current_dataset_reports_no_legacy_tracker_root(tmp_path: Path) -> None:
    """The negative half: the query is not vacuously true."""
    manifest = new_dataset_manifest(name="current", base_dir=tmp_path / "current")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)

    assert legacy_tracking_roots(dataset.roots) == {}


def test_the_backfill_adds_only_what_is_absent(tmp_path: Path) -> None:
    """A manifest already declaring every root comes back unchanged.

    Asserted on the roots mapping rather than on the manifest bytes: ``save``
    does not reproduce ``new_dataset_manifest``'s comment header, which is
    pre-existing and has nothing to do with the backfill. Comparing files would
    fail for a reason this test is not about.
    """
    manifest = new_dataset_manifest(name="current", base_dir=tmp_path / "current")
    declared = dict(yaml.safe_load(manifest.read_text())["roots"])

    dataset = Dataset(manifest_path=manifest).load(ensure_roots=False)

    assert dataset.roots == declared
