"""The feature forward pass: recompute run_ids, re-address what the code moved.

Exercises ``Dataset.reconcile`` against a real dataset through the registry, so the
reconciler rebuilds features exactly as the runtime does. The load-bearing check is
the first one: a freshly-run feature reconciles to ``ok`` with its recomputed
identifier byte-identical to the one on disk -- the guard against a second hash
site drifting from the real one.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.identity_scheme import (
    FEATURE_IDENTITY_SCHEME,
    MARKER_NAME,
    read_identity_scheme,
    write_identity_scheme,
)
from mosaic.core.pipeline.index import (
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.run import run_feature
from mosaic.core.pipeline.tracks_identity import (
    TRACKS_IDENTITY_SCHEME,
    convert_variant_payload,
    tracks_run_id,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import tracks_index, tracks_index_path
from mosaic.core.pipeline.types import (
    Feature,
    InputRequire,
    Inputs,
    InputStream,
    Params,
    TrackInput,
)

from .conftest import add_tracks_variant


class _ReconcileFeature:
    """A minimal per-frame feature that reads tracks -- registered for each test.

    Follows the uniform ``FeatureCls(inputs, params)`` constructor the reconciler
    rebuilds through, unlike the workflow tests' local features.
    """

    name = "reconcile-test"
    version = "0.1"
    parallelizable = True
    scope_dependent = False
    consumed_roots: ClassVar[tuple[str, ...]] = ()

    class Inputs(Inputs[TrackInput]):
        _require: ClassVar[InputRequire] = "any"

    class Params(Params):
        pass

    def __init__(
        self,
        inputs: "Inputs[TrackInput] | None" = None,
        params: dict[str, object] | None = None,
    ) -> None:
        self.inputs = inputs if inputs is not None else self.Inputs(("tracks",))
        self.params = self.Params.from_overrides(params)

    def load_state(
        self,
        run_root: Path,
        artifact_paths: dict[str, Path],
        dependency_lookups: dict[str, dict[tuple[str, str], Path]],
    ) -> bool:
        del run_root, artifact_paths, dependency_lookups
        return True

    def fit(self, inputs: InputStream) -> None:
        del inputs

    def save_state(self, run_root: Path) -> None:
        del run_root

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"frame": df["frame"], "value": df["feat_a"] * 2})


@pytest.fixture
def registered_feature() -> Iterator[type[Feature]]:
    """Register ``_ReconcileFeature`` for the test, then remove it.

    The reconciler looks a run's class up in ``FEATURES`` by ``.name``, so a run's
    feature must be registered to be recomputed. Registering in a fixture keeps the
    global registry clean between tests.
    """
    from mosaic.behavior.feature_library.registry import FEATURES

    FEATURES[_ReconcileFeature.__name__] = _ReconcileFeature
    try:
        yield _ReconcileFeature
    finally:
        _ = FEATURES.pop(_ReconcileFeature.__name__, None)
        _ReconcileFeature.version = "0.1"  # undo any per-test version bump


def _only_finding(report: object, verdict: str) -> None:
    """Assert the report holds exactly one finding, of *verdict*."""
    assert hasattr(report, "findings")
    findings = getattr(report, "findings")
    assert len(findings) == 1, [f.verdict for f in findings]
    assert findings[0].verdict == verdict, findings[0]


def _storage(ds: Dataset) -> str:
    """The single feature storage directory name under ``features/``.

    Derived rather than hardcoded: ``("tracks",)`` inputs give the run the storage
    name ``reconcile-test__from__tracks``, not the bare slug.
    """
    root = ds.get_root("features")
    names = sorted(child.name for child in root.iterdir() if child.is_dir())
    assert len(names) == 1, names
    return names[0]


def _run(ds: Dataset) -> str:
    """Run the test feature and return its run_id."""
    return run_feature(ds, _ReconcileFeature()).run_id


def _run_root(ds: Dataset, run_id: str) -> Path:
    return feature_run_root(ds, _storage(ds), run_id)


def test_fresh_run_reconciles_ok(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A just-run feature recomputes to its own identifier: the second-hash-site guard."""
    del registered_feature
    _run(scenario_dataset)

    report = scenario_dataset.reconcile(only=("features",))

    assert not report.changed
    _only_finding(report, "ok")
    assert report.findings[0].old_address == report.findings[0].new_address


def test_stale_scheme_marker_is_refreshed(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """An older marker with an unchanged identifier is scheme_stale, and --apply heals it."""
    del registered_feature
    run_id = _run(scenario_dataset)
    run_root = _run_root(scenario_dataset, run_id)
    write_identity_scheme(run_root, "4")  # pretend it was minted under an older scheme

    report = scenario_dataset.reconcile()
    _only_finding(report, "scheme_stale")
    assert read_identity_scheme(run_root) == "4"  # dry run touched nothing

    applied = scenario_dataset.reconcile(apply=True)
    assert applied.findings[0].action == "marker_refreshed"
    assert read_identity_scheme(run_root) == FEATURE_IDENTITY_SCHEME
    # Idempotent: a second pass now finds it current.
    _only_finding(scenario_dataset.reconcile(), "ok")


def _fabricate_shifted_run(
    ds: Dataset, true_run_id: str, *, scheme: str | None
) -> tuple[str, Path]:
    """Rewrite a run onto a wrong digest under *scheme*, to simulate a moved id.

    A single test run always writes the current digest, so a genuine machinery
    move cannot be produced in one process. Instead: rename the run's directory and
    index rows onto a deliberately-wrong digest and stamp an older (or absent)
    marker. Recompute from the untouched ``params.json`` then yields the *true*
    digest, which differs -- exactly the shape a real scheme change leaves behind.

    Returns ``(fake_run_id, true_run_root)`` -- where an ``--apply`` should land it.
    """
    version, digest = true_run_id.rsplit("-", 1)
    flipped = ("1" if digest[-1] == "0" else "0") + digest[1:]
    fake_run_id = f"{version}-{flipped}"
    storage = _storage(ds)
    true_root = feature_run_root(ds, storage, true_run_id)
    fake_root = feature_run_root(ds, storage, fake_run_id)
    _ = shutil.move(str(true_root), str(fake_root))
    index = feature_index(feature_index_path(ds, storage))
    _ = index.remap_run_id(
        true_run_id,
        fake_run_id,
        path_rewrite=lambda stored: ds.relative_to_root(fake_root / Path(stored).name),
    )
    if scheme is None:
        (fake_root / MARKER_NAME).unlink(missing_ok=True)
    else:
        write_identity_scheme(fake_root, scheme)
    return fake_run_id, true_root


def test_version_bump_stays_ok(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A version bump is a new recipe, never a re-address of old bytes.

    The recomputed id keeps the run's recorded version, so bumping the class
    version leaves existing runs ``ok`` -- the new version is a fresh run the user
    makes deliberately, not a silent restamp of the old output.
    """
    ds = scenario_dataset
    old_run_id = _run(ds)
    old_root = _run_root(ds, old_run_id)

    registered_feature.version = "0.2"

    report = ds.reconcile(apply=True)
    _only_finding(report, "ok")
    assert old_root.exists()  # untouched, still under its recorded version


def test_scheme_change_relocates_the_run(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A digest that moved under an older scheme is a pure, reversible re-address."""
    ds = scenario_dataset
    true_run_id = _run(ds)
    fake_run_id, true_root = _fabricate_shifted_run(ds, true_run_id, scheme="4")
    assert not true_root.exists()  # the run currently sits at the wrong digest

    preview = ds.reconcile()
    _only_finding(preview, "identity_shift_relocatable")
    assert preview.findings[0].new_address == true_run_id
    assert not true_root.exists()  # dry run moved nothing

    applied = ds.reconcile(apply=True)
    assert applied.findings[0].action == "relocated"
    assert true_root.exists()
    assert not _run_root(ds, fake_run_id).exists()
    assert read_identity_scheme(true_root) == FEATURE_IDENTITY_SCHEME
    assert applied.backups  # the index was backed up before the rewrite

    frame = feature_index(feature_index_path(ds, _storage(ds))).read(run_id=true_run_id)
    assert len(frame) == 2  # seq_a, seq_b
    for stored in frame["abs_path"]:
        assert ds.resolve_path(str(stored)).exists()

    _only_finding(ds.reconcile(), "ok")  # idempotent


def test_predates_marker_with_moved_digest_recomputes(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A pre-marker run whose digest moved recomputes; it is never re-addressed."""
    del registered_feature
    ds = scenario_dataset
    true_run_id = _run(ds)
    _, true_root = _fabricate_shifted_run(ds, true_run_id, scheme=None)

    report = ds.reconcile(apply=True)
    _only_finding(report, "identity_shift_recompute")
    assert not true_root.exists()  # left where it is, for an ordinary run to redo


def test_unexplained_digest_under_current_scheme_is_declined(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A digest that moved under the *current* scheme with no upstream move is declined.

    That is a recorded recipe not reproducing its own identifier -- a corruption
    signal, not an honest shift -- so the reconciler refuses to guess at a move.
    """
    del registered_feature
    ds = scenario_dataset
    true_run_id = _run(ds)
    _, true_root = _fabricate_shifted_run(
        ds, true_run_id, scheme=FEATURE_IDENTITY_SCHEME
    )

    report = ds.reconcile(apply=True)
    _only_finding(report, "unresolvable_pre_provenance")
    assert not true_root.exists()  # nothing moved


def test_missing_provenance_is_never_relocated(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A run whose params.json is gone is reported, never re-addressed on a guess."""
    del registered_feature
    ds = scenario_dataset
    old_run_id = _run(ds)
    old_root = _run_root(ds, old_run_id)
    (old_root / "params.json").unlink()

    applied = ds.reconcile(apply=True)
    _only_finding(applied, "unresolvable_pre_provenance")
    assert old_root.exists()  # the run stays put
    assert applied.findings[0].action == "reported"


# --- Tracks variants -------------------------------------------------------

_TRACKS_OP = "convert-testfmt"
_TRACKS_VERSION = "0.1"
_TRACKS_PARAMS: dict[str, object] = {"threshold": 1}


def _make_tracks_variant(ds: Dataset, run_id: str | None = None) -> str:
    """Build a tracks variant with a proper sidecar: parquet, index row, params.json."""
    variant = run_id or tracks_run_id(
        _TRACKS_OP, _TRACKS_VERSION, convert_variant_payload(_TRACKS_PARAMS)
    )
    add_tracks_variant(ds, variant, "seq_a", "seq_b")
    _ = write_tracks_variant(
        ds.get_root("tracks"), variant, _TRACKS_OP, _TRACKS_VERSION, _TRACKS_PARAMS
    )
    return variant


def _set_variant_scheme(variant_root: Path, scheme: str) -> None:
    sidecar = variant_root / "params.json"
    data = json.loads(sidecar.read_text())
    data["identity_scheme"] = scheme
    sidecar.write_text(json.dumps(data))


def _fabricate_tracks_shift(ds: Dataset, true_id: str, *, scheme: str) -> str:
    """Move a variant onto a wrong digest under *scheme*, simulating a moved id."""
    prefix, digest = true_id.rsplit("-", 1)
    fake_id = f"{prefix}-{('1' if digest[-1] == '0' else '0')}{digest[1:]}"
    root = ds.get_root("tracks")
    _ = shutil.move(str(root / true_id), str(root / fake_id))
    _ = tracks_index(tracks_index_path(ds)).remap_run_id(
        true_id,
        fake_id,
        path_rewrite=lambda stored: ds.relative_to_root(
            root / fake_id / Path(stored).name
        ),
    )
    _set_variant_scheme(root / fake_id, scheme)
    return fake_id


def test_tracks_variant_reconciles_ok(scenario_dataset: Dataset) -> None:
    """A variant recomputes to its own id: the tracks second-hash-site guard."""
    _make_tracks_variant(scenario_dataset)
    report = scenario_dataset.reconcile(only=("tracks",))
    assert not report.changed
    _only_finding(report, "ok")


def test_tracks_scheme_stale_is_refreshed(scenario_dataset: Dataset) -> None:
    """An older sidecar scheme with an unchanged digest is scheme_stale; --apply heals it."""
    ds = scenario_dataset
    variant = _make_tracks_variant(ds)
    variant_root = ds.get_root("tracks") / variant
    _set_variant_scheme(variant_root, "1")

    _only_finding(ds.reconcile(only=("tracks",)), "scheme_stale")

    applied = ds.reconcile(apply=True, only=("tracks",))
    assert applied.findings[0].action == "marker_refreshed"
    refreshed = json.loads((variant_root / "params.json").read_text())
    assert refreshed["identity_scheme"] == TRACKS_IDENTITY_SCHEME
    _only_finding(ds.reconcile(only=("tracks",)), "ok")


def test_tracks_variant_relocates(scenario_dataset: Dataset) -> None:
    """A digest that moved under an older scheme is a pure, reversible re-address."""
    ds = scenario_dataset
    true_id = _make_tracks_variant(ds)
    fake_id = _fabricate_tracks_shift(ds, true_id, scheme="1")
    root = ds.get_root("tracks")
    assert not (root / true_id).exists()

    _only_finding(ds.reconcile(only=("tracks",)), "identity_shift_relocatable")

    applied = ds.reconcile(apply=True, only=("tracks",))
    assert applied.findings[0].action == "relocated"
    assert applied.findings[0].new_address == true_id
    assert (root / true_id).exists()
    assert not (root / fake_id).exists()
    frame = tracks_index(tracks_index_path(ds)).read(run_id=true_id)
    assert len(frame) == 2
    for stored in frame["abs_path"]:
        assert ds.resolve_path(str(stored)).exists()
    _only_finding(ds.reconcile(only=("tracks",)), "ok")


def test_tracks_missing_sidecar_is_unresolvable(scenario_dataset: Dataset) -> None:
    """A variant with no sidecar cannot be recomputed, so it is reported, never moved."""
    ds = scenario_dataset
    variant = tracks_run_id(
        _TRACKS_OP, _TRACKS_VERSION, convert_variant_payload(_TRACKS_PARAMS)
    )
    add_tracks_variant(ds, variant, "seq_a")  # no write_tracks_variant -> no sidecar
    _only_finding(ds.reconcile(only=("tracks",)), "unresolvable_pre_provenance")


def test_tracks_move_cascades_into_features(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A relocated tracks variant carries its feature consumers to new ids too."""
    del registered_feature
    ds = scenario_dataset
    # A variant sitting at a wrong digest under an older scheme, that a feature read.
    true_id = tracks_run_id(
        _TRACKS_OP, _TRACKS_VERSION, convert_variant_payload(_TRACKS_PARAMS)
    )
    prefix, digest = true_id.rsplit("-", 1)
    fake_id = f"{prefix}-{('1' if digest[-1] == '0' else '0')}{digest[1:]}"
    _ = _make_tracks_variant(ds, run_id=fake_id)
    _set_variant_scheme(ds.get_root("tracks") / fake_id, "1")

    feature_run_id = run_feature(ds, _ReconcileFeature()).run_id
    storage = _storage(ds)
    assert feature_run_root(ds, storage, feature_run_id).exists()

    applied = ds.reconcile(apply=True)
    by_key = {f.key: f for f in applied.findings}
    assert by_key["tracks"].verdict == "identity_shift_relocatable"
    assert by_key["tracks"].new_address == true_id
    assert by_key["features"].verdict == "identity_shift_relocatable"
    # The feature moved because its tracks input moved, not on its own.
    assert by_key["features"].old_address == feature_run_id
    assert by_key["features"].new_address != feature_run_id
    assert not feature_run_root(ds, storage, feature_run_id).exists()
    assert feature_run_root(ds, storage, by_key["features"].new_address).exists()
    # Idempotent once both have landed (identity kinds; the fixture's absolute
    # tracks paths are a separate, hygiene-pass concern).
    assert not ds.reconcile(only=("features", "tracks")).changed


# --- Composition of the index-hygiene passes ------------------------------


def test_full_reconcile_composes_hygiene_passes(
    scenario_dataset: Dataset, registered_feature: type[Feature]
) -> None:
    """A full run folds in dangling-row and non-portable-path hygiene.

    The scenario fixture writes absolute tracks paths, so a full reconcile surfaces
    them under ``repathed`` (``make_portable``) while the identity findings stay
    ``ok``; a narrowed ``only`` run skips hygiene entirely.
    """
    del registered_feature
    ds = scenario_dataset
    _run(ds)

    preview = ds.reconcile()
    assert preview.changed
    assert preview.repathed  # the fixture's absolute tracks paths are surfaced
    assert all(f.verdict == "ok" for f in preview.findings)

    # Narrowing to one kind is a question about that kind, not the whole tree.
    assert not ds.reconcile(only=("features",)).repathed

    applied = ds.reconcile(apply=True)
    assert applied.repathed
    # Now portable: a second full run is clean.
    assert not ds.reconcile().changed
