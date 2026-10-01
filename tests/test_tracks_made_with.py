"""Which tracks in a dataset were made with a given trained model.

A model is deleted only once nothing refers to it, and a tracks variant refers to
the model its identity payload names. The payloads here are built by each tool's
own settings function, so a tool that renames its model key is still found.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mosaic.core.dataset import Dataset
from mosaic.core.manifest import LibraryLink
from mosaic.core.pipeline.tracks_identity import (
    infer_variant_payload,
    read_tracks_variant,
    tracks_variant_root,
    write_tracks_variant,
)
from mosaic.core.pipeline.tracks_index import (
    TracksMadeWith,
    TracksMadeWithEntry,
    tracks_made_with,
)
from mosaic.tracking.common.mint import mint_tracker_run
from mosaic.tracking.model_refs import resolve_model
from mosaic.tracking.ops.infer import InferPoseOp, PoseInferParams
from mosaic.tracking.sleap.dataset_runs import run_sleap, sleap_settings
from mosaic.tracking.sleap.params import SleapParams
from mosaic.tracking.sleap.version import SLEAP_KIND, SLEAP_VERSION
from mosaic.tracking.trex.dataset_runs import trex_settings
from mosaic.tracking.trex.params import TrexParams
from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

from tests.helpers import (
    add_track_sequences,
    add_tracks_variant,
    make_dataset,
    register_trained_model,
    write_sleap_model,
)

MODEL = "train-sleap.0.2-abcdef0123"
OTHER = "train-sleap.0.2-0123456789"


def _sleap_variant(ds: Dataset, model: str, *sequences: str) -> tuple[str, str]:
    """A SLEAP run's tracks variant over *sequences*, made with *model*.

    Returns the variant and the run that produced it.
    """
    params = SleapParams(model_paths=[model])
    minted = mint_tracker_run(
        ds,
        kind=SLEAP_KIND,
        version=SLEAP_VERSION,
        settings=sleap_settings(params, model_id=model),
    )
    add_tracks_variant(
        ds, minted.tracks_variant, *sequences, producer_run_id=minted.run_id
    )
    return minted.tracks_variant, minted.run_id


def test_every_entry_of_a_variant_made_with_the_model_is_found(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path)
    variant, run_id = _sleap_variant(ds, MODEL, "s1", "s2")

    found = tracks_made_with(ds, MODEL)

    assert found.entries == (
        TracksMadeWithEntry(variant, "", "s1", SLEAP_KIND, run_id),
        TracksMadeWithEntry(variant, "", "s2", SLEAP_KIND, run_id),
    )
    assert found.unreadable_variants == ()


def test_a_variant_made_with_another_model_is_not_found(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path)
    mine, _ = _sleap_variant(ds, MODEL, "s1")
    _ = _sleap_variant(ds, OTHER, "s2")

    found = tracks_made_with(ds, MODEL)

    assert [entry.variant for entry in found.entries] == [mine]
    assert [entry.sequence for entry in found.entries] == ["s1"]


def test_both_of_trexs_models_are_found(tmp_path: Path) -> None:
    """TREx names its detection and identification models under keys of its own."""
    ds = make_dataset(tmp_path)
    detect, identify = "train-pose.0.2-abcdef0123", "train-identity.0.1-abcdef0123"
    minted = mint_tracker_run(
        ds,
        kind=TREX_KIND,
        version=TREX_VERSION,
        settings=trex_settings(
            TrexParams(), detect_model_id=detect, vi_model_id=identify
        ),
    )
    add_tracks_variant(ds, minted.tracks_variant, "s1")

    for model in (detect, identify):
        (entry,) = tracks_made_with(ds, model).entries
        assert entry.variant == minted.tracks_variant
        assert entry.producer == TREX_KIND


def test_a_model_served_by_a_linked_library_is_found(tmp_path: Path) -> None:
    """The payload names the model's run, not the dataset or path it came from."""
    model = "train-pose.0.2-abcdef0123"
    library = make_dataset(tmp_path / "libraries" / "7", name="library")
    weights = library.get_root("models") / "train-pose" / model / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    register_trained_model(library, "train-pose", model, weights)
    project = make_dataset(tmp_path / "52", name="project")
    _ = project.add_library(LibraryLink(id="group", path="../libraries/7"))

    # What an inference run writes for this model, resolved as the run resolves it.
    resolved = resolve_model(project, model, "train-pose")
    assert resolved.library_id == "group", "served by the library, not the project"
    params = PoseInferParams(model=model)
    identity = InferPoseOp().plan_identity(project, params, project.resolve_scope(None))
    _ = write_tracks_variant(
        project.get_root("tracks"),
        identity.tracks_variant,
        "infer-pose",
        InferPoseOp.version,
        infer_variant_payload(params.identity_dump(), resolved.model_id),
    )
    add_tracks_variant(project, identity.tracks_variant, "s1")

    (entry,) = tracks_made_with(project, model).entries

    assert entry.variant == identity.tracks_variant
    assert tracks_made_with(library, model).entries == (), (
        "the tracks are the project's"
    )


def test_a_variant_whose_record_is_absent_or_unreadable_is_reported(
    tmp_path: Path,
) -> None:
    """Either might name the model, so neither is taken to be safe."""
    ds = make_dataset(tmp_path)
    absent, _ = _sleap_variant(ds, MODEL, "s1")
    corrupt, _ = _sleap_variant(ds, OTHER, "s2")
    (tracks_variant_root(ds.get_root("tracks"), absent) / "params.json").unlink()
    _ = (
        tracks_variant_root(ds.get_root("tracks"), corrupt) / "params.json"
    ).write_text("{not json")

    found = tracks_made_with(ds, MODEL)

    assert found.entries == ()
    assert found.unreadable_variants == tuple(sorted((absent, corrupt)))


def test_unlabelled_tables_are_reported_as_the_empty_variant(tmp_path: Path) -> None:
    """Written before variants existed, nothing on disk says what made them."""
    ds = make_dataset(tmp_path)
    add_track_sequences(ds, "old")
    _ = _sleap_variant(ds, OTHER, "s1")

    found = tracks_made_with(ds, MODEL)

    assert found.entries == ()
    assert found.unreadable_variants == ("",)


def test_a_dataset_without_tracks_refers_to_nothing(tmp_path: Path) -> None:
    found = tracks_made_with(make_dataset(tmp_path), MODEL)

    assert found == TracksMadeWith(
        entries=(), unreadable_variants=(), unconfirmed_variants=()
    )


@pytest.mark.parametrize("reference", ["", "models/train-sleap/run", "yolo-pose"])
def test_a_reference_that_is_not_a_run_id_is_refused(
    tmp_path: Path, reference: str
) -> None:
    """A match anywhere in a payload is exact only for a string as distinct as a run id."""
    with pytest.raises(ValueError, match="not a run id"):
        _ = tracks_made_with(make_dataset(tmp_path), reference)


# --- models named by their content ---------------------------------------------


def _registered_sleap_model(ds: Dataset, run_id: str, weights: bytes) -> Path:
    """A finished ``train-sleap`` run whose model directory holds *weights*."""
    directory = write_sleap_model(
        ds.get_root("models") / "train-sleap" / run_id / "model", weights
    )
    register_trained_model(
        ds, "train-sleap", run_id, directory / "best.ckpt", directory=directory
    )
    return directory


def _variants(ds: Dataset) -> list[str]:
    """Every tracks variant recorded in *ds*, sorted."""
    return sorted(
        path.name
        for path in ds.get_root("tracks").iterdir()
        if (path / "params.json").is_file()
    )


def _sleap_run_variant(ds: Dataset, model_paths: list[str]) -> str:
    """The tracks variant that a SLEAP run with *model_paths* records, over s1.

    The run is real up to its media scope, which is empty, so the variant's record
    is the one a run writes.
    """
    _ = ds.index_media([ds.get_root(ds.resolve_media_root())])
    _ = run_sleap(ds, SleapParams(model_paths=model_paths))
    (variant,) = _variants(ds)
    add_tracks_variant(ds, variant, "s1")
    return variant


def test_each_member_of_a_sleap_model_set_is_found(tmp_path: Path) -> None:
    """A set of several references is named by one digest over its artifacts.

    No member's run id reaches the payload. The variant's provenance records each
    one, and a member is found through it.
    """
    ds = make_dataset(tmp_path)
    centroid, instance = MODEL, "train-sleap.0.2-fedcba9876"
    _ = _registered_sleap_model(ds, centroid, b"centroid")
    _ = _registered_sleap_model(ds, instance, b"instance")

    variant = _sleap_run_variant(ds, [centroid, instance])

    sidecar = read_tracks_variant(ds.get_root("tracks"), variant)
    assert sidecar is not None
    assert centroid not in repr(sidecar.params), "the set is named by its digest"
    for model in (centroid, instance):
        (entry,) = tracks_made_with(ds, model).entries
        assert (entry.variant, entry.sequence) == (variant, "s1")
    assert tracks_made_with(ds, OTHER).entries == ()


def test_a_set_run_again_by_path_is_still_found_by_its_members(
    tmp_path: Path,
) -> None:
    """The set's digest covers its artifacts, so both runs are one variant."""
    ds = make_dataset(tmp_path)
    centroid, instance = MODEL, "train-sleap.0.2-fedcba9876"
    by_path = [
        str(_registered_sleap_model(ds, centroid, b"centroid")),
        str(_registered_sleap_model(ds, instance, b"instance")),
    ]
    variant = _sleap_run_variant(ds, [centroid, instance])

    _ = run_sleap(ds, SleapParams(model_paths=by_path))

    assert _variants(ds) == [variant]
    for model in (centroid, instance):
        (entry,) = tracks_made_with(ds, model).entries
        assert entry.variant == variant


def test_a_sleap_set_recorded_without_its_member_runs_is_unconfirmed(
    tmp_path: Path,
) -> None:
    """A set's record from before member runs were kept names only its digest.

    That digest covers every member and equals no one model's, so the record can
    neither confirm nor rule out a member, and says so rather than finding
    nothing.
    """
    ds = make_dataset(tmp_path)
    centroid, instance = MODEL, "train-sleap.0.2-fedcba9876"
    _ = _registered_sleap_model(ds, centroid, b"centroid")
    _ = _registered_sleap_model(ds, instance, b"instance")
    variant = _sleap_run_variant(ds, [centroid, instance])
    record = tracks_variant_root(ds.get_root("tracks"), variant) / "params.json"
    sidecar = json.loads(record.read_text())
    del sidecar["observed"]["model_run_ids"]
    _ = record.write_text(json.dumps(sidecar))
    digest = resolve_model(ds, centroid, "train-sleap").digest

    for found in (
        tracks_made_with(ds, centroid),
        tracks_made_with(ds, centroid, digest=digest),
    ):
        assert found.entries == ()
        assert found.unconfirmed_variants == (variant,)
        assert found.unreadable_variants == ()


def test_a_variant_that_names_another_model_by_run_id_is_ruled_out(
    tmp_path: Path,
) -> None:
    ds = make_dataset(tmp_path)
    _ = _registered_sleap_model(ds, OTHER, b"other")
    _ = _sleap_run_variant(ds, [OTHER])

    found = tracks_made_with(ds, MODEL)

    assert (found.entries, found.unconfirmed_variants) == ((), ())


def test_a_sleap_model_handed_in_by_path_is_found_by_its_digest(
    tmp_path: Path,
) -> None:
    """A directory handed in by path is named by a digest over its files."""
    ds = make_dataset(tmp_path)
    _ = _registered_sleap_model(ds, MODEL, b"weights")
    copy = write_sleap_model(tmp_path / "elsewhere", b"weights")
    variant = _sleap_run_variant(ds, [str(copy)])
    digest = resolve_model(ds, MODEL, "train-sleap").digest

    unasked = tracks_made_with(ds, MODEL)
    assert unasked.entries == ()
    assert unasked.unconfirmed_variants == (variant,), "the same bytes, or not"
    found = tracks_made_with(ds, MODEL, digest=digest)
    (entry,) = found.entries
    assert entry.variant == variant
    assert found.unconfirmed_variants == ()


def test_a_weights_file_handed_in_by_path_is_found_by_its_digest(
    tmp_path: Path,
) -> None:
    """A weights file handed in by path is named by the digest of its bytes."""
    ds = make_dataset(tmp_path)
    model = "train-pose.0.2-abcdef0123"
    weights = ds.get_root("models") / "train-pose" / model / "best.pt"
    weights.parent.mkdir(parents=True)
    _ = weights.write_bytes(b"weights")
    register_trained_model(ds, "train-pose", model, weights)
    copy = tmp_path / "elsewhere" / "best.pt"
    copy.parent.mkdir()
    _ = copy.write_bytes(b"weights")
    # What an inference run handed the copy by path writes, resolved as it resolves.
    params = PoseInferParams(model=str(copy))
    identity = InferPoseOp().plan_identity(ds, params, ds.resolve_scope(None))
    _ = write_tracks_variant(
        ds.get_root("tracks"),
        identity.tracks_variant,
        "infer-pose",
        InferPoseOp.version,
        infer_variant_payload(
            params.identity_dump(), resolve_model(ds, str(copy), "train-pose").model_id
        ),
    )
    add_tracks_variant(ds, identity.tracks_variant, "s1")
    digest = resolve_model(ds, model, "train-pose").digest

    assert tracks_made_with(ds, model).entries == ()
    (entry,) = tracks_made_with(ds, model, digest=digest).entries
    assert entry.variant == identity.tracks_variant
    other = resolve_model(ds, str(weights), "train-pose").digest.replace("0", "1")
    assert tracks_made_with(ds, model, digest=other) == TracksMadeWith(
        entries=(), unreadable_variants=(), unconfirmed_variants=()
    ), "a weights digest names one model, so one that differs rules it out"


def _scheme_one_infer_variant(ds: Dataset, model: str, name: str) -> str:
    """An inference variant as tracks identity scheme 1 recorded one, over s1.

    Scheme 1 named a model handed in by path by a digest of the path string,
    and one handed in by run id by that run id.
    """
    variant = f"infer-pose.0.3-{name}"
    record = write_tracks_variant(
        ds.get_root("tracks"),
        variant,
        "infer-pose",
        "0.3",
        {"params": {"model": model, "conf": 0.25}, "model": model},
    )
    sidecar = json.loads(record.read_text())
    sidecar["identity_scheme"] = "1"
    _ = record.write_text(json.dumps(sidecar))
    add_tracks_variant(ds, variant, "s1")
    return variant


def test_a_scheme_one_inference_variant_of_a_path_is_unconfirmed(
    tmp_path: Path,
) -> None:
    """Its model term digests the path string, which no search can match."""
    ds = make_dataset(tmp_path)
    path_named = _scheme_one_infer_variant(ds, "0123456789abcdef", "0123456789")
    by_run = _scheme_one_infer_variant(ds, MODEL, "abcdef0123")

    found = tracks_made_with(ds, MODEL)

    assert [entry.variant for entry in found.entries] == [by_run]
    assert found.unconfirmed_variants == (path_named,)
    assert found.unreadable_variants == ()


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "abcdef012",
        "abcdef01234",
        "abcdef0123456789a",
        "ABCDEF0123",
        "abcdef012g",
        "train-pose.0.2-abcdef0123",
    ],
)
def test_a_digest_that_is_not_lowercase_hex_of_ten_or_sixteen_is_refused(
    tmp_path: Path, digest: str
) -> None:
    """No model is named by any other value, which could equal a setting."""
    with pytest.raises(ValueError, match="not a model digest"):
        _ = tracks_made_with(make_dataset(tmp_path), MODEL, digest=digest)


@pytest.mark.parametrize("digest", ["abcdef0123", "abcdef0123456789"])
def test_a_digest_of_ten_or_sixteen_lowercase_hex_is_accepted(
    tmp_path: Path, digest: str
) -> None:
    """The lengths a model directory's digest and a weights file's digest have."""
    found = tracks_made_with(make_dataset(tmp_path), MODEL, digest=digest)

    assert found.entries == ()
