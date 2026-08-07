"""What names one variant of a dataset's standardized tracks.

Three producers write into ``tracks/`` -- a registered converter, the TREx
tracker, and an inference op -- and until now none of them recorded which recipe
produced a table. All five write sites targeted one flat
``tracks/<group>__<seq>.parquet`` behind an ``exists()`` skip, so a second
producer's output was discarded with a success return.

The value minted here is the Stage 3.1 tracks hash in full, not a placeholder:
Stage 3.2 promotes it into the directory the parquets live in, so it has to be
right now rather than migrated twice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from mosaic.core.dataset import Dataset, new_dataset_manifest
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.core.pipeline.tracks_identity import (
    TRACKS_IDENTITY_SCHEME,
    convert_variant_payload,
    converter_op,
    infer_variant_payload,
    tracks_run_id,
    tracks_variant_root,
    tracker_variant_payload,
    write_tracks_variant,
)
from mosaic.core.track_converter import TrackConvertParams
from mosaic.core.track_library.sleap import (
    SleapAnalysisH5Converter,
    SleapConvertParams,
)
from mosaic.core.track_library.trex import TrexNpzConverter

VARIANT = re.compile(r"^[a-z0-9_-]+\.[0-9]+(?:\.[0-9]+)*-[0-9a-f]{10}$")


# --- The identity -------------------------------------------------------------


def test_a_variant_names_its_producer_and_version() -> None:
    """Readable on disk, so two producers are told apart without a lookup."""
    minted = tracks_run_id(
        "convert-sleap_analysis_h5", "0.1", {"params": {"fps": 30.0}}
    )

    assert VARIANT.match(minted), minted
    assert minted.startswith("convert-sleap_analysis_h5.0.1-")


def test_it_parses_as_an_op_run_id() -> None:
    """One format for every run identifier in the codebase, not two.

    Including when the producer segment carries an underscore, which every
    converted variant does -- raw formats are spelled ``sleap_analysis_h5``.
    """
    parsed = parse_op_run_id(tracks_run_id("trex", "0.1", {"a": 1}))
    assert parsed is not None
    assert (parsed.kind, parsed.version) == ("trex", "0.1")

    converted = parse_op_run_id(
        tracks_run_id("convert-sleap_analysis_h5", "0.1", {"a": 1})
    )
    assert converted is not None
    assert converted.kind == "convert-sleap_analysis_h5"


def test_a_different_recipe_is_a_different_variant() -> None:
    a = tracks_run_id("convert-sleap_analysis_h5", "0.1", {"params": {"fps": 30.0}})
    b = tracks_run_id("convert-sleap_analysis_h5", "0.1", {"params": {"fps": 60.0}})

    assert a != b


def test_a_version_bump_moves_the_variant_without_moving_the_digest() -> None:
    old = tracks_run_id("trex", "0.1", {"a": 1})
    new = tracks_run_id("trex", "0.2", {"a": 1})

    assert old != new
    assert old.rsplit("-", 1)[1] == new.rsplit("-", 1)[1]


def test_an_absent_upstream_is_omitted_rather_than_hashed_empty() -> None:
    """The term can be added later without moving an unchained variant.

    ``json.dumps(..., sort_keys=True)`` digests an absent key differently from
    one whose value is empty -- the same mechanism that lets ``compute_run_id``
    add a scope term only for scope-dependent features.
    """
    unchained = tracks_run_id("trex", "0.1", {"a": 1})
    explicit_empty = tracks_run_id("trex", "0.1", {"a": 1, "upstream": ""})
    chained = tracks_run_id("trex", "0.1", {"a": 1}, upstream="trex.0.1-aaaaaaaaaa")

    assert unchained != explicit_empty
    assert unchained != chained


def test_the_producer_segment_says_a_conversion_is_a_conversion() -> None:
    """So a converted variant cannot collide with an op of the same name."""
    assert converter_op("trex_npz") == "convert-trex_npz"


# --- The record beside the tables ---------------------------------------------


def test_a_variant_records_what_it_is(tmp_path: Path) -> None:
    run_id = tracks_run_id(
        "convert-sleap_analysis_h5", "0.1", {"params": {"fps": 30.0}}
    )

    path = write_tracks_variant(
        tmp_path, run_id, "convert-sleap_analysis_h5", "0.1", {"fps": 30.0}
    )
    record = json.loads(path.read_text())

    assert path == tracks_variant_root(tmp_path, run_id) / "params.json"
    assert record["op"] == "convert-sleap_analysis_h5"
    assert record["version"] == "0.1"
    assert record["params"] == {"fps": 30.0}
    assert record["identity_scheme"] == TRACKS_IDENTITY_SCHEME


def test_the_observed_tool_version_is_provenance_not_identity(
    tmp_path: Path,
) -> None:
    """An upstream upgrade that produces identical output must not re-derive.

    So what the installed tool reports is written down, and hashed by nothing.
    """
    payload = {"params": {"fps": 30.0}}
    plain = tracks_run_id("trex", "0.1", payload)

    path = write_tracks_variant(
        tmp_path, plain, "trex", "0.1", payload, {"trex_build": "2.1.3-abcdef"}
    )
    record = json.loads(path.read_text())

    assert record["observed"] == {"trex_build": "2.1.3-abcdef"}
    assert tracks_run_id("trex", "0.1", payload) == plain


def test_recording_a_variant_twice_is_idempotent(tmp_path: Path) -> None:
    """One variant is described once, however many sequences it covers."""
    run_id = tracks_run_id("trex", "0.1", {"a": 1})

    first = write_tracks_variant(tmp_path, run_id, "trex", "0.1", {"a": 1})
    before = first.read_text()
    second = write_tracks_variant(tmp_path, run_id, "trex", "0.1", {"a": 1})

    assert first == second
    assert second.read_text() == before


# --- The dataset seam ---------------------------------------------------------


def _dataset(tmp_path: Path, name: str) -> Dataset:
    manifest = new_dataset_manifest(name=name, base_dir=tmp_path / name)
    return Dataset(manifest_path=manifest).load(ensure_roots=True)


def test_one_recipe_is_one_variant_across_every_sequence(tmp_path: Path) -> None:
    """The point of a params-only, scope-free identity.

    A per-sequence value would mint as many variants as there are sequences,
    which is what P2d says an identifier must not do.
    """
    dataset = _dataset(tmp_path, "one-recipe")
    converter = SleapAnalysisH5Converter()
    params = SleapConvertParams(fps=30.0)

    first = dataset._tracks_variant(converter, params)
    second = dataset._tracks_variant(converter, params)

    assert first == second


def test_two_converters_are_two_variants(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, "two-converters")

    sleap = dataset._tracks_variant(SleapAnalysisH5Converter(), SleapConvertParams())
    trex = dataset._tracks_variant(TrexNpzConverter(), TrackConvertParams())

    assert sleap != trex
    assert sleap.startswith("convert-sleap_analysis_h5.")
    assert trex.startswith("convert-trex_npz.")


def test_validation_strictness_does_not_mint_a_second_variant(
    tmp_path: Path,
) -> None:
    """``strict_schema`` is HASH_EXCLUDE, so it reaches no identity."""
    dataset = _dataset(tmp_path, "strictness")
    converter = SleapAnalysisH5Converter()

    lenient = dataset._tracks_variant(
        converter, SleapConvertParams(strict_schema=False)
    )
    strict = dataset._tracks_variant(converter, SleapConvertParams(strict_schema=True))

    assert lenient == strict


def test_converting_records_the_variant_beside_the_tracks(tmp_path: Path) -> None:
    """Explicable from disk, not merely comparable."""
    dataset = _dataset(tmp_path, "recorded")
    run_id = dataset._tracks_variant(
        SleapAnalysisH5Converter(), SleapConvertParams(fps=30.0)
    )

    record = json.loads(
        (
            tracks_variant_root(dataset.get_root("tracks"), run_id) / "params.json"
        ).read_text()
    )

    assert record["op"] == "convert-sleap_analysis_h5"
    assert record["params"]["fps"] == 30.0


# --- the payload builders --------------------------------------------------
#
# Each producer's payload is a named function rather than a dict literal at its
# mint site, so the golden corpus can pin the wrapper and not just the digest.


def test_a_trex_variant_is_the_tracker_run_it_came_from() -> None:
    """Byte-identical to ``trex_run_id`` for the same settings, on purpose.

    At Stage 3.2 ``tracks/trex.<v>-<d>/`` and ``trex/trex.<v>-<d>/`` then read as
    obviously one run. Wrapping the settings would mint a second digest for one
    recipe and produce two near-identical directory names.
    """
    from mosaic.tracking.trex.dataset_runs import trex_run_id
    from mosaic.tracking.trex.version import TREX_KIND, TREX_VERSION

    settings = {"track_max_individuals": 4, "cm_per_pixel": 0.5}
    variant = tracks_run_id(TREX_KIND, TREX_VERSION, tracker_variant_payload(settings))

    assert variant == trex_run_id(settings)


def test_an_infer_variant_is_the_inference_run_it_came_from() -> None:
    """Same payload as ``infer_run_id``, so the two coincide term for term."""
    from mosaic.tracking.ops.infer import PointInferParams, infer_run_id

    params = PointInferParams(model="models/points/best.pt")
    model_id = "train-points.0.1-aaaaaaaaaa"
    variant = tracks_run_id(
        "infer-points",
        "0.1",
        infer_variant_payload(params.identity_dump(), model_id),
    )

    assert variant == infer_run_id("infer-points", "0.1", params, model_id)


def test_the_convert_payload_wraps_params_under_one_named_key() -> None:
    """The key name is load-bearing: renaming it moves every converted variant."""
    assert convert_variant_payload({"neck_idx": 3}) == {"params": {"neck_idx": 3}}


def test_a_variant_is_scope_free_across_the_sequences_it_covers() -> None:
    """One recipe, one value -- however many sequences the run touched."""
    settings = {"track_max_individuals": 4}
    assert tracker_variant_payload(settings) == tracker_variant_payload(dict(settings))
    assert tracks_run_id(
        "trex", "0.1", tracker_variant_payload(settings)
    ) == tracks_run_id("trex", "0.1", tracker_variant_payload(settings))
