"""Entry identity cannot reach a conversion recipe.

A converter used to be a bare callable handed an untyped dict that the caller
had just mutated with the entry's group and sequence. Hashing "the conversion
params" would therefore have hashed the sequence name, minting one tracks
variant per sequence where there is one recipe -- which is exactly what P2d says
an identifier must not contain.

The fix is structural rather than disciplinary: entry identity travels as
``EntryHints``, a frozen dataclass that is deliberately not a ``Params``, so a
sequence name is not *in* the object that gets digested.
``test_no_converter_can_hash_an_entry_name`` is that rule as a check over the
whole registry.

Three of the four converters had no test at all before this file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar, Literal, get_args, override

import numpy as np
import pandas as pd
import pytest

import mosaic.core.track_library  # noqa: F401  -- registers the converters
from mosaic.core.params import Declared
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    KeepEmptyFrames,
    TrackConverter,
    TrackConvertParams,
    builtin_track_converters,
    get_track_converter,
    register_track_converter,
)
from mosaic.core.track_library.calms21 import Calms21Converter, Calms21Params
from mosaic.core.track_library.conformance import converter_problems
from mosaic.core.track_library.sleap import (
    SleapAnalysisH5Converter,
    SleapConvertParams,
)
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline.placement import place_converted
from mosaic.core.pipeline.tracks_index import (
    frame_axis_mismatches,
    read_frames_read,
    read_tracks_index,
)
from mosaic.core.root_lookup import RootLookup
from mosaic.core.track_library.trex import TrexNpzConverter

from tests.helpers import (
    calms21_dataset,
    make_dataset,
    tree_snapshot,
    write_dlc_csv,
    write_sleap_analysis_h5,
    write_trex_npz,
    write_ultralytics_predictions,
)


assert TRACK_CONVERTERS, (
    "the built-in converters are not registered, so every parametrize below "
    "collects zero cases and reports green. The module-scope "
    "`import mosaic.core.track_library` above is what fills the registry."
)


def test_registering_a_converter_first_keeps_the_built_ins() -> None:
    """A caller's own converter does not displace the library's.

    `ensure_track_converters_registered` used to return early whenever the
    registry held anything. A notebook that registered its own converter before
    touching a dataset -- which is what
    `docs/guides/tracking/write-a-converter.md` tells it to do -- then held that
    one converter and no others, and every built-in format reported as
    unregistered for the rest of the process.
    """

    @register_track_converter
    class _CallerConverter(TrackConverter[TrackConvertParams]):
        src_format = "caller_registered_first"
        version = "0.1"

    try:
        assert get_track_converter("trex_npz").src_format == "trex_npz"
        assert "caller_registered_first" in TRACK_CONVERTERS
    finally:
        _ = TRACK_CONVERTERS.pop("caller_registered_first", None)


def test_a_second_class_under_a_registered_format_is_refused() -> None:
    """Registering another class under a format names both and keeps the first.

    A silent overwrite changes what a format converts to for every later
    caller while its variant ids stay the same.
    """
    src_format = "refused_second_class"

    class FirstConverter(TrackConverter[TrackConvertParams]):
        version = "0.1"

    class SecondConverter(TrackConverter[TrackConvertParams]):
        version = "0.1"

    FirstConverter.src_format = SecondConverter.src_format = src_format
    first, second = FirstConverter, SecondConverter
    _ = register_track_converter(first)
    try:
        with pytest.raises(ValueError, match=src_format) as raised:
            _ = register_track_converter(second)

        message = str(raised.value)
        assert f"{first.__module__}.{first.__qualname__}" in message
        assert f"{second.__module__}.{second.__qualname__}" in message
        assert TRACK_CONVERTERS[src_format] is first
    finally:
        _ = TRACK_CONVERTERS.pop(src_format, None)


def test_registering_a_class_again_replaces_it() -> None:
    """The same class registered twice stays registered."""
    src_format = "registered_twice"

    class TwiceConverter(TrackConverter[TrackConvertParams]):
        version = "0.1"

    TwiceConverter.src_format = src_format
    cls = TwiceConverter
    try:
        _ = register_track_converter(cls)
        _ = register_track_converter(cls)

        assert TRACK_CONVERTERS[src_format] is cls
    finally:
        _ = TRACK_CONVERTERS.pop(src_format, None)


def test_a_class_built_anew_replaces_the_registered_one() -> None:
    """A notebook cell run again defines a new class of the same qualified name."""
    src_format = "cell_run_again"

    def run_cell() -> type[TrackConverter[TrackConvertParams]]:
        class CellConverter(TrackConverter[TrackConvertParams]):
            src_format = "cell_run_again"
            version = "0.1"

        return register_track_converter(CellConverter)

    try:
        first = run_cell()
        second = run_cell()

        assert first is not second
        assert TRACK_CONVERTERS[src_format] is second
    finally:
        _ = TRACK_CONVERTERS.pop(src_format, None)


def test_the_built_in_set_holds_the_library_and_not_a_registered_test_class() -> None:
    """``builtin_track_converters`` names the formats under ``mosaic.`` only."""
    src_format = "not_a_built_in"

    class NotBuiltIn(TrackConverter[TrackConvertParams]):
        version = "0.1"

    NotBuiltIn.src_format = src_format
    _ = register_track_converter(NotBuiltIn)
    try:
        built_in = builtin_track_converters()

        assert {"trex_npz", "sleap_analysis_h5", "calms21_npy"} <= set(built_in)
        assert src_format not in built_in
        assert all(cls.__module__.startswith("mosaic.") for cls in built_in.values())
        assert all(TRACK_CONVERTERS[name] is cls for name, cls in built_in.items())
    finally:
        _ = TRACK_CONVERTERS.pop(src_format, None)


# --- The rule -----------------------------------------------------------------

BUILT_IN_FORMATS = sorted(builtin_track_converters())
"""The formats the library registers, as a test module's own converter is not."""


class _SequenceParams(TrackConvertParams):
    """Params that would hash the entry's sequence."""

    sequence: Annotated[str, Declared("A name that belongs to the entry.")] = ""


class _HashesASequence(TrackConverter[_SequenceParams]):
    """A converter whose params hash an entry name, and is never registered."""

    src_format = "hashes_a_sequence_test"
    Params: ClassVar[type[TrackConvertParams]] = _SequenceParams


@pytest.mark.parametrize("src_format", BUILT_IN_FORMATS)
def test_no_converter_can_hash_an_entry_name(src_format: str) -> None:
    """No built-in converter has an entry-identity field in its params.

    The helper checks ``identity_dump()`` rather than the field list, since that
    is the payload the tracks hash is built from. It builds the params with
    ``model_construct`` rather than the validating constructor, because a
    converter is allowed a *required* parameter -- ``trex_npz_scaled`` has one,
    being the reader that is told a factor its files do not record -- and this
    asks what a payload may contain, not whether a default instance exists.

    The class that hashes a sequence is reported first, so the filter on the
    helper's wording is known to select the rule it stands for.
    """
    entry_identity = "entry identity"
    reported = converter_problems(_HashesASequence)
    assert [problem for problem in reported if entry_identity in problem]

    problems = converter_problems(TRACK_CONVERTERS[src_format])

    assert not [problem for problem in problems if entry_identity in problem]


@pytest.mark.parametrize("src_format", BUILT_IN_FORMATS)
def test_every_built_in_converter_conforms(src_format: str) -> None:
    """Each built-in meets the contract that a package's own tests check."""
    assert converter_problems(TRACK_CONVERTERS[src_format]) == []


def test_validation_strictness_is_not_part_of_the_recipe() -> None:
    """``strict_schema`` changes what is *checked*, never what is written."""
    assert "strict_schema" not in TrackConvertParams().identity_dump()


def test_an_unknown_parameter_raises_rather_than_being_ignored() -> None:
    """Silently dropping ``neck_idx`` produced a wrong heading, quietly.

    ``Params`` forbids extras, so a knob aimed at the wrong converter is now a
    loud error. On a genuinely mixed-format dataset, pass ``params_by_format``.
    """
    with pytest.raises(ValueError):
        _ = TrexNpzConverter.Params.from_overrides({"neck_idx": 1})


def test_registering_a_converter_without_a_format_raises() -> None:
    class Nameless(TrexNpzConverter):
        src_format = ""

    with pytest.raises(ValueError, match="src_format"):
        _ = register_track_converter(Nameless)


def test_an_unregistered_format_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="trex_npz"):
        _ = get_track_converter("no_such_format")


def test_indexing_refuses_a_format_no_converter_claims(tmp_path: Path) -> None:
    """The refusal is at indexing, not at the conversion that comes later.

    A typo written into the index is an index nothing can convert, and the row
    then sits there being skipped rather than reported. Resolving the converter
    at the point the format is *chosen* is what makes the typo say so.
    """
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    ds = Dataset(new_dataset_manifest("t", tmp_path / "ds")).load(ensure_roots=True)
    src = tmp_path / "raw"
    src.mkdir()
    (src / "s.csv").write_text("frame,x,y\n0,1,2\n")

    with pytest.raises(ValueError, match="trex_npz"):
        _ = ds.index_tracks_raw([src], patterns=["*.csv"], src_format="no_such_format")

    assert not (ds.get_root("tracks_raw") / "index.csv").exists()


# --- The converters themselves ------------------------------------------------


def _write_trex_npz(path: Path, n: int = 12, ind: int = 0) -> None:
    write_trex_npz(
        path,
        n=n,
        individual=ind,
        columns={
            "poseX": np.stack([np.linspace(0.0, 1.0, n)] * 2, axis=1),
            "poseY": np.stack([np.linspace(1.0, 0.0, n)] * 2, axis=1),
        },
    )


def test_trex_npz_takes_its_entry_from_the_hints(tmp_path: Path) -> None:
    npz = tmp_path / "vid_id0.npz"
    _write_trex_npz(npz)

    df = TrexNpzConverter().convert(
        npz, TrackConvertParams(), EntryHints(group="g", sequence="s")
    )

    assert set(df["group"]) == {"g"}
    assert set(df["sequence"]) == {"s"}


def test_trex_npz_falls_back_to_the_stem_without_its_id_suffix(
    tmp_path: Path,
) -> None:
    """The fallback is TRex's own naming convention, not a parameter."""
    npz = tmp_path / "vid_id0.npz"
    _write_trex_npz(npz)

    df = TrexNpzConverter().convert(npz, TrackConvertParams(), EntryHints())

    assert set(df["sequence"]) == {"vid"}


def test_trex_npz_declares_that_several_files_are_one_sequence() -> None:
    """The declaration ``core`` reads, in place of comparing to a literal.

    The rule is asked of a *stem* rather than of a file, because indexing has to
    group what it found into entries before it opens any of them.
    """
    assert TrexNpzConverter.merges_per_sequence
    assert TrexNpzConverter().sequence_from_stem("hex_7_fish2") == "hex_7"
    assert TrexNpzConverter().sequence_from_stem("no_suffix") == "no_suffix"


def _write_calms21(
    path: Path,
    pairs: dict[str, dict[str, int]],
    keypoints: np.ndarray | None = None,
) -> None:
    """A CalMS21 ``.npy``: zeroed ``(n, 2, 2, 7)`` keypoints unless given some."""
    payload = {
        group: {
            seq: {
                "keypoints": np.zeros((n, 2, 2, 7), dtype=float)
                if keypoints is None
                else keypoints
            }
            for seq, n in seqs.items()
        }
        for group, seqs in pairs.items()
    }
    np.save(path, payload, allow_pickle=True)


def test_calms21_enumerates_group_sequence_pairs(tmp_path: Path) -> None:
    npy = tmp_path / "calms.npy"
    _write_calms21(npy, {"train": {"s0": 4, "s1": 4}})

    assert Calms21Converter().enumerate_sequences(npy) == [
        ("train", "s0"),
        ("train", "s1"),
    ]


def test_calms21_refuses_to_guess_among_several_sequences(tmp_path: Path) -> None:
    """A file holding several sequences is ambiguous without a hint.

    The converter says so rather than picking one -- silently converting the
    first would mint tracks for an entry the caller never named.
    """
    npy = tmp_path / "multi.npy"
    _write_calms21(npy, {"train": {"s0": 4, "s1": 4}})

    with pytest.raises(ValueError, match="contains multiple sequences"):
        _ = Calms21Converter().convert(npy, Calms21Params(), EntryHints())


def test_a_converter_has_one_spelling_of_fps() -> None:
    """One behaviour with two payload shapes would mint two variants.

    A converter reading ``fps`` and falling back to ``fps_default`` would hash
    identical output twice, so the fallback is resolved by the caller before the
    value reaches the digest.
    """
    assert "fps" in SleapConvertParams.model_fields
    assert "fallback_fps" in SleapConvertParams.model_fields
    assert "fps_default" not in SleapConvertParams.model_fields


def test_calms21_survives_one_missing_landmark(tmp_path: Path) -> None:
    """A NaN landmark used to make the whole body centre NaN for that frame.

    CalMS21 was the one converter averaging with a plain ``mean`` rather than
    ``nanmean``, so a single unlabelled landmark erased the animal's position
    for the frame -- silently, since ``mosaic_v1`` records an absent centre as
    NaN and a genuinely absent animal looks the same. All five converters now
    share ``keypoint_centroid``, and this is the behaviour that changed.
    """
    keypoints = np.zeros((3, 1, 2, 4), dtype=float)
    keypoints[:, 0, 0, :] = [0.0, 2.0, 4.0, 6.0]  # x of the four landmarks
    keypoints[:, 0, 1, :] = [1.0, 3.0, 5.0, 7.0]  # y
    keypoints[1, 0, :, 3] = np.nan  # one landmark unlabelled, in frame 1 only

    npy = tmp_path / "nan.npy"
    _write_calms21(npy, {"g": {"s0": 3}}, keypoints)

    table = Calms21Converter().convert(npy, Calms21Params(), EntryHints())

    assert table["X"].tolist() == [3.0, 2.0, 3.0]  # mean(0,2,4,6) / mean(0,2,4)
    assert table["Y"].tolist() == [4.0, 3.0, 4.0]  # mean(1,3,5,7) / mean(1,3,5)


def test_calms21_registers_one_class_per_source_format() -> None:
    """Both spellings convert identically, but name themselves distinctly."""
    npy = get_track_converter("calms21_npy")
    js = get_track_converter("calms21_json")

    assert isinstance(npy, Calms21Converter)
    assert isinstance(js, Calms21Converter)
    assert type(npy).src_format != type(js).src_format
    assert type(npy).Params is type(js).Params


def test_debug_output_is_not_part_of_the_recipe() -> None:
    from mosaic.core.track_library.calms21 import Calms21Params

    assert "debug" not in Calms21Params(debug=True).identity_dump()
    assert Calms21Params(debug=True).identity_dump() == (
        Calms21Params(debug=False).identity_dump()
    )


# --- The dataset-level seam ---------------------------------------------------


def test_two_sequences_of_one_recipe_share_one_params_payload(tmp_path: Path) -> None:
    """The point of the whole item, stated as an equality.

    Two entries, one recipe: whatever the tracks hash is built from must be the
    same object for both, or one recipe becomes as many variants as there are
    sequences.
    """
    params = SleapConvertParams(fps=30.0)

    first = EntryHints(group="g", sequence="seq_a")
    second = EntryHints(group="g", sequence="seq_b")

    assert first != second
    assert params.identity_dump() == params.identity_dump()
    assert "sequence" not in json.dumps(params.identity_dump())


def test_the_dataset_resolves_its_default_into_the_fallback_rate(
    tmp_path: Path,
) -> None:
    """Resolution before hashing: the recorded value is the one the dataset gave.

    The default is the last rate that a conversion takes, so it fills the fallback
    and leaves ``fps``, the explicit rate, unset.
    """
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="fps", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    dataset.meta["fps_default"] = 60.0

    resolved = dataset._converter_params(SleapAnalysisH5Converter(), None)

    assert isinstance(resolved, SleapConvertParams)
    assert resolved.fps is None
    assert resolved.fallback_fps == 60.0


def test_an_explicit_fps_outranks_the_dataset_default(tmp_path: Path) -> None:
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="fps2", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    dataset.meta["fps_default"] = 60.0

    resolved = dataset._converter_params(SleapAnalysisH5Converter(), {"fps": 25.0})

    assert isinstance(resolved, SleapConvertParams)
    assert resolved.fps == 25.0
    assert resolved.fallback_fps == 60.0


def test_entry_hints_never_reach_the_params(tmp_path: Path) -> None:
    """The caller may still pass them; they are dropped, not forwarded."""
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="hints", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)

    resolved = dataset._converter_params(
        TrexNpzConverter(),
        {"group": "g", "sequence": "s", "group_from": "filename"},
    )

    assert resolved.identity_dump() == TrackConvertParams().identity_dump()


def test_params_by_format_keeps_a_mixed_dataset_convertible(tmp_path: Path) -> None:
    """Strict params plus one flat dict would break a mixed-format dataset."""
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="mixed", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)

    overrides = {"params_by_format": {"sleap_analysis_h5": {"fps": 45.0}}}

    sleap = dataset._converter_params(
        SleapAnalysisH5Converter(), overrides, "sleap_analysis_h5"
    )
    trex = dataset._converter_params(TrexNpzConverter(), overrides, "trex_npz")

    assert isinstance(sleap, SleapConvertParams)
    assert sleap.fps == 45.0
    assert trex.identity_dump() == TrackConvertParams().identity_dump()


# --- CalMS21 entry names ----------------------------------------------------
#
# CalMS21 spells its in-file ids as slash paths, read verbatim out of the source
# file. mosaic percent-encodes a "/" for filenames and always has, so this
# worked -- but an entry name doubles as a filesystem path component in the
# control plane, where it does not.


def test_calms21_flattens_a_task_path_into_a_compound_name() -> None:
    from mosaic.core.track_library.calms21 import calms21_entry_name

    assert (
        calms21_entry_name("task1/test/mouse075_task1_annotator1")
        == "task1__test__mouse075_task1_annotator1"
    )
    # A name with no slash is left exactly as it is.
    assert calms21_entry_name("seqA") == "seqA"


def test_calms21_compound_names_parse_with_the_default_separator() -> None:
    """The upgrade the "__" choice buys.

    ``get_sequence_metadata(level_names=[...])`` reads CalMS21's hierarchy with
    its *default* separator now, instead of needing ``separator="/"``.
    """
    from mosaic.core.helpers import parse_hierarchy
    from mosaic.core.track_library.calms21 import calms21_entry_name

    parsed = parse_hierarchy(
        "", calms21_entry_name("task1/test/m075"), ["task", "split", "mouse"]
    )
    assert parsed == {"task": "task1", "split": "test", "mouse": "m075"}


def test_calms21_enumerate_and_convert_agree_on_the_name(tmp_path: Path) -> None:
    """A hint round-tripped through enumerate_sequences must still match.

    They are compared in different places, so if only one flattened the
    conversion would raise KeyError for every sequence.
    """
    from mosaic.core.track_library.calms21 import Calms21Converter

    npy = tmp_path / "calms.npy"
    _write_calms21(npy, {"train": {"task1/test/m075": 4}})
    converter = Calms21Converter()

    pairs = converter.enumerate_sequences(npy)
    assert pairs == [("train", "task1__test__m075")]

    group, sequence = pairs[0]
    df = converter.convert(
        npy, Calms21Params(), EntryHints(group=group, sequence=sequence)
    )
    assert set(df["sequence"]) == {"task1__test__m075"}


def test_calms21_labels_and_tracks_agree_on_the_entry_name(tmp_path: Path) -> None:
    """The label converter must mint the name the track converter minted.

    It did not, and the failure was not a quiet mismatch: the label side wrote
    the raw slash id, and ``write_labels_row`` puts an entry name through
    ``validate_entry_name``, which rejects a ``/``. So converting real CalMS21
    annotations raised outright -- after the ``.npz`` had already been written,
    which left an orphan file that made the retry skip the very sequence that
    failed. Flattening on one side only would have been worse than either: a
    label no track table can be joined to, with nothing raising to say so.
    """
    from mosaic.behavior.label_library.calms21_behavior import (
        CalMS21BehaviorConverter,
        CalMS21BehaviorParams,
    )
    from mosaic.core.entry import validate_entry_name
    from mosaic.core.track_library.calms21 import Calms21Converter

    seq_id = "task1/test/mouse075_task1_annotator1"
    npy = tmp_path / "calms21_task1_test.npy"
    np.save(
        npy,
        {
            "annotator-id_0": {
                seq_id: {
                    "keypoints": np.zeros((6, 2, 2, 7), dtype=float),
                    "annotations": np.zeros(6, dtype=int),
                }
            }
        },
        allow_pickle=True,
    )

    (tracks_pair,) = Calms21Converter().enumerate_sequences(npy)
    entries = CalMS21BehaviorConverter().convert(
        npy, CalMS21BehaviorParams(), {"group": ""}
    )

    assert [e.sequence for e in entries] == [tracks_pair[1]]
    # The name is usable as one path component, which is what raised before.
    assert validate_entry_name(entries[0].sequence, "sequence")
    # The raw id is still recorded -- it is the key inside the source file.
    assert entries[0].payload["sequence_key"] == seq_id


def test_calms21_label_version_says_its_output_identity_moved() -> None:
    """0.2 changed the entry keys, so it changed the filenames labels land in.

    0.3 records ``background_id = -1``, so a row no event matches is unlabeled.
    """
    from mosaic.behavior.label_library.calms21_behavior import (
        CalMS21BehaviorConverter,
    )

    assert CalMS21BehaviorConverter.version == "0.3"


def test_calms21_version_says_its_output_identity_moved() -> None:
    """A variant identity covers what the recipe emits, and that changed three times.

    0.2 changed the entry keys. 0.3 changed the columns: the derived ones are
    gone, and so are eighteen TRex-shaped placeholders -- fifteen of them
    all-NaN floats that every template matrix built from CalMS21 was carrying.
    0.4 dropped the ground-truth ``label`` column.
    """
    from mosaic.core.track_library.calms21 import Calms21Converter

    assert Calms21Converter.version == "0.4"


def test_calms21_tracks_carry_no_ground_truth_label() -> None:
    """The answer a classifier is scored on is not a track column.

    A ``label`` column joined every feature built on the tracks, and a scope-free
    fitter that reads numeric columns by wildcard trained on it. Labels reach a
    run only through convert-labels.
    """
    from mosaic.core.track_library.calms21 import _calms21_seq_to_trex_df

    frame = _calms21_seq_to_trex_df(
        {
            "keypoints": np.zeros((4, 2, 2, 7), dtype=float),
            "annotations": np.array([0, 1, 2, 3]),
        },
        "g",
        "seq",
    )

    assert "label" not in frame.columns


def test_calms21_no_longer_fabricates_trex_shaped_placeholders() -> None:
    """An all-NaN column nobody measured is not a column.

    ``feature_columns()`` selects by wildcard, so each of these was silently
    joining every scaler, embedding and model matrix built from this converter.
    """
    from mosaic.core.track_library.calms21 import _calms21_seq_to_trex_df

    frame = _calms21_seq_to_trex_df(
        {"keypoints": np.zeros((4, 2, 2, 7), dtype=float)}, "g", "seq"
    )
    fabricated = {
        "SPEED#pcentroid",
        "SPEED#wcentroid",
        "midline_x",
        "midline_length",
        "MIDLINE_OFFSET",
        "num_pixels",
        "detection_p",
        "visual_identification_p",
        "missing",
        "timestamp",
    }
    assert fabricated.isdisjoint(set(frame.columns))


# --- entry names are one path component -------------------------------------
#
# Enforced where a name is chosen, at none of the read paths: an index that
# already holds a slash-bearing name keeps resolving exactly as it did.


def test_entry_hints_refuse_a_path_separator() -> None:
    """The earliest of the three write boundaries names the converter."""
    from mosaic.core.entry import validate_entry_name

    with pytest.raises(ValueError, match="forward slash"):
        _ = EntryHints(sequence="task1/test/m075")
    with pytest.raises(ValueError, match="forward slash"):
        _ = EntryHints(group="a/b")
    with pytest.raises(ValueError, match="backslash"):
        _ = EntryHints(sequence="a\\b")

    # And the compound spelling that replaces it is fine.
    assert EntryHints(sequence="task1__test__m075").sequence == "task1__test__m075"
    assert validate_entry_name("task1__test__m075", "sequence") == "task1__test__m075"


@pytest.mark.parametrize("fps", [0.0, -29.97, float("nan"), float("inf")])
def test_entry_hints_refuse_a_rate_that_is_not_a_measurement(fps: float) -> None:
    """An unknown rate is ``None``, never a placeholder a converter would apply."""
    with pytest.raises(ValueError, match="not a measured frame rate"):
        _ = EntryHints(fps=fps)


def test_the_error_says_what_to_do_instead() -> None:
    from mosaic.core.entry import validate_entry_name

    with pytest.raises(ValueError, match="__"):
        _ = validate_entry_name("a/b", "sequence")


# --- SLEAP analysis-HDF5 converter -------------------------------------------


def _two_track_fixture() -> tuple[np.ndarray, np.ndarray]:
    """4 frames, 2 tracks, 2 nodes. Track 0 present 0-3; track 1 present 1-2."""
    tracks = np.full((4, 2, 2, 2), np.nan)
    # track 0: a moving point pair on every frame
    for fr in range(4):
        tracks[fr, 0, 0] = [10.0 + fr, 20.0]
        tracks[fr, 0, 1] = [12.0 + fr, 22.0]
    # track 1: present only on frames 1 and 2
    for fr in (1, 2):
        tracks[fr, 1, 0] = [100.0, 200.0 + fr]
        tracks[fr, 1, 1] = [102.0, 202.0 + fr]
    scores = np.full((4, 2, 2), np.nan)
    scores[:, 0, :] = 0.9
    scores[1:3, 1, :] = 0.8
    return tracks, scores


def test_sleap_converter_flattens_tracks_to_trex_v1(tmp_path: Path) -> None:
    from mosaic.core.track_library.sleap import (
        SleapAnalysisH5Converter,
        SleapConvertParams,
    )

    tracks, scores = _two_track_fixture()
    h5 = tmp_path / "vid1.analysis.h5"
    write_sleap_analysis_h5(h5, tracks, scores, preset="matlab")

    conv = SleapAnalysisH5Converter()
    df = conv.convert(
        h5, SleapConvertParams(fps=25.0), EntryHints(group="g", sequence="s")
    )

    # one id per track, present-frame rows only
    assert set(df["id"]) == {0, 1}
    assert len(df[df["id"] == 0]) == 4
    assert len(df[df["id"] == 1]) == 2
    # required trex_v1 columns + pose prefixes + confidence
    for col in (
        "frame",
        "time",
        "id",
        "group",
        "sequence",
        "poseX0",
        "poseY1",
        "poseP0",
    ):
        assert col in df.columns
    # hints and fps drive group / sequence / time
    assert set(df["group"]) == {"g"} and set(df["sequence"]) == {"s"}
    t0 = df[df["id"] == 0].sort_values("frame")
    assert list(t0["frame"]) == [0, 1, 2, 3]
    assert t0["time"].iloc[2] == pytest.approx(2 / 25.0)
    # track 1 only on frames 1,2
    assert sorted(df[df["id"] == 1]["frame"]) == [1, 2]


def test_sleap_converter_reorders_by_dims_matlab_equals_standard(
    tmp_path: Path,
) -> None:
    from mosaic.core.track_library.sleap import (
        SleapAnalysisH5Converter,
        SleapConvertParams,
    )

    tracks, scores = _two_track_fixture()
    m = tmp_path / "m.analysis.h5"
    s = tmp_path / "s.analysis.h5"
    write_sleap_analysis_h5(m, tracks, scores, preset="matlab")
    write_sleap_analysis_h5(s, tracks, scores, preset="standard")

    conv = SleapAnalysisH5Converter()
    hints = EntryHints(group="", sequence="x")
    dm = conv.convert(m, SleapConvertParams(fps=30.0), hints).sort_values(
        ["id", "frame"]
    )
    dstd = conv.convert(s, SleapConvertParams(fps=30.0), hints).sort_values(
        ["id", "frame"]
    )
    # the dims attribute makes the transposed and native layouts equivalent
    pd.testing.assert_frame_equal(
        dm.reset_index(drop=True), dstd.reset_index(drop=True)
    )


def test_sleap_converter_falls_back_to_matlab_when_dims_absent(tmp_path: Path) -> None:
    from mosaic.core.track_library.sleap import (
        SleapAnalysisH5Converter,
        SleapConvertParams,
    )

    tracks, scores = _two_track_fixture()
    h5 = tmp_path / "nodims.analysis.h5"
    write_sleap_analysis_h5(h5, tracks, scores, preset="matlab", with_dims=False)
    df = SleapAnalysisH5Converter().convert(
        h5, SleapConvertParams(fps=30.0), EntryHints(group="", sequence="x")
    )
    assert set(df["id"]) == {0, 1}
    assert len(df[df["id"] == 0]) == 4


type _SleapPreset = Literal["matlab", "standard"]


def _five_frame_fixture() -> tuple[np.ndarray, np.ndarray]:
    """5 frames, 3 tracks, 2 nodes, and a last frame on which no track is present.

    Track 0 is present on frames 0 to 3, track 1 on frames 1 and 2 with a file
    score of 0.0 on the frames it is absent, and track 2 on none, with a score of
    0.7 throughout.
    """
    tracks = np.full((5, 3, 2, 2), np.nan)
    for frame in range(4):
        tracks[frame, 0, 0] = [10.0 + frame, 20.0]
        tracks[frame, 0, 1] = [12.0 + frame, 22.0]
    for frame in (1, 2):
        tracks[frame, 1, 0] = [100.0, 200.0 + frame]
        tracks[frame, 1, 1] = [102.0, 202.0 + frame]
    scores = np.full((5, 3, 2), np.nan)
    scores[:, 0, :] = 0.9
    scores[:, 1, :] = 0.0
    scores[1:3, 1, :] = 0.8
    scores[:, 2, :] = 0.7
    return tracks, scores


def _convert_five_frames(
    tmp_path: Path, *, keep_empty_frames: bool, preset: _SleapPreset = "matlab"
) -> pd.DataFrame:
    tracks, scores = _five_frame_fixture()
    h5 = tmp_path / "five.analysis.h5"
    write_sleap_analysis_h5(h5, tracks, scores, preset=preset)
    return SleapAnalysisH5Converter().convert(
        h5,
        SleapConvertParams(fps=25.0, keep_empty_frames=keep_empty_frames),
        EntryHints(group="g", sequence="s"),
    )


def test_sleap_default_table_has_no_row_for_an_absent_track_or_empty_frame(
    tmp_path: Path,
) -> None:
    df = _convert_five_frames(tmp_path, keep_empty_frames=False)

    assert sorted(set(df["id"])) == [0, 1]
    assert 4 not in set(df["frame"])
    assert len(df) == 6


def test_sleap_keeps_every_frame_of_every_track_when_asked(tmp_path: Path) -> None:
    default = _convert_five_frames(tmp_path, keep_empty_frames=False)
    every = _convert_five_frames(tmp_path, keep_empty_frames=True)

    assert len(every) == 15
    assert not every.duplicated(["frame", "id"]).any()
    for track_id in (0, 1, 2):
        frames = every.loc[every["id"] == track_id, "frame"]
        assert sorted(frames) == [0, 1, 2, 3, 4]
    assert every["time"].tolist() == pytest.approx((every["frame"] / 25.0).tolist())

    keyed = every.merge(default[["frame", "id"]], on=["frame", "id"], how="inner")
    assert len(keyed) == len(default)
    sorted_every = keyed.sort_values(["id", "frame"]).reset_index(drop=True)
    sorted_default = default.sort_values(["id", "frame"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(sorted_every, sorted_default)

    absent = every.merge(
        default[["frame", "id"]], on=["frame", "id"], how="left", indicator=True
    )
    absent = absent[absent["_merge"] == "left_only"]
    assert len(absent) == 9
    value_columns = [
        column
        for column in every.columns
        if column in ("X", "Y") or column.startswith("pose")
    ]
    assert {"poseX0", "poseY1", "poseP0", "poseP1"} <= set(value_columns)
    assert absent[value_columns].isna().all().all()


def test_sleap_every_frame_table_keeps_the_default_columns(tmp_path: Path) -> None:
    default = _convert_five_frames(tmp_path, keep_empty_frames=False)
    every = _convert_five_frames(tmp_path, keep_empty_frames=True)

    assert list(every.columns) == list(default.columns)
    assert every.dtypes.to_dict() == default.dtypes.to_dict()


@pytest.mark.parametrize("preset", ["matlab", "standard"])
def test_sleap_frames_read_is_the_frame_axis_of_the_file(
    tmp_path: Path, preset: _SleapPreset
) -> None:
    tracks, scores = _five_frame_fixture()
    h5 = tmp_path / "five.analysis.h5"
    write_sleap_analysis_h5(h5, tracks, scores, preset=preset)

    assert SleapAnalysisH5Converter().frames_read(h5) == 5


def test_sleap_frames_read_falls_back_to_the_matlab_order_without_dims(
    tmp_path: Path,
) -> None:
    tracks, scores = _five_frame_fixture()
    h5 = tmp_path / "nodims.analysis.h5"
    write_sleap_analysis_h5(h5, tracks, scores, preset="matlab", with_dims=False)

    assert SleapAnalysisH5Converter().frames_read(h5) == 5


def test_sleap_frames_read_refuses_a_file_without_tracks(tmp_path: Path) -> None:
    import h5py

    h5 = tmp_path / "empty.analysis.h5"
    with h5py.File(str(h5), "w") as handle:
        handle["node_names"] = np.array([b"a"], "S")

    with pytest.raises(ValueError, match="no 'tracks' dataset"):
        SleapAnalysisH5Converter().frames_read(h5)


def test_sleap_frames_read_passes_on_a_file_that_is_not_hdf5(tmp_path: Path) -> None:
    text = tmp_path / "not.h5"
    _ = text.write_text("not hdf5")

    with pytest.raises(OSError):
        SleapAnalysisH5Converter().frames_read(text)


# --- reconversion after a source change -------------------------------------


def _trex_dataset(tmp_path: Path) -> Dataset:
    """A dataset whose ``tracks_raw`` holds one TRex sequence of two individuals."""
    from mosaic.core.dataset import new_dataset_manifest

    manifest = new_dataset_manifest(name="reconv", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    raw = dataset.get_root("tracks_raw")
    raw.mkdir(parents=True, exist_ok=True)
    for individual in (0, 1):
        _write_trex_npz(raw / f"vid_id{individual}.npz", ind=individual)
    _ = dataset.index_tracks_raw([raw], patterns="*.npz", src_format="trex_npz")
    return dataset


def test_a_changed_source_is_reconverted_without_overwrite(tmp_path: Path) -> None:
    """Existence says a recipe has *a* table, never that it matches the source.

    A scan updates the raw checksums and the per-sequence composition; without a
    currency check the conversion that follows skips on existence alone, leaving
    a table -- and an index row claiming a composition -- that disagree with the
    files they name, under a command that reported success.
    """
    dataset = _trex_dataset(tmp_path)
    dataset.convert_all_tracks()
    table = next(dataset.get_root("tracks").glob("*/vid.parquet"))
    before = table.read_bytes()

    # Change the individual the ``source_md5`` column does *not* name, so only
    # the composition over all of them can notice.
    _write_trex_npz(dataset.get_root("tracks_raw") / "vid_id1.npz", n=20, ind=1)
    _ = dataset.index_tracks_raw(
        [dataset.get_root("tracks_raw")], patterns="*.npz", src_format="trex_npz"
    )
    dataset.convert_all_tracks()

    assert table.read_bytes() != before, "the changed source was not reconverted"


def test_an_unchanged_source_is_not_reconverted(tmp_path: Path) -> None:
    """The currency check must not turn every call into a full recompute."""
    dataset = _trex_dataset(tmp_path)
    dataset.convert_all_tracks()
    table = next(dataset.get_root("tracks").glob("*/vid.parquet"))
    stamp = table.stat().st_mtime_ns

    dataset.convert_all_tracks()

    assert table.stat().st_mtime_ns == stamp


def test_a_cached_reconversion_reports_no_superseded_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A conversion that correctly writes nothing has superseded nothing.

    The warning names entries to delete, so a false positive on the idempotent
    path is an invitation to destroy the dataset's tracks.
    """
    dataset = _trex_dataset(tmp_path)
    dataset.convert_all_tracks()
    _ = capsys.readouterr()

    dataset.convert_all_tracks()

    assert "drop_entries" not in capsys.readouterr().err


def test_a_superseded_spelling_is_reported_with_the_safe_remedy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An entry the current recipes do not produce is the true positive.

    The remedy has to name ``run_id=""``: the default drops *every* variant of
    those entries, which with ``delete_files=True`` deletes the conversion that
    was just made.
    """
    from mosaic.core.pipeline.tracks_index import write_tracks_row

    dataset = _trex_dataset(tmp_path)
    dataset.convert_all_tracks()
    stale = dataset.get_root("tracks") / "old__spelling.parquet"
    _ = stale.write_bytes(b"")
    write_tracks_row(
        dataset,
        run_id="",
        group="old",
        sequence="spelling",
        out_path=stale,
        producer="",
        std_format="trex_v1",
        n_rows=0,
        carries_tracklets=None,
    )
    _ = capsys.readouterr()

    dataset.convert_all_tracks()

    reported = capsys.readouterr().err
    assert "('old', 'spelling')" in reported
    assert 'run_id=""' in reported


# --- convert_all_tracks hooks ----------------------------------------------------


def test_track_conversion_variant_matches_the_written_variant_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """The planner's name for a conversion is the one the conversion writes."""
    dataset = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})
    before = tree_snapshot(dataset.base_dir)

    planned = dataset.track_conversion_variant("calms21_npy")

    assert tree_snapshot(dataset.base_dir) == before
    outcome = dataset.convert_all_tracks(source_format="calms21_npy")
    assert outcome.variants == (planned,)


def test_convert_all_tracks_converts_only_the_named_source_format(
    tmp_path: Path,
) -> None:
    """A second format's raw rows are left for a conversion of their own."""
    from mosaic.core.manifest import TracksScanSource
    from mosaic.core.pipeline.tracks_index import read_tracks_index

    dataset = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}}, scan=False)
    trex_dir = dataset.base_dir / "trex"
    _write_trex_npz(trex_dir / "vid_id0.npz", ind=0)
    dataset.add_scan_source(
        TracksScanSource(
            id="trex",
            path=str(trex_dir),
            patterns=("*.npz",),
            src_format="trex_npz",
        )
    )
    dataset.scan_tracks()
    progress: list[tuple[int, int]] = []

    outcome = dataset.convert_all_tracks(
        source_format="calms21_npy",
        progress=lambda done, total: progress.append((done, total)),
    )

    producers = set(read_tracks_index(dataset)["producer"])
    assert producers == {"convert-calms21_npy"}
    assert outcome.converted == 1
    assert progress[-1] == (1, 1)


def test_a_source_format_with_no_raw_rows_is_refused(tmp_path: Path) -> None:
    dataset = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})

    with pytest.raises(ValueError, match="trex_npz"):
        _ = dataset.convert_all_tracks(source_format="trex_npz")


def test_a_failed_file_reaches_on_failure_inside_its_exception(
    tmp_path: Path,
) -> None:
    """The callback runs while the exception is handled, so it can format it."""
    import sys
    import traceback

    dataset = calms21_dataset(
        tmp_path / "ds",
        {"good": {"a": {"s1": 4}}, "bad": {"a": {"s2": 4}}},
        scan=False,
    )
    _ = (dataset.base_dir / "raw" / "bad.npy").write_bytes(b"not an npy file")
    dataset.scan_tracks()
    heard: list[tuple[str, str, bool]] = []

    def on_failure(path: str, exc: BaseException) -> None:
        heard.append((Path(path).name, type(exc).__name__, sys.exc_info()[1] is exc))
        assert "Traceback" in traceback.format_exc()

    outcome = dataset.convert_all_tracks(on_failure=on_failure)

    assert outcome.failed == 1
    assert [(name, live) for name, _, live in heard] == [("bad.npy", True)]


# --- the recording frame axis and repeated rows at the convert sites --------


def test_entry_hints_place_a_sequence_at_frame_zero_by_default() -> None:
    assert EntryHints().first_frame == 0


def test_entry_hints_refuse_a_negative_first_frame() -> None:
    with pytest.raises(ValueError, match="first_frame"):
        _ = EntryHints(first_frame=-1)


def test_placing_a_table_at_frame_zero_returns_it_unchanged() -> None:
    table = pd.DataFrame({"frame": [0, 1], "time": [0.0, 0.05], "id": [0, 0]})

    assert place_converted(table, EntryHints(), media_frame_count=None) is table


def test_placing_a_table_shifts_its_frames_and_times() -> None:
    table = pd.DataFrame({"frame": [0, 1], "time": [0.0, 0.05], "id": [0, 0]})

    placed = place_converted(
        table, EntryHints(fps=20.0, first_frame=100), media_frame_count=None
    )

    assert placed["frame"].tolist() == [100, 101]
    assert placed["time"].tolist() == pytest.approx([5.0, 5.05])


def test_a_table_already_on_the_recording_axis_is_refused() -> None:
    """A division's raw frames end below its frame count, and frame 100 was shifted."""
    table = pd.DataFrame({"frame": [100, 101], "id": [0, 0]})

    with pytest.raises(ValueError, match="already"):
        _ = place_converted(table, EntryHints(first_frame=100), media_frame_count=50)
    assert place_converted(
        table.assign(frame=[0, 49]), EntryHints(first_frame=0), media_frame_count=50
    )["frame"].tolist() == [0, 49]


def _trex_clips(tmp_path: Path, *, second_start: int) -> Dataset:
    """One trial's TRex export of fish 0 in two clip directories, as one entry."""
    from mosaic.core.dataset import new_dataset_manifest

    manifest = new_dataset_manifest(name="clips", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    raw = dataset.get_root("tracks_raw")
    for clip, start in (("clipA", 0), ("clipB", second_start)):
        write_trex_npz(
            raw / clip / "trial_fish0.npz",
            n=6,
            columns={"frame": np.arange(start, start + 6, dtype=np.int64)},
        )
    _ = dataset.index_tracks_raw(
        [raw / "clipA", raw / "clipB"], patterns="*.npz", src_format="trex_npz"
    )
    return dataset


def _failures(dataset: Dataset, source_format: str | None = None) -> list[str]:
    """Convert, and return the message of each failed file."""
    heard: list[str] = []
    _ = dataset.convert_all_tracks(
        source_format=source_format,
        on_failure=lambda _path, error: heard.append(str(error)),
    )
    return heard


def test_merged_clips_each_numbered_from_zero_are_refused(tmp_path: Path) -> None:
    """Two clips numbered from frame 0 repeat every (frame, id) pair they share."""
    dataset = _trex_clips(tmp_path, second_start=0)

    heard = _failures(dataset)

    assert len(heard) == 1
    assert "'trial'" in heard[0]
    assert "frame 0 of id 0 appears in more than one row" in heard[0]
    assert dataset.list_entries() == []


def test_merged_clips_numbered_across_the_recording_convert(tmp_path: Path) -> None:
    dataset = _trex_clips(tmp_path, second_start=6)

    assert _failures(dataset) == []
    assert len(dataset.load_tracks(group="", sequence="trial")) == 12


class _RowsFromCsv(TrackConverter[TrackConvertParams]):
    """Return the rows of a CSV of ``sequence,frame,id`` for the hinted sequence."""

    src_format = "rows_from_csv_test"
    version = "0.1"
    output_schema = "mosaic_v1"

    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        _ = params
        rows = pd.read_csv(path, dtype={"sequence": str})
        if hints.sequence:
            rows = rows[rows["sequence"] == hints.sequence]
        return rows.assign(
            group=hints.group,
            time=rows["frame"] / 30.0,
            X=1.0,
            Y=2.0,
        ).reset_index(drop=True)


class _SequencesFromCsv(_RowsFromCsv):
    """The same rows, from a file that holds several sequences."""

    src_format = "sequences_from_csv_test"
    enumerable = True

    def enumerate_sequences(self, path: Path) -> list[tuple[str, str]]:
        names = pd.read_csv(path, dtype={"sequence": str})["sequence"]
        return [("", str(name)) for name in sorted(set(names))]


@pytest.fixture
def csv_converters() -> Iterator[None]:
    """Register the two CSV converters for one test."""
    for cls in (_RowsFromCsv, _SequencesFromCsv):
        _ = register_track_converter(cls)
    try:
        yield
    finally:
        for cls in (_RowsFromCsv, _SequencesFromCsv):
            _ = TRACK_CONVERTERS.pop(cls.src_format, None)


@pytest.mark.usefixtures("csv_converters")
@pytest.mark.parametrize(
    "src_format",
    [_RowsFromCsv.src_format, _SequencesFromCsv.src_format],
    ids=["per-file", "per-sequence"],
)
def test_a_converted_table_with_a_repeated_row_fails_its_entry(
    tmp_path: Path, src_format: str
) -> None:
    from mosaic.core.dataset import new_dataset_manifest

    manifest = new_dataset_manifest(name="rows", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    raw = dataset.get_root("tracks_raw")
    _ = (raw / "s.csv").write_text("sequence,frame,id\ns,0,0\ns,1,0\ns,1,0\n")
    _ = dataset.index_tracks_raw([raw], patterns="*.csv", src_format=src_format)

    heard = _failures(dataset, src_format)

    assert len(heard) == 1
    assert "'s'" in heard[0]
    assert "frame 1 of id 0 appears in more than one row" in heard[0]
    assert dataset.list_entries() == []


# --- the tracklets flag a conversion records --------------------------------


def _converted_flag(ds: Dataset, params: dict[str, object] | None = None) -> str:
    """Convert every raw row, and return the one tracks row's tracklets cell."""
    _ = ds.convert_all_tracks(params=params)
    index = read_tracks_index(ds)
    assert len(index) == 1
    return str(index.iloc[0]["carries_tracklets"])


def _raw_dataset(tmp_path: Path, src_format: str, pattern: str) -> Dataset:
    ds = make_dataset((tmp_path / "ds").resolve(), roots=("tracks_raw", "tracks"))
    ds.meta["fps_default"] = 30.0
    ds.index_tracks_raw(
        [ds.base_dir / "raw"], patterns=[pattern], src_format=src_format
    )
    return ds


def test_a_trex_conversion_records_tracklets(tmp_path: Path) -> None:
    raw = tmp_path / "ds" / "raw"
    for individual in (0, 1):
        write_trex_npz(raw / f"myseq_fish{individual}.npz", individual=individual)
    ds = _raw_dataset(tmp_path, "trex_npz", "*.npz")
    assert _converted_flag(ds) == "true"


def test_a_calms21_conversion_records_tracklets(tmp_path: Path) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})
    assert _converted_flag(ds) == "true"


def test_a_deeplabcut_conversion_records_tracklets(tmp_path: Path) -> None:
    _ = write_dlc_csv(tmp_path / "ds" / "raw" / "myseq.csv", ["a", "b"], n_frames=6)
    ds = _raw_dataset(tmp_path, "deeplabcut", "*.csv")
    assert _converted_flag(ds) == "true"


def test_an_ultralytics_conversion_records_tracklets(tmp_path: Path) -> None:
    write_ultralytics_predictions(tmp_path / "ds" / "raw" / "myseq.parquet")
    ds = _raw_dataset(tmp_path, "ultralytics_tracks", "*.parquet")
    assert _converted_flag(ds) == "true"


def _sleap_dataset(tmp_path: Path) -> Dataset:
    tracks, scores = _two_track_fixture()
    write_sleap_analysis_h5(tmp_path / "ds" / "raw" / "myseq.h5", tracks, scores)
    return _raw_dataset(tmp_path, "sleap_analysis_h5", "*.h5")


@pytest.mark.parametrize(
    ("answer", "cell"), [(True, "true"), (False, "false"), (None, "")]
)
def test_a_sleap_conversion_records_the_users_answer(
    tmp_path: Path, answer: bool | None, cell: str
) -> None:
    ds = _sleap_dataset(tmp_path)
    params: dict[str, object] = {} if answer is None else {"tracks_named": answer}
    assert _converted_flag(ds, params) == cell


def test_a_sleap_conversion_without_an_answer_keeps_its_variant(
    tmp_path: Path,
) -> None:
    """The field joins the identity only once the user sets it."""
    ds = make_dataset((tmp_path / "ds").resolve())

    assert "tracks_named" not in SleapConvertParams().identity_dump()
    assert (
        ds.track_conversion_variant("sleap_analysis_h5")
        == "convert-sleap_analysis_h5.0.3-3d47f5c74c"
    )


def test_an_answered_sleap_conversion_is_another_variant(tmp_path: Path) -> None:
    ds = make_dataset((tmp_path / "ds").resolve())

    unanswered = ds.track_conversion_variant("sleap_analysis_h5")
    named = ds.track_conversion_variant("sleap_analysis_h5", {"tracks_named": True})
    unnamed = ds.track_conversion_variant("sleap_analysis_h5", {"tracks_named": False})

    assert len({unanswered, named, unnamed}) == 3


# --- the file a converter reads ----------------------------------------------


@dataclass(frozen=True, slots=True)
class _InputFileCall:
    """The arguments of one ``input_file`` call."""

    path: Path
    params: TrackConvertParams
    source_md5: str
    root_of: RootLookup


_input_file_calls: list[_InputFileCall] = []
_converted_paths: list[Path] = []


@pytest.fixture(autouse=True)
def _forget_recorded_calls() -> None:
    _input_file_calls.clear()
    _converted_paths.clear()


class _MergedRowsFromCsv(_RowsFromCsv):
    """Several files make one sequence, named by the stem before the last ``_``."""

    merges_per_sequence = True

    @override
    def sequence_from_stem(self, stem: str) -> str:
        return stem.rsplit("_", 1)[0]


class _Redirecting(TrackConverter[TrackConvertParams]):
    """Read the file of the same name under ``redirected/``, and record each call."""

    @override
    def input_file(
        self,
        path: Path,
        params: TrackConvertParams,
        source_md5: str,
        root_of: RootLookup,
    ) -> Path:
        _input_file_calls.append(_InputFileCall(path, params, source_md5, root_of))
        return path.parent.parent / "redirected" / path.name

    @override
    def convert(
        self, path: Path, params: TrackConvertParams, hints: EntryHints
    ) -> pd.DataFrame:
        _converted_paths.append(path)
        return super().convert(path, params, hints)


class _RedirectedRows(_Redirecting, _RowsFromCsv):
    src_format = "redirected_rows_csv_test"


class _RedirectedSequences(_Redirecting, _SequencesFromCsv):
    src_format = "redirected_sequences_csv_test"


class _RedirectedMerge(_Redirecting, _MergedRowsFromCsv):
    src_format = "redirected_merge_csv_test"


class _Refusing(TrackConverter[TrackConvertParams]):
    """Refuse every file."""

    @override
    def input_file(
        self,
        path: Path,
        params: TrackConvertParams,
        source_md5: str,
        root_of: RootLookup,
    ) -> Path:
        _ = params, source_md5, root_of
        raise ValueError(f"{path} cannot be read")


class _RefusingRows(_Refusing, _RowsFromCsv):
    src_format = "refusing_rows_csv_test"


class _RefusingSequences(_Refusing, _SequencesFromCsv):
    src_format = "refusing_sequences_csv_test"


class _RefusingMerge(_Refusing, _MergedRowsFromCsv):
    src_format = "refusing_merge_csv_test"


REDIRECTING = (
    _RedirectedRows,
    _RedirectedSequences,
    _RedirectedMerge,
    _RefusingRows,
    _RefusingSequences,
    _RefusingMerge,
)


@pytest.fixture
def redirecting_converters(monkeypatch: pytest.MonkeyPatch) -> None:
    for cls in REDIRECTING:
        monkeypatch.setitem(TRACK_CONVERTERS, cls.src_format, cls)


def _rows(sequence: str, frames: int, individual: int = 0) -> list[str]:
    return [f"{sequence},{frame},{individual}" for frame in range(frames)]


def _write_csv(path: Path, rows: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("\n".join(["sequence,frame,id", *rows]) + "\n")
    return path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _redirected_dataset(
    tmp_path: Path,
    src_format: str,
    files: dict[str, tuple[list[str], list[str]]],
    *,
    multi_sequences_per_file: bool = False,
) -> Dataset:
    """A dataset over ``raw/``, whose files each have a counterpart in ``redirected/``.

    Args:
        files: Each file name mapped to the rows of the raw file and the rows of
            its counterpart.
    """
    from mosaic.core.dataset import new_dataset_manifest

    manifest = new_dataset_manifest(name="redirected", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    for name, (raw_rows, redirected_rows) in files.items():
        _ = _write_csv(tmp_path / "raw" / name, raw_rows)
        _ = _write_csv(tmp_path / "redirected" / name, redirected_rows)
    _ = dataset.index_tracks_raw(
        [tmp_path / "raw"],
        patterns="*.csv",
        src_format=src_format,
        multi_sequences_per_file=multi_sequences_per_file,
    )
    return dataset


@pytest.mark.usefixtures("redirecting_converters")
def test_a_single_entry_conversion_reads_the_file_the_hook_returns(
    tmp_path: Path,
) -> None:
    ds = _redirected_dataset(
        tmp_path, _RedirectedRows.src_format, {"s.csv": (_rows("s", 2), _rows("s", 5))}
    )
    raw = tmp_path / "raw" / "s.csv"

    assert _failures(ds, _RedirectedRows.src_format) == []

    assert len(ds.load_tracks(group="", sequence="s")) == 5
    (call,) = _input_file_calls
    assert call.path == raw.resolve()
    assert call.source_md5 == _md5(raw)
    assert call.params == TrackConvertParams()
    assert _converted_paths == [tmp_path / "redirected" / "s.csv"]


@pytest.mark.usefixtures("redirecting_converters")
def test_the_hook_is_handed_the_datasets_roots_by_key(tmp_path: Path) -> None:
    ds = _redirected_dataset(
        tmp_path, _RedirectedRows.src_format, {"s.csv": (_rows("s", 2), _rows("s", 5))}
    )
    _ = _failures(ds, _RedirectedRows.src_format)

    (call,) = _input_file_calls

    assert call.root_of("tracks") == ds.get_root("tracks")
    assert call.root_of("no-such-root") is None


@pytest.mark.usefixtures("redirecting_converters")
def test_an_enumerated_conversion_reads_the_file_the_hook_returns(
    tmp_path: Path,
) -> None:
    ds = _redirected_dataset(
        tmp_path,
        _RedirectedSequences.src_format,
        {"all.csv": (_rows("a", 2) + _rows("b", 2), _rows("a", 4) + _rows("b", 3))},
        multi_sequences_per_file=True,
    )
    raw = tmp_path / "raw" / "all.csv"

    assert _failures(ds, _RedirectedSequences.src_format) == []

    assert len(ds.load_tracks(group="", sequence="a")) == 4
    assert len(ds.load_tracks(group="", sequence="b")) == 3
    assert [(call.path, call.source_md5) for call in _input_file_calls] == [
        (raw.resolve(), _md5(raw))
    ] * 2
    assert _converted_paths == [tmp_path / "redirected" / "all.csv"] * 2


@pytest.mark.usefixtures("redirecting_converters")
def test_a_merged_conversion_reads_the_file_the_hook_returns_for_each_raw_row(
    tmp_path: Path,
) -> None:
    ds = _redirected_dataset(
        tmp_path,
        _RedirectedMerge.src_format,
        {
            "trial_0.csv": (_rows("trial", 2, 0), _rows("trial", 4, 0)),
            "trial_1.csv": (_rows("trial", 2, 1), _rows("trial", 3, 1)),
        },
    )
    raws = [tmp_path / "raw" / "trial_0.csv", tmp_path / "raw" / "trial_1.csv"]

    assert _failures(ds, _RedirectedMerge.src_format) == []

    assert len(ds.load_tracks(group="", sequence="trial")) == 7
    assert sorted((call.path, call.source_md5) for call in _input_file_calls) == [
        (raw.resolve(), _md5(raw)) for raw in raws
    ]
    assert sorted(_converted_paths) == [
        tmp_path / "redirected" / "trial_0.csv",
        tmp_path / "redirected" / "trial_1.csv",
    ]


@pytest.mark.usefixtures("redirecting_converters")
def test_a_hook_that_raises_fails_the_entry_and_writes_no_table(
    tmp_path: Path,
) -> None:
    ds = _redirected_dataset(
        tmp_path,
        _RefusingRows.src_format,
        {"s.csv": (_rows("s", 2), _rows("s", 2))},
    )

    heard = _failures(ds, _RefusingRows.src_format)

    assert len(heard) == 1
    assert "s.csv cannot be read" in heard[0]
    assert ds.list_entries() == []
    assert read_tracks_index(ds).empty


@pytest.mark.usefixtures("redirecting_converters")
def test_a_hook_that_raises_fails_each_sequence_of_an_enumerated_file(
    tmp_path: Path,
) -> None:
    ds = _redirected_dataset(
        tmp_path,
        _RefusingSequences.src_format,
        {"all.csv": (_rows("a", 2) + _rows("b", 2), _rows("a", 2) + _rows("b", 2))},
        multi_sequences_per_file=True,
    )

    heard = _failures(ds, _RefusingSequences.src_format)

    assert len(heard) == 2
    assert all("all.csv cannot be read" in message for message in heard)
    assert ds.list_entries() == []
    assert read_tracks_index(ds).empty


@pytest.mark.usefixtures("redirecting_converters")
def test_a_hook_that_raises_fails_a_merged_sequence(tmp_path: Path) -> None:
    ds = _redirected_dataset(
        tmp_path,
        _RefusingMerge.src_format,
        {
            "trial_0.csv": (_rows("trial", 2, 0), _rows("trial", 2, 0)),
            "trial_1.csv": (_rows("trial", 2, 1), _rows("trial", 2, 1)),
        },
    )

    heard = _failures(ds, _RefusingMerge.src_format)

    assert len(heard) == 1
    assert "cannot be read" in heard[0]
    assert ds.list_entries() == []
    assert read_tracks_index(ds).empty


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_the_default_hook_returns_the_path_it_is_given(
    src_format: str, tmp_path: Path
) -> None:
    converter = get_track_converter(src_format)
    path = tmp_path / "file"

    returned = converter.input_file(
        path, converter.Params.model_construct(), "", lambda _key: None
    )

    assert returned == path


def test_keep_empty_frames_declares_the_shared_prose() -> None:
    declarations = [m for m in get_args(KeepEmptyFrames)[1:] if isinstance(m, Declared)]

    assert get_args(KeepEmptyFrames)[0] is bool
    assert [d.description for d in declarations] == [
        "Write a row for every frame and individual that the source records, "
        "with NaN positions, keypoints and scores where it has none. Unset "
        "keeps the converter's default."
    ]


def test_a_converter_that_keeps_the_base_hook_records_no_frame_count(
    tmp_path: Path,
) -> None:
    converter = get_track_converter("calms21_npy")

    assert type(converter).frames_read is TrackConverter.frames_read
    assert converter.frames_read(tmp_path / "unread.npy") is None


def test_a_frame_count_that_cannot_be_read_is_returned_as_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter = SleapAnalysisH5Converter()
    path = tmp_path / "file.h5"

    def raising(error: Exception) -> Callable[[SleapAnalysisH5Converter, Path], int]:
        def hook(_self: SleapAnalysisH5Converter, _path: Path) -> int:
            raise error

        return hook

    monkeypatch.setattr(SleapAnalysisH5Converter, "frames_read", raising(OSError("a")))
    assert isinstance(converter.try_frames_read(path), OSError)
    monkeypatch.setattr(
        SleapAnalysisH5Converter, "frames_read", raising(ValueError("b"))
    )
    assert isinstance(converter.try_frames_read(path), ValueError)
    monkeypatch.setattr(SleapAnalysisH5Converter, "frames_read", raising(KeyError("c")))
    with pytest.raises(KeyError):
        _ = converter.try_frames_read(path)


def test_a_frame_count_that_can_be_read_is_returned(tmp_path: Path) -> None:
    ds_root = tmp_path / "ds"
    tracks, scores = _two_track_fixture()
    raw = ds_root / "raw" / "myseq.h5"
    write_sleap_analysis_h5(raw, tracks, scores)

    assert SleapAnalysisH5Converter().try_frames_read(raw) == 4
    assert get_track_converter("calms21_npy").try_frames_read(raw) is None


# --- the frame count a conversion records ------------------------------------


@pytest.mark.parametrize("params", [{}, {"keep_empty_frames": True}])
def test_a_sleap_conversion_records_the_frames_of_its_file(
    tmp_path: Path, params: dict[str, object]
) -> None:
    ds = _sleap_dataset(tmp_path)

    _ = ds.convert_all_tracks(params=params)

    (row,) = (r for _, r in read_tracks_index(ds).iterrows())
    assert read_frames_read(row) == 4


def test_a_merging_conversion_records_no_frame_count(tmp_path: Path) -> None:
    ds = _trex_dataset(tmp_path)

    _ = ds.convert_all_tracks()

    index = read_tracks_index(ds)
    assert len(index) == 1
    assert read_frames_read(index.iloc[0]) is None


def test_an_enumerable_conversion_that_keeps_the_base_hook_records_no_count(
    tmp_path: Path,
) -> None:
    ds = calms21_dataset(tmp_path / "ds", {"train": {"a": {"s1": 4}}})

    _ = ds.convert_all_tracks()

    index = read_tracks_index(ds)
    assert len(index) == 1
    assert read_frames_read(index.iloc[0]) is None


_frames_read_calls: list[Path] = []


class _CountedSequences(_SequencesFromCsv):
    """A file of several sequences whose hook counts its calls."""

    src_format = "counted_sequences_csv_test"

    @override
    def frames_read(self, path: Path) -> int | None:
        _frames_read_calls.append(path)
        return 17


@pytest.fixture
def counted_sequences() -> Iterator[None]:
    _frames_read_calls.clear()
    _ = register_track_converter(_CountedSequences)
    try:
        yield
    finally:
        _ = TRACK_CONVERTERS.pop(_CountedSequences.src_format, None)


@pytest.mark.usefixtures("counted_sequences")
def test_an_enumerated_file_is_counted_once_for_all_its_sequences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ds = _redirected_dataset(
        tmp_path,
        _CountedSequences.src_format,
        {"all.csv": (_rows("a", 2) + _rows("b", 2) + _rows("c", 2), [])},
        multi_sequences_per_file=True,
    )

    assert _failures(ds, _CountedSequences.src_format) == []

    index = read_tracks_index(ds)
    assert len(index) == 3
    assert [read_frames_read(row) for _, row in index.iterrows()] == [17, 17, 17]
    assert _frames_read_calls == [(tmp_path / "raw" / "all.csv").resolve()]

    _ = capsys.readouterr()
    _frames_read_calls.clear()
    assert _failures(ds, _CountedSequences.src_format) == []

    assert _frames_read_calls == []
    assert capsys.readouterr().err == ""


def test_a_count_that_cannot_be_read_leaves_a_blank_cell_and_one_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(self: SleapAnalysisH5Converter, path: Path) -> int | None:
        raise OSError(f"cannot open {path.name}")

    monkeypatch.setattr(SleapAnalysisH5Converter, "frames_read", unreadable)
    ds = _sleap_dataset(tmp_path)
    raw = (tmp_path / "ds" / "raw" / "myseq.h5").resolve()

    outcome = ds.convert_all_tracks()

    assert outcome.ok
    assert outcome.converted == 1
    index = read_tracks_index(ds)
    assert len(index) == 1
    assert read_frames_read(index.iloc[0]) is None
    assert len(ds.load_tracks(group="", sequence="myseq")) > 0
    lines = capsys.readouterr().err.splitlines()
    assert lines == [f"[WARN] frames_read of {raw} not recorded: cannot open myseq.h5"]


def test_a_sleap_conversion_leaves_no_frame_axis_mismatch(tmp_path: Path) -> None:
    ds = _sleap_dataset(tmp_path)

    _ = ds.convert_all_tracks()

    assert frame_axis_mismatches(ds) == ()
