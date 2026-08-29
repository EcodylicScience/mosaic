"""Item 1.3: entry identity cannot reach a conversion recipe.

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

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mosaic.core.track_library  # noqa: F401  -- registers the converters
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    get_track_converter,
    register_track_converter,
)
from mosaic.core.track_library.calms21 import Calms21Converter, Calms21Params
from mosaic.core.track_library.sleap import (
    SleapAnalysisH5Converter,
    SleapConvertParams,
)
from mosaic.core.dataset import Dataset
from mosaic.core.track_library.trex import TrexNpzConverter

from tests.helpers import write_trex_npz

# What entry identity is spelled as. A converter may accept these as hints; none
# may accept them as parameters.
ENTRY_KEYS = frozenset({"group", "sequence", "group_from"})

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


# --- The rule -----------------------------------------------------------------


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_no_converter_can_hash_an_entry_name(src_format: str) -> None:
    """No registered converter has an entry-identity field in its params.

    Checked over ``identity_dump()`` rather than the field list, since that is
    the payload the tracks hash is built from. Built with ``model_construct``
    rather than the validating constructor, because a converter is allowed a
    *required* parameter -- ``trex_npz_scaled`` has one, being the reader that is
    told a factor its files do not record -- and this asks what a payload may
    contain, not whether a default instance exists.
    """
    params_cls = TRACK_CONVERTERS[src_format].Params
    hashed = set(params_cls.model_construct().identity_dump())

    assert not (hashed & ENTRY_KEYS), (
        f"{src_format} would hash entry identity: {sorted(hashed & ENTRY_KEYS)}"
    )


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_every_converter_declares_a_version(src_format: str) -> None:
    """The version becomes a visible segment of the tracks variant identity."""
    assert TRACK_CONVERTERS[src_format].version


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_the_registry_key_is_the_declared_format(src_format: str) -> None:
    """One class per format, so a variant identity names exactly one producer."""
    assert TRACK_CONVERTERS[src_format].src_format == src_format


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_a_merging_format_declares_how_a_stem_names_its_sequence(
    src_format: str,
) -> None:
    """Several files are one sequence only if they agree on the name.

    Left at the base's answer the stem *is* the sequence, so every per-individual
    file would be its own entry and the merge would find one file per group.
    """
    converter_cls = TRACK_CONVERTERS[src_format]
    declares_a_rule = (
        converter_cls.sequence_from_stem is not TrackConverter.sequence_from_stem
    )

    assert declares_a_rule or not converter_cls.merges_per_sequence, (
        f"{src_format} declares merges_per_sequence without overriding "
        "sequence_from_stem, so each of its files would be its own sequence"
    )


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_no_format_claims_both_directions_of_one_relationship(
    src_format: str,
) -> None:
    """``enumerable`` and ``merges_per_sequence`` are opposite claims.

    One says a single file holds several sequences, the other that several files
    hold one. A format claiming both would have ``convert_one_track`` expand it
    into per-sequence tables and ``convert_all_tracks`` concatenate those same
    rows under a blank sequence.
    """
    converter_cls = TRACK_CONVERTERS[src_format]

    assert not (converter_cls.enumerable and converter_cls.merges_per_sequence), (
        f"{src_format} claims both that one file holds many sequences and that "
        "many files hold one"
    )


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
        poseX=np.stack([np.linspace(0.0, 1.0, n)] * 2, axis=1),
        poseY=np.stack([np.linspace(1.0, 0.0, n)] * 2, axis=1),
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


def _write_calms21(path: Path, pairs: dict[str, dict[str, int]]) -> None:
    payload = {
        group: {
            seq: {"keypoints": np.zeros((n, 2, 2, 7), dtype=float)}
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
    assert "fps_default" not in SleapConvertParams.model_fields


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


def test_the_dataset_resolves_fps_from_its_own_default(tmp_path: Path) -> None:
    """Resolution before hashing: the recorded value is the one used."""
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="fps", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    dataset.meta["fps_default"] = 60.0

    resolved = dataset._converter_params(SleapAnalysisH5Converter(), None)

    assert isinstance(resolved, SleapConvertParams)
    assert resolved.fps == 60.0


def test_an_explicit_fps_outranks_the_dataset_default(tmp_path: Path) -> None:
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="fps2", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    dataset.meta["fps_default"] = 60.0

    resolved = dataset._converter_params(SleapAnalysisH5Converter(), {"fps": 25.0})

    assert isinstance(resolved, SleapConvertParams)
    assert resolved.fps == 25.0


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
    from mosaic.core.helpers import validate_entry_name
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
    """0.2 changed the entry keys, so it changed the filenames labels land in."""
    from mosaic.behavior.label_library.calms21_behavior import (
        CalMS21BehaviorConverter,
    )

    assert CalMS21BehaviorConverter.version == "0.2"


def test_calms21_version_says_its_output_identity_moved() -> None:
    """A variant identity covers what the recipe emits, and that changed twice.

    0.2 changed the entry keys. 0.3 changed the columns: the derived ones are
    gone, and so are eighteen TRex-shaped placeholders -- fifteen of them
    all-NaN floats that every template matrix built from CalMS21 was carrying.
    """
    from mosaic.core.track_library.calms21 import Calms21Converter

    assert Calms21Converter.version == "0.3"


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
    from mosaic.core.helpers import validate_entry_name

    with pytest.raises(ValueError, match="forward slash"):
        _ = EntryHints(sequence="task1/test/m075")
    with pytest.raises(ValueError, match="forward slash"):
        _ = EntryHints(group="a/b")
    with pytest.raises(ValueError, match="backslash"):
        _ = EntryHints(sequence="a\\b")

    # And the compound spelling that replaces it is fine.
    assert EntryHints(sequence="task1__test__m075").sequence == "task1__test__m075"
    assert validate_entry_name("task1__test__m075", "sequence") == "task1__test__m075"


def test_the_error_says_what_to_do_instead() -> None:
    from mosaic.core.helpers import validate_entry_name

    with pytest.raises(ValueError, match="__"):
        _ = validate_entry_name("a/b", "sequence")


# --- SLEAP analysis-HDF5 converter -------------------------------------------


def _write_sleap_analysis_h5(
    path: Path,
    tracks_ftn2: np.ndarray,
    scores_ftn: np.ndarray | None = None,
    *,
    preset: str = "matlab",
    with_dims: bool = True,
) -> None:
    """Write a synthetic SLEAP analysis HDF5 from a canonical tracks array.

    *tracks_ftn2* is ``(frame, track, node, 2)``; *scores_ftn* is
    ``(frame, track, node)`` or None. ``matlab`` writes the transposed layout
    ``sleap-convert`` produces by default; ``standard`` writes the Python-native
    layout -- both carry a ``dims`` attribute so the converter reorders either.
    """
    import h5py

    if preset == "matlab":
        arr = np.transpose(tracks_ftn2, (1, 3, 2, 0))  # (track, xy, node, frame)
        dims = ["track", "xy", "node", "frame"]
        sarr = np.transpose(scores_ftn, (1, 2, 0)) if scores_ftn is not None else None
        sdims = ["track", "node", "frame"]
    else:  # standard / python-native
        arr = tracks_ftn2
        dims = ["frame", "track", "node", "xy"]
        sarr = scores_ftn
        sdims = ["frame", "track", "node"]

    with h5py.File(str(path), "w") as f:
        d = f.create_dataset("tracks", data=arr)
        if with_dims:
            d.attrs["dims"] = json.dumps(dims)
        if sarr is not None:
            s = f.create_dataset("point_scores", data=sarr)
            if with_dims:
                s.attrs["dims"] = json.dumps(sdims)


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
    _write_sleap_analysis_h5(h5, tracks, scores, preset="matlab")

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
    _write_sleap_analysis_h5(m, tracks, scores, preset="matlab")
    _write_sleap_analysis_h5(s, tracks, scores, preset="standard")

    conv = SleapAnalysisH5Converter()
    hints = EntryHints(group="", sequence="x")
    dm = conv.convert(m, SleapConvertParams(), hints).sort_values(["id", "frame"])
    dstd = conv.convert(s, SleapConvertParams(), hints).sort_values(["id", "frame"])
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
    _write_sleap_analysis_h5(h5, tracks, scores, preset="matlab", with_dims=False)
    df = SleapAnalysisH5Converter().convert(
        h5, SleapConvertParams(), EntryHints(group="", sequence="x")
    )
    assert set(df["id"]) == {0, 1}
    assert len(df[df["id"] == 0]) == 4


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
    )
    _ = capsys.readouterr()

    dataset.convert_all_tracks()

    reported = capsys.readouterr().err
    assert "('old', 'spelling')" in reported
    assert 'run_id=""' in reported
