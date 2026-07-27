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
import pytest

import mosaic.core.track_library  # noqa: F401  -- registers the converters
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConvertParams,
    get_track_converter,
    register_track_converter,
)
from mosaic.core.track_library.calms21 import Calms21Converter, Calms21Params
from mosaic.core.track_library.mabe22 import Mabe22Converter, Mabe22Params
from mosaic.core.track_library.trex import TrexNpzConverter

# What entry identity is spelled as. A converter may accept these as hints; none
# may accept them as parameters.
ENTRY_KEYS = frozenset({"group", "sequence", "group_from"})


# --- The rule -----------------------------------------------------------------


@pytest.mark.parametrize("src_format", sorted(TRACK_CONVERTERS))
def test_no_converter_can_hash_an_entry_name(src_format: str) -> None:
    """No registered converter has an entry-identity field in its params.

    Checked over ``identity_dump()`` rather than the field list, since that is
    the payload the tracks hash is built from.
    """
    params_cls = TRACK_CONVERTERS[src_format].Params
    hashed = set(params_cls().identity_dump())

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
    with pytest.raises(KeyError, match="trex_npz"):
        _ = get_track_converter("no_such_format")


# --- The converters themselves ------------------------------------------------


def _write_trex_npz(path: Path, n: int = 12, ind: int = 0) -> None:
    np.savez(
        path,
        frame=np.arange(n, dtype=np.int64),
        time=np.arange(n, dtype=float) / 30.0,
        id=np.array([ind]),
        X=np.linspace(0.0, 1.0, n),
        Y=np.linspace(1.0, 0.0, n),
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


def _write_mabe22(path: Path, sequences: dict[str, int]) -> None:
    payload = {
        "vocabulary": {"a": 0},
        "sequences": {
            name: {"keypoints": np.zeros((n, 2, 3, 2), dtype=float)}
            for name, n in sequences.items()
        },
    }
    np.save(path, payload, allow_pickle=True)


def test_mabe22_selects_the_hinted_sequence(tmp_path: Path) -> None:
    npy = tmp_path / "mouse_triplet_train.npy"
    _write_mabe22(npy, {"seq_a": 5, "seq_b": 7})

    df = Mabe22Converter().convert(
        npy, Mabe22Params(fps=30.0), EntryHints(sequence="seq_b")
    )

    assert set(df["sequence"]) == {"seq_b"}
    assert set(df["group"]) == {"mouse_triplet_train"}


def test_mabe22_refuses_to_guess_among_several_sequences(tmp_path: Path) -> None:
    npy = tmp_path / "multi.npy"
    _write_mabe22(npy, {"seq_a": 5, "seq_b": 7})

    with pytest.raises(ValueError, match="contains 2 sequences"):
        _ = Mabe22Converter().convert(npy, Mabe22Params(), EntryHints())


def test_mabe22_converts_a_lone_sequence_without_a_hint(tmp_path: Path) -> None:
    """Preserved behaviour: one sequence is unambiguous."""
    npy = tmp_path / "single.npy"
    _write_mabe22(npy, {"only": 5})

    df = Mabe22Converter().convert(npy, Mabe22Params(), EntryHints())

    assert set(df["sequence"]) == {"only"}


def test_mabe22_enumerates_its_sequences(tmp_path: Path) -> None:
    npy = tmp_path / "grp.npy"
    _write_mabe22(npy, {"seq_a": 5, "seq_b": 7})

    assert Mabe22Converter().enumerate_sequences(npy) == [
        ("grp", "seq_a"),
        ("grp", "seq_b"),
    ]


def test_mabe22_has_one_spelling_of_fps() -> None:
    """The old converter read ``fps`` falling back to ``fps_default``.

    One behaviour with two payload shapes would mint two tracks variants for
    identical output, so the fallback moved to the caller, before hashing.
    """
    assert "fps" in Mabe22Params.model_fields
    assert "fps_default" not in Mabe22Params.model_fields


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
    params = Mabe22Params(fps=30.0)

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

    resolved = dataset._converter_params(Mabe22Converter(), None)

    assert isinstance(resolved, Mabe22Params)
    assert resolved.fps == 60.0


def test_an_explicit_fps_outranks_the_dataset_default(tmp_path: Path) -> None:
    from mosaic.core.dataset import Dataset, new_dataset_manifest

    manifest = new_dataset_manifest(name="fps2", base_dir=tmp_path / "ds")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    dataset.meta["fps_default"] = 60.0

    resolved = dataset._converter_params(Mabe22Converter(), {"fps": 25.0})

    assert isinstance(resolved, Mabe22Params)
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

    overrides = {"params_by_format": {"mabe22_npy": {"fps": 45.0}}}

    mabe = dataset._converter_params(Mabe22Converter(), overrides, "mabe22_npy")
    trex = dataset._converter_params(TrexNpzConverter(), overrides, "trex_npz")

    assert isinstance(mabe, Mabe22Params)
    assert mabe.fps == 45.0
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


def test_calms21_version_says_its_output_identity_moved() -> None:
    """A variant identity covers what the recipe emits, and that changed."""
    from mosaic.core.track_library.calms21 import Calms21Converter

    assert Calms21Converter.version == "0.2"


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
