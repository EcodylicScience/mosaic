"""Mapping a model's keypoints onto a pose, by name, by override or by index."""

from __future__ import annotations

import math

import pytest

from mosaic.core.annotations.keypoint_map import (
    KeypointMap,
    KeypointMapParams,
    resolve_keypoint_map,
)
from mosaic.core.annotations.model import Keypoint, KeypointSchema
from mosaic.core.params import DeclaredModel, Params

TARGET = KeypointSchema(names=("nose", "left_ear", "right_ear"))


def _placed(x: float, score: float | None = None) -> Keypoint:
    return Keypoint(x=x, y=x + 0.5, visibility=2, score=score)


# --- name match ----------------------------------------------------------------


def test_a_source_with_the_target_names_maps_each_to_its_own_position() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("nose", "left_ear", "right_ear"))

    assert keypoint_map.sources == (0, 1, 2)
    assert keypoint_map.coverage == ("nose", "left_ear", "right_ear")
    assert keypoint_map.unmapped_sources == ()
    assert keypoint_map.is_identity


def test_a_reordered_source_maps_each_name_to_its_source_position() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("right_ear", "nose", "left_ear"))

    assert keypoint_map.sources == (1, 2, 0)
    assert not keypoint_map.is_identity
    applied = keypoint_map.apply((_placed(3.0), _placed(1.0), _placed(2.0)))
    assert [point.x for point in applied] == [1.0, 2.0, 3.0]


def test_a_partial_source_leaves_the_missing_target_unmapped() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("nose", "tail", "left_ear"))

    assert keypoint_map.sources == (0, 2, None)
    assert keypoint_map.coverage == ("nose", "left_ear")
    assert keypoint_map.unmapped_sources == ("tail",)
    assert not keypoint_map.is_identity


def test_names_compare_exactly() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("Nose", "left_ear", "right_ear"))

    assert keypoint_map.sources == (None, 1, 2)
    assert keypoint_map.unmapped_sources == ("Nose",)


def test_without_by_name_only_the_overrides_map() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET,
        ("nose", "left_ear", "right_ear"),
        overrides={"left_ear": "left_ear"},
        by_name=False,
    )

    assert keypoint_map.sources == (None, 1, None)
    assert keypoint_map.coverage == ("left_ear",)
    assert keypoint_map.unmapped_sources == ("nose", "right_ear")


# --- overrides -----------------------------------------------------------------


def test_an_override_by_name_maps_a_differently_named_source() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET, ("snout", "left_ear", "right_ear"), overrides={"nose": "snout"}
    )

    assert keypoint_map.sources == (0, 1, 2)
    assert not keypoint_map.is_identity


def test_an_override_by_index_maps_that_source_position() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET, ("snout", "left_ear", "right_ear"), overrides={"nose": 0}
    )

    assert keypoint_map.sources == (0, 1, 2)


def test_an_override_wins_over_the_name_match() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET, ("nose", "snout", "left_ear", "right_ear"), overrides={"nose": "snout"}
    )

    assert keypoint_map.sources == (1, 2, 3)
    assert keypoint_map.unmapped_sources == ("nose",)


def test_an_override_can_swap_two_same_named_keypoints() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET,
        ("nose", "left_ear", "right_ear"),
        overrides={"left_ear": "right_ear", "right_ear": "left_ear"},
    )

    assert keypoint_map.sources == (0, 2, 1)
    assert not keypoint_map.is_identity


def test_an_index_keyed_map_serves_positionally_named_weights() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET,
        ("kp0", "kp1", "kp2", "kp3"),
        overrides={"nose": 2, "left_ear": 0, "right_ear": 3},
    )

    assert keypoint_map.sources == (2, 0, 3)
    assert keypoint_map.coverage == ("nose", "left_ear", "right_ear")
    assert keypoint_map.unmapped_sources == ("kp1",)


def test_the_order_overrides_are_written_in_does_not_change_the_map() -> None:
    sources = ("a", "b", "c")
    forward = resolve_keypoint_map(
        TARGET, sources, overrides={"nose": "a", "left_ear": "b", "right_ear": "c"}
    )
    backward = resolve_keypoint_map(
        TARGET, sources, overrides={"right_ear": "c", "left_ear": "b", "nose": "a"}
    )

    assert forward == backward


# --- refusals ------------------------------------------------------------------


def test_an_override_naming_no_target_keypoint_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'tail'.*nose"):
        _ = resolve_keypoint_map(
            TARGET, ("nose", "left_ear", "right_ear"), overrides={"tail": "nose"}
        )


def test_an_override_naming_no_source_keypoint_is_refused() -> None:
    with pytest.raises(ValueError, match="no keypoint named 'snoot'"):
        _ = resolve_keypoint_map(
            TARGET, ("snout", "left_ear", "right_ear"), overrides={"nose": "snoot"}
        )


def test_an_override_index_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = resolve_keypoint_map(TARGET, ("kp0", "kp1", "kp2"), overrides={"nose": 3})


def test_a_negative_override_index_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = resolve_keypoint_map(TARGET, ("kp0", "kp1", "kp2"), overrides={"nose": -1})


def test_one_source_mapped_to_two_targets_is_refused() -> None:
    with pytest.raises(ValueError, match=r"'left_ear'.*'nose'.*'left_ear'"):
        _ = resolve_keypoint_map(
            TARGET, ("nose", "left_ear", "right_ear"), overrides={"nose": "left_ear"}
        )


def test_two_overrides_naming_one_source_are_refused() -> None:
    with pytest.raises(ValueError, match="two target keypoints"):
        _ = resolve_keypoint_map(
            TARGET, ("kp0", "kp1", "kp2"), overrides={"nose": 1, "right_ear": "kp1"}
        )


def test_a_map_that_maps_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="maps none"):
        _ = resolve_keypoint_map(TARGET, ("kp0", "kp1", "kp2"))


def test_a_map_without_by_name_or_overrides_is_refused() -> None:
    with pytest.raises(ValueError, match="maps none"):
        _ = resolve_keypoint_map(
            TARGET, ("nose", "left_ear", "right_ear"), by_name=False
        )


def test_a_map_whose_sources_do_not_align_to_the_target_is_refused() -> None:
    with pytest.raises(ValueError, match="one source per target keypoint"):
        _ = KeypointMap(target=TARGET, source_names=("nose",), sources=(0,))


def test_a_map_whose_source_position_is_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        _ = KeypointMap(target=TARGET, source_names=("nose",), sources=(0, 1, None))


# --- apply ---------------------------------------------------------------------


def test_apply_keeps_each_score_and_fills_an_unmapped_target_as_absent() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("left_ear", "tail", "nose"))
    source = (_placed(1.0, score=0.75), _placed(2.0, score=0.5), _placed(3.0))

    applied = keypoint_map.apply(source)

    assert len(applied) == TARGET.num_keypoints
    assert applied[0] == source[2]
    assert applied[0].score is None
    assert applied[1] == source[0]
    assert applied[1].score == 0.75
    assert applied[2].visibility == 0
    assert math.isnan(applied[2].x)
    assert math.isnan(applied[2].y)
    assert applied[2].score is None


def test_apply_refuses_keypoints_not_aligned_to_the_source() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("nose", "left_ear", "right_ear"))

    with pytest.raises(ValueError, match="2 keypoints"):
        _ = keypoint_map.apply((_placed(1.0), _placed(2.0)))


# --- identity ------------------------------------------------------------------


def test_a_source_with_other_names_is_never_the_identity() -> None:
    keypoint_map = resolve_keypoint_map(
        TARGET,
        ("kp0", "kp1", "kp2"),
        overrides={"nose": 0, "left_ear": 1, "right_ear": 2},
    )

    assert keypoint_map.sources == (0, 1, 2)
    assert not keypoint_map.is_identity


def test_the_identity_payload_names_both_poses_and_the_positions() -> None:
    keypoint_map = resolve_keypoint_map(TARGET, ("left_ear", "tail", "nose"))

    assert keypoint_map.identity_payload() == {
        "target": ["nose", "left_ear", "right_ear"],
        "source": ["left_ear", "tail", "nose"],
        "sources": [2, 0, None],
    }


def test_the_identity_payload_does_not_depend_on_how_the_map_was_spelled() -> None:
    sources = ("right_ear", "nose", "left_ear")
    by_name = resolve_keypoint_map(TARGET, sources)
    by_table = resolve_keypoint_map(
        TARGET,
        sources,
        overrides={"nose": "nose", "left_ear": "left_ear", "right_ear": "right_ear"},
        by_name=False,
    )
    by_index = resolve_keypoint_map(
        TARGET, sources, overrides={"nose": 1, "left_ear": 2, "right_ear": 0}
    )

    assert by_name.identity_payload() == by_table.identity_payload()
    assert by_name.identity_payload() == by_index.identity_payload()


def test_maps_resolving_to_other_pairs_have_other_identity_payloads() -> None:
    sources = ("nose", "left_ear", "right_ear")
    straight = resolve_keypoint_map(TARGET, sources)
    swapped = resolve_keypoint_map(
        TARGET, sources, overrides={"left_ear": "right_ear", "right_ear": "left_ear"}
    )

    assert straight.identity_payload() != swapped.identity_payload()


# --- parameters ----------------------------------------------------------------


def test_the_parameters_default_to_a_name_match() -> None:
    params = KeypointMapParams()

    assert params.keypoints == {}
    assert params.by_name


def test_the_parameters_keep_a_name_and_an_index_apart() -> None:
    params = KeypointMapParams.model_validate_json(
        '{"keypoints": {"nose": 0, "left_ear": "kp1"}, "by_name": false}'
    )

    assert params.keypoints == {"nose": 0, "left_ear": "kp1"}
    assert not params.by_name


def test_the_parameters_refuse_a_boolean_where_an_index_belongs() -> None:
    """A JSON ``true`` is not coerced to source position 1."""
    with pytest.raises(ValueError, match="keypoints.nose"):
        _ = KeypointMapParams.model_validate_json('{"keypoints": {"nose": true}}')
    with pytest.raises(ValueError, match="keypoints.nose"):
        _ = KeypointMapParams.model_validate({"keypoints": {"nose": True}})


def test_the_parameters_refuse_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="Extra inputs"):
        _ = KeypointMapParams.model_validate({"keypoint": {}})


def test_the_parameters_are_a_declared_model_and_not_a_parameter_model() -> None:
    """A nested configuration model, which the parameter-model ratchets do not count."""
    assert issubclass(KeypointMapParams, DeclaredModel)
    assert not issubclass(KeypointMapParams, Params)


def test_every_parameter_publishes_a_description() -> None:
    properties: dict[str, dict[str, object]] = KeypointMapParams.model_json_schema()[
        "properties"
    ]

    assert sorted(properties) == ["by_name", "keypoints"]
    for name, schema in properties.items():
        assert schema.get("description"), name
