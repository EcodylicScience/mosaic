"""Test the ``media`` parameter on the seven consumers of media, and its refusal.

A consumer that names a media variant reads a video already cut to the variant's
frame range. Every frame number that a user types is a source frame, while the
tool applies a consumer's frame window to the variant's frames. One validator on
:class:`MediaInputParams` refuses the combination, for the typed window fields and
for the same settings passed through a tool's extra-settings dictionary.

These tests check each op against the windows that it declares, pin the refusal's
wording, show that a recipe step combining the two is refused before any step
runs, and show that ``media`` changes an identity only when it is set.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline._utils import ResolvedScope
from mosaic.core.pipeline.graph import (
    Recipe,
    RecipeInvalid,
    check_recipe,
    plan_pipeline,
)
from mosaic.core.pipeline.media_input import MediaInputParams, media_identity_terms
from mosaic.core.pipeline.ops import OPS, OpIdentity
from mosaic.tracking import register_ops
from tests.helpers import make_dataset

register_ops()

VARIANT = "preprocess.0.1-3f2a9c01d4"
OTHER_VARIANT = "preprocess.0.1-0123456789"

CONSUMERS: dict[str, dict[str, JsonValue]] = {
    "trex": {},
    "sleap": {"model_paths": ["train-sleap.0.1-aaaaaaaaaa"]},
    "litpose": {"model_path": "train-litpose.0.1-aaaaaaaaaa"},
    "ultralytics": {"model_path": "train-pose.0.1-aaaaaaaaaa"},
    "infer-pose": {"model": "train-pose.0.1-aaaaaaaaaa"},
    "infer-points": {"model": "train-points.0.1-aaaaaaaaaa"},
    "infer-localizer": {"model": "train-localizer.0.1-aaaaaaaaaa"},
}
"""Each op with a ``media`` parameter, and the params that it needs to validate.

The models are named by training run identifiers. An identity plans on an empty dataset,
because a run identifier serves as the model identity and the planner does not read
weights to find it.
"""

_INFER_WINDOW: dict[str, JsonValue] = {
    "start_frame": 10,
    "end_frame": 200,
    "frame_step": 2,
    "max_frames": 50,
}

WINDOWS: dict[str, dict[str, JsonValue]] = {
    "trex": {"analysis_range": [10, 200]},
    "sleap": {"analysis_range": [10, 200]},
    "litpose": {},
    "ultralytics": {"start_frame": 10, "end_frame": 200, "frame_step": 2},
    "infer-pose": _INFER_WINDOW,
    "infer-points": _INFER_WINDOW,
    "infer-localizer": _INFER_WINDOW,
}
"""Each op's window fields, with a value that differs from the field's default."""

_INFER_WINDOW_DEFAULTS: dict[str, JsonValue] = {
    "start_frame": 0,
    "end_frame": None,
    "frame_step": 1,
    "max_frames": None,
}

WINDOW_DEFAULTS: dict[str, dict[str, JsonValue]] = {
    "trex": {"analysis_range": None},
    "sleap": {"analysis_range": None},
    "litpose": {},
    "ultralytics": {"start_frame": 0, "end_frame": None, "frame_step": 1},
    "infer-pose": _INFER_WINDOW_DEFAULTS,
    "infer-points": _INFER_WINDOW_DEFAULTS,
    "infer-localizer": _INFER_WINDOW_DEFAULTS,
}
"""The same fields restated at their defaults, which counts as unset."""

_TREX_FRAME_SETTINGS = (
    "analysis_range",
    "analysis_stop_after",
    "gui_stop_after",
    "video_conversion_range",
)

EXTRA_SETTINGS_WINDOWS: dict[str, dict[str, tuple[str, ...]]] = {
    "trex": {
        "convert_extra_settings": _TREX_FRAME_SETTINGS,
        "track_extra_settings": _TREX_FRAME_SETTINGS,
    },
    "sleap": {"sleap_extra_settings": ("frames",)},
}
"""The tool settings that select frames, per pass-through dictionary."""

SETTING_VALUES: dict[str, JsonValue] = {
    "analysis_range": [10, 200],
    "analysis_stop_after": 200,
    "gui_stop_after": 200,
    "video_conversion_range": [10, 200],
    "frames": "10-200",
}
"""A value of the shape that each tool setting takes."""

WINDOW_CASES = [
    pytest.param(kind, field, value, id=f"{kind}-{field}")
    for kind, window in WINDOWS.items()
    for field, value in window.items()
]

EXTRA_SETTINGS_CASES = [
    pytest.param(kind, settings_field, key, id=f"{kind}-{settings_field}-{key}")
    for kind, fields in EXTRA_SETTINGS_WINDOWS.items()
    for settings_field, keys in fields.items()
    for key in keys
]

SETTINGS_FIELD_CASES = [
    pytest.param(kind, settings_field, id=f"{kind}-{settings_field}")
    for kind, fields in EXTRA_SETTINGS_WINDOWS.items()
    for settings_field in fields
]

EXPECTED_REFUSAL = (
    "infer-pose: `start_frame` cannot be combined with `media`. `media` names "
    "preprocess.0.1-3f2a9c01d4, which is derived media: a video already cut to "
    "its frame range. Put the range in that variant with a `trim` or "
    "`decimate` step, or leave `media` empty to read the original recording from "
    "`media_raw` with this frame range."
)


def _params_type(kind: str) -> type[MediaInputParams]:
    """Return the registered params model of *kind*, which must take ``media``."""
    declared = OPS[kind].Params
    assert issubclass(declared, MediaInputParams)
    return declared


def _params(kind: str, **values: JsonValue) -> MediaInputParams:
    """Validate *kind*'s params from its required values plus *values*."""
    return _params_type(kind).model_validate({**CONSUMERS[kind], **values})


def _refusal(kind: str, **values: JsonValue) -> str:
    """Return the one validation message that *values* are refused with."""
    with pytest.raises(ValidationError) as raised:
        _ = _params(kind, **values)
    errors = raised.value.errors()
    assert len(errors) == 1, errors
    return errors[0]["msg"].removeprefix("Value error, ")


# --- the ops that take media, and their declarations --------------------------


def test_the_seven_consumers_of_media_take_the_parameter() -> None:
    """Only the trackers and the inference ops take ``media``."""
    taking = {
        kind for kind, op in OPS.items() if issubclass(op.Params, MediaInputParams)
    }

    assert taking == set(CONSUMERS)


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_a_refusal_names_the_op_that_refused(kind: str) -> None:
    """Each params model names its op kind instead of the base's empty kind."""
    assert _params_type(kind).op_kind == OPS[kind].kind


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_each_consumer_declares_every_window_it_takes(kind: str) -> None:
    """Each op declares every window field, because an undeclared one is not refused."""
    declared = _params_type(kind)

    assert set(declared.window_fields) == set(WINDOWS[kind])
    assert {
        field: set(keys) for field, keys in declared.extra_settings_window_keys.items()
    } == {
        field: set(keys) for field, keys in EXTRA_SETTINGS_WINDOWS.get(kind, {}).items()
    }


# --- the refusal --------------------------------------------------------------


def test_the_refusal_says_what_derived_media_is_and_where_the_range_goes() -> None:
    """The message names the variant, the reason for the refusal, and both remedies."""
    assert _refusal("infer-pose", media=VARIANT, start_frame=10) == EXPECTED_REFUSAL


@pytest.mark.parametrize(("kind", "field", "value"), WINDOW_CASES)
def test_a_window_is_refused_on_derived_media(
    kind: str, field: str, value: JsonValue
) -> None:
    message = _refusal(kind, media=VARIANT, **{field: value})

    assert message.startswith(f"{kind}: `{field}` cannot be combined with `media`")
    assert "derived media" in message
    assert "`media_raw`" in message
    assert VARIANT in message


@pytest.mark.parametrize(("kind", "field", "value"), WINDOW_CASES)
def test_the_same_window_without_media_is_kept(
    kind: str, field: str, value: JsonValue
) -> None:
    params = _params(kind, **{field: value})

    assert field in params.model_fields_set
    assert params.media == ""


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_media_with_every_window_restated_at_its_default_is_accepted(
    kind: str,
) -> None:
    """A named field counts as set only when it differs from its default."""
    params = _params(kind, media=VARIANT, **WINDOW_DEFAULTS[kind])

    assert params.media == VARIANT


def test_every_window_set_is_named_in_one_refusal() -> None:
    message = _refusal("ultralytics", media=VARIANT, start_frame=10, frame_step=2)

    assert message.startswith(
        "ultralytics: `start_frame`, `frame_step` cannot be combined with `media`"
    )


@pytest.mark.parametrize(("kind", "settings_field", "key"), EXTRA_SETTINGS_CASES)
def test_a_frame_setting_passed_through_is_refused_on_derived_media(
    kind: str, settings_field: str, key: str
) -> None:
    """The same range is refused when sent to the tool around the typed field."""
    settings: dict[str, JsonValue] = {key: SETTING_VALUES[key]}
    message = _refusal(kind, media=VARIANT, **{settings_field: settings})

    assert message.startswith(
        f"{kind}: `{key}` in `{settings_field}` cannot be combined with `media`"
    )
    assert "derived media" in message
    assert "`media_raw`" in message


@pytest.mark.parametrize(("kind", "settings_field", "key"), EXTRA_SETTINGS_CASES)
def test_a_frame_setting_left_null_is_accepted_beside_media(
    kind: str, settings_field: str, key: str
) -> None:
    """A null does not set a frame window, like a field restated at its default."""
    settings: dict[str, JsonValue] = {key: None}
    params = _params(kind, media=VARIANT, **{settings_field: settings})

    assert params.media == VARIANT


@pytest.mark.parametrize(("kind", "settings_field", "key"), EXTRA_SETTINGS_CASES)
def test_the_same_setting_without_media_is_kept(
    kind: str, settings_field: str, key: str
) -> None:
    settings: dict[str, JsonValue] = {key: SETTING_VALUES[key]}
    params = _params(kind, **{settings_field: settings})

    assert settings_field in params.model_fields_set


@pytest.mark.parametrize(("kind", "settings_field"), SETTINGS_FIELD_CASES)
def test_other_settings_pass_through_beside_media(
    kind: str, settings_field: str
) -> None:
    """Only the settings that select frames are refused."""
    settings: dict[str, JsonValue] = {"output_format": "npz"}
    params = _params(kind, media=VARIANT, **{settings_field: settings})

    assert params.media == VARIANT


# --- the frame window a run reads ---------------------------------------------


@pytest.mark.parametrize(("kind", "field", "value"), WINDOW_CASES)
def test_a_window_field_narrows_the_frames_read(
    kind: str, field: str, value: JsonValue
) -> None:
    assert _params(kind, **{field: value}).frame_window == (f"`{field}`",)


@pytest.mark.parametrize(("kind", "settings_field", "key"), EXTRA_SETTINGS_CASES)
def test_a_frame_setting_passed_through_narrows_the_frames_read(
    kind: str, settings_field: str, key: str
) -> None:
    settings: dict[str, JsonValue] = {key: SETTING_VALUES[key]}
    params = _params(kind, **{settings_field: settings})

    assert params.frame_window == (f"`{key}` in `{settings_field}`",)


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_every_window_at_its_default_reads_every_frame(kind: str) -> None:
    assert _params(kind, **WINDOW_DEFAULTS[kind]).frame_window == ()


# --- the identity term --------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
@pytest.mark.parametrize(
    ("media", "terms"), [("", {}), (VARIANT, {"media": VARIANT})], ids=["empty", "set"]
)
def test_media_reaches_identity_only_when_set(
    kind: str, media: str, terms: dict[str, str]
) -> None:
    """An empty ``media`` does not add a term, and older identifiers stay unchanged."""
    assert media_identity_terms(_params(kind, media=media)) == terms


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_media_is_left_out_of_the_hashed_params(kind: str) -> None:
    """``identity_dump`` omits the ``media`` key, set or not."""
    assert "media" not in _params(kind, media=VARIANT).identity_dump()


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_setting_media_moves_the_run_and_its_tracks_variant(
    kind: str, tmp_path: Path
) -> None:
    """The identity is planned through the op, as both the graph and a run plan it."""
    dataset = make_dataset(tmp_path / "planned")
    op = OPS[kind]

    def identity(**values: JsonValue) -> OpIdentity:
        return op().plan_identity(dataset, _params(kind, **values), ResolvedScope())

    original = identity()
    derived = identity(media=VARIANT)

    assert derived.run_id != original.run_id
    assert derived.tracks_variant != original.tracks_variant
    assert identity(media=OTHER_VARIANT) != derived


# --- recipes ------------------------------------------------------------------

_PREPROCESS_STEP: dict[str, JsonValue] = {
    "id": "prep",
    "type": "op",
    "kind": "preprocess",
    "params": {
        "steps": [{"step": "crop", "x": 0, "y": 0, "width": 320, "height": 240}]
    },
}


def _recipe(kind: str, params: dict[str, JsonValue]) -> Recipe:
    """Return a recipe of a preprocess step and a *kind* step run with *params*."""
    consumer: dict[str, JsonValue] = {
        "id": "consume",
        "type": "op",
        "kind": kind,
        "params": {**CONSUMERS[kind], **params},
    }
    return Recipe.model_validate({"steps": [_PREPROCESS_STEP, consumer]})


_MEDIA_SPELLINGS = [
    pytest.param(VARIANT, id="run-id"),
    pytest.param({"step": "prep"}, id="step-reference"),
]


@pytest.mark.parametrize("media", _MEDIA_SPELLINGS)
@pytest.mark.parametrize(("kind", "field", "value"), WINDOW_CASES)
def test_a_recipe_step_combining_a_window_with_media_is_refused(
    kind: str, field: str, value: JsonValue, media: JsonValue
) -> None:
    """A reference validates with a stand-in identifier, which is not empty."""
    problems = check_recipe(_recipe(kind, {"media": media, field: value}))

    assert [problem.step for problem in problems] == ["consume"]
    assert f"`{field}` cannot be combined with `media`" in problems[0].message


@pytest.mark.parametrize(("kind", "field", "value"), WINDOW_CASES)
def test_a_recipe_refusal_never_names_the_stand_in_identifier(
    kind: str, field: str, value: JsonValue
) -> None:
    """A step reference is checked with a stand-in, which is not a variant's name."""
    recipe = _recipe(kind, {"media": {"step": "prep"}, field: value})

    (problem,) = check_recipe(recipe)

    assert "0000000000" not in problem.message
    expected = "`media` names derived media (the output of a preprocess step): "
    assert expected in problem.message


@pytest.mark.parametrize("kind", sorted(CONSUMERS))
def test_a_recipe_step_naming_a_variant_by_reference_is_accepted(kind: str) -> None:
    assert check_recipe(_recipe(kind, {"media": {"step": "prep"}})) == ()


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    [
        pytest.param(kind, *next(iter(window.items())), id=kind)
        for kind, window in WINDOWS.items()
        if window
    ],
)
def test_a_plan_is_refused_before_the_dataset_is_read(
    kind: str, field: str, value: JsonValue, tmp_path: Path
) -> None:
    recipe = _recipe(kind, {"media": {"step": "prep"}, field: value})

    with pytest.raises(RecipeInvalid) as raised:
        _ = plan_pipeline(make_dataset(tmp_path / "bare"), recipe)

    assert "derived media" in str(raised.value)
