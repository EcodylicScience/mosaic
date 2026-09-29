"""Test the pre-processing op's parameters and the run identifier that they mint.

A variant's identity is a function of its parameters alone: the steps in order,
each with its version, the upstream variant, the labeled rate when one is set,
the codec and the quality that the encode resolves to. A permission that leaves
the encoded output unchanged stays out of it.
"""

from __future__ import annotations

import math
import re

import pytest
from pydantic import ValidationError

from mosaic.core.media.preprocess import (
    ClaheStep,
    CropStep,
    MaskStep,
    MediaStepSpec,
    TrimStep,
)
from mosaic.core.pipeline.preprocess_layout import PREPROCESS_KIND
from mosaic.core.pipeline.preprocess import (
    AV1_DEFAULT_QUALITY,
    H264_DEFAULT_CRF,
    PREPROCESS_VERSION,
    PreprocessParams,
    VariantCodec,
    preprocess_identity,
    preprocess_identity_payload,
    resolved_quality,
)

_CROP = CropStep(x=120, y=40, width=320, height=240)
_TRIM = TrimStep(start=100, stop=400)
_CLAHE = ClaheStep()


def _run_id(steps: list[MediaStepSpec], **fields: object) -> str:
    params = PreprocessParams.model_validate({"steps": steps, **fields})
    return preprocess_identity(params).run_id


# --- refusals ----------------------------------------------------------------


def test_an_empty_step_list_is_refused_naming_the_entry_media() -> None:
    with pytest.raises(ValidationError, match="Leave `media` empty"):
        _ = PreprocessParams(steps=[])


@pytest.mark.parametrize("fps", [0.0, -30.0, math.nan, math.inf])
def test_a_rate_that_is_not_a_positive_finite_number_is_refused(fps: float) -> None:
    with pytest.raises(ValidationError, match="fps"):
        _ = PreprocessParams(steps=[_CROP], fps=fps)


@pytest.mark.parametrize(("codec", "quality"), [("av1", 64), ("h264", 52)])
def test_a_quality_above_the_codec_scale_is_refused(
    codec: VariantCodec, quality: int
) -> None:
    with pytest.raises(ValidationError, match=codec):
        _ = PreprocessParams(steps=[_CROP], codec=codec, quality=quality)


@pytest.mark.parametrize("codec", ["av1", "h264"])
def test_a_negative_quality_is_refused(codec: VariantCodec) -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        _ = PreprocessParams(steps=[_CROP], codec=codec, quality=-1)


def test_the_schema_publishes_the_lowest_quality() -> None:
    """Both scales start at 0, and an editor can bound the control from them."""
    quality = PreprocessParams.model_json_schema()["properties"]["quality"]

    assert {"type": "integer", "minimum": 0} in quality["anyOf"]


@pytest.mark.parametrize(
    ("codec", "quality"),
    [("av1", 0), ("av1", 63), ("h264", 0), ("h264", 51)],
)
def test_a_quality_at_either_end_of_the_codec_scale_is_accepted(
    codec: VariantCodec, quality: int
) -> None:
    params = PreprocessParams(steps=[_CROP], codec=codec, quality=quality)

    assert resolved_quality(params) == quality


def test_an_unknown_step_is_refused() -> None:
    with pytest.raises(ValidationError, match="blur"):
        _ = PreprocessParams.model_validate({"steps": [{"step": "blur"}]})


def test_the_upstream_variant_is_not_checked_here() -> None:
    """Recipe validation puts a feature-shaped placeholder where a reference goes."""
    params = PreprocessParams(steps=[_CROP], media="0.0-0000000000")

    assert params.media == "0.0-0000000000"


# --- validation --------------------------------------------------------------


def test_a_json_step_list_validates_into_step_objects() -> None:
    params = PreprocessParams.model_validate_json(
        '{"steps": [{"step": "crop", "x": 120, "y": 40, "width": 320, '
        '"height": 240}, {"step": "trim", "start": 100, "stop": 400}]}'
    )

    assert params.steps == [_CROP, _TRIM]
    assert isinstance(params.steps[0], CropStep)
    assert isinstance(params.steps[1], TrimStep)


# --- identity ----------------------------------------------------------------


def test_the_run_id_is_the_op_kind_and_version_over_a_digest() -> None:
    run_id = _run_id([_CROP])

    assert (PREPROCESS_KIND, PREPROCESS_VERSION) == ("preprocess", "0.1")
    assert re.fullmatch(r"preprocess\.0\.1-[0-9a-f]{10}", run_id)


def test_every_field_identity_keeps_reaches_the_payload() -> None:
    """The payload is built by hand, and a new field must be added to it."""
    params = PreprocessParams(steps=[_CROP], fps=15.0)

    assert set(preprocess_identity_payload(params)) == set(params.identity_dump())


def test_reordering_two_steps_moves_the_run_id() -> None:
    assert _run_id([_CROP, _TRIM]) != _run_id([_TRIM, _CROP])


def test_allowing_hardware_leaves_the_run_id() -> None:
    assert _run_id([_CROP], allow_hardware=True) == _run_id([_CROP])


def test_a_step_version_is_in_the_payload_and_moves_the_run_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _run_id([_CROP, _TRIM])
    payload = preprocess_identity_payload(PreprocessParams(steps=[_CROP, _TRIM]))

    monkeypatch.setattr(CropStep, "version", "0.2")

    assert payload["steps"] == [
        {
            "step": "crop",
            "version": "0.1",
            "x": 120,
            "y": 40,
            "width": 320,
            "height": 240,
        },
        {"step": "trim", "version": "0.1", "start": 100, "stop": 400},
    ]
    assert _run_id([_CROP, _TRIM]) != before


def test_a_mask_polygon_enters_the_payload_as_json() -> None:
    mask = MaskStep(polygon=[(0, 0), (10, 0), (10, 10)])

    payload = preprocess_identity_payload(PreprocessParams(steps=[mask]))

    assert payload["steps"] == [
        {
            "step": "mask",
            "version": "0.1",
            "polygon": [[0, 0], [10, 0], [10, 10]],
            "keep": True,
        }
    ]


@pytest.mark.parametrize(
    ("codec", "default"), [("av1", AV1_DEFAULT_QUALITY), ("h264", H264_DEFAULT_CRF)]
)
def test_the_resolved_quality_is_in_the_payload(
    codec: VariantCodec, default: int
) -> None:
    unset = PreprocessParams(steps=[_CROP], codec=codec)
    explicit = PreprocessParams(steps=[_CROP], codec=codec, quality=default)

    assert preprocess_identity_payload(unset)["quality"] == default
    assert preprocess_identity(unset) == preprocess_identity(explicit)


def test_the_defaults_are_mosaic_constants() -> None:
    assert (AV1_DEFAULT_QUALITY, H264_DEFAULT_CRF) == (14, 16)


def test_an_unset_rate_adds_no_payload_key() -> None:
    unset = preprocess_identity_payload(PreprocessParams(steps=[_CROP]))
    labeled = preprocess_identity_payload(PreprocessParams(steps=[_CROP], fps=15.0))

    assert "fps" not in unset
    assert labeled["fps"] == 15.0
    assert _run_id([_CROP], fps=15) == _run_id([_CROP], fps=15.0)
    assert _run_id([_CROP], fps=15.0) != _run_id([_CROP])


def test_an_upstream_variant_moves_the_run_id() -> None:
    upstream = _run_id([_CROP])

    assert _run_id([_CLAHE], media=upstream) != _run_id([_CLAHE])


def test_the_codec_moves_the_run_id() -> None:
    assert _run_id([_CROP], codec="h264") != _run_id([_CROP])
