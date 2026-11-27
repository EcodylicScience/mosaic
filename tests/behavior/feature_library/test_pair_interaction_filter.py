"""``pair-interaction-filter`` reads the position columns its params name.

``trex_npz`` tracks keep TRex's body centroid in ``X``/``Y`` and its head point in
``X#head``/``Y#head``. ``position_columns`` names the pair the distance check reads,
so the head point can be the position without a pixel column or a keypoint.
"""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from mosaic.behavior.feature_library.pair_interaction_filter import (
    PairInteractionFilter,
)
from mosaic.core.json_value import JsonValue
from mosaic.core.pipeline.column_contract import (
    MissingInputColumns,
    check_input_columns,
)
from mosaic.core.pipeline.graph import build_feature
from tests.helpers import run_id_digest

_FRAMES = 40
_HEAD: tuple[str, str] = ("X#head", "Y#head")

_PARAMS: dict[str, object] = {
    "min_run_frames": 5,
    "morphological_structure_size": 0,
    "frame_padding": 0,
    "shift_dist": 0.0,
    "require_facing": False,
}


def _tracks() -> pd.DataFrame:
    """Two animals whose centroids are far apart and whose head points are close.

    ``x_pixels`` places them close too, so a run that read it would find the
    interaction the centroid and the head columns disagree about.
    """
    rows: list[dict[str, object]] = []
    for frame in range(_FRAMES):
        for animal, (centroid, head) in enumerate(((0.0, 0.0), (300.0, 20.0))):
            rows.append(
                {
                    "frame": frame,
                    "time": frame / 30.0,
                    "id": animal,
                    "X": centroid,
                    "Y": 0.0,
                    "X#head": head,
                    "Y#head": 0.0,
                    "x_pixels": head / 2.0,
                    "y_pixels": 0.0,
                    "ANGLE": 0.0,
                    "sequence": "s1",
                    "group": "g1",
                    "fps": 30.0,
                }
            )
    return pd.DataFrame(rows)


def _frames(params: dict[str, object], table: pd.DataFrame | None = None) -> int:
    feature = PairInteractionFilter(params={**_PARAMS, **params})
    out = feature.apply(_tracks() if table is None else table)
    return int(out["frame"].nunique()) if len(out) else 0


def test_the_named_columns_are_the_position() -> None:
    assert _frames({"position_columns": _HEAD}) == _FRAMES


def test_the_named_columns_are_read_instead_of_the_centroid_and_the_pixels() -> None:
    assert _frames({"use_pixel_coords": False}) == 0
    assert _frames({}) == _FRAMES
    assert _frames({"position_columns": ("X", "Y")}) == 0


def test_a_missing_named_column_raises_without_a_fallback() -> None:
    table = _tracks().drop(columns=["Y#head"])
    with pytest.raises(ValueError, match="Y#head"):
        _frames({"position_columns": _HEAD}, table)


def test_a_named_column_that_is_absent_does_not_fall_back_to_the_centroid() -> None:
    table = _tracks().drop(columns=["X#head", "Y#head"])
    with pytest.raises(ValueError, match="X#head"):
        _frames({"position_columns": _HEAD}, table)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"pose_head_index": 2}, "pose_head_index"),
        ({"use_pixel_coords": False}, "use_pixel_coords"),
        ({"position_columns": ("X#head", "X#head")}, "twice"),
        ({"position_columns": ("X#head",)}, "Field required"),
        ({"position_columns": ("X#head", "Y#head", "Z#head")}, "at most 2 items"),
        ({"position_columns": ("", "Y#head")}, "empty"),
    ],
    ids=["pose-head", "centroid", "same-column", "one", "three", "empty-name"],
)
def test_a_named_pair_refuses_what_it_contradicts(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        PairInteractionFilter.Params.from_overrides(
            {"position_columns": _HEAD, **overrides}
        )


def test_an_unset_pair_leaves_the_identity() -> None:
    params = PairInteractionFilter.Params
    assert "position_columns" not in params().identity_dump()
    assert "position_columns" not in params(position_columns=None).identity_dump()
    assert run_id_digest(params()) == run_id_digest(params(position_columns=None))
    named = params(position_columns=_HEAD)
    assert named.identity_dump()["position_columns"] == _HEAD
    assert run_id_digest(named) != run_id_digest(params())
    assert run_id_digest(named) != run_id_digest(params(position_columns=("X", "Y")))


def test_a_json_list_names_the_pair() -> None:
    params = PairInteractionFilter.Params.from_overrides(
        {"position_columns": ["X#head", "Y#head"]}
    )
    assert params.position_columns == _HEAD


def _lacks(params: dict[str, JsonValue], present: list[str]) -> None:
    check_input_columns(
        build_feature("pair-interaction-filter", ["tracks"], params), present
    )


_BASE = ["frame", "time", "id", "group", "sequence", "ANGLE"]


def test_the_declared_inputs_are_the_named_columns_alone() -> None:
    check_input_columns(
        build_feature(
            "pair-interaction-filter",
            ["tracks"],
            {"position_columns": ["X#head", "Y#head"]},
        ),
        [*_BASE, "X#head", "Y#head"],
    )


@pytest.mark.parametrize("missing", ["X#head", "Y#head"])
def test_a_table_lacking_a_named_column_is_refused_before_apply(missing: str) -> None:
    present = [*_BASE, "X#head", "Y#head", "X", "Y", "x_pixels", "y_pixels"]
    present.remove(missing)
    with pytest.raises(MissingInputColumns, match=f"lacks '{missing}'"):
        _lacks({"position_columns": ["X#head", "Y#head"]}, present)


def test_the_unset_declaration_is_unchanged() -> None:
    _lacks({}, [*_BASE, "X", "Y"])
    with pytest.raises(MissingInputColumns, match="lacks"):
        _lacks({}, [*_BASE, "X#head", "Y#head"])
