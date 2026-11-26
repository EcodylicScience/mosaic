"""A built-in track format name is one that a conversion's run id can carry.

A conversion's variant is ``convert-<format>.<version>-<digest>``, and
``parse_op_run_id`` reads the kind up to the version's dot. A format outside
the charset below is a kind that the parser cannot return. A built-in contains
no ``__`` either, because that is the separator of the namespace an installed
converter's format carries, so a later built-in cannot take an installed
format's name.
"""

from __future__ import annotations

import re

import pytest

from mosaic.core.pipeline.op_identity import op_run_id, parse_op_run_id
from mosaic.core.pipeline.tracks_identity import converter_op
from mosaic.core.track_converter import builtin_track_converters

FORMAT_NAME = re.compile(r"[a-z0-9]+(_[a-z0-9]+)*")
BUILT_IN_FORMATS = sorted(builtin_track_converters())


@pytest.mark.parametrize("src_format", BUILT_IN_FORMATS)
def test_a_built_in_format_is_lowercase_words_joined_by_single_underscores(
    src_format: str,
) -> None:
    assert FORMAT_NAME.fullmatch(src_format)
    assert "__" not in src_format


@pytest.mark.parametrize("src_format", BUILT_IN_FORMATS)
def test_a_conversion_run_id_parses_back_to_its_kind_and_version(
    src_format: str,
) -> None:
    version = builtin_track_converters()[src_format].version

    parsed = parse_op_run_id(op_run_id(converter_op(src_format), version, {}))

    assert parsed is not None
    assert parsed.kind == converter_op(src_format)
    assert parsed.version == version
