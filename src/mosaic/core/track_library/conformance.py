"""The contract of a track converter, as a list of breaches.

:func:`converter_problems` returns one sentence per way a converter class breaks
the contract, and an empty list when it conforms. The registry tests of mosaic
call it for each built-in, and a package that ships a converter calls it from its
own tests, so the suite and the package check one contract.

Without a sample file it checks that:

- ``version``, ``carries_tracklets`` and ``output_schema`` are each declared by
  the class or a parent below :class:`~mosaic.core.track_converter.TrackConverter`,
  not left to the base class;
- the class declares a ``src_format``, and ``version`` comes back from the run
  identifier that a conversion of that format would mint;
- the registry, when it holds a class under ``src_format``, holds this one (a
  class that nothing registered gets no verdict from this check);
- a format with ``merges_per_sequence`` overrides ``sequence_from_stem``;
- a format is not both ``enumerable`` and ``merges_per_sequence``;
- ``Params`` names none of ``group``, ``sequence`` and ``group_from`` in the
  payload that a tracks variant hashes.

Given a sample file it also converts it with ``EntryHints()`` and checks that
the table passes ``ensure_track_schema(..., strict=True)`` against
``output_schema``. A conversion that raises is a problem that names the
exception.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from mosaic.core.schema import ensure_track_schema
from mosaic.core.track_converter import (
    TRACK_CONVERTERS,
    EntryHints,
    TrackConverter,
    TrackConvertParams,
    ensure_track_converters_registered,
)

__all__ = ["ENTRY_IDENTITY_KEYS", "converter_problems"]

ENTRY_IDENTITY_KEYS: Final = frozenset({"group", "sequence", "group_from"})
"""What entry identity is spelled as. A converter may accept these as hints, and
none may accept them as parameters, because a parameter reaches the tracks hash."""

_DECLARED_ATTRIBUTES: Final = ("version", "carries_tracklets", "output_schema")


def _declares(cls: type[TrackConverter[Any]], attribute: str) -> bool:
    """Whether *cls* or a parent below ``TrackConverter`` sets *attribute*."""
    return any(
        attribute in vars(parent)
        for parent in cls.__mro__
        if parent is not TrackConverter
    )


def converter_problems(
    cls: type[TrackConverter[Any]],
    sample: Path | None = None,
    params: TrackConvertParams | None = None,
) -> list[str]:
    """Return the ways *cls* breaks the track-converter contract.

    Args:
        cls: The converter class to check.
        sample: A raw file of the converter's format. Given one, the converter
            converts it and the table is validated against ``output_schema``.
        params: The conversion params for the sample. Defaults to
            ``cls.Params()``.

    Returns:
        One sentence per breach, empty when *cls* conforms.
    """
    # Imported here because importing a submodule of ``mosaic.core.pipeline``
    # loads the package, and importing the track library loads neither the
    # dataset nor the pipeline (``tests/core/track_library/test_window_converter.py``).
    from mosaic.core.pipeline.op_identity import op_run_id, parse_op_run_id
    from mosaic.core.pipeline.tracks_identity import converter_op

    name = cls.__name__
    problems: list[str] = []

    for attribute in _DECLARED_ATTRIBUTES:
        if not _declares(cls, attribute):
            problems.append(
                f"{name} does not declare {attribute}, so it takes the base "
                "class's value."
            )

    src_format = getattr(cls, "src_format", "")
    if not src_format:
        problems.append(f"{name} does not declare a src_format.")
    else:
        run_id = op_run_id(converter_op(src_format), cls.version, {})
        parsed = parse_op_run_id(run_id)
        if parsed is None or parsed.version != cls.version:
            problems.append(
                f"{name} has version {cls.version!r} and src_format "
                f"{src_format!r}, whose run identifier does not return that "
                "version. A version is digits separated by dots, and a "
                "src_format is lowercase letters, digits, hyphens and "
                "underscores."
            )
        ensure_track_converters_registered()
        registered = TRACK_CONVERTERS.get(src_format)
        if registered is not None and registered is not cls:
            problems.append(
                f"{name} declares src_format {src_format!r}, which the registry "
                f"holds for {registered.__module__}.{registered.__qualname__}."
            )

    if cls.merges_per_sequence and (
        cls.sequence_from_stem is TrackConverter.sequence_from_stem
    ):
        problems.append(
            f"{name} declares merges_per_sequence without overriding "
            "sequence_from_stem, so each of its files would be its own sequence."
        )
    if cls.enumerable and cls.merges_per_sequence:
        problems.append(
            f"{name} declares both enumerable and merges_per_sequence, which "
            "claim that one file holds many sequences and that many files hold "
            "one."
        )

    hashed = sorted(
        set(cls.Params.model_construct().identity_dump()) & ENTRY_IDENTITY_KEYS
    )
    if hashed:
        problems.append(
            f"{name}'s Params would hash entry identity: {', '.join(hashed)}. "
            "Entry identity travels as EntryHints, which is never hashed."
        )

    if sample is not None:
        problems.extend(_sample_problems(cls, sample, params))
    return problems


def _sample_problems(
    cls: type[TrackConverter[Any]], sample: Path, params: TrackConvertParams | None
) -> list[str]:
    """Convert *sample* and return what is wrong with the table."""
    name = cls.__name__
    try:
        table = cls().convert(sample, params or cls.Params(), EntryHints())
    except Exception as exc:
        return [f"{name} raised {type(exc).__name__} converting {sample.name}: {exc}"]
    try:
        _ = ensure_track_schema(
            table, cls.output_schema, strict=True, source=sample.name
        )
    except Exception as exc:
        return [
            f"{name} converted {sample.name} to a table that fails the "
            f"{cls.output_schema} schema: {exc}"
        ]
    return []
