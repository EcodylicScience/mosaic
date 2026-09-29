"""Declare the ``media`` parameter that every consumer of media shares.

A tracker or an inference op reads an entry's original recording from
``media_raw`` unless ``media`` names a media variant, the run identifier of a
``preprocess`` run. Then it reads that derived video instead, and its tracks are
mapped back into the entry's own pixels and frames.

A variant is already cut to a frame range by its ``trim`` and ``decimate`` steps,
while every frame number that a user types on a consumer is a source frame. A tool
applies a consumer's frame window to the frames that it reads, which for a variant
are variant frames. :class:`MediaInputParams` therefore refuses the combination.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

from pydantic import model_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.params import HASH_EXCLUDE, Declared, Params
from mosaic.core.pipeline.op_identity import parse_op_run_id
from mosaic.core.pipeline.preprocess_layout import PREPROCESS_KIND

__all__ = ["MediaInputParams", "media_identity_terms"]

_MEDIA_DESCRIPTION = (
    "The run id of a preprocess media variant to read in place of the entry's "
    "original media. The tool reads that derived video, and its tracks are "
    "published in the entry's own pixels and frames. Empty reads the original "
    "recording from media_raw. A frame window cannot be combined with it. The "
    "variant's trim and decimate steps set the range."
)


class MediaInputParams(Params):
    """Declare ``media`` for every op that reads an entry's media, as a mixin.

    ``media`` is ``HASH_EXCLUDE``, and ``identity_dump()`` omits it. Each consumer
    adds :func:`media_identity_terms` to the payload that it mints from. The term
    is absent when ``media`` is empty. Every identifier minted before the field
    existed is unchanged.

    A subclass declares the fields and tool settings that select frames, and the
    op kind that a refusal names.
    """

    media: Annotated[str, HASH_EXCLUDE, Declared(_MEDIA_DESCRIPTION)] = ""

    window_fields: ClassVar[tuple[str, ...]] = ()
    """The fields that select the frames that the op reads."""

    extra_settings_window_keys: ClassVar[dict[str, tuple[str, ...]]] = {}
    """Per pass-through settings field, the tool settings that select frames."""

    op_kind: ClassVar[str] = ""
    """The op these parameters belong to, named in a refusal."""

    @model_validator(mode="after")
    def _refuse_a_frame_window_on_derived_media(self) -> Self:
        """Refuse a frame window when ``media`` names a variant.

        A field counts as set when it differs from its default. Restating a
        default is not refused. A settings key counts as set when its value is not
        null. The tools read null as the setting left unset.

        The refusal names the variant only when ``media`` is a preprocess run
        identifier. A recipe step that refers to another step's variant is checked
        with a stand-in identifier in place of a preprocess run id. The refusal
        then describes ``media`` as the output of a preprocess step.
        """
        if not self.media:
            return self
        fields = type(self).model_fields
        offending: list[str] = []
        for name in self.window_fields:
            value: object = getattr(self, name)
            default: object = fields[name].get_default(call_default_factory=True)
            if value != default:
                offending.append(f"`{name}`")
        for settings_field, keys in self.extra_settings_window_keys.items():
            settings: JsonValue = getattr(self, settings_field)
            if not isinstance(settings, dict):
                continue
            offending.extend(
                f"`{key}` in `{settings_field}`"
                for key in keys
                if settings.get(key) is not None
            )
        if not offending:
            return self
        parsed = parse_op_run_id(self.media)
        named = (
            f"{self.media}, which is derived media"
            if parsed is not None and parsed.kind == PREPROCESS_KIND
            else f"derived media (the output of a {PREPROCESS_KIND} step)"
        )
        refused = (
            f"{self.op_kind}: {', '.join(offending)} cannot be combined with "
            f"`media`. `media` names {named}: a video already cut to its frame "
            f"range. Put the range in that variant with a `trim` or `decimate` "
            f"step, or leave `media` empty to read the original recording from "
            f"`media_raw` with this frame range."
        )
        raise ValueError(refused)


def media_identity_terms(params: MediaInputParams) -> dict[str, str]:
    """Return the identity term that ``media`` contributes to a consumer's run id.

    Args:
        params: A consumer's parameters.

    Returns:
        ``{"media": params.media}`` when a variant is named, and an empty dict
        when ``media`` is empty.
    """
    return {"media": params.media} if params.media else {}
