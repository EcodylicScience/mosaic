"""``convert-tracks`` and ``convert-labels``: a dataset's conversions as recipe steps.

A recipe used to start from tracks and labels that something outside it had
already converted. These two ops make the conversion a step, so one request can
convert a dataset's raw sources and then compute on what it converted.

**Each op names one variant.** Its identity is the variant
:meth:`~mosaic.core.dataset.Dataset.track_conversion_variant` or
:meth:`~mosaic.core.dataset.Dataset.label_conversion_variant` mints from the
converter and its params, and that minting reads no index, so a plan names the
variant before the conversion runs. A downstream ``tracks`` or ``labels``
reference then pins it.

**A conversion reads every raw row of its format.** Neither op takes a scope.
What a dataset converts is decided by its scan sources, which stay with the
user, and the plan refuses a format that ``tracks_raw`` or ``labels_raw`` holds
no row of.

**A file that fails is reported and then fails the attempt**, as
``resample-tracks`` does: each failure is an ``entry_error`` on the run-log, and
a rerun converts only what is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import JsonValue, field_validator, model_validator

from mosaic.core.params import Declared, Params
from mosaic.core.pipeline.job import JobContext
from mosaic.core.pipeline.ops import Op, OpIdentity, register_op

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline._utils import ResolvedScope

__all__ = [
    "ConvertLabelsOp",
    "ConvertLabelsParams",
    "ConvertTracksOp",
    "ConvertTracksParams",
]

_LEGACY_KEYS = frozenset({"fps_default"})
"""Keys a converter's overrides accept beyond its params model, as the dataset
reads them."""


def _unknown_keys(
    given: dict[str, JsonValue], fields: frozenset[str], refused: frozenset[str]
) -> list[str]:
    """The keys of *given* that name no field, sorted, plus those always refused."""
    return sorted(
        key for key in given if key in refused or key not in fields | _LEGACY_KEYS
    )


_SOURCE_FORMAT = (
    "The registered converter's src_format, such as 'calms21_npy'. One step "
    "converts one format and names one variant."
)
_TRACK_PARAMS = (
    "The converter's own params, such as fps. A key the converter does not "
    "declare is refused."
)
_MERGE = "Merge a sequence's several files into one table. Unset asks the converter."
_GROUP_FROM = (
    "Which group a converted entry is filed under: 'filename' (the scan's "
    "file-level group, the default), 'infile' or 'both'."
)
_LABEL_KIND = "The label kind to convert into, such as 'behavior'."
_LABEL_PARAMS = (
    "The label converter's own params, such as fps or group_from. A key the "
    "converter does not declare is refused."
)


class ConvertTracksParams(Params):
    """Parameters for ``convert-tracks``.

    ``strict_schema`` travels inside ``params``, where the converter declares it
    and leaves it out of the variant's identity.
    """

    source_format: Annotated[str, Declared(_SOURCE_FORMAT)]
    params: Annotated[dict[str, JsonValue], Declared(_TRACK_PARAMS)] = {}
    merge_per_sequence: Annotated[bool | None, Declared(_MERGE)] = None
    group_from: Annotated[
        Literal["filename", "infile", "both"] | None, Declared(_GROUP_FROM)
    ] = None

    @model_validator(mode="after")
    def _converter_keys(self) -> ConvertTracksParams:
        from mosaic.core.track_converter import get_track_converter

        converter = get_track_converter(self.source_format)
        fields = frozenset(type(converter).Params.model_fields)
        unknown = _unknown_keys(
            self.params, fields, frozenset({"params_by_format", "group", "sequence"})
        )
        if unknown:
            msg = (
                f"params names {unknown}, which the {self.source_format!r} "
                f"converter does not declare. It declares {sorted(fields)}."
            )
            raise ValueError(msg)
        return self


@register_op
class ConvertTracksOp(Op[ConvertTracksParams]):
    """Convert one format's raw tracks into a tracks variant."""

    kind = "convert-tracks"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    writes_tracks = True
    scope_takes = "none"
    scope_dependent = False
    Params = ConvertTracksParams

    def target(self, params: ConvertTracksParams, scope: ResolvedScope) -> str:
        return params.source_format

    def plan_identity(
        self,
        ds: Dataset,
        params: ConvertTracksParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """The variant this conversion writes, minted from its params alone."""
        variant = ds.track_conversion_variant(params.source_format, params.params)
        return OpIdentity(run_id=variant, tracks_variant=variant)

    def run(
        self,
        ds: Dataset,
        params: ConvertTracksParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        identity = self.plan_identity(ds, params, scope)
        ctx.set_run_id(identity.run_id)

        def progress(done: int, total: int) -> None:
            if total != ctx.total:
                ctx.set_total(total)
            ctx.heartbeat(done)

        outcome = ds.convert_all_tracks(
            params=dict(params.params),
            overwrite=overwrite,
            merge_per_sequence=params.merge_per_sequence,
            group_from=params.group_from,
            source_format=params.source_format,
            progress=progress,
            cancel_check=ctx.check_cancel,
            on_failure=ctx.entry_failed,
        )
        # No ``tracks_variant`` event: it records the variants an attempt read,
        # and a conversion reads raw files. The variant it wrote is its run id.
        ctx.entries_written(max(outcome.converted - outcome.failed, 0))
        if outcome.failed:
            # A missing entry here is missing from everything downstream, and a
            # rerun skips what was written, so the attempt fails and resumes.
            msg = (
                f"convert-tracks failed on {outcome.failed} raw file(s) of format "
                f"{params.source_format!r}; variant {identity.tracks_variant} is "
                "incomplete. The run-log names each file. Fix them and re-run "
                "to resume."
            )
            raise RuntimeError(msg)
        return identity.run_id


class ConvertLabelsParams(Params):
    """Parameters for ``convert-labels``."""

    kind: Annotated[str, Declared(_LABEL_KIND)] = "behavior"
    source_format: Annotated[str, Declared(_SOURCE_FORMAT)]
    params: Annotated[dict[str, JsonValue], Declared(_LABEL_PARAMS)] = {}

    @field_validator("kind")
    @classmethod
    def _lower_kind(cls, value: str) -> str:
        """The kind as the label root spells it, which is lower case."""
        return value.lower()

    @model_validator(mode="after")
    def _converter_keys(self) -> ConvertLabelsParams:
        from mosaic.core.label_converter import get_label_converter

        converter = get_label_converter(self.source_format, self.kind)
        fields = frozenset(type(converter).Params.model_fields)
        unknown = _unknown_keys(
            self.params,
            fields,
            frozenset({"params_by_format", "group", "sequence", "source_format"}),
        )
        if unknown:
            msg = (
                f"params names {unknown}, which the {self.source_format!r} "
                f"label converter does not declare. It declares {sorted(fields)}."
            )
            raise ValueError(msg)
        return self


@register_op
class ConvertLabelsOp(Op[ConvertLabelsParams]):
    """Convert one format's raw labels into a labels variant of one kind."""

    kind = "convert-labels"
    category = "convert"
    domain = "tracking"
    version = "0.1"
    writes_tracks = False
    writes_labels = True
    scope_takes = "none"
    scope_dependent = False
    Params = ConvertLabelsParams

    def target(self, params: ConvertLabelsParams, scope: ResolvedScope) -> str:
        return f"{params.kind}/{params.source_format}"

    def plan_identity(
        self,
        ds: Dataset,
        params: ConvertLabelsParams,
        scope: ResolvedScope,
        *,
        require_data: bool = True,
    ) -> OpIdentity:
        """The labels variant this conversion writes, minted from its params."""
        variant = ds.label_conversion_variant(
            params.kind, params.source_format, params.params
        )
        return OpIdentity(run_id=variant, labels_variant=variant)

    def run(
        self,
        ds: Dataset,
        params: ConvertLabelsParams,
        scope: ResolvedScope,
        overwrite: bool,
        ctx: JobContext,
    ) -> str:
        identity = self.plan_identity(ds, params, scope)
        ctx.set_run_id(identity.run_id)

        def progress(done: int, total: int) -> None:
            if total != ctx.total:
                ctx.set_total(total)
            ctx.heartbeat(done)

        outcome = ds.convert_all_labels(
            kind=params.kind,
            overwrite=overwrite,
            params=dict(params.params),
            source_format=params.source_format,
            progress=progress,
            cancel_check=ctx.check_cancel,
        )
        ctx.entries_written(outcome.written + outcome.skipped)
        return identity.run_id
