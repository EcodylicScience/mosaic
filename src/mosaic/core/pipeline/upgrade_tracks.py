"""Bringing already-converted TRex tables to pixels, without their raw files.

Reconverting is the ordinary way to move a dataset onto ``trex_v2``, and it is
the better one: it re-reads the source, so nothing depends on what a previous
conversion happened to keep. This exists for the case where reconverting is not
possible -- ``mosaic sweep-tracking`` reclaims a finished tracker run once it is
past its retention window, and a dataset whose TRex output has been swept has
tables and no longer has the ``.npz`` files they came from.

**It works because the factor survived in the table.** TRex writes
``cm_per_pixel`` into every export, the old converter was a pure passthrough that
kept every field it found, and the flattener pads a one-element array to full
length rather than dropping it. So a table converted before any of this carries,
in its first row, the number needed to undo the scaling applied to it. That is a
happy accident rather than a design, which is why a test pins the behavior it
depends on.

**It refuses rather than guessing.** A table that does not record its factor
cannot be converted by anything here -- the whole difficulty is that centimetres
and pixels are indistinguishable once you have lost the number -- so it is
reported and left alone rather than assumed to be unscaled.

The output lands under the variant a *reconversion* would have produced, not a
new one of its own. Two paths to one recipe should agree on where it lives, so a
later reconversion of the same entry finds its table already there instead of
writing a second one beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


from mosaic.core.helpers import make_entry_key, text_cell
from mosaic.core.schema import ensure_track_schema, schema_family
from mosaic.core.track_converter import get_track_converter
from mosaic.core.track_library.trex import (
    CALIBRATION_COLUMN,
    calibration_from_frame,
    name_the_body_centre,
    unscale_to_pixels,
)

from .tracks_identity import (
    convert_variant_payload,
    converter_op,
    tracks_run_id,
    tracks_variant_root,
    write_tracks_variant,
)
from .tracks_index import read_tracks_index, write_tracks_row
from .writers import read_parquet_table, write_parquet_atomic

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

__all__ = ["UpgradeOutcome", "UpgradeReport", "upgrade_trex_tables"]

_TREX_SOURCE_FORMAT = "trex_npz"

# Which producers wrote a table this can act on. A tracker run and a conversion
# of an uploaded export are the same bytes under two producer names, so both
# qualify; every other producer wrote a table in pixels already and needs a
# reconversion rather than a rescale, because its *columns* change too.
_TREX_PRODUCERS = frozenset({"trex", converter_op(_TREX_SOURCE_FORMAT)})


@dataclass(frozen=True, slots=True)
class UpgradeOutcome:
    """What happened to one table."""

    group: str
    sequence: str
    run_id: str
    status: str
    """``"upgraded"``, ``"refused"`` or ``"skipped"``."""
    detail: str = ""


@dataclass
class UpgradeReport:
    """What happened to every table considered."""

    outcomes: list[UpgradeOutcome] = field(default_factory=list)
    target_variant: str = ""

    @property
    def upgraded(self) -> list[UpgradeOutcome]:
        return [o for o in self.outcomes if o.status == "upgraded"]

    @property
    def refused(self) -> list[UpgradeOutcome]:
        return [o for o in self.outcomes if o.status == "refused"]

    @property
    def skipped(self) -> list[UpgradeOutcome]:
        return [o for o in self.outcomes if o.status == "skipped"]


def _target_variant(ds: Dataset, *, apply: bool) -> str:
    """The variant a reconversion under the current converter would mint."""
    converter = get_track_converter(_TREX_SOURCE_FORMAT)
    params = converter.Params()
    op = converter_op(_TREX_SOURCE_FORMAT)
    run_id = tracks_run_id(
        op, type(converter).version, convert_variant_payload(params.identity_dump(), 0)
    )
    if apply:
        _ = write_tracks_variant(
            ds.get_root("tracks"),
            run_id,
            op,
            type(converter).version,
            params.identity_dump(),
        )
    return run_id


def upgrade_trex_tables(ds: Dataset, *, apply: bool = False) -> UpgradeReport:
    """Rescale this dataset's centimetre-era TRex tables into pixels.

    Args:
        ds: The dataset.
        apply: Write the results. The default reports what would happen and
            touches nothing, because this rewrites data rather than an index.

    Returns:
        One outcome per table considered.
    """
    converter = get_track_converter(_TREX_SOURCE_FORMAT)
    target = _target_variant(ds, apply=apply)
    report = UpgradeReport(target_variant=target)

    index = read_tracks_index(ds)
    for _, series in index.iterrows():
        row: dict[str, object] = {str(k): v for k, v in series.items()}
        group = text_cell(row.get("group", ""))
        sequence = text_cell(row.get("sequence", ""))
        run_id = text_cell(row.get("run_id", ""))
        producer = text_cell(row.get("producer", ""))
        recorded = text_cell(row.get("std_format", ""))

        def outcome(status: str, detail: str = "") -> None:
            report.outcomes.append(
                UpgradeOutcome(group, sequence, run_id, status, detail)
            )

        if producer not in _TREX_PRODUCERS:
            outcome("skipped", f"producer {producer!r} is not TRex")
            continue
        # An unrecorded schema is the legacy one -- the column was added after
        # trex_v1 was the only schema there was.
        if schema_family(recorded or "trex_v1") != "trex_v1":
            outcome("skipped", f"already {recorded!r}")
            continue

        path = ds.resolve_path(str(row.get("abs_path", "")))
        if not path.exists():
            outcome("refused", f"table is missing from disk: {path}")
            continue

        table = read_parquet_table(path)
        try:
            cm_per_pixel = calibration_from_frame(table)
        except ValueError as exc:
            outcome("refused", str(exc))
            continue
        if cm_per_pixel is None:
            outcome(
                "refused",
                f"records no {CALIBRATION_COLUMN}, so whether its positions are "
                "centimetres cannot be established. Reconvert from the .npz, which "
                "carries the factor, or re-export from the .results.",
            )
            continue

        try:
            converted = name_the_body_centre(
                unscale_to_pixels(table, cm_per_pixel), source=path
            )
        except ValueError as exc:
            outcome("refused", str(exc))
            continue

        if not apply:
            outcome("upgraded", f"would rescale by 1/{cm_per_pixel}")
            continue

        out_path = (
            tracks_variant_root(ds.get_root("tracks"), target)
            / f"{make_entry_key(group, sequence)}.parquet"
        )
        _ = ensure_track_schema(
            converted,
            converter.output_schema,
            strict=True,
            source=f"{group}/{sequence} (upgraded)",
        )
        _ = write_parquet_atomic(converted, out_path)
        write_tracks_row(
            ds,
            run_id=target,
            group=group,
            sequence=sequence,
            out_path=out_path,
            producer=converter_op(_TREX_SOURCE_FORMAT),
            std_format=converter.output_schema,
            n_rows=int(len(converted)),
            source=path,
            consumed_source_roots=("tracks",),
        )
        outcome("upgraded", f"rescaled by 1/{cm_per_pixel}")

    return report
