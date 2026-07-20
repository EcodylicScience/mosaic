"""Multi-video transcode job with bidirectional derivative links.

``TranscodeOp`` is a registered op (``kind="transcode"``, ``domain="media"``) run
through :func:`mosaic.core.pipeline.ops.run_op`. It transcodes the originals of
one ``(group, sequence)`` entry for the analysis or playback target, running the
minimum operation each source needs. When a source is already clean for the
target, nothing is written for it.

Each performed transcode writes a per-target derivative under the ``media`` root
(``<entry>.analysis.mp4`` or ``<entry>.playback.mp4``, so the two coexist) and
links it both ways:

* forward -- the original's row in the ``media_raw`` index gets its per-target
  forward-link column (``analysis_derivative_path`` or
  ``playback_derivative_path``) set to the derivative path relative to the
  ``media`` root, leaving the other target's column untouched;
* back -- the ``media`` index (one row per derivative) records the derivative's
  re-probed facts, with ``source_path`` pointing at the original relative to the
  ``media_raw`` root.

:meth:`Dataset.resolve_media` reads the ``analysis_derivative_path`` link to
route analysis reads to the clean derivative.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
from mosaic_media import (
    CHROME_149,
    DEFAULT_THRESHOLDS,
    MediaFacts,
    Verdict,
    derive,
    probe_media,
)
from mosaic_media.transcode import (
    ANALYSIS_ENCODING,
    PLAYBACK_ENCODING,
    TranscodeError,
    TranscodeProgress,
    TranscodeResult,
    run_transcode,
)

from mosaic.core.helpers import make_entry_key, to_safe_name
from mosaic.core.media.facts_columns import (
    MEDIA_INDEX_COLUMNS,
    derivative_column_for_target,
    facts_to_row,
    series_facts_or_none,
)
from mosaic.core.pipeline._utils import hash_params
from mosaic.core.pipeline.ops import Op, register_op
from mosaic.core.pipeline.types import Params

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.job import JobContext

# Progress denominator per source file: fraction in [0, 1] maps onto this many
# ticks so the aggregate advances smoothly across all N sources.
_TICKS_PER_SOURCE = 1000

# Numeric media-index columns; every other column is a text cell that must round
# trip through CSV as an empty string rather than a float NaN.
_NUMERIC_INDEX_COLUMNS = frozenset(
    {"size_bytes", "width", "height", "fps", "frame_count", "video_order"}
)


class TranscodeParams(Params):
    """Parameters for one entry's transcode job."""

    entry: tuple[str, str]  # (group, sequence)
    target: Literal["analysis", "playback"] = "analysis"
    allow_hardware: bool = False


def _suffix_for_multi(index: int, n_sources: int) -> str:
    """Per-file stem suffix: empty for a single source, ``_<index>`` for many."""
    return "" if n_sources <= 1 else f"_{index}"


def _relative_to(path: Path, anchor: Path) -> str:
    """POSIX-style path of *path* relative to *anchor* (falls back to relpath)."""
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _transcode_run_id(
    params: TranscodeParams, sources: list[tuple[int, str, int]]
) -> str:
    """Content run_id: params identity plus an ordered per-source (rel path, size) digest.

    *sources* is ``(video_order, path relative to the media_raw root, size_bytes)``
    per source, in ``video_order``. Relative paths and sizes (not absolute paths or
    mtimes) keep the digest copy-stable across machines.
    """
    fingerprint = {
        "params": params.identity_dump(),
        "sources": [
            [order, rel_path, size]
            for order, rel_path, size in sorted(sources, key=lambda item: item[0])
        ],
    }
    return f"transcode-{hash_params(fingerprint)}"


def _load_index(index_path: Path) -> pd.DataFrame:
    """Read a media index CSV into the full schema, text cells as object ``""``.

    Missing columns are added, and text columns (everything non-numeric) are
    coerced to an object dtype with NaN replaced by ``""`` so later cell writes
    do not trip pandas' incompatible-dtype warning on all-empty float columns.
    """
    if index_path.exists():
        df = pd.read_csv(index_path)
    else:
        df = pd.DataFrame(columns=MEDIA_INDEX_COLUMNS)
    for column in MEDIA_INDEX_COLUMNS:
        if column not in df.columns:
            df[column] = ""
        if column not in _NUMERIC_INDEX_COLUMNS:
            df[column] = df[column].astype("object").where(df[column].notna(), "")
    return df


def _set_forward_link(
    ds: "Dataset",
    source: Path,
    derivative_rel: str,
    target: Literal["analysis", "playback"],
) -> None:
    """Point the original's ``media_raw`` row at its per-target derivative.

    Writes only the column for *target* (``analysis_derivative_path`` or
    ``playback_derivative_path``), leaving the other target's link untouched.
    Idempotent.
    """
    raw_root = ds.get_root(ds.resolve_media_root())
    index_path = raw_root / "index.csv"
    df = _load_index(index_path)
    source_resolved = source.resolve()
    matches = df["abs_path"].map(
        lambda value: ds.resolve_path(str(value)).resolve() == source_resolved
    )
    df.loc[matches, derivative_column_for_target(target)] = derivative_rel
    df[MEDIA_INDEX_COLUMNS].to_csv(index_path, index=False)


def _derivative_row(
    ds: "Dataset",
    group: str,
    sequence: str,
    source: Path,
    output_path: Path,
    facts: MediaFacts,
    verdict: Verdict,
    video_order: int,
) -> dict[str, object]:
    """Build the ``media`` index row describing one derivative."""
    stat = output_path.stat()
    raw_root = ds.get_root(ds.resolve_media_root())
    row: dict[str, object] = {
        "name": output_path.name,
        "group": group,
        "sequence": sequence,
        "group_safe": to_safe_name(group) if group else "",
        "sequence_safe": to_safe_name(sequence),
        "abs_path": ds.relative_to_root(output_path),
        "size_bytes": stat.st_size,
        "mtime_iso": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "width": facts.width,
        "height": facts.height,
        "fps": facts.fps,
        "codec": facts.codec_name,
        "media_type": "video",
        **facts_to_row(facts, verdict),
        # facts_to_row leaves source_path empty; the back-link records the origin.
        "source_path": _relative_to(source, raw_root),
        "video_order": video_order,
    }
    return row


def _set_back_link(
    ds: "Dataset",
    group: str,
    sequence: str,
    source: Path,
    output_path: Path,
    facts: MediaFacts,
    verdict: Verdict,
    video_order: int,
) -> None:
    """Record (or replace) the derivative's ``media`` index row (idempotent)."""
    index_path = ds.get_root("media") / "index.csv"
    df = _load_index(index_path)
    row = _derivative_row(
        ds, group, sequence, source, output_path, facts, verdict, video_order
    )
    abs_value = str(row["abs_path"])
    if not df.empty:
        df = df[df["abs_path"].astype(str) != abs_value]
    new_row = pd.DataFrame([row], columns=MEDIA_INDEX_COLUMNS)
    combined = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    combined[MEDIA_INDEX_COLUMNS].to_csv(index_path, index=False)


@register_op
class TranscodeOp(Op[TranscodeParams]):
    """Transcode one entry's originals for a target and link the derivatives both ways."""

    kind = "transcode"
    domain = "media"
    category = "transcode"
    version = "0.1"
    Params = TranscodeParams

    def target(self, params: TranscodeParams) -> str:
        group, sequence = params.entry
        return f"{group}/{sequence}"

    def run(self, ds: "Dataset", params: TranscodeParams, ctx: "JobContext") -> str:
        group, sequence = params.entry
        matched = ds.match_media_rows(group, sequence)
        sources = [
            (int(row.get("video_order", 0) or 0), ds.resolve_path(row["abs_path"]), row)
            for _, row in matched.iterrows()
        ]
        raw_root = ds.get_root(ds.resolve_media_root())
        fingerprint_sources = [
            (order, _relative_to(path, raw_root), int(row.get("size_bytes", 0) or 0))
            for order, path, row in sources
        ]
        run_id = _transcode_run_id(params, fingerprint_sources)
        ctx.set_run_id(run_id)

        n_sources = len(sources)
        encoding = (
            ANALYSIS_ENCODING if params.target == "analysis" else PLAYBACK_ENCODING
        )
        media_root = ds.get_root("media")

        ctx.set_total(n_sources * _TICKS_PER_SOURCE)
        for i, (video_order, source, row) in enumerate(sources):
            ctx.check_cancel()
            ctx.progress.on_phase(
                "transcode", f"{group}/{sequence}[{i}]: {params.target}"
            )

            facts = series_facts_or_none(row)
            if facts is None:
                facts = probe_media(source)
            verdict = derive(facts, CHROME_149, DEFAULT_THRESHOLDS)

            dest = media_root / (
                f"{make_entry_key(group, sequence)}"
                f"{_suffix_for_multi(i, n_sources)}.{params.target}.mp4"
            )

            def _on_progress(progress: TranscodeProgress, index: int = i) -> None:
                if progress.fraction is not None:
                    done = index * _TICKS_PER_SOURCE + int(
                        progress.fraction * _TICKS_PER_SOURCE
                    )
                    ctx.heartbeat(done=done)

            result: TranscodeResult = run_transcode(
                source,
                dest,
                params.target,
                facts,
                verdict,
                profile=CHROME_149,
                thresholds=DEFAULT_THRESHOLDS,
                encoding=encoding,
                allow_hardware=params.allow_hardware,
                on_progress=_on_progress,
                cancel_check=ctx.cancel_token.is_cancelled,
            )

            ctx.heartbeat(done=(i + 1) * _TICKS_PER_SOURCE)
            if not result.performed or result.output_path is None:
                continue
            if result.output_facts is None or result.output_verdict is None:
                message = (
                    f"transcode of {source} reported performed but returned no "
                    f"output facts/verdict"
                )
                raise TranscodeError(message)

            output_path = result.output_path
            derivative_rel = _relative_to(output_path, media_root)
            _set_forward_link(ds, source, derivative_rel, params.target)
            _set_back_link(
                ds,
                group,
                sequence,
                source,
                output_path,
                result.output_facts,
                result.output_verdict,
                video_order,
            )

        return run_id
