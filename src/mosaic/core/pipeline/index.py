from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pandas as pd

from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset


def feature_run_root(ds: Dataset, feature_name: str, run_id: str) -> Path:
    return ds.get_root("features") / feature_name / run_id


def feature_index_path(ds: Dataset, feature_name: str) -> Path:
    return ds.get_root("features") / feature_name / "index.csv"


def missing_outputs_error(
    feature_name: str, run_id: str, missing: list[Path], total: int
) -> FileNotFoundError:
    """Build an actionable error for a feature run whose outputs all resolve missing.

    Raised when *every* output file for a run is unreachable after
    ``Dataset.resolve_path`` — the classic "dataset moved / synced with
    non-portable absolute paths" signal. Preferred over a raw
    ``FileNotFoundError`` (loud + fixable) and over silently skipping every
    entry (which would compute a downstream feature over an empty manifest).

    Args:
        feature_name: Storage name of the feature whose run is stale.
        run_id: The run whose outputs are missing.
        missing: Resolved paths that do not exist (non-empty).
        total: Total number of output rows examined for the run.
    """
    first = missing[0]
    return FileNotFoundError(
        f"Feature {feature_name!r} run {run_id!r}: all {len(missing)} of "
        f"{total} output file(s) are missing, first: {first}. The index likely "
        f"points at another machine's paths (dataset moved, or synced with "
        f"non-portable absolute paths). Repair a moved/synced dataset with "
        f"ds.make_portable() on the machine whose root matches the stored "
        f"paths, or ds.rewrite_index_paths({{old_prefix: new_prefix}}); if the "
        f"outputs were deleted, recompute the feature (or ds.reindex_features() "
        f"to drop the stale index rows)."
    )


_MAX_LISTED_MATCHES: Final = 5
"""How many matches an ambiguity names before it counts the rest.

A run root holds one output per sequence, so a hundred-sequence dataset would
otherwise put a hundred filenames in one message."""


def ambiguous_artifact_error(
    field_name: str, feature_name: str, run_root: Path, pattern: str, files: list[Path]
) -> ValueError:
    """Build an actionable error for a pattern that names more than one file.

    Args:
        field_name: The params field holding the reference.
        feature_name: Storage name of the feature that wrote the run root.
        run_root: The producer's run directory that was searched.
        pattern: The glob that matched.
        files: The matches, sorted (more than one).
    """
    listed = ", ".join(path.name for path in files[:_MAX_LISTED_MATCHES])
    if len(files) > _MAX_LISTED_MATCHES:
        listed = f"{listed}, and {len(files) - _MAX_LISTED_MATCHES} more"
    return ValueError(
        f"Field {field_name!r} references feature {feature_name!r}, whose run root "
        f"{run_root} holds {len(files)} files matching {pattern!r}: {listed}. A run "
        f"root is not a directory of named state files -- it holds one per-entry "
        f"output parquet per sequence beside whatever the feature saved -- so this "
        f"pattern does not name one artifact. Name the file: on a recipe reference "
        f'write {{"step": ..., "pattern": "<filename>"}}, and in library code use '
        f"the producing feature's pinned artifact class."
    )


def resolve_artifact_file(
    field_name: str, feature_name: str, run_root: Path, pattern: str
) -> Path | None:
    """The one file *pattern* names in *run_root*, or None when it names none.

    The single funnel every artifact reference resolves through, so a graph-driven
    run and a library-driven one cannot disagree. More than one match is refused
    rather than guessed: this used to take ``sorted(...)[0]``, which for the
    derived ``*.parquet`` glob is a per-entry output rather than the artifact the
    reference meant, and every layer downstream read it as the real thing.

    Returning ``None`` for no match is deliberate and is not the same question. A
    consumer branches on whether its field resolved, so an absent artifact is a
    state features already handle; an ambiguous one is a wrong answer.

    Args:
        field_name: The params field holding the reference.
        feature_name: Storage name of the feature that wrote the run root.
        run_root: The producer's run directory to search.
        pattern: The glob to match, already defaulted by ``ArtifactSpec``.

    Raises:
        ValueError: The pattern matched more than one file.
    """
    files = sorted(run_root.glob(pattern))
    if len(files) > 1:
        raise ambiguous_artifact_error(
            field_name, feature_name, run_root, pattern, files
        )
    return files[0] if files else None


# --- Feature Index ---


@dataclass(frozen=True, slots=True)
class FeatureIndexRow(RunIndexRowBase):
    """Typed row for the feature index CSV."""

    feature: str
    version: str
    group: str
    sequence: str
    params_hash: str
    n_rows: int = 0
    # Which hashing contract minted this row's run_id. Defaulted rather than
    # required so the five sibling row types are unaffected and every existing
    # construction site keeps working. Rows written before the marker existed
    # read back as "" -- an honest "predates the scheme", which is a more useful
    # answer than stamping the current one onto history that cannot be verified.
    # Reconstructed rows (Dataset.reindex_features) cannot know the historical
    # value and must leave it empty for the same reason.
    identity_scheme: str = ""
    # What this entry was made from, recorded and never hashed -- item 5.1's
    # features half, which item 4.4 makes computable.
    #
    # ``consumed_roots`` is the feature's declaration (comma-joined, sorted);
    # ``consumed_composition`` is this entry's composition under those roots at
    # the moment it was computed. Both are empty for the forty features that
    # declare no root, which is the ordinary case and not a gap.
    #
    # Recorded rather than hashed, and the distinction is the whole design. A
    # per-frame identifier names a directory holding *every* entry, so a
    # per-entry fact in it would rename the directory that holds another
    # sequence's already-correct output (rule P2d). The composition each entry
    # consumed belongs beside that entry's row, where item 6.2 can turn it into a
    # per-entry delete set -- which is what H3 case 1 actually asks for: "tracks
    # and features **go**", deleted, not re-identified.
    consumed_roots: str = ""
    consumed_composition: str = ""
    # The source composition of the *tracks* this entry read, copied from the tracks
    # row that recorded it. A separate column rather than folded into
    # ``consumed_composition`` on purpose: folding would make every row written
    # before this existed read as one-side-known, hence stale, and force a full
    # recompute of every tracks-consuming feature on every dataset. Empty here means
    # "predates the column" and is served with a warning, so only rows written from
    # now on are checkable.
    consumed_tracks_composition: str = ""


def adopt_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Bring a feature index read off disk up to the current schema, in memory.

    ``feature_index`` had no ``adopt`` hook, which was survivable while every
    added column was a string: ``_read_frame`` passes ``keep_default_na=False``
    and ``to_csv`` writes NaN as empty, so a missing string column round-trips as
    ``""``. It stops being survivable the moment one is numeric -- an absent
    column concatenated against a real row widens the integer, and ``40`` reaches
    disk as ``40.0``. The hook is cheap now and load-bearing later.

    Same shape as ``tracks_index.adopt_legacy_columns``: every column built with
    an explicit ``object`` dtype, missing ones added empty, NaN coerced, and
    off-schema columns dropped.
    """
    out = pd.DataFrame(index=df.index)
    for column in FEATURE_INDEX_COLUMNS:
        if column in df.columns:
            cells = ["" if pd.isna(cell) else cell for cell in df[column]]
        else:
            cells = [""] * len(df)
        out[column] = pd.Series(cells, index=df.index, dtype="object")
    return out


FEATURE_INDEX_COLUMNS: list[str] = [field.name for field in fields(FeatureIndexRow)]
"""The schema, in CSV order. Derived from the row so the two cannot drift."""


def feature_index(path: Path) -> IndexCSV[FeatureIndexRow]:
    """Factory: return an IndexCSV configured for the feature index schema."""
    return IndexCSV(
        path,
        FeatureIndexRow,
        dedup_keys=["run_id", "group", "sequence"],
        adopt=adopt_feature_columns,
    )


def list_feature_runs(ds: Dataset, feature_name: str) -> pd.DataFrame:
    return feature_index(feature_index_path(ds, feature_name)).list_runs()


def latest_feature_run_root(ds: Dataset, feature_name: str) -> tuple[str, Path]:
    idx = feature_index(feature_index_path(ds, feature_name))
    run_id = idx.latest_run_id()
    return run_id, feature_run_root(ds, feature_name, run_id)


def recorded_tracks_composition(
    ds: Dataset, feature_name: str, run_id: str
) -> dict[tuple[str, str], str]:
    """Per entry, the tracks composition each row recorded reading.

    Its own reader rather than a third element on :func:`recorded_consumption`,
    which drops every row whose ``consumed_roots`` is empty -- and that is the forty
    features this exists for.
    """
    index = feature_index(feature_index_path(ds, feature_name))
    if not index.path.exists():
        return {}
    try:
        rows = index.read(run_id=run_id)
    except FileNotFoundError:
        return {}
    return {
        (str(row["group"]), str(row["sequence"])): str(
            row.get("consumed_tracks_composition", "")
        )
        for _, row in rows.iterrows()
    }


def recorded_consumption(
    ds: Dataset, feature_name: str, run_id: str
) -> dict[tuple[str, str], tuple[tuple[str, ...], str]]:
    """What each entry of *run_id* recorded consuming: its roots, and its digest.

    One reader for the two callers that need it -- :func:`drifted_entries`, which
    compares the digest against the present, and ``run_feature``'s cache-hit
    pre-pass, which carries it forward rather than restamping a skipped entry.
    Two loops over the same three cells would be two answers to item 6.2's walk,
    which is the argument ``encode_entry_composition`` already makes one level
    down.

    An **absent** entry is not one recording ``""``. The first has no row at all:
    an output on disk that nothing describes, whose provenance cannot be stated.
    The second is a row written before item 5.1, or one whose root was not
    establishable. Neither is evidence of change; both are item 6.2's to fail
    closed on. Entries declaring no root are dropped here, having nothing to say.
    """
    from .sequence_index import decode_consumed_roots

    index = feature_index(feature_index_path(ds, feature_name))
    if not index.path.exists():
        return {}
    try:
        rows = index.read(run_id=run_id)
    except FileNotFoundError:
        return {}

    recorded: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {}
    for _, row in rows.iterrows():
        roots = decode_consumed_roots(str(row.get("consumed_roots", "")))
        if not roots:
            continue
        entry = (str(row["group"]), str(row["sequence"]))
        recorded[entry] = (roots, str(row.get("consumed_composition", "")))
    return recorded


def drifted_entries(
    ds: Dataset, feature_name: str, run_id: str
) -> tuple[tuple[str, str], ...]:
    """Entries whose source has moved since this run recorded what it consumed.

    Item 5.2's chain-runner half. Both sides are already on disk, so this is two
    small index reads and no probe: the run's rows carry the composition each
    entry was built from (item 5.1's features half), and ``<root>/sequences.csv``
    carries what that entry is made of now (item 4.4). A difference between them
    is a source that moved under a finished run.

    **Recorded against recorded, never recomputed.** A value recomputed from the
    present agrees with itself by construction, which is the argument
    ``sequence_index`` makes for storing compositions at all. It also keeps
    ``Pipeline.status`` free of any filesystem measurement -- a status display
    that probed would be unusable on the corpora this exists for.

    **Both sides must be non-empty to count**, and that is the honest-empty rule
    rather than caution. An empty recorded cell means the run predates item 5.1
    or its root could not be established; an empty current one means the
    projection has not been written or is unestablishable now. Neither is
    evidence of change, and reporting either as drift would light up every
    pre-Stage-5 run in the display. Unknown is item 6.2's to fail closed on, not
    this function's to guess at.
    """
    from .sequence_index import encode_entry_composition, read_entry_compositions

    recorded = recorded_consumption(ds, feature_name, run_id)
    if not recorded:
        return ()

    current = read_entry_compositions(ds, recorded.keys())
    drifted: list[tuple[str, str]] = []
    for entry, (roots, was) in sorted(recorded.items()):
        now = encode_entry_composition(current.get(entry, {}), roots)
        if was and now and was != now:
            drifted.append(entry)
    return tuple(drifted)


# Item 6.1: reconciled through the shared registry, beside the tracker
# indexes, so one pass covers every root that has an ``IndexCSV`` behind it.
register_reconcilable_index("features", feature_index)
