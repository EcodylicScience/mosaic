from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import pandas as pd

from mosaic.core.entry import Entry
from mosaic.core.pipeline.composition import (
    cached_entry_disposition,
    input_root_part,
    join_input_cells,
)
from mosaic.core.pipeline.dataset_indexes import register_reconcilable_index
from mosaic.core.pipeline.index_csv import IndexCSV, RunIndexRowBase

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.inventory.params import RunParams


class DatasetRoots(Protocol):
    """A dataset's named roots, the one part of it that feature paths read."""

    def get_root(self, key: str, /) -> Path: ...


def feature_run_root(ds: DatasetRoots, feature_name: str, run_id: str) -> Path:
    return ds.get_root("features") / feature_name / run_id


def feature_index_path(ds: DatasetRoots, feature_name: str) -> Path:
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
    # now on are checkable. A run with a feature input records its inputs' cells
    # joined with ``+`` in input order, a feature input's cell being its upstream
    # row's own (``build_manifest``).
    consumed_tracks_composition: str = ""
    # The declared-root compositions of a run's feature inputs: per input in input
    # order, joined with ``+``, the upstream row's ``consumed_composition`` and
    # ``consumed_input_composition`` as ``own/inherited`` (``input_root_part`` in
    # ``composition.py``). A ``tracks`` input's part is blank, and so is the cell of
    # a run whose every part is blank. Kept apart from ``consumed_composition``,
    # which holds this run's own declared roots.
    consumed_input_composition: str = ""
    # The neighbors an overlapped run read for this entry, each by entry key with
    # its tracks composition (``neighbor_compositions`` in ``run.py``). Empty at
    # ``overlap_frames=0``, and out of the run id, like the cells above.
    consumed_neighbors: str = ""


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


@dataclass(frozen=True, slots=True)
class RecordedTracksReads:
    """What one entry's feature row records reading from the tracks tables.

    Attributes:
        composition: The entry's ``consumed_tracks_composition``.
        neighbors: Its ``consumed_neighbors``, empty without overlap.
        inputs: Its ``consumed_input_composition``, empty without a feature
            input whose upstream records a declared root.
    """

    composition: str = ""
    neighbors: str = ""
    inputs: str = ""


def recorded_tracks_reads(
    ds: Dataset, feature_name: str, run_id: str
) -> dict[tuple[str, str], RecordedTracksReads]:
    """Per entry, the tracks, neighbors and input cells each row recorded reading.

    Its own reader rather than a third element on :func:`recorded_consumption`,
    which drops every row whose ``consumed_roots`` is empty -- and that is the forty
    features this exists for.
    """
    return _tracks_reads_of(_run_rows(ds, feature_name, run_id))


def _run_rows(ds: Dataset, feature_name: str, run_id: str) -> pd.DataFrame:
    """One run's rows of its feature index, empty when there are none."""
    index = feature_index(feature_index_path(ds, feature_name))
    if not index.path.exists():
        return pd.DataFrame()
    try:
        return index.read(run_id=run_id)
    except FileNotFoundError:
        return pd.DataFrame()


def _tracks_reads_of(rows: pd.DataFrame) -> dict[Entry, RecordedTracksReads]:
    """Per entry, the tracks, neighbors and input cells of one run's *rows*."""
    return {
        (str(row["group"]), str(row["sequence"])): RecordedTracksReads(
            composition=str(row.get("consumed_tracks_composition", "")),
            neighbors=str(row.get("consumed_neighbors", "")),
            inputs=str(row.get("consumed_input_composition", "")),
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
    return _consumption_of(_run_rows(ds, feature_name, run_id))


def _consumption_of(rows: pd.DataFrame) -> dict[Entry, tuple[tuple[str, ...], str]]:
    """Per entry of one run's *rows* that declares a root, its roots and digest."""
    from .sequence_index import decode_consumed_roots

    recorded: dict[Entry, tuple[tuple[str, ...], str]] = {}
    for _, row in rows.iterrows():
        roots = decode_consumed_roots(str(row.get("consumed_roots", "")))
        if not roots:
            continue
        entry = (str(row["group"]), str(row["sequence"]))
        recorded[entry] = (roots, str(row.get("consumed_composition", "")))
    return recorded


def _read_feature_frame(path: Path) -> pd.DataFrame:
    """A feature storage's whole index, empty when it has none."""
    index = feature_index(path)
    return index.read() if index.path.exists() else pd.DataFrame()


class DriftReads:
    """The index reads that several :func:`drifted_entries` calls share.

    An inventory scan builds one and passes it to the call for every run: each
    feature storage's index is read once, and the tracks cells of each set of
    variants are computed once. A call given none builds one for itself.

    Args:
        ds: The dataset whose indexes are read.
        feature_frame: Reads a feature storage's whole index from its path. The
            default reads it through :func:`feature_index`. A scan passes the
            reader it reads the indexes with, and the file is read once.
    """

    def __init__(
        self,
        ds: Dataset,
        feature_frame: Callable[[Path], pd.DataFrame] = _read_feature_frame,
    ) -> None:
        self._ds: Dataset = ds
        self._feature_frame: Callable[[Path], pd.DataFrame] = feature_frame
        self._frames: dict[str, pd.DataFrame] = {}
        self._tracks: dict[tuple[str, ...], dict[Entry, str]] = {}
        self._held: dict[tuple[str, ...], frozenset[Entry]] = {}

    def _rows(self, storage: str, run_id: str) -> pd.DataFrame:
        frame = self._frames.get(storage)
        if frame is None:
            frame = self._feature_frame(feature_index_path(self._ds, storage))
            self._frames[storage] = frame
        if frame.empty or "run_id" not in frame.columns:
            return pd.DataFrame()
        return frame[frame["run_id"] == run_id]

    def tracks_reads(
        self, storage: str, run_id: str
    ) -> dict[Entry, RecordedTracksReads]:
        """What :func:`recorded_tracks_reads` returns, from the shared frame."""
        return _tracks_reads_of(self._rows(storage, run_id))

    def consumption(
        self, storage: str, run_id: str
    ) -> dict[Entry, tuple[tuple[str, ...], str]]:
        """What :func:`recorded_consumption` returns, from the shared frame."""
        return _consumption_of(self._rows(storage, run_id))

    def tracks_cells(self, variants: Sequence[str]) -> dict[Entry, str]:
        """The tracks cells of *variants*, as ``tracks_compositions`` builds them."""
        from .tracks_index import tracks_compositions

        key = tuple(sorted(set(variants)))
        cells = self._tracks.get(key)
        if cells is None:
            cells = tracks_compositions(self._ds, key)
            self._tracks[key] = cells
        return cells

    def tracks_entries(self, variants: Sequence[str]) -> frozenset[Entry]:
        """The entries the tracks index holds a row for under *variants*.

        An unlabelled row counts beside a row of *variants*, because a run
        records only the labelled variants it read. A row whose composition is
        blank counts too: it is absent from :meth:`tracks_cells` but present here.
        """
        from .tracks_index import read_tracks_index

        key = tuple(sorted(set(variants)))
        held = self._held.get(key)
        if held is None:
            readable = {"", *key}
            held = frozenset(
                (str(row.get("group", "")), str(row.get("sequence", "")))
                for _, row in read_tracks_index(self._ds).iterrows()
                if str(row.get("run_id", "")) in readable
            )
            self._held[key] = held
        return held


def drifted_entries(
    ds: Dataset,
    feature_name: str,
    run_id: str,
    run_params: RunParams | None,
    reads: DriftReads | None = None,
) -> tuple[tuple[str, str], ...]:
    """Entries whose sources have moved since this run recorded what it consumed.

    Each entry's row is compared, channel by channel, with what the dataset
    records now, under :func:`~.composition.cached_entry_disposition`, the rule
    ``run_feature`` applies before it serves a cached entry. An entry is drifted
    when any channel says ``recompute``. The entries reported are those a run
    would recompute:

    - its declared roots: the row's ``consumed_composition`` against the entry's
      composition under those roots in ``<root>/sequences.csv``;
    - its tracks: the row's ``consumed_tracks_composition`` against the tracks
      rows of the run's variants (:func:`~.tracks_index.tracks_compositions`) for
      a run whose only input is ``tracks``, or against its inputs' cells joined
      in input order (:func:`~.composition.join_input_cells`), a feature input's
      cell being its upstream row's ``consumed_tracks_composition``;
    - its upstreams' roots: the row's ``consumed_input_composition`` against its
      feature inputs' parts (:func:`~.composition.input_root_part`), joined the
      same way.

    The neighbors cell is compared by ``run_feature`` alone. An entry the run's
    inputs no longer resolve, dropped from the tracks index for instance, is not
    reported: a run would recompute nothing for it. A run that intersects its
    inputs resolves an entry every input holds, and a stacked run one that any
    input holds. A ``params.json`` written before it recorded stacking counts an
    entry any input holds: a false drift there lasts one run, which records the
    stacking, while counting only the entries every input holds would hide the
    real drift of a stacked run that nothing then recomputes.

    **Both sides are recorded values.** A value recomputed from the present
    would always agree with itself. Reading recorded values also keeps a status
    display free of any filesystem measurement.

    **One known side is drift**, and two blank sides are not. A row written before
    its cell existed against a present that records one is recomputed once, after
    which the row records the present. Two blanks cannot converge: the run
    serves them with a warning, and reporting them here would ask for a recompute
    that changes nothing.

    Args:
        ds: The dataset holding the run.
        feature_name: The run's feature storage.
        run_id: The run.
        run_params: The run's ``params.json``, which names its inputs and tracks
            variants. ``None`` compares the declared roots alone.
        reads: The index reads shared with other calls, or ``None`` to read
            them for this call alone.

    Returns:
        The drifted entries, sorted.
    """
    from .sequence_index import encode_entry_composition, read_entry_compositions

    shared = DriftReads(ds) if reads is None else reads
    recorded_roots = shared.consumption(feature_name, run_id)
    declared = tuple(run_params.scope.consumed_roots) if run_params else ()
    recorded_reads = shared.tracks_reads(feature_name, run_id) if run_params else {}
    entries = sorted(set(recorded_roots) | set(recorded_reads))
    if not entries:
        return ()
    current_roots = read_entry_compositions(ds, entries)
    tracks_now, inputs_now, held = (
        _current_input_cells(shared, run_params)
        if run_params
        else ({}, {}, frozenset(entries))
    )

    drifted: list[tuple[str, str]] = []
    for entry in entries:
        if entry not in held:
            continue
        roots, was = recorded_roots.get(entry, (declared, ""))
        channels = [
            (was, encode_entry_composition(current_roots.get(entry, {}), roots))
            if roots
            else ("", "")
        ]
        if run_params is not None:
            read = recorded_reads.get(entry, RecordedTracksReads())
            channels.append((read.composition, tracks_now.get(entry, "")))
            channels.append((read.inputs, inputs_now.get(entry, "")))
        if any(
            cached_entry_disposition(recorded, current) == "recompute"
            for recorded, current in channels
        ):
            drifted.append(entry)
    return tuple(drifted)


def _current_input_cells(
    reads: DriftReads, run_params: RunParams
) -> tuple[dict[Entry, str], dict[Entry, str], frozenset[Entry]]:
    """Per entry, the tracks cell and the upstream-roots cell a run would compare now.

    Built as ``build_manifest`` builds them: a run whose only input is ``tracks``
    compares its variants' tracks rows and no upstream roots, and a run with a
    feature input joins each input's part in input order, a ``tracks`` input's
    roots part being blank. The third element holds the entries the inputs
    still resolve: those every input holds a row for, or any input for a stacked
    run or one whose file does not say, whether or not the row records a
    composition.
    """
    tracks_variants, _ = run_params.variant_ids()
    sources = run_params.input_sources()
    if all(source == "tracks" for source in sources):
        return (
            reads.tracks_cells(tracks_variants),
            {},
            reads.tracks_entries(tracks_variants),
        )
    per_input_tracks: list[Mapping[Entry, str]] = []
    per_input_roots: list[Mapping[Entry, str]] = []
    per_input_held: list[frozenset[Entry]] = []
    for source in sources:
        if source == "tracks":
            per_input_tracks.append(reads.tracks_cells(tracks_variants))
            per_input_roots.append({})
            per_input_held.append(reads.tracks_entries(tracks_variants))
            continue
        storage, upstream_run = source
        upstream_reads = reads.tracks_reads(storage, upstream_run)
        per_input_held.append(frozenset(upstream_reads))
        upstream_roots = reads.consumption(storage, upstream_run)
        per_input_tracks.append(
            {entry: read.composition for entry, read in upstream_reads.items()}
        )
        per_input_roots.append(
            {
                entry: input_root_part(
                    upstream_roots.get(entry, ((), ""))[1], read.inputs
                )
                for entry, read in upstream_reads.items()
            }
        )
    held = (
        frozenset[Entry]().union(*per_input_held)
        if run_params.scope.stacks_inputs is not False
        else per_input_held[0].intersection(*per_input_held[1:])
    )
    return (
        join_input_cells(per_input_tracks),
        join_input_cells(per_input_roots),
        held,
    )


# Item 6.1: reconciled through the shared registry, beside the tracker
# indexes, so one pass covers every root that has an ``IndexCSV`` behind it.
register_reconcilable_index("features", feature_index)
