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
from mosaic.core.pipeline.dataset_indexes import (
    feature_storages,
    register_reconcilable_index,
)
from mosaic.core.pipeline.definitions.store import MODEL_FIELD
from mosaic.core.pipeline.index_csv import (
    IndexCSV,
    RunIndexRowBase,
    filter_index_rows,
)
from mosaic.core.pipeline.linked_rows import merge_linked_rows
from mosaic.core.scope import Scope

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.pipeline.arena_index import ArenaLookup
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


class AmbiguousArtifactFile(ValueError):
    """A pattern that matches more than one file of a run root."""


def ambiguous_artifact_error(
    field_name: str, feature_name: str, run_root: Path, pattern: str, files: list[Path]
) -> AmbiguousArtifactFile:
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
    return AmbiguousArtifactFile(
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
        AmbiguousArtifactFile: The pattern matched more than one file.
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
    # The digest of the saved fit state that an apply row read, copied from the
    # ``fit_scope.json`` of the run its ``model`` names. Blank on a row of a run
    # that sets no ``model``, and on a row written before the column existed. A
    # text cell and not a path, so ``merge_linked_rows`` leaves it as stored. Out
    # of the run id, like the cells above.
    consumed_model_state: str = ""


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


def _read_feature_frame(path: Path) -> pd.DataFrame:
    """The whole index at *path*, which exists."""
    return feature_index(path).read()


_FEATURE_ROW_KEYS: Final = ("run_id", "group", "sequence")
"""The columns that identify a feature index row."""

_FEATURE_ROWS_PATH_COLUMNS: Final = ("abs_path",)
"""The feature index columns that name one file each."""


def _read_index_if_present(
    path: Path, read_frame: Callable[[Path], pd.DataFrame]
) -> pd.DataFrame:
    """The index at *path*, or the schema's empty frame when there is none."""
    if not path.exists():
        return pd.DataFrame(columns=FEATURE_INDEX_COLUMNS)
    return read_frame(path)


def has_feature_index(ds: Dataset, storage: str) -> bool:
    """Whether *ds* or a dataset it links has an index for the feature *storage*."""
    return any(
        feature_index_path(linked.dataset, storage).exists()
        for linked in ds.lookup_order("features")
    )


def feature_rows(
    ds: Dataset,
    storage: str,
    *,
    run_id: str | None = None,
    read_frame: Callable[[Path], pd.DataFrame] = _read_feature_frame,
) -> pd.DataFrame:
    """The feature index of *storage* merged with those of the datasets *ds* links.

    Reads the index of each dataset of ``ds.lookup_order("features")`` and
    merges the frames with :func:`~mosaic.core.pipeline.linked_rows.merge_linked_rows`,
    keyed on ``(run_id, group, sequence)``: an own row shadows a linked row of the same
    run and entry, and an earlier link shadows a later one. The ``abs_path`` of a linked
    row is made absolute against its own dataset, and own cells stay as stored. Writes
    nothing.

    Args:
        ds: The reading dataset.
        storage: The feature's storage directory name.
        run_id: When set, only the rows of this run.
        read_frame: Reads the whole index at a path that exists. A caller that
            reads many indexes passes one that remembers what it read.

    Returns:
        The merged rows with a ``link_id`` column that names the link of each
        row and is ``""`` for an own row. Empty, with the schema's columns, when
        no dataset of the lookup holds a row.
    """
    merged = merge_linked_rows(
        [
            (
                linked.link_id,
                linked.dataset.base_dir,
                _read_index_if_present(
                    feature_index_path(linked.dataset, storage), read_frame
                ),
            )
            for linked in ds.lookup_order("features")
        ],
        _FEATURE_ROW_KEYS,
        _FEATURE_ROWS_PATH_COLUMNS,
    )
    if run_id is None:
        return merged
    return merged[merged["run_id"] == run_id].reset_index(drop=True)


def feature_run_rows(
    ds: Dataset,
    storage: str,
    run_id: str,
    *,
    filter_ext: str | None = None,
    scope: Scope | None = None,
) -> pd.DataFrame:
    """The merged rows of one run, as :meth:`IndexCSV.read` returns one index's.

    Args:
        ds: The reading dataset.
        storage: The feature's storage directory name.
        run_id: The run whose rows to read.
        filter_ext: Keep only rows whose ``abs_path`` ends with this suffix.
        scope: Which rows to keep. ``None`` keeps every row.

    Returns:
        The rows of :func:`feature_rows` for the run, filtered.

    Raises:
        FileNotFoundError: No dataset of the lookup has an index for *storage*,
            or none holds a row of *run_id*.
    """
    path = feature_index_path(ds, storage)
    if not has_feature_index(ds, storage):
        raise FileNotFoundError(f"Index not found: {path}")
    rows = feature_rows(ds, storage, run_id=run_id)
    if rows.empty:
        raise FileNotFoundError(f"No entries for run_id '{run_id}' in {path}")
    return filter_index_rows(rows, filter_ext=filter_ext, scope=scope)


def feature_run_roots(ds: Dataset, storage: str, run_id: str) -> tuple[Path, ...]:
    """Each existing root of a run, in lookup order, for a site that globs a pattern.

    A run directory that *ds* created and a linked one that holds the rest of
    the same run are both listed, *ds* first once it exists.

    Returns:
        The run root of each dataset of ``ds.lookup_order("features")`` whose
        directory exists. Empty when none does.
    """
    roots = (
        feature_run_root(linked.dataset, storage, run_id)
        for linked in ds.lookup_order("features")
    )
    return tuple(root for root in roots if root.is_dir())


def feature_run_file(ds: Dataset, storage: str, run_id: str, name: str) -> Path | None:
    """The run-level file *name* in the first run root that holds it.

    A run-level file such as ``params.json``, a fit or a report has no entry
    key, so a read goes to the first root of :func:`feature_run_roots` that
    contains it.

    Returns:
        The file's path, or ``None`` when no root holds it.
    """
    for root in feature_run_roots(ds, storage, run_id):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


type ModelReference = tuple[str, str, str | None]
"""A pinned ``model`` reference: the fit's storage, its run id and its pattern.

The pattern is the glob an ``ArtifactSpec`` names, and ``None`` for a
``DefinitionModel``, whose fit saved its state into the run root itself.
"""


def model_state_root(ds: Dataset, reference: ModelReference) -> Path | None:
    """The run root a ``model`` reference loads its state from, in lookup order.

    For a reference with a pattern, the first root of :func:`feature_run_roots`
    in which :func:`resolve_artifact_file` finds it. For one without, the first
    root with the fit's ``fit_scope.json``. A fit writes that file after it
    saves its state.

    Returns:
        The root, or ``None`` when no root holds the state.

    Raises:
        AmbiguousArtifactFile: The pattern matches more than one file of a root.
    """
    from mosaic.core.pipeline.fit_scope import FIT_SCOPE_NAME

    storage, run_id, pattern = reference
    if pattern is None:
        fit_scope = feature_run_file(ds, storage, run_id, FIT_SCOPE_NAME)
        return None if fit_scope is None else fit_scope.parent
    for root in feature_run_roots(ds, storage, run_id):
        if resolve_artifact_file(MODEL_FIELD, storage, root, pattern) is not None:
            return root
    return None


def readable_feature_storages(ds: Dataset) -> list[str]:
    """The feature storages of *ds* and of the datasets it links, each once.

    Returns:
        The storage directories of each dataset of ``ds.lookup_order("features")``
        in that order, sorted within a dataset, the first occurrence of a name
        kept.
    """
    seen: dict[str, None] = {}
    for linked in ds.lookup_order("features"):
        seen.update(dict.fromkeys(feature_storages(linked.dataset)))
    return list(seen)


def _latest_recorded_run_id(path: Path) -> str | None:
    """The most recent run of the index at *path*, or ``None`` when it records none."""
    index = feature_index(path)
    if not index.path.exists():
        return None
    runs = index.list_runs()
    return None if runs.empty else str(runs.iloc[0]["run_id"])


def latest_feature_run_root(ds: Dataset, feature_name: str) -> tuple[str, Path]:
    """The latest run of *feature_name* and its run root.

    The latest run of *ds* itself when its index records one, and otherwise the latest
    run of the first linked dataset whose index does. A later run in a link does not
    displace *ds*'s own work. A finished run comes before an unfinished one, as in
    :meth:`~mosaic.core.pipeline.index_csv.IndexCSV.latest_run_id`.

    Returns:
        The run's identifier and the root of the dataset that records it.

    Raises:
        FileNotFoundError: No dataset records a run and *ds* has no index.
        ValueError: No dataset records a run and *ds*'s index is empty.
    """
    for linked in ds.lookup_order("features"):
        run_id = _latest_recorded_run_id(
            feature_index_path(linked.dataset, feature_name)
        )
        if run_id is not None:
            return run_id, feature_run_root(linked.dataset, feature_name, run_id)
    run_id = feature_index(feature_index_path(ds, feature_name)).latest_run_id()
    return run_id, feature_run_root(ds, feature_name, run_id)


@dataclass(frozen=True, slots=True)
class RecordedTracksReads:
    """What one entry's feature row records reading from the tracks tables.

    Attributes:
        composition: The entry's ``consumed_tracks_composition``.
        neighbors: Its ``consumed_neighbors``, empty without overlap.
        model_state: Its ``consumed_model_state``, empty without a ``model``.
        inputs: Its ``consumed_input_composition``, empty without a feature
            input whose upstream records a declared root.
    """

    composition: str = ""
    neighbors: str = ""
    model_state: str = ""
    inputs: str = ""


def recorded_tracks_reads(
    ds: Dataset, feature_name: str, run_id: str
) -> dict[tuple[str, str], RecordedTracksReads]:
    """Per entry, the tracks, neighbors, model state and input cells each row recorded.

    Its own reader rather than a third element on :func:`recorded_consumption`,
    which drops every row whose ``consumed_roots`` is empty -- and that is the forty
    features this exists for.
    """
    return _tracks_reads_of(_run_rows(ds, feature_name, run_id))


def _run_rows(ds: Dataset, feature_name: str, run_id: str) -> pd.DataFrame:
    """One run's rows of its feature index and its links', empty when there are none."""
    return feature_rows(ds, feature_name, run_id=run_id)


def _tracks_reads_of(rows: pd.DataFrame) -> dict[Entry, RecordedTracksReads]:
    """Per entry, the tracks, neighbors, model state and input cells of *rows*."""
    return {
        (str(row["group"]), str(row["sequence"])): RecordedTracksReads(
            composition=str(row.get("consumed_tracks_composition", "")),
            neighbors=str(row.get("consumed_neighbors", "")),
            model_state=str(row.get("consumed_model_state", "")),
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


class DriftReads:
    """The index reads that several :func:`drifted_entries` calls share.

    An inventory scan builds one and passes it to the call for every run: each feature
    storage's index is read once and divided by run once, the tracks cells of each set
    of variants are computed once, each root's ``sequences.csv`` is read once, each
    fit's ``fit_scope.json`` is read once, and the arenas lookup is taken once. A
    call given none builds one for itself.

    Args:
        ds: The dataset whose indexes are read.
        feature_frame: Reads a feature storage's whole index from a path that
            exists, in this dataset and in the datasets it links. The default
            reads it through :func:`feature_index`. A scan passes the reader it
            reads the indexes with, and the file is read once.
    """

    def __init__(
        self,
        ds: Dataset,
        feature_frame: Callable[[Path], pd.DataFrame] = _read_feature_frame,
    ) -> None:
        self._ds: Dataset = ds
        self._feature_frame: Callable[[Path], pd.DataFrame] = feature_frame
        self._runs: dict[str, dict[str, pd.DataFrame]] = {}
        self._tracks: dict[tuple[str, ...], dict[Entry, str]] = {}
        self._held: dict[tuple[str, ...], frozenset[Entry]] = {}
        self._arena_lookup: ArenaLookup | None = None
        self._recorded: dict[Entry, dict[str, str]] | None = None
        self._model_states: dict[ModelReference, str] = {}

    def _rows(self, storage: str, run_id: str) -> pd.DataFrame:
        runs = self._runs.get(storage)
        if runs is None:
            frame = feature_rows(self._ds, storage, read_frame=self._feature_frame)
            runs = (
                {}
                if frame.empty or "run_id" not in frame.columns
                else {
                    str(key): rows for key, rows in frame.groupby("run_id", sort=False)
                }
            )
            self._runs[storage] = runs
        return runs.get(run_id, pd.DataFrame())

    def tracks_reads(
        self, storage: str, run_id: str
    ) -> dict[Entry, RecordedTracksReads]:
        """What :func:`recorded_tracks_reads` returns, from the shared frame."""
        return _tracks_reads_of(self._rows(storage, run_id))

    def model_state(self, reference: ModelReference) -> str:
        """The fit-state digest of the fit *reference* names, read once per fit.

        Read from the root :func:`model_state_root` resolves, through the
        datasets the dataset links. A fit that no reachable root holds, a
        ``fit_scope.json`` that cannot be read and a pattern that matches more
        than one file of a root (:class:`AmbiguousArtifactFile`) read as blank:
        a status read does not raise.

        Returns:
            The digest, and ``""`` when none is recorded or readable.
        """
        from .fit_scope import FitStateUnreadable, read_fit_state_digest

        digest = self._model_states.get(reference)
        if digest is None:
            try:
                root = model_state_root(self._ds, reference)
                digest = "" if root is None else read_fit_state_digest(root)
            except (AmbiguousArtifactFile, FitStateUnreadable):
                digest = ""
            self._model_states[reference] = digest
        return digest

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

    def recorded_compositions(self) -> dict[Entry, dict[str, str]]:
        """Every entry's :func:`~.sequence_index.recorded_compositions`, read once."""
        from .sequence_index import recorded_compositions

        if self._recorded is None:
            self._recorded = recorded_compositions(self._ds)
        return self._recorded

    def arena_lookup(self) -> ArenaLookup:
        """The :func:`~.arena_index.cached_arena_lookup` of the dataset, taken once."""
        from .arena_index import cached_arena_lookup

        if self._arena_lookup is None:
            self._arena_lookup = cached_arena_lookup(self._ds)
        return self._arena_lookup

    def tracks_entries(self, variants: Sequence[str]) -> frozenset[Entry]:
        """The entries the tracks index holds a row for under *variants*.

        An unlabelled row counts beside a row of *variants*, because a run
        records only the labelled variants it read. A row whose composition is
        blank counts too: it is absent from :meth:`tracks_cells` but present here.
        """
        from .tracks_index import tracks_rows

        key = tuple(sorted(set(variants)))
        held = self._held.get(key)
        if held is None:
            readable = {"", *key}
            held = frozenset(
                (str(row.get("group", "")), str(row.get("sequence", "")))
                for _, row in tracks_rows(self._ds).iterrows()
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
      composition under those roots in ``<root>/sequences.csv``, and its layout
      digest when a row or the run declares ``arenas``
      (:func:`~.arena_index.entry_compositions`);
    - its tracks: the row's ``consumed_tracks_composition`` against the tracks
      rows of the run's variants (:func:`~.tracks_index.tracks_compositions`) for
      a run whose only input is ``tracks``, or against its inputs' cells joined
      in input order (:func:`~.composition.join_input_cells`), a feature input's
      cell being its upstream row's ``consumed_tracks_composition``;
    - its upstreams' roots: the row's ``consumed_input_composition`` against its
      feature inputs' parts (:func:`~.composition.input_root_part`), joined the
      same way;
    - its model: for a run whose ``params.json`` records a ``model``, the row's
      ``consumed_model_state`` against the digest the fit records now
      (:meth:`DriftReads.model_state`). A fit under an unreachable link, or whose
      record cannot be read, reads as blank.

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
    from .arena_index import ARENAS_ROOT, entry_compositions
    from .sequence_index import encode_entry_composition

    shared = DriftReads(ds) if reads is None else reads
    recorded_roots = shared.consumption(feature_name, run_id)
    declared = tuple(run_params.scope.consumed_roots) if run_params else ()
    recorded_reads = shared.tracks_reads(feature_name, run_id) if run_params else {}
    model = run_params.model_reference() if run_params else None
    model_now = "" if model is None else shared.model_state(model)
    entries = sorted(set(recorded_roots) | set(recorded_reads))
    if not entries:
        return ()
    in_play = set(declared).union(*(roots for roots, _ in recorded_roots.values()))
    lookup = shared.arena_lookup() if ARENAS_ROOT in in_play else None
    current_roots = entry_compositions(
        ds, entries, in_play, lookup=lookup, recorded=shared.recorded_compositions()
    )
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
            if model is not None:
                channels.append((read.model_state, model_now))
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
