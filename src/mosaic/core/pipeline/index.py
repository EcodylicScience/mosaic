from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

import pandas as pd

from mosaic.core.entry import Entry, entry_token
from mosaic.core.pipeline.composition import (
    cached_entry_disposition,
    input_part,
    join_input_cells,
)
from mosaic.core.pipeline.dataset_indexes import (
    feature_storages,
    register_reconcilable_index,
)
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
    from mosaic.core.pipeline.fit_scope import FitScope
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
    # What this entry was made from, recorded and never hashed.
    #
    # ``consumed_roots`` is the feature's declaration (comma-joined, sorted);
    # ``consumed_composition`` is this entry's composition under those roots at
    # the moment it was computed. Both are empty for the forty features that
    # declare no root, which is the ordinary case and not a gap.
    #
    # Recorded rather than hashed, and the distinction is the whole design. A
    # per-frame identifier names a directory holding *every* entry, so a
    # per-entry fact in it would rename the directory that holds another
    # sequence's already-correct output. The composition each entry consumed
    # belongs beside that entry's row, where the delete set can read it and
    # delete that entry's tracks and features, not re-identify them.
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
    # What a run's feature inputs recorded: per input in input order, joined with
    # ``+``, the upstream row's ``consumed_composition`` and
    # ``consumed_input_composition`` as ``own/inherited``, led by its
    # ``consumed_model_state`` as ``state|own/inherited`` when that is not blank
    # (``input_part`` in ``composition.py``). A ``tracks`` input's part is blank,
    # and so is the cell of a run whose every part is blank. Kept apart from
    # ``consumed_composition``, which holds this run's own declared roots.
    consumed_input_composition: str = ""
    # The neighbors an overlapped run read for this entry, each by entry key with
    # its tracks composition (``neighbor_compositions`` in ``run.py``). Empty at
    # ``overlap_frames=0``, and out of the run id, like the cells above.
    consumed_neighbors: str = ""
    # The fit states a row read, spelled by ``fit_state_cell``: one part per fit
    # reference of its run, keyed by params field, holding the ``state_digest`` in
    # that fit's ``fit_scope.json``, and a ``_state`` part holding the digest of
    # the state the run itself loaded or saved when its run root has a fit
    # record. A fit's own rows carry the ``_state`` part. Blank on a row that
    # read no fit state, and on a row written before the column existed. A text
    # cell and not a path, so ``merge_linked_rows`` leaves it as stored. Out of
    # the run id, like the cells above.
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
    located = feature_run_file_location(ds, storage, run_id, name)
    return None if located is None else located[1]


def feature_run_file_location(
    ds: Dataset, storage: str, run_id: str, name: str
) -> tuple[str, Path] | None:
    """The run-level file *name* in the first root that holds it, and the link.

    Returns:
        The link id of the dataset of ``ds.lookup_order("features")`` whose run
        root holds the file, ``""`` for *ds* itself, and the file's path.
        ``None`` when no root holds it.
    """
    for linked in ds.lookup_order("features"):
        candidate = feature_run_root(linked.dataset, storage, run_id) / name
        if candidate.is_file():
            return linked.link_id, candidate
    return None


@dataclass(frozen=True)
class FitReference:
    """A params field that names a pinned fit: the state a run reads from it.

    Attributes:
        field: The top-level params field holding the reference.
        feature: The fit's feature storage.
        run_id: The fit's run id.
        pattern: The glob an ``ArtifactSpec`` names in the fit's root, and
            ``None`` for a ``DefinitionModel``, whose fit saved its state into
            the run root itself.
    """

    field: str
    feature: str
    run_id: str
    pattern: str | None


@dataclass(frozen=True)
class EntryCells:
    """The three cells an entry's row records of what it consumed.

    Attributes:
        roots: The entry's composition under the run's declared roots.
        tracks: The tracks cell: the tracks rows of the run's variants, or its
            inputs' cells joined in input order.
        inputs: The upstream-roots cell: its feature inputs' parts joined in
            input order, blank for a run whose only input is ``tracks``.
    """

    roots: str
    tracks: str
    inputs: str


OWN_STATE_PART: Final = "_state"
"""The part of a fit-state cell that holds the digest of the run's own state.

No params field takes the name, since a Pydantic field cannot begin with an
underscore.
"""


def fit_state_cell(states: Mapping[str, str]) -> str:
    """The ``consumed_model_state`` cell of a row that read *states*.

    Args:
        states: Each fit the row read, keyed by its params field, and the run's
            own state under :data:`OWN_STATE_PART`, each mapped to its digest.

    Returns:
        A bare digest for one part, sorted ``field=digest`` pairs for several,
        and ``""`` when every part is blank.
    """
    from .sequence_index import encode_entry_composition

    return encode_entry_composition(states, states.keys())


def fit_state_root(ds: Dataset, reference: FitReference) -> Path | None:
    """The run root a fit reference loads its state from, in lookup order.

    For a reference with a pattern, the first root of :func:`feature_run_roots`
    in which :func:`resolve_artifact_file` finds it. For one without, the first
    root with the fit's ``fit_scope.json``. A fit writes that file after it
    saves its state.

    Returns:
        The root, or ``None`` when no root holds the state.

    Raises:
        AmbiguousArtifactFile: The pattern matches more than one file of a root.
            The message names the reference's field.
    """
    from mosaic.core.pipeline.fit_scope import FIT_SCOPE_NAME

    if reference.pattern is None:
        fit_scope = feature_run_file(
            ds, reference.feature, reference.run_id, FIT_SCOPE_NAME
        )
        return None if fit_scope is None else fit_scope.parent
    for root in feature_run_roots(ds, reference.feature, reference.run_id):
        found = resolve_artifact_file(
            reference.field, reference.feature, root, reference.pattern
        )
        if found is not None:
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
        model_state: Its ``consumed_model_state``, empty for a row that read no
            fit state.
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
    Two loops over the same three cells would be two answers to the delete set's
    walk. ``encode_entry_composition`` already makes the same argument one level
    down.

    An **absent** entry is not one recording ``""``. The first has no row at all:
    an output on disk that nothing describes, whose provenance cannot be stated.
    The second is a row written before the column existed, or one whose root was
    not establishable. Neither is evidence of change; both are for the delete set
    to fail closed on. Entries declaring no root are dropped here, having nothing
    to say.
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
        strict: :meth:`fit_record` raises for a record that cannot be read
            instead of reading it as none.
    """

    def __init__(
        self,
        ds: Dataset,
        feature_frame: Callable[[Path], pd.DataFrame] = _read_feature_frame,
        *,
        strict: bool = False,
    ) -> None:
        self._ds: Dataset = ds
        self._feature_frame: Callable[[Path], pd.DataFrame] = feature_frame
        self._strict: bool = strict
        self._fit_records: dict[
            tuple[str, str], tuple[str, FitScope | None] | None
        ] = {}
        self._runs: dict[str, dict[str, pd.DataFrame]] = {}
        self._tracks: dict[tuple[str, ...], dict[Entry, str]] = {}
        self._held: dict[tuple[str, ...], frozenset[Entry]] = {}
        self._arena_lookup: ArenaLookup | None = None
        self._recorded: dict[Entry, dict[str, str]] | None = None
        self._fit_states: dict[FitReference, str] = {}

    @property
    def dataset(self) -> Dataset:
        """The dataset whose indexes are read."""
        return self._ds

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

    def fit_state(self, reference: FitReference) -> str:
        """The fit-state digest of the fit *reference* names, read once per fit.

        Read from the root :func:`fit_state_root` resolves, through the
        datasets the dataset links. A fit that no reachable root holds, a
        ``fit_scope.json`` that cannot be read and a pattern that matches more
        than one file of a root (:class:`AmbiguousArtifactFile`) read as blank:
        a status read does not raise.

        Returns:
            The digest, and ``""`` when none is recorded or readable.
        """
        from .fit_scope import FitStateUnreadable, read_fit_state_digest

        digest = self._fit_states.get(reference)
        if digest is None:
            try:
                root = fit_state_root(self._ds, reference)
                digest = "" if root is None else read_fit_state_digest(root)
            except (AmbiguousArtifactFile, FitStateUnreadable):
                digest = ""
            self._fit_states[reference] = digest
        return digest

    def fit_record(self, storage: str, run_id: str) -> FitScope | None:
        """The ``fit_scope.json`` of a run, read once per run.

        Read through :meth:`located_fit_record`.

        Returns:
            The record, and ``None`` when no dataset holds one, or, unless the
            reads are strict, when it cannot be read.

        Raises:
            FitStateUnreadable: The reads are strict and the record cannot be
                read.
        """
        located = self.located_fit_record(storage, run_id)
        return None if located is None else located[1]

    def located_fit_record(
        self, storage: str, run_id: str
    ) -> tuple[str, FitScope | None] | None:
        """The ``fit_scope.json`` of a run and the link it is under, read once.

        Read through :func:`~.fit_scope.read_located_fit_scope`, in the first
        dataset of the lookup that holds one.

        Returns:
            The link id of the dataset holding the record, ``""`` for the
            reading dataset, and the record, ``None`` when it cannot be read
            and the reads are not strict. ``None`` when no dataset holds one.

        Raises:
            FitStateUnreadable: The reads are strict and the record cannot be
                read.
        """
        from .fit_scope import read_located_fit_scope

        key = (storage, run_id)
        if key not in self._fit_records:
            self._fit_records[key] = read_located_fit_scope(
                self._ds, storage, run_id, strict=self._strict
            )
        return self._fit_records[key]

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
    fit_drifted: bool | None = None,
) -> tuple[tuple[str, str], ...]:
    """Entries whose sources have moved since this run recorded what it consumed.

    Each entry's row is compared, channel by channel, with what the dataset
    records now, under :func:`~.composition.cached_entry_disposition`, the rule
    ``run_feature`` applies before it serves a cached entry. An entry is
    drifted when any channel says ``recompute``. The entries reported are those
    a run would recompute:

    - its declared roots: the row's ``consumed_composition`` against the entry's
      composition under those roots in ``<root>/sequences.csv``, and its layout
      digest when a row or the run declares ``arenas``
      (:func:`~.arena_index.entry_compositions`);
    - its tracks: the row's ``consumed_tracks_composition`` against the tracks
      rows of the run's variants (:func:`~.tracks_index.tracks_compositions`) for
      a run whose only input is ``tracks``, or against its inputs' cells joined
      in input order (:func:`~.composition.join_input_cells`), a feature input's
      cell being its upstream row's ``consumed_tracks_composition``;
    - its upstreams' roots and fit states: the row's
      ``consumed_input_composition`` against its feature inputs' parts
      (:func:`~.composition.input_part` of each upstream row's declared roots,
      inputs cell and ``consumed_model_state``), joined the same way, which
      recomputes an entry after a fit its upstream read was fitted again;
    - its fits: the row's ``consumed_model_state`` against the cell of
      :func:`fit_state_cell` over the digest each fit reference of the run
      records now (:meth:`DriftReads.fit_state`), and the run's own digest
      under :data:`OWN_STATE_PART` when the run holds a fit record
      (:meth:`DriftReads.fit_record`). A fit under an unreachable link, or
      whose record cannot be read, reads as blank.

    The neighbors cell is compared by ``run_feature`` alone. An entry the run's
    inputs no longer resolve, dropped from the tracks index for instance, is
    not reported: a run would recompute nothing for it. A run that intersects
    its inputs resolves an entry every input holds, and a stacked run one that
    any input holds. A ``params.json`` written before it recorded stacking
    counts an entry any input holds: a false drift there lasts one run, which
    records the stacking, while counting only the entries every input holds
    misses the drift of a stacked run that nothing then recomputes.

    **A drifted fit reports every entry its rows hold.** A run whose fit has
    drifted (:func:`fit_drift`) is fitted again, which recomputes every entry
    it is asked for. The comparison above is skipped and each entry of its
    rows is reported, whether or not its inputs still resolve it.

    **Both sides are recorded values.** A value recomputed from the present
    would always agree with itself. Reading recorded values also keeps a status
    display free of any filesystem measurement.

    **One known side is drift**, and two blank sides are not. A row written
    before its cell existed against a present that records one is recomputed
    once, after which the row records the present. Two blanks cannot converge:
    the run serves them with a warning, and reporting them here would ask for a
    recompute that changes nothing.

    Args:
        ds: The dataset holding the run.
        feature_name: The run's feature storage.
        run_id: The run.
        run_params: The run's ``params.json``, which names its inputs and tracks
            variants. ``None`` compares the declared roots alone.
        reads: The index reads shared with other calls, or ``None`` to read
            them for this call alone.
        fit_drifted: Whether the run's fit has drifted, for a caller that
            measured it with :func:`fit_drift`, or ``None`` to measure it here.

    Returns:
        The drifted entries, sorted.
    """
    shared = DriftReads(ds) if reads is None else reads
    recorded_roots = shared.consumption(feature_name, run_id)
    recorded_reads = shared.tracks_reads(feature_name, run_id) if run_params else {}
    fits_now = _current_fit_cell(shared, feature_name, run_id, run_params)
    entries = sorted(set(recorded_roots) | set(recorded_reads))
    if not entries:
        return ()
    if fit_drifted is None:
        fit = fit_drift(ds, feature_name, run_id, run_params, shared)
        fit_drifted = fit is not None and fit.drifted
    if fit_drifted:
        return tuple(entries)
    declared = tuple(run_params.scope.consumed_roots) if run_params else ()
    cells_now, held = current_entry_cells(
        shared,
        run_params,
        entries,
        recorded_roots={entry: roots for entry, (roots, _) in recorded_roots.items()},
    )

    drifted: list[tuple[str, str]] = []
    for entry in entries:
        if entry not in held:
            continue
        roots, was = recorded_roots.get(entry, (declared, ""))
        cells = cells_now[entry]
        channels = [(was, cells.roots) if roots else ("", "")]
        if run_params is not None:
            read = recorded_reads.get(entry, RecordedTracksReads())
            channels.append((read.composition, cells.tracks))
            channels.append((read.inputs, cells.inputs))
            channels.append((read.model_state, fits_now))
        if any(
            cached_entry_disposition(recorded, current) == "recompute"
            for recorded, current in channels
        ):
            drifted.append(entry)
    return tuple(drifted)


def current_entry_cells(
    reads: DriftReads,
    run_params: RunParams | None,
    entries: Sequence[Entry],
    *,
    recorded_roots: Mapping[Entry, Sequence[str]] | None = None,
) -> tuple[dict[Entry, EntryCells], frozenset[Entry]]:
    """The cells each of *entries* would record for a run of *run_params* now.

    The one construction of an entry's cells from the dataset's indexes, built
    as ``build_manifest`` builds them: the entry's composition under its
    declared roots, the tracks cell and the upstream-roots cell
    (:class:`EntryCells`). The comparison of :func:`drifted_entries` and the
    record of a fit's inputs both read it.

    A run whose only input is ``tracks`` compares its variants' tracks rows and
    no upstream roots. A run with a feature input joins each input's part in
    input order, a ``tracks`` input's roots part being blank. An entry no input
    holds has blank tracks and upstream-roots cells.

    Args:
        reads: The index reads shared with other calls.
        run_params: The run's ``params.json``, which names its declared roots,
            inputs and tracks variants. ``None`` builds the roots cell alone,
            under *recorded_roots*, and holds every entry.
        entries: The entries to build the cells of.
        recorded_roots: The roots an entry's row recorded, which replace the
            declared roots for that entry. An entry it leaves out is built
            under the declared roots.

    Returns:
        The cells of each of *entries*, and the entries the run's inputs still
        resolve: those every input holds a row for, or any input for a stacked
        run or one whose file does not say, whether or not the row records a
        composition. An entry outside it has no input to read.
    """
    from .arena_index import ARENAS_ROOT, entry_compositions
    from .sequence_index import encode_entry_composition

    declared = tuple(run_params.scope.consumed_roots) if run_params else ()
    recorded: Mapping[Entry, Sequence[str]] = (
        {} if recorded_roots is None else recorded_roots
    )
    in_play = set(declared).union(*recorded.values())
    lookup = reads.arena_lookup() if ARENAS_ROOT in in_play else None
    compositions = entry_compositions(
        reads.dataset,
        entries,
        in_play,
        lookup=lookup,
        recorded=reads.recorded_compositions(),
    )
    tracks_now, inputs_now, held = (
        _current_input_cells(reads, run_params)
        if run_params
        else ({}, {}, frozenset(entries))
    )
    cells: dict[Entry, EntryCells] = {}
    for entry in entries:
        roots = tuple(recorded.get(entry, declared))
        cells[entry] = EntryCells(
            roots=encode_entry_composition(compositions.get(entry, {}), roots)
            if roots
            else "",
            tracks=tracks_now.get(entry, ""),
            inputs=inputs_now.get(entry, ""),
        )
    return cells, held


def _current_fit_cell(
    reads: DriftReads, storage: str, run_id: str, run_params: RunParams | None
) -> str:
    """The fit-state cell a run of *storage* would record for a row now.

    Built as ``run_feature`` builds it: each fit reference of the run's
    ``params.json`` with the digest its fit records now, and the run's own
    digest when the run holds a fit record. ``""`` without params.
    """
    if run_params is None:
        return ""
    states = {
        reference.field: reads.fit_state(reference)
        for reference in run_params.fit_references()
    }
    own = reads.fit_record(storage, run_id)
    if own is not None:
        states[OWN_STATE_PART] = own.state_digest
    return fit_state_cell(states)


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
                entry: input_part(
                    upstream_roots.get(entry, ((), ""))[1],
                    read.inputs,
                    read.model_state,
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


@dataclass(frozen=True)
class FitDrift:
    """How a fit's recorded inputs compare with what the dataset holds now.

    Attributes:
        link_id: The link of the dataset that holds the fit's record, and
            ``""`` for the reading dataset.
        recorded: Whether the record holds the inputs of the fit. A fit saved
            before it did holds none and does not read as drifted.
        moved: The recorded entries whose cells moved, sorted.
        dropped: The training entries the run's inputs no longer hold, sorted.
            Always empty for a run whose entry set is in its run id.
        refitted: The params fields whose fit records another state than the
            fit read, sorted.
    """

    link_id: str
    recorded: bool
    moved: tuple[Entry, ...]
    dropped: tuple[Entry, ...]
    refitted: tuple[str, ...]

    @property
    def drifted(self) -> bool:
        """Whether the fit read inputs that have changed since."""
        return self.recorded and bool(self.moved or self.dropped or self.refitted)


def fit_drift(
    ds: Dataset,
    storage: str,
    run_id: str,
    run_params: RunParams | None,
    reads: DriftReads | None = None,
) -> FitDrift | None:
    """Compare the inputs a run's fit recorded with what the dataset holds now.

    The run's ``fit_scope.json`` is read from the first dataset of the lookup
    that holds it (:meth:`DriftReads.located_fit_record`). Each recorded entry
    the run's inputs still hold has its cells compared with those
    :func:`current_entry_cells` builds now, and each fit the run read has the
    digest it recorded compared with the one that fit records now
    (:meth:`DriftReads.fit_state`), each under
    :func:`~.composition.cached_entry_disposition`: a side blank on both is
    not drift, and one known side is.

    A recorded entry the inputs no longer hold is dropped for a run whose entry
    set is not in its run id, a fit pinned to its training set. A
    scope-dependent run resolves a new run id once an entry leaves, and the
    entry is left out of its comparison.

    Both sides are recorded values. The read writes nothing, takes no lock and
    imports no feature definition.

    Args:
        ds: The dataset reading.
        storage: The run's feature storage.
        run_id: The run.
        run_params: The run's ``params.json``, which names its inputs, its fit
            references and whether its scope is in its run id. ``None``
            compares the fits read alone.
        reads: The index reads shared with other calls, or ``None`` to read
            them for this call alone. Strict reads raise for a record that
            cannot be read.

    Returns:
        The drift, or ``None`` when no dataset holds a readable record of the
        run.

    Raises:
        FitStateUnreadable: *reads* are strict and the record cannot be read.
    """
    shared = DriftReads(ds) if reads is None else reads
    located = shared.located_fit_record(storage, run_id)
    if located is None:
        return None
    link_id, record = located
    if record is None:
        return None
    references = (
        {reference.field: reference for reference in run_params.fit_references()}
        if run_params
        else {}
    )
    refitted = {
        read.field
        for read in record.via
        if cached_entry_disposition(
            read.state_digest,
            shared.fit_state(
                references.get(
                    read.field,
                    FitReference(
                        field=read.field,
                        feature=read.feature,
                        run_id=read.run_id,
                        pattern=None,
                    ),
                )
            ),
        )
        == "recompute"
    }
    moved: set[Entry] = set()
    dropped: set[Entry] = set()
    if record.inputs and run_params is not None:
        recorded = {(item.group, item.sequence): item.cells for item in record.inputs}
        cells_now, held = current_entry_cells(shared, run_params, sorted(recorded))
        for entry, was in recorded.items():
            if entry not in held:
                if not run_params.scope.scope_dependent:
                    dropped.add(entry)
                continue
            now = cells_now[entry]
            channels = (
                (was.roots, now.roots),
                (was.tracks, now.tracks),
                (was.inputs, now.inputs),
            )
            if any(
                cached_entry_disposition(before, after) == "recompute"
                for before, after in channels
            ):
                moved.add(entry)
    return FitDrift(
        link_id=link_id,
        recorded=record.inputs is not None,
        moved=tuple(sorted(moved)),
        dropped=tuple(sorted(dropped)),
        refitted=tuple(sorted(refitted)),
    )


def refit_sample(drift: FitDrift, *, sample: int = 5) -> tuple[str, ...]:
    """The entries of *drift* a summary names: up to *sample*, moved before dropped.

    Each is named by :func:`~mosaic.core.entry.entry_token` and each group is
    sorted.
    """
    entries = [*sorted(drift.moved), *sorted(drift.dropped)]
    return tuple(
        entry_token(group, sequence) for group, sequence in entries[: max(sample, 0)]
    )


def describe_refit(
    moved: int, dropped: int, refitted: Iterable[str], named: Sequence[str]
) -> str:
    """One line saying what changed under a fit, from its counts and its sample.

    ``<moved> moved, <dropped> gone, fitted again: <fields>``, a zero count and
    an empty field list left out, then the entries *named* in parentheses with
    the count of those left out.

    Args:
        moved: How many recorded entries moved.
        dropped: How many training entries the inputs no longer hold.
        refitted: The params fields whose fit was fitted again.
        named: The entries to name, as :func:`refit_sample` lists them.

    Returns:
        The summary, and ``""`` when nothing changed.
    """
    parts: list[str] = []
    if moved:
        parts.append(f"{moved} moved")
    if dropped:
        parts.append(f"{dropped} gone")
    fields = sorted(refitted)
    if fields:
        parts.append(f"fitted again: {', '.join(fields)}")
    summary = ", ".join(parts)
    if not named:
        return summary
    rest = moved + dropped - len(named)
    listed = ", ".join(named) + (f" and {rest} more" if rest else "")
    return f"{summary} ({listed})"


def refit_summary(drift: FitDrift, *, sample: int = 5) -> str:
    """One line saying what changed under a fit.

    :func:`describe_refit` over the counts of *drift* and up to *sample* of its
    entries.

    Args:
        drift: The fit's drift.
        sample: How many entries to name at most.

    Returns:
        The summary, and ``""`` for a drift that names nothing.
    """
    return describe_refit(
        len(drift.moved),
        len(drift.dropped),
        drift.refitted,
        refit_sample(drift, sample=sample),
    )


# Registered with the shared registry beside the tracker indexes. One reconcile
# pass then covers every root that has an ``IndexCSV`` behind it.
register_reconcilable_index("features", feature_index)
