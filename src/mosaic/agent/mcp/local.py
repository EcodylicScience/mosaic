"""The tool backend that answers mosaic's shared tools by calling the library here.

:class:`LocalBackend` serves the datasets of an allowlist, each addressed by its
manifest ``uuid``. It opens every dataset without creating its roots, keeps one
:class:`~mosaic.core.pipeline.inventory.InventoryCache` per dataset behind a
lock, and builds the step-type catalog on a background thread. A submission is
recorded in the dataset and run by a detached driver
(:func:`~mosaic.core.pipeline.graph.start_detached_driver`), which outlives the
server.

Every tool that reads a catalog or a recipe runs in the definition scope of the
dataset it names, which never loads a definition's code. On a backend built
without ``allow_feature_definitions`` that scope is disabled: the catalogs list
no ``def:`` step type, and a recipe naming one is refused before it is validated.

Two rules guard the tools that change a dataset. A dataset whose manifest names
a managing service (``managed_by``) refuses them unless the backend was built
with ``allow_managed``. ``submit_pipeline`` and ``cancel_request`` refuse a call
that meets a confirmation condition of their
:class:`~mosaic.agent.mcp.spec.ToolSpec` until it is repeated with ``confirm``.

The module imports the library inside the methods that use it, so building a
backend loads neither pandas nor the feature library.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from mosaic.core.media.division_wire import DivisionReportWire
from mosaic.core.pipeline.error_envelope import ErrorEnvelope, error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError, ErrorKind
from mosaic.core.pipeline.parallelism import ParallelLimits, ParallelRefused
from mosaic.core.scan_findings import ScanFinding, missing_path_finding
from mosaic.core.strict_model import terse

from .dataset_tools import missing_field
from .spec import (
    LISTED_STATUS_NOT,
    LISTING_MISSING_SAMPLE,
    MAX_STEP_TYPE_LISTING_CHARS,
    PLAN_MISSING_SAMPLE,
    PLAN_RETIRABLE_SAMPLE,
    PLAN_RETIRED_SAMPLE,
    PLAN_SUPERSEDED_SAMPLE,
    SUBMITTED_STEP_SAMPLE,
    ActivityPage,
    AttemptDiagnosis,
    CancelRequestInput,
    ConfirmationFinding,
    DatasetChange,
    DatasetList,
    DatasetListItem,
    DescribeDatasetInput,
    DescribeStepTypesInput,
    DiagnoseFailuresInput,
    FailureDiagnosis,
    GetInventoryInput,
    GetRequestStatusInput,
    ListActivityInput,
    ListDatasetsInput,
    ListSequencesInput,
    ManageDatasetInput,
    PatchRecipeInput,
    PlanRecipeInput,
    PreviewOutputInput,
    RunSummary,
    ScopeSelector,
    SequencePage,
    SequenceRow,
    SubmitPipelineInput,
    SubmittedDuplicate,
    SubmittedPipeline,
    SubmittedStep,
    StepTypeDetailLevel,
    SuggestStepsInput,
    ToolName,
    ToolResult,
    ValidateRecipeInput,
    cleanup_not_allowed,
    confirmation_required,
    dataset_managed,
    feature_definitions_not_allowed,
    invalid_cursor,
    unknown_dataset,
    validation_errors,
)

if TYPE_CHECKING:
    from mosaic.core.annotations.sequence_tag_wire import SequenceTagImport
    from mosaic.core.dataset import Dataset
    from mosaic.core.dataset_summary import DatasetSummary
    from mosaic.core.manifest import DatasetManifest
    from mosaic.core.media.store_division import DivisionRefused
    from mosaic.core.pipeline.cancel import RequestCancel
    from mosaic.core.pipeline.definitions.store import DefinitionRecord
    from mosaic.core.pipeline.graph import (
        CatalogEntry,
        CatalogIndex,
        CleanupPolicy,
        DeclarationCatalog,
        DetachedDriver,
        Plan,
        PlannedStep,
        PlanWire,
        Recipe,
        RecordedErrorWire,
        Request,
        RequestStatusWire,
        ValidationReport,
    )
    from mosaic.core.pipeline.inventory import InventoryCache, InventoryWire
    from mosaic.core.pipeline.label_series import LabelRevisionRef
    from mosaic.core.pipeline.parallelism import ParallelSettings
    from mosaic.core.pipeline.preview import ArtifactPreview
    from mosaic.core.pipeline.run_summary import RunSummaryWire
    from mosaic.core.pipeline.tag_reads import ResolvedTagTable
    from mosaic.core.scope import Scope
    from mosaic.core.selections import SelectionSummary
    from mosaic.runlog import AttemptOutcome, RunLogSnapshot

    from .config import CleanupClass
    from .outputs import PatchedRecipe, PlanPage, StepSuggestions, StepTypes

__all__ = [
    "POLL_AFTER_MIN_SECONDS",
    "POLL_AFTER_SECONDS",
    "POLL_AFTER_TRAINING_SECONDS",
    "DriverStarter",
    "LocalBackend",
    "ToolFailure",
    "answer",
    "discovered_manifests",
]

_LOG = logging.getLogger(__name__)

POLL_AFTER_MIN_SECONDS: Final = 10
"""How long a client waits before polling a submitted request.

The shortest wait a request status reports
(``mosaic.core.pipeline.graph.POLL_AFTER_MIN_SECONDS``), which a test keeps equal.
"""

POLL_AFTER_SECONDS: Final = POLL_AFTER_MIN_SECONDS
"""The earlier name of :data:`POLL_AFTER_MIN_SECONDS`."""

POLL_AFTER_TRAINING_SECONDS: Final = 60
"""How long a client waits before polling a request that trains on a GPU."""

_EVALUATE_LABELS: Final = "evaluate-labels"
"""The one feature that scores a model's output against labels."""

type DriverStarter = Callable[
    [Dataset, Request, ParallelSettings, tuple[str, ...] | None], DetachedDriver
]
"""What starts the process driving a submitted request.

It is given the request's parallelism and the cards the operator allows its GPU
steps, ``None`` for every visible card.
"""


class ToolFailure(Exception):
    """A tool's failure, as the envelope the backend returns.

    :func:`answer` turns it into the envelope. A reader of the served datasets
    beside the tools, such as a server's resources, raises it the same way.
    """

    def __init__(self, envelope: ErrorEnvelope) -> None:
        super().__init__(envelope.message)
        self.envelope: ErrorEnvelope = envelope


@dataclass
class _Served:
    """One dataset of the allowlist, opened without creating its roots.

    Attributes:
        uuid: The manifest's ``uuid``.
        manifest: The manifest file.
        dataset: The dataset, opened once.
        lock: Held while the inventory cache is read or refreshed. The cache is
            not thread-safe.
        cache: The dataset's inventory, built on first use.
    """

    uuid: str
    manifest: Path
    dataset: Dataset
    lock: threading.Lock = field(default_factory=threading.Lock)
    cache: InventoryCache | None = None


class LocalBackend:
    """Answer the shared tools over the datasets of an allowlist, in this process.

    A manifest of the allowlist that does not open, has no ``uuid`` or repeats
    the ``uuid`` of an earlier one is left out and logged. The datasets are
    opened on first use, or on the warming thread.
    """

    def __init__(
        self,
        manifests: Iterable[Path],
        *,
        allow_managed: bool = False,
        warm: bool = True,
        start_driver: DriverStarter | None = None,
        allow_cleanup: frozenset[CleanupClass] = frozenset(),
        cleanup_policy: CleanupPolicy | None = None,
        parallel_limits: ParallelLimits | None = None,
        gpu_cards: Sequence[str] | None = None,
        dataset_roots: Sequence[Path] = (),
        source_roots: Sequence[Path] = (),
        recipe_roots: Sequence[Path] = (),
        allow_feature_definitions: bool = False,
    ) -> None:
        """Serve the datasets of *manifests*.

        Args:
            manifests: The manifest files, or the directories holding them.
            allow_managed: Whether the tools that change a dataset may change one
                that a service manages.
            allow_cleanup: The artifacts the tools may remove. With
                ``feature-runs``, ``submit_pipeline`` removes superseded feature
                runs when called with ``cleanup: superseded``.
            cleanup_policy: Which superseded runs a cleanup keeps, fixed by the
                operator. The default policy when ``None``.
            parallel_limits: The parallelism a submission may ask for, fixed by
                the operator. One worker per physical core, in threads,
                above a floor of 10% available memory, when ``None``. A submission
                may name ``process`` workers.
            gpu_cards: The cards a submission's drivers may lease to GPU steps,
                as ``nvidia-smi`` indexes or UUIDs, fixed by the operator.
                Every visible card when ``None``.
            warm: Whether to open the datasets and build the step-type catalog
                on a background thread now. Otherwise the first call that needs
                each builds it.
            start_driver: What starts the driver of a submitted request.
                :func:`~mosaic.core.pipeline.graph.start_detached_driver` when
                ``None``.
            dataset_roots: The directories inside which ``manage_dataset`` may
                create a dataset, fixed by the operator. A dataset found below
                one is served, and one created while serving is served at once.
            source_roots: The directories inside which ``manage_dataset`` may
                declare a source, fixed by the operator.
            recipe_roots: The directories inside which a recipe argument
                ``{"path": ...}`` is read, fixed by the operator.
            allow_feature_definitions: Whether the tools read and run the
                datasets' feature definitions and ``manage_dataset`` defines one,
                fixed by the operator. Off, every call runs under a disabled
                definition scope.
        """
        self._manifests: tuple[Path, ...] = tuple(manifests)
        self._allow_managed: bool = allow_managed
        self._allow_cleanup: frozenset[CleanupClass] = allow_cleanup
        self._cleanup_policy: CleanupPolicy | None = cleanup_policy
        self._parallel_limits: ParallelLimits = (
            ParallelLimits() if parallel_limits is None else parallel_limits
        )
        self._start_driver: DriverStarter | None = start_driver
        self._gpu_cards: tuple[str, ...] | None = (
            None if gpu_cards is None else tuple(gpu_cards)
        )
        self._dataset_roots: tuple[Path, ...] = tuple(
            root.resolve() for root in dataset_roots
        )
        self._source_roots: tuple[Path, ...] = tuple(
            root.resolve() for root in source_roots
        )
        self._recipe_roots: tuple[Path, ...] = tuple(
            root.resolve() for root in recipe_roots
        )
        self._allow_feature_definitions: bool = allow_feature_definitions
        self._served: dict[str, _Served] | None = None
        self._looked_at: set[Path] = set()
        self._served_lock: threading.Lock = threading.Lock()
        self._catalog_lock: threading.Lock = threading.Lock()
        self._contributors_registered: bool = False
        self._warmer: threading.Thread | None = None
        if warm:
            self._warmer = threading.Thread(
                target=self._warm, name="mosaic-mcp-warm", daemon=True
            )
            self._warmer.start()

    @classmethod
    def discover(
        cls,
        root: Path,
        *,
        max_depth: int | None = None,
        allow_managed: bool = False,
        warm: bool = True,
        start_driver: DriverStarter | None = None,
        allow_cleanup: frozenset[CleanupClass] = frozenset(),
        cleanup_policy: CleanupPolicy | None = None,
        parallel_limits: ParallelLimits | None = None,
        gpu_cards: Sequence[str] | None = None,
        allow_feature_definitions: bool = False,
    ) -> LocalBackend:
        """Serve the datasets found in *root* and below it.

        A manifest that does not read or has no ``uuid`` is left out and logged.

        Args:
            root: The directory to search.
            max_depth: How many levels below *root* to search, the discovery
                default when ``None``.
            allow_managed: As for :class:`LocalBackend`.
            warm: As for :class:`LocalBackend`.
            start_driver: As for :class:`LocalBackend`.
            allow_cleanup: As for :class:`LocalBackend`.
            cleanup_policy: As for :class:`LocalBackend`.
            parallel_limits: As for :class:`LocalBackend`.
            gpu_cards: As for :class:`LocalBackend`.
            allow_feature_definitions: As for :class:`LocalBackend`.
        """
        manifests = discovered_manifests(root, max_depth=max_depth)
        return cls(
            manifests,
            allow_managed=allow_managed,
            warm=warm,
            start_driver=start_driver,
            allow_cleanup=allow_cleanup,
            cleanup_policy=cleanup_policy,
            parallel_limits=parallel_limits,
            gpu_cards=gpu_cards,
            allow_feature_definitions=allow_feature_definitions,
        )

    def dataset(self, uuid: str) -> Dataset:
        """Return the served dataset *uuid*, as opened without creating roots.

        Raises:
            ToolFailure: The allowlist names no dataset *uuid*.
        """
        return self._dataset(uuid).dataset

    @contextmanager
    def definitions(self, dataset: Dataset | None) -> Generator[None]:
        """Run the block in the definition scope a call on *dataset* runs in.

        The scope never loads a definition's code, and it is disabled unless the
        backend allows feature definitions. A caller's disabled scope stays
        disabled. A resource read of a served dataset enters it as a tool does.
        """
        from mosaic.core.pipeline.definitions.scope import definition_scope

        with definition_scope(
            dataset, load=False, allowed=self._allow_feature_definitions
        ):
            yield

    def catalog(self) -> CatalogIndex:
        """Return the step-type catalog index, building it once across threads."""
        return self._catalog_index()

    def wait_until_warm(self, timeout: float | None = None) -> None:
        """Block until the warming thread, if any, has finished.

        A summary-mode ``preview_output`` waits here before it imports scipy.
        """
        if self._warmer is not None:
            self._warmer.join(timeout)

    # ------------------------------------------------------------------ tools

    def list_datasets(self, args: ListDatasetsInput) -> ToolResult[DatasetList]:
        """Answer ``list_datasets``."""
        return answer(lambda: self._list_datasets(args))

    def describe_dataset(
        self, args: DescribeDatasetInput
    ) -> ToolResult[DatasetSummary]:
        """Answer ``describe_dataset``."""
        return answer(lambda: self._describe_dataset(args))

    def list_sequences(self, args: ListSequencesInput) -> ToolResult[SequencePage]:
        """Answer ``list_sequences``."""
        return answer(lambda: self._list_sequences(args))

    def get_inventory(self, args: GetInventoryInput) -> ToolResult[InventoryWire]:
        """Answer ``get_inventory``."""
        return answer(lambda: self._get_inventory(args))

    def describe_step_types(
        self, args: DescribeStepTypesInput
    ) -> ToolResult[StepTypes]:
        """Answer ``describe_step_types``."""
        return answer(
            lambda: self._scoped(
                "describe_step_types",
                args.dataset,
                lambda: self._describe_step_types(args),
            )
        )

    def validate_recipe(
        self, args: ValidateRecipeInput
    ) -> ToolResult[ValidationReport]:
        """Answer ``validate_recipe``."""
        return answer(
            lambda: self._scoped(
                "validate_recipe", args.dataset, lambda: self._validate_recipe(args)
            )
        )

    def suggest_steps(self, args: SuggestStepsInput) -> ToolResult[StepSuggestions]:
        """Answer ``suggest_steps``."""
        return answer(
            lambda: self._scoped(
                "suggest_steps", args.dataset, lambda: self._suggest_steps(args)
            )
        )

    def patch_recipe(self, args: PatchRecipeInput) -> ToolResult[PatchedRecipe]:
        """Answer ``patch_recipe``."""
        return answer(
            lambda: self._scoped(
                "patch_recipe", args.dataset, lambda: self._patch_recipe(args)
            )
        )

    def plan_recipe(self, args: PlanRecipeInput) -> ToolResult[PlanPage]:
        """Answer ``plan_recipe``."""
        return answer(
            lambda: self._scoped(
                "plan_recipe", args.dataset, lambda: self._plan_recipe(args)
            )
        )

    def submit_pipeline(
        self, args: SubmitPipelineInput
    ) -> ToolResult[SubmittedPipeline]:
        """Answer ``submit_pipeline``."""
        return answer(
            lambda: self._scoped(
                "submit_pipeline", args.dataset, lambda: self._submit_pipeline(args)
            )
        )

    def get_request_status(
        self, args: GetRequestStatusInput
    ) -> ToolResult[RequestStatusWire]:
        """Answer ``get_request_status``."""
        return answer(lambda: self._get_request_status(args))

    def cancel_request(self, args: CancelRequestInput) -> ToolResult[RequestCancel]:
        """Answer ``cancel_request``."""
        return answer(lambda: self._cancel_request(args))

    def list_activity(self, args: ListActivityInput) -> ToolResult[ActivityPage]:
        """Answer ``list_activity``."""
        return answer(lambda: self._list_activity(args))

    def diagnose_failures(
        self, args: DiagnoseFailuresInput
    ) -> ToolResult[FailureDiagnosis]:
        """Answer ``diagnose_failures``."""
        return answer(lambda: self._diagnose_failures(args))

    def preview_output(
        self, args: PreviewOutputInput
    ) -> ToolResult[ArtifactPreview | RunSummaryWire]:
        """Answer ``preview_output``."""
        return answer(lambda: self._preview_output(args))

    def manage_dataset(self, args: ManageDatasetInput) -> ToolResult[DatasetChange]:
        """Answer ``manage_dataset``."""
        return answer(
            lambda: self._scoped(
                "manage_dataset", args.dataset, lambda: self._manage_dataset(args)
            )
        )

    def _scoped[T](
        self, tool: ToolName, dataset: str | None, compute: Callable[[], T]
    ) -> T:
        """Return *compute*'s result, computed in the definition scope of *dataset*.

        On a backend that does not allow feature definitions, a definition that
        the call names is refused with the operator's setting named.

        Raises:
            ToolFailure: *dataset* is not served, or the call names a definition
                that this backend does not allow.
        """
        from mosaic.core.pipeline.definitions.store import (
            FeatureDefinitionsNotAllowed,
        )

        served = None if dataset is None else self._dataset(dataset).dataset
        try:
            with self.definitions(served):
                return compute()
        except FeatureDefinitionsNotAllowed as refused:
            if self._allow_feature_definitions:
                raise
            raise ToolFailure(
                feature_definitions_not_allowed(tool, refused.slug)
            ) from refused

    # ------------------------------------------------------------- datasets

    def _list_datasets(self, args: ListDatasetsInput) -> DatasetList:
        from mosaic.core.pipeline.inventory import reportable_universe

        self._serve_new_datasets()
        current = [
            (held, _current_manifest(held)) for held in self._datasets().values()
        ]
        page, next_cursor = _page(
            current,
            lambda listed: (listed[1].name, listed[0].uuid),
            args.limit,
            args.cursor,
        )
        items = tuple(
            DatasetListItem(
                uuid=held.uuid,
                name=manifest.name,
                n_sequences=len(reportable_universe(held.dataset)),
                api_managed=manifest.is_managed,
            )
            for held, manifest in page
        )
        return DatasetList(items=items, total=len(current), next_cursor=next_cursor)

    def _describe_dataset(self, args: DescribeDatasetInput) -> DatasetSummary:
        from mosaic.core.dataset import open_dataset
        from mosaic.core.dataset_summary import dataset_summary

        held = self._dataset(args.dataset)
        with held.lock:
            found = self._inventory(held).get()
        # Opened again so that the manifest fields are those on disk now.
        return dataset_summary(
            open_dataset(held.manifest, ensure_roots=False), found=found, raw=args.raw
        )

    def _list_sequences(self, args: ListSequencesInput) -> SequencePage:
        from mosaic.core.dataset import open_dataset
        from mosaic.core.pipeline.index_csv import index_records
        from mosaic.core.pipeline.tracks_index import (
            count_cell,
            read_frame_extent,
            read_tracks_index,
            select_variant_rows,
        )
        from mosaic.core.recording import division_placement, recording_name

        # Opened again so that each recording follows the declaration on disk now.
        dataset = open_dataset(self._dataset(args.dataset).manifest, ensure_roots=False)
        index = read_tracks_index(dataset)
        if args.group is not None:
            index = index[index["group"] == args.group]
        try:
            selected = select_variant_rows(index, args.tracks_variant)
        except ValueError as error:
            raise ToolFailure(
                ErrorEnvelope(
                    kind="invalid",
                    reason="variant_ambiguous",
                    message=str(error),
                    detail={"field": "tracks_variant"},
                )
            ) from error
        entries = dataset.recording_entries()
        placed = dataset.division_facts(entries)
        rows: list[SequenceRow] = []
        for record in index_records(selected):
            extent = read_frame_extent(record)
            group = record.get("group", "")
            sequence = record.get("sequence", "")
            rows.append(
                SequenceRow(
                    group=group,
                    sequence=sequence,
                    recording=recording_name(dataset.recording_of(group, sequence)),
                    tracks_variant=record.get("run_id", ""),
                    producer=record.get("producer", ""),
                    std_format=record.get("std_format", ""),
                    n_rows=count_cell(record.get("n_rows")),
                    n_keypoints=count_cell(record.get("n_keypoints")),
                    frame_min=None if extent is None else extent[0],
                    frame_max=None if extent is None else extent[1],
                    **division_placement(placed.get((group, sequence))),
                )
            )
        if args.tracks_variant is None:
            tracked = {(row.group, row.sequence) for row in rows}
            rows.extend(
                SequenceRow(
                    group=group,
                    sequence=sequence,
                    recording=recording_name(dataset.recording_of(group, sequence)),
                    tracks_variant=None,
                    producer=None,
                    std_format=None,
                    n_rows=None,
                    n_keypoints=None,
                    frame_min=None,
                    frame_max=None,
                    **division_placement(placed.get((group, sequence))),
                )
                for group, sequence in sorted(entries - tracked)
                if args.group is None or group == args.group
            )
        page, next_cursor = _page(
            rows, lambda row: (row.group, row.sequence), args.limit, args.cursor
        )
        return SequencePage(items=tuple(page), total=len(rows), next_cursor=next_cursor)

    def inventory_document(self, args: GetInventoryInput) -> InventoryWire:
        """Return the inventory that *args* names, each feature run with its params.

        Not a tool. The ``inventory`` resource reads it, and keeps what a tool's
        listing leaves to a ``ref``: the recorded params and the default sample
        of missing keys.
        """
        return self._get_inventory(args, whole=True)

    def _get_inventory(
        self, args: GetInventoryInput, *, whole: bool = False
    ) -> InventoryWire:
        from mosaic.core.pipeline.inventory import MISSING_SAMPLE_WIRE, inventory_wire

        detailed = whole or args.ref is not None

        held = self._dataset(args.dataset)
        with held.lock:
            found = self._inventory(held).get()
        return inventory_wire(
            held.dataset,
            found,
            kinds=args.kinds,
            scope=_entry_scope(held.dataset, args.entries),
            tracks_run_id=args.tracks_variant,
            status_not=args.status_not
            if args.status_not is not None or args.ref is not None
            else LISTED_STATUS_NOT,
            limit=args.limit,
            cursor=args.cursor,
            ref=args.ref,
            with_params=detailed,
            missing_sample=MISSING_SAMPLE_WIRE if detailed else LISTING_MISSING_SAMPLE,
        )

    # -------------------------------------------------------------- catalog

    def _describe_step_types(self, args: DescribeStepTypesInput) -> StepTypes:
        from mosaic.core.pipeline.graph import (
            catalog_listing,
            computes_in_track_space,
            search_catalog,
        )

        from .outputs import StepTypes

        index = self._catalog_index()
        level = args.level
        if args.names is not None:
            return self._named_step_types(args.names, level)
        searched = search_catalog(index.entries, args.query)
        matched = [
            match
            for match in searched
            if (args.type is None or match.entry.type == args.type)
            and (args.category is None or match.entry.category == args.category)
            and not (args.available_only and match.entry.unavailable)
            and (
                args.track_space is None
                or computes_in_track_space(match.entry, args.track_space)
            )
        ]
        page, next_cursor = _page(
            matched, lambda match: match.key, args.limit, args.cursor
        )
        return StepTypes(
            schema_version=index.schema_version,
            mosaic_version=index.mosaic_version,
            detail=level,
            listing=catalog_listing(
                (match.entry for match in page), MAX_STEP_TYPE_LISTING_CHARS
            ),
            total=len(matched),
            next_cursor=next_cursor,
            hint=None if searched else self._no_match_hint(),
        )

    def _no_match_hint(self) -> str:
        """Return the route to a measure that no step type computes."""
        route = (
            "manage_dataset with define_feature"
            if self._allow_feature_definitions
            else "propose mosaic features define to the user"
        )
        lead = "No step type matches these words. To compute the measure, "
        return f"{lead}write a feature definition: {route}."

    def _named_step_types(
        self, names: list[str], level: StepTypeDetailLevel
    ) -> StepTypes:
        """Describe each step type of *names*, listing those no step type has.

        Raises:
            ToolFailure: No name is a step type's.
        """
        from mosaic.core.pipeline.graph import StepTypeNotFound, step_type_detail

        from .outputs import StepTypes

        index = self._catalog_index()
        self._declarations()
        rows = {entry.name: entry for entry in index.entries}
        asked = list(dict.fromkeys(names))
        known = [name for name in asked if name in rows]
        unknown = tuple(name for name in asked if name not in rows)
        if not known:
            envelope = error_envelope(StepTypeNotFound(unknown[0]))
            raise ToolFailure(
                envelope.model_copy(
                    update={
                        "message": (
                            f"no step type named {', '.join(map(repr, unknown))}; "
                            "describe_step_types with a query finds them"
                        ),
                        "detail": {"names": list(unknown)},
                    }
                )
            )
        return StepTypes(
            schema_version=index.schema_version,
            mosaic_version=index.mosaic_version,
            detail=level,
            entries=tuple(rows[name] for name in known) if level == "summary" else (),
            details=()
            if level == "summary"
            else tuple(step_type_detail(name, detail=level) for name in known),
            unknown=unknown,
            total=len(known),
            next_cursor=None,
        )

    def step_type_entry(self, name: str) -> CatalogEntry:
        """Return the editor-catalog entry of *name*, params schema included.

        Not a tool. The ``mosaic://catalog/{name}`` resource reads it.

        Raises:
            ToolFailure: The catalog has no step type of that name.
        """
        from mosaic.core.pipeline.graph import StepTypeNotFound, describe_step_type

        self._declarations()
        try:
            return describe_step_type(name)
        except StepTypeNotFound as error:
            raise ToolFailure(error_envelope(error)) from error

    def _validate_recipe(self, args: ValidateRecipeInput) -> ValidationReport:
        from mosaic.core.pipeline.graph import (
            RecipeInvalid,
            canonical_recipe,
            merge_cautions,
            read_recipe,
            recipe_reference,
            validation_report,
        )
        from mosaic.core.pipeline.graph.validate import refuse_disallowed_definitions

        document = self._recipe_document(args.recipe)
        if recipe_reference(document) is None:
            # Sent whole or read from a file, it is reported and not refused when
            # it does not parse.
            try:
                refuse_disallowed_definitions(read_recipe(document))
            except RecipeInvalid:
                pass
            report = validation_report(document, self._declarations())
        else:
            recipe = self._recipe(document, args.dataset)
            report = validation_report(canonical_recipe(recipe), self._declarations())
        return report.model_copy(update={"cautions": merge_cautions(report.cautions)})

    def _suggest_steps(self, args: SuggestStepsInput) -> StepSuggestions:
        from mosaic.core.pipeline.graph import (
            attach_candidates,
            input_candidates,
            insert_candidates,
        )

        from .outputs import StepSuggestions

        recipe = self._recipe(args.recipe, args.dataset)
        catalog = self._declarations()
        if args.mode == "input":
            if args.step is None:
                raise ToolFailure(
                    ErrorEnvelope(
                        kind="invalid",
                        reason="step_required",
                        message="mode 'input' names the step to add an input to",
                        detail={"field": "step"},
                    )
                )
            report = input_candidates(recipe, args.step, catalog)
            allowed = {
                name: candidate
                for name, candidate in report.candidates.items()
                if candidate.allowed
            }
            return StepSuggestions(
                mode=args.mode,
                step=args.step,
                steps={},
                inputs=allowed,
                children=(),
                still_needs=report.still_needs,
                refused=len(report.candidates) - len(allowed),
            )
        if args.mode == "insert":
            inserted = insert_candidates(recipe, args.step, catalog)
            candidates, children = inserted.candidates, inserted.children
        else:
            candidates = attach_candidates(recipe, args.step, catalog).candidates
            children = ()
        steps = {
            name: attachment
            for name, attachment in candidates.items()
            if attachment.allowed
        }
        return StepSuggestions(
            mode=args.mode,
            step=args.step,
            steps=steps,
            inputs={},
            children=children,
            still_needs=(),
            refused=len(candidates) - len(steps),
        )

    def _patch_recipe(self, args: PatchRecipeInput) -> PatchedRecipe:
        from mosaic.core.pipeline.graph import (
            canonical_recipe,
            merge_cautions,
            patch_recipe,
            patch_validation_report,
            save_recipe,
        )
        from mosaic.core.pipeline.graph.validate import refuse_disallowed_definitions

        from .outputs import PatchedRecipe

        held = self._dataset(args.dataset)
        dataset = self._changeable(held, "patch_recipe")
        base = self._recipe(args.base, args.dataset)
        patched = patch_recipe(base, args.ops)
        # An edit may add a definition the base did not name.
        refuse_disallowed_definitions(patched)
        _ = save_recipe(dataset.base_dir, patched)
        report, unchanged = patch_validation_report(
            canonical_recipe(base), canonical_recipe(patched), self._declarations()
        )
        merged = report.model_copy(update={"cautions": merge_cautions(report.cautions)})
        return PatchedRecipe(
            **merged.model_dump(),
            name=patched.name,
            steps_total=len(patched.steps),
            cautions_unchanged=len(merge_cautions(unchanged)),
        )

    def _recipe_document(self, document: object) -> object:
        """The JSON in the file that *document* names as ``{"path": ...}``, else *document*.

        The file is read inside the recipe roots, once, so a later edit of it
        changes nothing that the call planned or submitted.

        Raises:
            ToolFailure: It names a file outside the recipe roots, or the server
                has none.
            JsonFileUnreadable: The file does not read as JSON.
        """
        from mosaic.core.pipeline.graph import read_json_file

        from .dataset_tools import PathRefused, confined_path

        named = _recipe_file(document)
        if named is None:
            return document
        try:
            path = confined_path(named, self._recipe_roots, "recipe_roots")
        except PathRefused as refused:
            raise ToolFailure(refused.envelope) from refused
        return read_json_file(path)

    def _recipe(self, document: object, dataset: str | None) -> Recipe:
        """Parse *document*, or read the recipe file or saved recipe it names.

        A file named ``{"path": ...}`` is read inside the recipe roots. A saved
        recipe is read from *dataset*.

        Raises:
            ToolFailure: It names a file outside the recipe roots, or a saved
                recipe and no dataset is given.
            FeatureDefinitionsNotAllowed: A step names a feature definition and
                the definition scope is disabled.
            JsonFileUnreadable: The file it names does not read as JSON.
            RecipeInvalid: It does not parse.
            RecipeNotSaved: The dataset has no saved recipe it names.
            RecipeNameAmbiguous: It names several saved recipes by name.
        """
        from mosaic.core.pipeline.graph import (
            read_recipe,
            recipe_reference,
            resolve_saved_recipe,
        )
        from mosaic.core.pipeline.graph.validate import refuse_disallowed_definitions

        document = self._recipe_document(document)
        reference = recipe_reference(document)
        if reference is None:
            recipe = read_recipe(document)
            refuse_disallowed_definitions(recipe)
            return recipe
        if dataset is None:
            raise ToolFailure(
                ErrorEnvelope(
                    kind="invalid",
                    reason="dataset_required",
                    message=(
                        "a recipe named by digest or saved name is read from a "
                        "dataset: pass its dataset"
                    ),
                    detail={"field": "dataset"},
                )
            )
        recipe = resolve_saved_recipe(
            self._dataset(dataset).dataset.base_dir, reference
        )
        refuse_disallowed_definitions(recipe)
        return recipe

    # ------------------------------------------------------------- planning

    def _plan_recipe(self, args: PlanRecipeInput) -> PlanPage:
        from mosaic.core.pipeline.graph import (
            assume_added_cautions,
            plan_wire,
            plan_with_cleanup,
            requests_matching,
            retirable_recipes,
        )
        from mosaic.core.dataset import open_dataset
        from mosaic.core.pipeline.index import feature_run_root
        from mosaic.core.scope import Scope

        held = self._dataset(args.dataset)
        recipe = self._recipe(args.recipe, args.dataset)
        catalog = self._declarations()
        # Opened again, as ``_describe_dataset`` does, so that the manifest fields
        # (the split roles above all) are those on disk now and not at start.
        dataset = open_dataset(held.manifest, ensure_roots=False)
        tags = _tag_table(dataset, args.entries, args.step_entries)
        scope = _scope(dataset, args.entries, tags)
        step_scopes = _step_scopes(dataset, args.step_entries, tags)
        with held.lock:
            # A read, so it needs no --allow-cleanup. Planned with the binds, as
            # the submission plans, so its fingerprint is the submission's.
            plan = plan_with_cleanup(
                dataset,
                recipe,
                cleanup=self._cleanup(args.cleanup),
                request=_planning_request(
                    recipe, args, scope, step_scopes, None if tags is None else tags.ref
                ),
                inventory=self._inventory(held),
                catalog=catalog,
                refuse=False,
                retire=args.retire,
                applies_request_id=args.applies_request_id,
            )
        # Read outside the lock: it reads request files and run-logs only.
        unended = requests_matching(
            held.dataset.base_dir,
            plan.recipe_digest,
            Scope() if scope is None else scope,
        )
        retirable, retirable_omitted = (
            retirable_recipes(
                held.dataset.base_dir,
                exclude_digest=plan.recipe_digest,
                limit=PLAN_RETIRABLE_SAMPLE,
                present=lambda storage, run_id: feature_run_root(
                    held.dataset, storage, run_id
                ).is_dir(),
            )
            if plan.cleanup is not None
            else ((), 0)
        )
        wire = plan_wire(
            plan,
            missing_sample=PLAN_MISSING_SAMPLE,
            open_requests=unended,
            superseded_sample=PLAN_SUPERSEDED_SAMPLE,
            retired_sample=PLAN_RETIRED_SAMPLE,
            retirable=retirable,
            retirable_omitted=retirable_omitted,
            parallel=self._parallel_limits,
            extra_cautions=assume_added_cautions(
                recipe, plan, args.assume_added, scope=scope
            ),
        )
        return _plan_page(wire, args.limit, args.cursor)

    def _submit_pipeline(self, args: SubmitPipelineInput) -> SubmittedPipeline:
        from mosaic.core.pipeline.graph import (
            GPU_TRAIN_LANE,
            CleanupNeedsPlan,
            PlanChanged,
            ReviewedPlan,
            plan_fingerprint,
            plan_with_cleanup,
            start_detached_driver,
            submit_request,
        )
        from mosaic.core.scope import Scope

        held = self._dataset(args.dataset)
        dataset = self._changeable(held, "submit_pipeline")
        if args.cleanup == "superseded" and "feature-runs" not in self._allow_cleanup:
            raise ToolFailure(cleanup_not_allowed("submit_pipeline"))
        if args.cleanup == "superseded" and args.plan_fingerprint is None:
            raise CleanupNeedsPlan()
        cleanup = self._cleanup(args.cleanup)
        try:
            parallel = self._parallel_limits.settings_for(
                args.workers, args.parallel_mode
            )
        except ParallelRefused as refused:
            raise ToolFailure(error_envelope(refused)) from refused
        recipe = self._recipe(args.recipe, args.dataset)
        catalog = self._declarations()
        tags = _tag_table(dataset, args.entries, args.step_entries)
        tag_revision = None if tags is None else tags.ref
        narrowing = _scope(dataset, args.entries, tags) or Scope()
        step_scopes = _step_scopes(dataset, args.step_entries, tags)
        with held.lock:
            cache = self._inventory(held)
            # Planned with the submission's binds, as submit_request plans.
            plan = plan_with_cleanup(
                dataset,
                recipe,
                cleanup=cleanup,
                request=_planning_request(
                    recipe, args, narrowing, step_scopes, tag_revision
                ),
                inventory=cache,
                catalog=catalog,
                retire=args.retire,
                applies_request_id=args.applies_request_id,
            )
            found = plan_fingerprint(plan)
            # The submission records this plan unless an index has moved since.
            reviewed = ReviewedPlan(plan, cache.generation)
            if args.plan_fingerprint is not None and found != args.plan_fingerprint:
                raise PlanChanged(expected=args.plan_fingerprint, found=found)
            findings = _submission_findings(dataset, plan, args, catalog, narrowing)
            if findings and not args.confirm:
                raise ToolFailure(confirmation_required("submit_pipeline", findings))
            # An unconfirmed submission is checked for open duplicates again
            # under the submission lock, which another process also takes.
            submitted = submit_request(
                dataset,
                recipe,
                scope=narrowing,
                bind=args.bind,
                allow_partial=args.allow_partial,
                owner=args.owner,
                inventory=cache,
                catalog=catalog,
                # Without a reviewed fingerprint, the plan the findings were
                # judged on, so a plan that moves before the write still refuses.
                expect_plan=found
                if args.plan_fingerprint is None
                else args.plan_fingerprint,
                refuse_duplicates=not args.confirm,
                step_scopes=step_scopes,
                selections=_selection_names(args.entries, args.step_entries),
                cleanup=cleanup,
                retire=args.retire,
                tracks_run_id=args.tracks_run_id,
                tag_revision=tag_revision,
                pins=args.pins,
                applies_request_id=args.applies_request_id,
                reviewed=reviewed,
            )
        request = submitted.request
        start = self._start_driver or start_detached_driver
        try:
            driver = start(dataset, request, parallel, self._gpu_cards)
        except ClassifiedError as error:
            envelope = error_envelope(error)
            detail = {**(envelope.detail or {}), "request_id": request.request_id}
            raise ToolFailure(envelope.model_copy(update={"detail": detail})) from error
        lanes = {planned.lane for planned in submitted.plan.steps}
        to_compute = submitted.plan.pending
        return SubmittedPipeline(
            request_id=request.request_id,
            recipe_digest=request.recipe_digest,
            plan_fingerprint=plan_fingerprint(submitted.plan),
            counts=dict(Counter(planned.status for planned in submitted.plan.steps)),
            to_compute=len(to_compute),
            first_steps=tuple(
                SubmittedStep(
                    step=planned.step_id,
                    runs=planned.runs,
                    execution_id=request.execution_of(planned.step_id),
                    lane=planned.lane,
                    run_id=planned.run_id,
                    status=planned.status,
                )
                for planned in to_compute[:SUBMITTED_STEP_SAMPLE]
            ),
            truncated=len(to_compute) > SUBMITTED_STEP_SAMPLE,
            duplicates=tuple(
                SubmittedDuplicate(
                    request_id=duplicate.request_id,
                    steps=duplicate.named_steps,
                    steps_omitted=duplicate.steps_omitted,
                )
                for duplicate in submitted.duplicates
            ),
            driver_pid=driver.pid,
            driver_log=str(driver.log_path),
            poll_after_seconds=POLL_AFTER_TRAINING_SECONDS
            if GPU_TRAIN_LANE in lanes
            else POLL_AFTER_MIN_SECONDS,
            cleanup_runs=0
            if request.cleanup is None
            else len(request.cleanup.approved),
            cleanup_bytes=_approved_bytes(submitted.plan, request),
            workers=parallel.workers,
            parallel_mode=parallel.mode,
        )

    # ------------------------------------------------------------- datasets

    def _manage_dataset(self, args: ManageDatasetInput) -> DatasetChange:
        from mosaic.core.dataset_summary import dataset_next_steps

        from .dataset_tools import PathRefused

        if args.action == "define_feature" and not self._allow_feature_definitions:
            raise ToolFailure(feature_definitions_not_allowed("manage_dataset"))
        try:
            if args.action == "create":
                held, message = self._create_dataset(args)
                return self._change(args, held, message)
            if args.dataset is None:
                raise ToolFailure(missing_field(args.action, "dataset"))
            held = self._dataset(args.dataset)
            if args.action == "list_sequence_tags":
                return self._list_sequence_tags(args, held)
            dataset = self._changeable(held, "manage_dataset")
            scanned = _Scanned()
            orphaned: int | None = None
            defined: DefinitionRecord | None = None
            imported: SequenceTagImport | None = None
            saved: SelectionSummary | None = None
        except PathRefused as refused:
            raise ToolFailure(refused.envelope) from refused
        try:
            match args.action:
                case "add_source":
                    message = self._add_source(dataset, args)
                case "remove_source":
                    orphaned = self._remove_source(dataset, args)
                    message = (
                        f"removed the {args.kind} source {args.source_id!r}; "
                        f"{orphaned} index rows it claimed stay"
                    )
                case "scan":
                    scanned = _scan(dataset, args)
                    message = (
                        f"scanned {', '.join(scanned.rows) or 'no declared source'}"
                    )
                case "set_tags":
                    message = _set_tags(dataset, args)
                case "set_arenas":
                    message = _set_arenas(dataset, args)
                case "save_selection":
                    saved = _save_selection(dataset, args)
                    message = (
                        f"saved the selection {saved.name!r} with "
                        f"{saved.n_entries} entries"
                    )
                case "define_feature":
                    defined = _define_feature(dataset, args)
                    message = (
                        f"{'reused' if defined.reused else 'defined'} the feature "
                        f"definition {defined.name!r} at {defined.digest[:12]}; "
                        f"recipes name it def:{defined.name}"
                    )
                case "import_sequence_tags":
                    imported = self._import_sequence_tags(dataset, args)
                    message = _tag_import_message(imported)
        except PathRefused as refused:
            raise ToolFailure(refused.envelope) from refused
        finally:
            # A scan that failed on a later kind kept what the earlier wrote.
            self._reopened(held)
        change = self._change(args, held, message)
        return change.model_copy(
            update={
                "scanned": scanned.rows,
                "divisions": len(scanned.divisions.divisions),
                "held_back": scanned.divisions.held_back,
                "refused": scanned.divisions.refused,
                "findings": (
                    scanned.findings
                    if imported is None
                    else (*scanned.findings, *imported.findings)
                ),
                "orphaned_rows": orphaned,
                "definition": defined,
                "tag_import": (
                    None
                    if imported is None
                    else imported.model_copy(update={"findings": ()})
                ),
                "selection": saved,
                "next_steps": dataset_next_steps(held.dataset),
            }
        )

    def _change(
        self, args: ManageDatasetInput, held: _Served, message: str
    ) -> DatasetChange:
        from mosaic.core.dataset_summary import dataset_next_steps

        return DatasetChange(
            action=args.action,
            dataset=held.uuid,
            name=held.dataset.manifest.name,
            manifest=str(held.manifest),
            message=message,
            next_steps=dataset_next_steps(held.dataset),
        )

    def _create_dataset(self, args: ManageDatasetInput) -> tuple[_Served, str]:
        from mosaic.core.dataset import new_dataset_manifest

        from .dataset_tools import confined_path, create_findings

        if args.path is None:
            raise ToolFailure(missing_field("create", "path"))
        target = confined_path(args.path, self._dataset_roots, "dataset_roots")
        manifest = target / "dataset.yaml"
        taken = target.exists() and (not target.is_dir() or any(target.iterdir()))
        if taken:
            raise ToolFailure(_dataset_exists(target))
        name = args.name or target.name
        findings = create_findings(name, target)
        if not args.confirm:
            raise ToolFailure(confirmation_required("manage_dataset", findings))
        target.mkdir(parents=True, exist_ok=True)
        try:
            # Claimed exclusively, so two calls for one path make one dataset.
            manifest.open("x").close()
        except FileExistsError as exists:
            raise ToolFailure(_dataset_exists(target)) from exists
        try:
            written = new_dataset_manifest(name, target)
        except BaseException:
            manifest.unlink(missing_ok=True)
            raise
        held = self._attach(written)
        if held is None:
            msg = f"the dataset created at {written} does not open"
            raise RuntimeError(msg)
        return held, f"created the dataset {name!r} at {str(target)!r}"

    def _add_source(self, dataset: Dataset, args: ManageDatasetInput) -> str:
        from mosaic.core.source_fields import (
            SourceFieldsInvalid,
            scan_source_from_fields,
        )

        from .dataset_tools import (
            confined_path,
            refuse_escaping_globs,
            source_fields_of,
            source_findings,
        )

        if args.kind is None:
            raise ToolFailure(missing_field("add_source", "kind"))
        if args.path is None:
            raise ToolFailure(missing_field("add_source", "path"))
        path = confined_path(args.path, self._source_roots, "source_roots")
        refuse_escaping_globs(args)
        try:
            source = scan_source_from_fields(source_fields_of(args, path))
        except SourceFieldsInvalid as invalid:
            raise ToolFailure(_invalid_source(str(invalid))) from invalid
        findings = source_findings("add_source", args.kind, source.id, str(path))
        if not args.confirm:
            raise ToolFailure(confirmation_required("manage_dataset", findings))
        try:
            dataset.declare_scan_source(source)
        except ValueError as invalid:
            raise ToolFailure(_invalid_source(str(invalid))) from invalid
        declared = (
            f"declared the {args.kind} source {source.id!r} ({source.mode}) "
            f"at {str(path)!r}"
        )
        missing = missing_path_finding(
            args.kind, source.id, dataset.resolve_source_path(source)
        )
        if missing is None:
            return declared
        return f"{declared}; warning: {missing.message}"

    def _import_sequence_tags(
        self, dataset: Dataset, args: ManageDatasetInput
    ) -> SequenceTagImport:
        """Write the CSV *args* name as the next tag table revision.

        Raises:
            ToolFailure: The call names no path, the path lies outside the
                operator's roots or holds no file, a replacement is not
                confirmed, or the library refuses the import.
        """
        from mosaic.core.annotations.sequence_tags import (
            import_sequence_tags,
            parse_key_option,
        )

        from .dataset_tools import confined_path, replace_findings

        if args.path is None:
            raise ToolFailure(missing_field("import_sequence_tags", "path"))
        path = confined_path(args.path, self._source_roots, "source_roots")
        if not path.is_file():
            raise ToolFailure(
                ErrorEnvelope(
                    kind="not_found",
                    reason="csv_not_found",
                    message=f"no CSV file at {str(path)!r}",
                    detail={"path": args.path},
                )
            )
        if args.replace and not args.dry_run and not args.confirm:
            raise ToolFailure(
                confirmation_required("manage_dataset", replace_findings(path))
            )
        try:
            key = None if args.key is None else parse_key_option(args.key)
            return import_sequence_tags(
                dataset,
                path,
                key=key,
                types=args.types,
                replace=args.replace,
                dry_run=args.dry_run,
            )
        except ValueError as refused:
            raise ToolFailure(_sequence_tags_refused(refused)) from refused

    def _list_sequence_tags(
        self, args: ManageDatasetInput, held: _Served
    ) -> DatasetChange:
        """List the sequence tags of *held*, which a managed dataset answers too."""
        from mosaic.core.annotations.sequence_tags import list_sequence_tags
        from mosaic.core.dataset import open_dataset

        dataset = open_dataset(held.manifest, ensure_roots=False)
        try:
            listing = list_sequence_tags(dataset)
        except ValueError as refused:
            raise ToolFailure(_sequence_tags_refused(refused)) from refused
        message = (
            f"{len(listing.tags)} sequence tags at revision {listing.revision}"
            if listing.revision is not None
            else "the dataset has no sequence tags"
        )
        change = self._change(args, held, message)
        return change.model_copy(update={"sequence_tags": listing})

    def _remove_source(self, dataset: Dataset, args: ManageDatasetInput) -> int:
        from .dataset_tools import source_findings

        if args.kind is None:
            raise ToolFailure(missing_field("remove_source", "kind"))
        if args.source_id is None:
            raise ToolFailure(missing_field("remove_source", "source_id"))
        findings = source_findings("remove_source", args.kind, args.source_id, None)
        if not args.confirm:
            raise ToolFailure(confirmation_required("manage_dataset", findings))
        try:
            return dataset.undeclare_scan_source(args.kind, args.source_id)
        except KeyError as unknown:
            raise ToolFailure(
                ErrorEnvelope(
                    kind="not_found",
                    reason="unknown_source",
                    message=str(unknown.args[0]),
                    detail={"kind": args.kind, "source_id": args.source_id},
                )
            ) from unknown

    # ------------------------------------------------------------- requests

    def _get_request_status(self, args: GetRequestStatusInput) -> RequestStatusWire:
        from mosaic.core.pipeline.cancel import read_request
        from mosaic.core.pipeline.graph import request_status_wire

        held = self._dataset(args.dataset)
        request = read_request(held.dataset.base_dir, args.request_id)
        # The wait runs on the worker thread that answers the call, and holds no
        # lock of the dataset.
        return request_status_wire(
            held.dataset.base_dir,
            request,
            brief=args.brief,
            wait_seconds=args.wait_seconds,
        )

    def _cancel_request(self, args: CancelRequestInput) -> RequestCancel:
        from mosaic.core.pipeline.cancel import cancel_request, read_request

        held = self._dataset(args.dataset)
        dataset = self._changeable(held, "cancel_request")
        request = read_request(dataset.base_dir, args.request_id)
        findings = self._cancel_findings(dataset, request)
        if findings and not args.confirm:
            raise ToolFailure(confirmation_required("cancel_request", findings))
        return cancel_request(dataset, args.request_id)

    def _list_activity(self, args: ListActivityInput) -> ActivityPage:
        from mosaic.core.pipeline.graph import list_requests
        from mosaic.runlog import list_runs, run_log_dir

        held = self._dataset(args.dataset)
        base = held.dataset.base_dir
        as_of = datetime.now(UTC)
        if args.view == "requests":
            requests = list_requests(
                base, limit=args.limit, after=args.cursor, changed_since=args.since
            )
            return ActivityPage(
                view=args.view,
                requests=requests.items,
                next_cursor=requests.next_after,
                as_of=as_of,
            )
        runs = list_runs(
            run_log_dir(base),
            limit=args.limit,
            after=args.cursor,
            changed_since=args.since,
        )
        return ActivityPage(
            view=args.view,
            runs=tuple(_run_summary(snapshot) for snapshot in runs["items"]),
            next_cursor=runs["next_after"],
            as_of=as_of,
        )

    def _diagnose_failures(self, args: DiagnoseFailuresInput) -> FailureDiagnosis:
        from mosaic.core.pipeline.cancel import AttemptNotFound, read_request
        from mosaic.core.pipeline.graph import request_status_wire, run_status_wire

        if (args.request_id is None) == (args.execution_id is None):
            raise ToolFailure(
                ErrorEnvelope(
                    kind="invalid",
                    reason="one_id_required",
                    message="name exactly one of request_id and execution_id",
                    detail={"fields": ["request_id", "execution_id"]},
                )
            )
        held = self._dataset(args.dataset)
        base = held.dataset.base_dir
        if args.execution_id is not None:
            attempt = run_status_wire(base, args.execution_id)
            if attempt is None:
                raise AttemptNotFound(args.execution_id)
            diagnosed = _diagnosis(
                base, "", attempt.execution_id, attempt.outcome, attempt.error
            )
            return FailureDiagnosis(
                request_id=None,
                verdict=None,
                attempts=() if diagnosed is None else (diagnosed,),
            )
        request = read_request(base, args.request_id or "")
        status = request_status_wire(base, request)
        diagnosed = (
            _diagnosis(base, step.step, step.execution_id, step.outcome, step.error)
            for step in status.steps
        )
        attempts = tuple(attempt for attempt in diagnosed if attempt is not None)
        return FailureDiagnosis(
            request_id=status.request_id, verdict=status.verdict, attempts=attempts
        )

    def _preview_output(
        self, args: PreviewOutputInput
    ) -> ArtifactPreview | RunSummaryWire:
        from mosaic.core.pipeline.preview import preview_artifact, preview_ref

        held = self._dataset(args.dataset)
        if args.mode == "summary":
            # Importing scipy beside the warming thread can raise _DeadlockError.
            self.wait_until_warm()
            return _summary(held.dataset, args)
        # The input model requires both in entry mode.
        assert args.group is not None and args.sequence is not None
        return preview_artifact(
            held.dataset,
            preview_ref(args.ref),
            (args.group, args.sequence),
            columns=args.columns,
            limit=args.limit,
            max_columns=args.max_columns,
            listed=args.listed,
        )

    # -------------------------------------------------------------- helpers

    def _warm(self) -> None:
        """Open the datasets and build the catalog, so the first calls do not wait."""
        _ = self._datasets()
        _ = self._catalog_index()

    def _datasets(self) -> dict[str, _Served]:
        """Return the served datasets by ``uuid``, opening them on first use."""
        with self._served_lock:
            if self._served is None:
                self._served = _open_allowlist(self._manifests)
            # A copy, so a reader iterates a snapshot while _attach adds one.
            return dict(self._served)

    def _dataset(self, uuid: str) -> _Served:
        """Return the served dataset *uuid*.

        A dataset made below a dataset root since the last look is served first.

        Raises:
            ToolFailure: The allowlist names no dataset *uuid*.
        """
        held = self._datasets().get(uuid)
        if held is None and self._dataset_roots:
            self._serve_new_datasets()
            held = self._datasets().get(uuid)
        if held is None:
            raise ToolFailure(unknown_dataset(uuid))
        return held

    def _serve_new_datasets(self) -> None:
        """Serve each dataset below a dataset root that is not served yet."""
        from mosaic.core.dataset_discovery import (
            DEFAULT_DISCOVERY_DEPTH,
            discover_datasets,
        )

        if not self._dataset_roots:
            return
        served = {held.manifest.resolve() for held in self._datasets().values()}
        for root in self._dataset_roots:
            if not root.is_dir():
                continue
            for found in discover_datasets(root, max_depth=DEFAULT_DISCOVERY_DEPTH):
                manifest = Path(found.manifest).resolve()
                if (
                    found.error is None
                    and found.uuid
                    and manifest not in served
                    and manifest not in self._looked_at
                ):
                    # Looked at once: a copy repeating a served uuid is not
                    # opened again on every listing.
                    self._looked_at.add(manifest)
                    _ = self._attach(manifest)
                    served.add(manifest)

    def _attach(self, manifest: Path) -> _Served | None:
        """Serve the dataset of *manifest* from now on, and return it.

        ``None`` when it does not open or has no uuid. A uuid already served
        keeps the dataset served first.
        """
        added = _open_allowlist([manifest])
        _ = self._datasets()
        with self._served_lock:
            assert self._served is not None
            for uuid, held in added.items():
                if uuid in self._served:
                    return self._served[uuid]
                self._served = {**self._served, uuid: held}
                return held
        return None

    def _reopened(self, held: _Served) -> None:
        """Read *held*'s manifest again after a change, so every tool sees it."""
        from mosaic.core.dataset import open_dataset

        dataset = open_dataset(held.manifest, ensure_roots=False)
        with held.lock:
            held.dataset = dataset
            held.cache = None

    def _cleanup(self, mode: str) -> CleanupPolicy | None:
        """The operator's cleanup policy when *mode* asks for one, else ``None``."""
        from mosaic.core.pipeline.graph import CleanupPolicy

        if mode != "superseded":
            return None
        return self._cleanup_policy or CleanupPolicy()

    def _changeable(self, held: _Served, tool: ToolName) -> Dataset:
        """Return *held* opened again, refusing a dataset that a service manages.

        The manifest is read again, so a marker written since the server started
        refuses the change.

        Raises:
            ToolFailure: The dataset is managed and the backend does not allow
                changing a managed one.
        """
        from mosaic.core.dataset import open_dataset

        dataset = open_dataset(held.manifest, ensure_roots=False)
        if dataset.manifest.is_managed and not self._allow_managed:
            raise ToolFailure(
                dataset_managed(tool, held.uuid, dataset.manifest.managed_by)
            )
        return dataset

    def _inventory(self, held: _Served) -> InventoryCache:
        """Return *held*'s inventory cache, refreshed. Call with ``held.lock`` held."""
        from mosaic.core.pipeline.inventory import InventoryCache

        if held.cache is None:
            self._register_contributors()
            held.cache = InventoryCache(held.dataset)
        else:
            _ = held.cache.revalidate()
        return held.cache

    def _register_contributors(self) -> None:
        """Register the tracking kinds of the inventory, once."""
        from mosaic.tracking import register_inventory_contributors

        with self._served_lock:
            if not self._contributors_registered:
                register_inventory_contributors()
                self._contributors_registered = True

    def _declarations(self) -> DeclarationCatalog:
        """Return the declaration catalog of the scope in force.

        The built-in catalog is built once across threads, under the lock. The
        scope's definitions are joined outside it. Calls on two datasets therefore
        do not wait on each other's reads.
        """
        from mosaic.core.pipeline.graph import declaration_catalog
        from mosaic.core.pipeline.graph.resolve import builtin_declaration_catalog

        with self._catalog_lock:
            _ = builtin_declaration_catalog()
        return declaration_catalog()

    def _catalog_index(self) -> CatalogIndex:
        """Return the catalog index of the scope in force.

        The built-in editor catalog is built once across threads, under the lock,
        and the scope's definitions are joined outside it.
        """
        from mosaic.core.pipeline.graph import catalog_index
        from mosaic.core.pipeline.graph.resolve import builtin_editor_catalog

        with self._catalog_lock:
            _ = builtin_editor_catalog()
        return catalog_index()

    def _cancel_findings(
        self, dataset: Dataset, request: Request
    ) -> list[ConfirmationFinding]:
        """Return the running steps of *request* that lose their work when cancelled."""
        from mosaic.behavior.feature_library.runner_client import RunnerInterpreter
        from mosaic.core.narrowing import is_object_tuple
        from mosaic.core.pipeline.graph import (
            FeatureStepSpec,
            feature_class_for_slug,
            load_recipe_for_request,
            request_rollup,
            step_liveness,
        )

        # A step whose process is provably gone loses nothing more to a cancel,
        # which only records it.
        running = {
            attempt.step_id
            for attempt in request_rollup(dataset.base_dir, request).steps
            if attempt.started
            and not attempt.terminal
            and step_liveness(attempt) != "dead"
        }
        if not running:
            return []
        catalog = self._declarations()
        findings: list[ConfirmationFinding] = []
        for step in load_recipe_for_request(dataset.base_dir, request).steps:
            if step.id not in running:
                continue
            if isinstance(step, FeatureStepSpec):
                feature = feature_class_for_slug(step.feature)
                declared: object = getattr(feature, "prerequisites", ())
                if is_object_tuple(declared) and any(
                    isinstance(item, RunnerInterpreter) for item in declared
                ):
                    findings.append(
                        ConfirmationFinding(
                            "hard_cancel",
                            f"step {step.id!r} runs {step.feature} in its runner, "
                            "which a cancel stops at once",
                        )
                    )
                continue
            declaration = catalog.get(step.kind)
            if declaration is not None and declaration.category == "train":
                findings.append(
                    ConfirmationFinding(
                        "hard_cancel",
                        f"step {step.id!r} trains with {step.kind}, which loses "
                        "the epoch in progress",
                    )
                )
        return findings


def _summary(dataset: Dataset, args: PreviewOutputInput) -> RunSummaryWire:
    """Summarize the feature run *args* names, over the entries it selects.

    Sequence tags select against the revision that ``tag_revision`` names. The
    summary's tag keys read that table, and the summary records its revision
    whether or not a key is a tag.

    Raises:
        TagSelectionInvalid: The tags select no entry, or name an unknown tag or a
            value their type refuses.
        SequenceTagTableUnavailable: The table cannot be read as one revision of
            one dataset.
    """
    from mosaic.core.pipeline.run_summary import summarize_ref
    from mosaic.core.pipeline.summary_arguments import read_where
    from mosaic.core.scope import Scope
    from mosaic.core.selections import selection_scope

    scope = None if args.groups is None else Scope(groups=args.groups)
    tag_revision = args.tag_revision
    tag_table = None
    if args.selection is not None:
        scope = selection_scope(dataset, args.selection)
    elif args.sequence_tags is not None:
        from mosaic.core.annotations.tag_selection import (
            selection_tag_table,
            tagged_scope,
        )

        tags = selection_tag_table(dataset, tag_revision)
        scope = tagged_scope(tags, args.sequence_tags, groups=args.groups)
        tag_revision = None
        tag_table = tags
    return summarize_ref(
        dataset,
        args.ref,
        by=args.by,
        columns=args.columns,
        tests=args.tests,
        where=None if args.where is None else read_where(args.where),
        scope=scope,
        allow_partial=args.allow_partial,
        tag_revision=tag_revision,
        tag_table=tag_table,
    )


def answer[T: BaseModel](compute: Callable[[], T]) -> T | ErrorEnvelope:
    """Return *compute*'s result, or the envelope of the failure it raised."""
    try:
        return compute()
    except ToolFailure as failure:
        return failure.envelope
    except ClassifiedError as error:
        return error_envelope(error)


def discovered_manifests(root: Path, *, max_depth: int | None = None) -> list[Path]:
    """Return the manifests of the datasets in *root* and below it.

    A manifest that does not read or has no ``uuid`` is left out and logged.

    Args:
        root: The directory to search.
        max_depth: How many levels below *root* to search, the discovery
            default when ``None``.
    """
    from mosaic.core.dataset_discovery import (
        DEFAULT_DISCOVERY_DEPTH,
        discover_datasets,
    )

    depth = DEFAULT_DISCOVERY_DEPTH if max_depth is None else max_depth
    manifests: list[Path] = []
    for found in discover_datasets(root, max_depth=depth):
        if found.error is not None or found.uuid is None:
            _LOG.warning(
                "not serving %s: %s",
                found.manifest,
                found.error or "its manifest has no uuid",
            )
            continue
        manifests.append(Path(found.manifest))
    return manifests


def _open_allowlist(manifests: Sequence[Path]) -> dict[str, _Served]:
    """Open each manifest without creating roots, keyed by its ``uuid``.

    A manifest that does not open, has no ``uuid`` or repeats one is logged and
    left out.
    """
    from mosaic.core.dataset import open_dataset
    from mosaic.user_paths import user_path

    served: dict[str, _Served] = {}
    for given in manifests:
        path = user_path(given)
        try:
            dataset = open_dataset(path, ensure_roots=False)
        except _manifest_errors() as error:
            _LOG.warning("not serving %s: %s", path, error)
            continue
        uuid = dataset.manifest.uuid
        if not uuid:
            _LOG.warning("not serving %s: its manifest has no uuid", path)
            continue
        if uuid in served:
            _LOG.warning(
                "not serving %s: uuid %s is already served from %s",
                path,
                uuid,
                served[uuid].manifest,
            )
            continue
        served[uuid] = _Served(
            uuid=uuid, manifest=dataset.manifest_path, dataset=dataset
        )
    return served


def _manifest_errors() -> tuple[type[Exception], ...]:
    """Return the errors that a manifest which does not read raises."""
    import yaml

    return (OSError, ValueError, ClassifiedError, yaml.YAMLError)


def _current_manifest(held: _Served) -> DatasetManifest:
    """Return *held*'s manifest as it reads now, the one opened when it does not."""
    from mosaic.core.manifest import read_manifest

    try:
        return read_manifest(held.manifest)
    except _manifest_errors() as error:
        _LOG.warning("listing %s as opened: %s", held.manifest, error)
        return held.dataset.manifest


def _page[T](
    items: Sequence[T],
    key: Callable[[T], tuple[str, ...]],
    limit: int,
    cursor: str | None,
) -> tuple[list[T], str | None]:
    """Return the page of *items*, ordered by *key*, that starts after *cursor*.

    The cursor is the key of the last item of the previous page, so an item
    added or removed between pages neither repeats nor skips one that stays.

    Raises:
        ToolFailure: *cursor* is not a key that this function wrote.
    """
    after = None if cursor is None else _cursor_key(cursor)
    remaining = [
        item for item in sorted(items, key=key) if after is None or key(item) > after
    ]
    page = remaining[:limit]
    more = len(remaining) > limit
    return page, json.dumps(list(key(page[-1]))) if more else None


def _plan_page(wire: PlanWire, limit: int | None, cursor: str | None) -> PlanPage:
    """Return the window of *wire*'s steps that starts after *cursor*.

    The cursor is ``[last step id, fingerprint]``, because a plan lists its steps
    in topological order and not in a sortable one. Every page carries the whole
    plan's fingerprint. The first page carries the plan-level blocks and a later
    page leaves them empty.

    Raises:
        ToolFailure: *cursor* names no step of this plan, or was written for a
            plan with another fingerprint.
    """
    from .outputs import PlanPage

    start = 0
    if cursor is not None:
        parts = _cursor_key(cursor)
        ids = [step.step for step in wire.steps]
        if len(parts) != 2 or parts[0] not in ids or parts[1] != wire.fingerprint:
            raise ToolFailure(invalid_cursor(cursor))
        start = ids.index(parts[0]) + 1
    stop = len(wire.steps) if limit is None else start + limit
    more = stop < len(wire.steps)
    first = cursor is None
    return PlanPage(
        recipe_digest=wire.recipe_digest,
        fingerprint=wire.fingerprint,
        scope_size=wire.scope_size,
        complete=wire.complete,
        steps=wire.steps[start:stop],
        problems=wire.problems if first else (),
        cautions=wire.cautions if first else (),
        open_requests=wire.open_requests if first else (),
        cleanup=wire.cleanup if first else None,
        retirable=wire.retirable if first else (),
        retirable_omitted=wire.retirable_omitted if first else 0,
        parallel=wire.parallel if first else None,
        locked_steps=wire.locked_steps if first else (),
        steps_total=len(wire.steps),
        next_cursor=(
            json.dumps([wire.steps[stop - 1].step, wire.fingerprint]) if more else None
        ),
    )


def _cursor_key(cursor: str) -> tuple[str, ...]:
    """Return the key that *cursor* encodes.

    Raises:
        ToolFailure: *cursor* is not a JSON array of strings.
    """
    from mosaic.core.narrowing import is_object_list

    try:
        parsed: object = json.loads(cursor)
    except ValueError as error:
        raise ToolFailure(invalid_cursor(cursor)) from error
    if not is_object_list(parsed):
        raise ToolFailure(invalid_cursor(cursor))
    parts = [part for part in parsed if isinstance(part, str)]
    if len(parts) != len(parsed):
        raise ToolFailure(invalid_cursor(cursor))
    return tuple(parts)


def _recipe_file(document: object) -> str | None:
    """The file that a recipe argument ``{"path": ...}`` names, else ``None``."""
    from mosaic.core.narrowing import is_str_mapping

    if not is_str_mapping(document) or len(document) != 1:
        return None
    path = document.get("path")
    return path if isinstance(path, str) else None


def _tag_table(
    dataset: Dataset,
    entries: list[tuple[str, str]] | ScopeSelector | None,
    step_entries: dict[str, list[tuple[str, str]] | ScopeSelector] | None = None,
) -> ResolvedTagTable | None:
    """Read the latest tag table once when any selector names sequence tags.

    Returns:
        The table that :func:`_scope` and :func:`_step_scopes` lower against, or
        ``None`` when no selector names a tag, so that a call without a tag
        selection reads nothing.

    Raises:
        TagSelectionInvalid: No revision of the sequence tags is indexed.
        SequenceTagTableUnavailable: The table cannot be read as one revision of
            one dataset.
    """
    selectors = [entries, *(step_entries or {}).values()]
    if not any(
        isinstance(chosen, ScopeSelector) and chosen.sequence_tags is not None
        for chosen in selectors
    ):
        return None
    from mosaic.core.annotations.tag_selection import selection_tag_table

    return selection_tag_table(dataset)


def _scope(
    dataset: Dataset,
    entries: list[tuple[str, str]] | ScopeSelector | None,
    tags: ResolvedTagTable | None,
) -> Scope | None:
    """Return the scope naming *entries*, ``None`` for the whole dataset.

    A bare list names entries, a selector is the library's scope, and a selector
    naming a saved selection is that selection's entries. Sequence tags are
    lowered against *tags*, the table :func:`_tag_table` read.

    Raises:
        SelectionNotFound: The selector names a selection the dataset lacks.
        TagSelectionInvalid: The tags select no entry, or name an unknown tag or
            a value their type refuses.
    """
    from mosaic.core.scope import Scope
    from mosaic.core.selections import selection_scope

    if entries is None:
        return None
    if isinstance(entries, ScopeSelector):
        if entries.selection is not None:
            return selection_scope(dataset, entries.selection)
        if entries.sequence_tags is not None:
            from mosaic.core.annotations.tag_selection import tagged_scope

            assert tags is not None
            return tagged_scope(tags, entries.sequence_tags)
        return Scope(
            entries=None if entries.entries is None else list(entries.entries),
            groups=entries.groups,
            sequences=entries.sequences,
        )
    return Scope(entries=list(entries))


def _entry_scope(
    dataset: Dataset, entries: list[tuple[str, str]] | ScopeSelector | None
) -> Scope | None:
    """Return the scope naming *entries*, a groups or sequences selector enumerated.

    The inventory narrows a tracks or labels variant by named entries alone,
    so a selector is resolved against the dataset's entries first, and every
    kind of artifact is narrowed alike. Sequence tags select against the latest
    revision of the tag table.
    """
    from mosaic.core.pipeline.inventory.scan import narrow_target, reportable_universe
    from mosaic.core.scope import Scope

    scope = _scope(dataset, entries, _tag_table(dataset, entries))
    if scope is None or scope.entries is not None:
        return scope
    return Scope(entries=sorted(narrow_target(reportable_universe(dataset), scope)))


def _step_scopes(
    dataset: Dataset,
    step_entries: dict[str, list[tuple[str, str]] | ScopeSelector] | None,
    tags: ResolvedTagTable | None,
) -> dict[str, Scope]:
    """Return the scope narrowing each step that *step_entries* names."""
    found: dict[str, Scope] = {}
    for step_id, chosen in (step_entries or {}).items():
        narrowed = _scope(dataset, chosen, tags)
        if narrowed is not None:
            found[step_id] = narrowed
    return found


def _selection_names(
    entries: list[tuple[str, str]] | ScopeSelector | None,
    step_entries: dict[str, list[tuple[str, str]] | ScopeSelector] | None,
) -> dict[str, str]:
    """Return the saved selection each scope came from, keyed as a request keys it.

    The key ``""`` is the request's scope and a step id is that step's.
    """
    names = {
        step_id: chosen.selection
        for step_id, chosen in (step_entries or {}).items()
        if isinstance(chosen, ScopeSelector) and chosen.selection is not None
    }
    if isinstance(entries, ScopeSelector) and entries.selection is not None:
        names[""] = entries.selection
    return names


@dataclass(frozen=True)
class _Scanned:
    """What a ``scan`` call indexed.

    Attributes:
        rows: The raw index rows of each kind scanned.
        divisions: What the dividing media sources divided, held back and
            refused.
        findings: What the scan dropped or could not place.
    """

    rows: dict[str, int] = field(default_factory=dict)
    divisions: DivisionReportWire = field(default_factory=DivisionReportWire)
    findings: tuple[ScanFinding, ...] = ()


def _scan(dataset: Dataset, args: ManageDatasetInput) -> _Scanned:
    """Scan what *args* name; the raw index rows of each kind scanned.

    The kind given, or with ``source_id`` alone the kinds that declare it, or
    every kind that declares a source. A media scan that divides a store again
    lists its tracking outputs, so the tracking kinds are registered first.
    """
    from mosaic.core.dataset_summary import raw_label_rows
    from mosaic.core.media.division_wire import division_report_wire
    from mosaic.core.media.store_division import DivisionRefused
    from mosaic.core.pipeline.tracks_raw_index import read_tracks_raw_index
    from mosaic.tracking import register_ops

    declared = {
        kind: {source.id for source in dataset.manifest.sources.of_kind(kind)}
        for kind in ("media", "tracks", "labels")
    }
    if args.kind is not None:
        kinds = [args.kind]
    elif args.source_id is not None:
        kinds = [kind for kind, ids in declared.items() if args.source_id in ids]
    else:
        kinds = [kind for kind, ids in declared.items() if ids]
    if args.source_id is not None and not kinds:
        raise ToolFailure(
            ErrorEnvelope(
                kind="not_found",
                reason="unknown_source",
                message=f"no source named {args.source_id!r} is declared",
                detail={"source_id": args.source_id},
            )
        )
    only = () if args.source_id is None else (args.source_id,)
    scanned: dict[str, int] = {}
    divisions = DivisionReportWire()
    findings: list[ScanFinding] = []
    register_ops()
    for kind in kinds:
        try:
            if kind == "media":
                media = dataset.scan_media(
                    only=only, include_partial=args.include_partial
                )
                divisions = division_report_wire(media.divisions)
                findings.extend(media.findings)
                scanned[kind] = len(dataset.read_media_index())
            elif kind == "tracks":
                tracks = dataset.scan_tracks(only=only)
                findings.extend(tracks.findings)
                scanned[kind] = len(read_tracks_raw_index(tracks.path))
            else:
                labels = dataset.scan_labels(only=only)
                findings.extend(labels.findings)
                scanned[kind] = raw_label_rows(dataset)
        except DivisionRefused as refused:
            raise ToolFailure(_division_refused(refused)) from refused
        except (KeyError, ValueError, NotImplementedError) as invalid:
            raise ToolFailure(_invalid_source(str(invalid))) from invalid
    return _Scanned(rows=scanned, divisions=divisions, findings=tuple(findings))


def _division_refused(refused: DivisionRefused) -> ErrorEnvelope:
    """The envelope of a media scan that refuses to divide a store."""
    return ErrorEnvelope(
        kind="invalid",
        reason=refused.reason,
        message=str(refused),
        detail={"refusal": refused.reason},
    )


def _define_feature(dataset: Dataset, args: ManageDatasetInput) -> DefinitionRecord:
    """Define ``args.source`` as the feature definition ``args.name`` of *dataset*.

    Raises:
        ToolFailure: The call gives no name or no source.
        ClassifiedError: The name or the source is refused, as ``define_feature``
            refuses them.
    """
    from mosaic.core.pipeline.definitions.define import define_feature

    if args.name is None:
        raise ToolFailure(missing_field("define_feature", "name"))
    if args.source is None:
        raise ToolFailure(missing_field("define_feature", "source"))
    return define_feature(dataset, args.name, args.source)


def _sequence_tags_refused(refused: ValueError) -> ErrorEnvelope:
    """The answer for a sequence-tag import or listing that the library refuses.

    A CSV or key option that cannot be read is ``invalid``, which a changed request
    answers. A table that is edited elsewhere, moved under the import or cannot be
    read as one revision disagrees with the dataset's state, so it is a ``conflict``.
    """
    from mosaic.core.annotations.sequence_tags import (
        SequenceTagImportRefused,
        SequenceTagsMoved,
        SequenceTagTableUnavailable,
    )

    kind: ErrorKind = "conflict"
    if isinstance(refused, SequenceTagImportRefused):
        reason = "sequence_tags_refused"
    elif isinstance(refused, SequenceTagsMoved):
        reason = "sequence_tags_moved"
    elif isinstance(refused, SequenceTagTableUnavailable):
        reason = "sequence_tags_unavailable"
    else:
        kind, reason = "invalid", "sequence_tags_invalid"
    return ErrorEnvelope(kind=kind, reason=reason, message=terse(refused))


def _tag_import_message(report: SequenceTagImport) -> str:
    """Say in a sentence what a sequence-tag import did."""
    if report.dry_run:
        return f"dry run: {report.rows_matched} rows name entries, nothing written"
    if report.written:
        return (
            f"wrote revision {report.revision} of the tag table; "
            f"{report.rows_matched} rows name entries"
        )
    return f"nothing changed; the tag table stays at revision {report.revision}"


def _set_tags(dataset: Dataset, args: ManageDatasetInput) -> str:
    """Set and remove the tags *args* name, and say what changed."""
    if not args.tags and not args.remove_tags:
        raise ToolFailure(missing_field("set_tags", "tags or remove_tags"))
    try:
        dataset.update_tags(args.tags or {}, remove=args.remove_tags)
    except (KeyError, ValueError) as invalid:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_tag",
                message=str(invalid.args[0]) if invalid.args else str(invalid),
                detail={
                    "tags": [name for name in sorted(args.tags or {})],
                    "remove_tags": [name for name in args.remove_tags],
                },
            )
        ) from invalid
    changed = [*sorted(args.tags or {}), *(f"-{name}" for name in args.remove_tags)]
    return f"tags changed: {', '.join(changed)}"


def _save_selection(dataset: Dataset, args: ManageDatasetInput) -> SelectionSummary:
    """Save the entries *args* name as the selection *args* name, and return it.

    Null ``entries`` name every entry of the dataset, and a selector that names
    groups or sequences, or a saved selection, is enumerated.

    Raises:
        ToolFailure: ``selection_name`` is missing, or the name or the entries are
            refused.
        ClassifiedError: A selection of that name exists, or the selection named
            in ``entries`` does not.
    """
    from mosaic.core.pipeline.inventory.scan import reportable_universe
    from mosaic.core.selections import save_selection

    if args.selection_name is None:
        raise ToolFailure(missing_field("save_selection", "selection_name"))
    scope = _entry_scope(dataset, args.entries)
    chosen = (
        sorted(reportable_universe(dataset))
        if scope is None
        else scope.entry_pairs or []
    )
    try:
        saved = save_selection(dataset, args.selection_name, chosen, args.note)
    except ClassifiedError:
        raise
    except ValueError as invalid:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_selection",
                message=str(invalid),
                detail={"selection_name": args.selection_name},
            )
        ) from invalid
    return saved.summary()


def _set_arenas(dataset: Dataset, args: ManageDatasetInput) -> str:
    """Declare the arena layout *args* names, and say what changed.

    The layout is declared for ``recording``, or for each recording that the
    entries holding ``sequence_tags`` belong to, never both. Every refusal comes
    before the one write.

    Raises:
        TagSelectionInvalid: The tags select no entry, name an unknown tag or a
            value their type refuses, or cover a divided recording in part.
        SequenceTagTableUnavailable: The table cannot be read as one revision of
            one dataset.
    """
    from mosaic.core.arenas import ArenaKey, ArenaLayout

    if args.recording is not None and args.sequence_tags is not None:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_arenas",
                message="set_arenas takes recording or sequence_tags, not both",
                detail={"recording": args.recording, "camera": args.camera},
            )
        )
    if args.recording is None and args.sequence_tags is None:
        raise ToolFailure(missing_field("set_arenas", "recording or sequence_tags"))
    if args.arena_layout is None:
        raise ToolFailure(missing_field("set_arenas", "arena_layout"))
    try:
        layout = ArenaLayout.model_validate(args.arena_layout)
    except ValidationError as invalid:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_arenas",
                message=(
                    f"the arena layout is invalid: {invalid.error_count()} problem(s)"
                ),
                detail={
                    "recording": args.recording,
                    "camera": args.camera,
                    "errors": validation_errors(invalid),
                },
            )
        ) from invalid
    keys: list[ArenaKey] = []
    if args.sequence_tags is not None:
        from mosaic.core.annotations.tag_selection import (
            selection_tag_table,
            tagged_recordings,
        )

        tags = selection_tag_table(dataset)
        selected = tagged_recordings(dataset, tags, args.sequence_tags)
        keys = [
            ArenaKey(found.group, found.sequence, args.camera) for found in selected
        ]
    try:
        if args.recording is not None:
            found = dataset.recording_named(args.recording)
            keys = [ArenaKey(found.group, found.sequence, args.camera)]
        _ = dataset.set_arena_layouts(layout, keys)
    except ValueError as invalid:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid",
                reason="invalid_arenas",
                message=str(invalid),
                detail={"recording": args.recording, "camera": args.camera},
            )
        ) from invalid
    where = f" of camera {args.camera!r}" if args.camera else ""
    count = len(layout.arenas)
    if args.recording is None:
        noun = "recording" if len(keys) == 1 else "recordings"
        return (
            f"set {count} arenas of {len(keys)} {noun} selected by sequence_tags{where}"
        )
    return f"set {count} arenas of {args.recording!r}{where}"


def _dataset_exists(target: Path) -> ErrorEnvelope:
    """Refuse a create where something already is."""
    return ErrorEnvelope(
        kind="conflict",
        reason="dataset_exists",
        message=(
            f"{str(target)!r} is not an empty directory; a dataset is created in "
            "a new or empty one"
        ),
        detail={"path": str(target)},
    )


def _planning_request(
    recipe: Recipe,
    args: PlanRecipeInput | SubmitPipelineInput,
    scope: Scope | None,
    step_scopes: dict[str, Scope],
    tag_revision: LabelRevisionRef | None,
) -> Request:
    """Return the request that a plan and the submission after it plan with.

    Raises:
        ToolFailure: A choice the request refuses, such as a step both bound
            and pinned.
    """
    from mosaic.core.pipeline.graph import planning_request

    try:
        return planning_request(
            recipe,
            scope=scope,
            bind=args.bind,
            step_scopes=step_scopes,
            tracks_run_id=args.tracks_run_id,
            tag_revision=tag_revision,
            pins=args.pins,
        )
    except ValueError as invalid:
        raise ToolFailure(
            ErrorEnvelope(
                kind="invalid", reason="invalid_arguments", message=str(invalid)
            )
        ) from invalid


def _invalid_source(message: str) -> ErrorEnvelope:
    """Refuse a source that does not declare or scan."""
    return ErrorEnvelope(kind="invalid", reason="invalid_source", message=message)


def _submission_findings(
    dataset: Dataset,
    plan: Plan,
    args: SubmitPipelineInput,
    catalog: DeclarationCatalog,
    scope: Scope,
) -> list[ConfirmationFinding]:
    """Return each confirmation condition that submitting *plan* over *scope* meets.

    *scope* is the submission's entries, a saved or tag selection resolved.
    """
    from mosaic.core.pipeline.graph import (
        GPU_LANES,
        computed_runs,
        open_duplicates,
        retired_recipe_names,
    )
    from mosaic.core.pipeline.graph.store import open_requests_sentence
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS

    findings: list[ConfirmationFinding] = []
    for planned in plan.steps:
        if planned.lane in GPU_LANES:
            findings.append(
                ConfirmationFinding(
                    "gpu_lane", f"step {planned.step_id!r} runs on lane {planned.lane}"
                )
            )
        if planned.kind == "op":
            declaration = catalog.get(planned.runs)
            if declaration is not None and declaration.category == "train":
                findings.append(
                    ConfirmationFinding(
                        "training_op",
                        f"step {planned.step_id!r} trains a model with {planned.runs}",
                    )
                )
            if planned.runs in TRACKING_ROOTS:
                findings.append(
                    ConfirmationFinding(
                        "tracking_op",
                        f"step {planned.step_id!r} runs {planned.runs}",
                    )
                )
    findings.extend(_sealed_scoring_findings(plan))
    if args.allow_partial:
        findings.append(
            ConfirmationFinding("allow_partial", "the submission sets allow_partial")
        )
    if args.bind:
        # A step id is any string, and the binds are checked only after this:
        # each is quoted, so none can add a line to the confirmation question.
        pinned = ", ".join(repr(step_id) for step_id in sorted(args.bind))
        findings.append(
            ConfirmationFinding("bind", f"the submission pins steps {pinned}")
        )
    removable = plan.cleanup.removable if plan.cleanup is not None else ()
    if plan.cleanup is not None and removable:
        storages = ", ".join(
            repr(name) for name in sorted({run.storage for run in removable})
        )
        gigabytes = sum(run.bytes for run in removable) / 1e9
        deleted = (
            f"the submission deletes {len(removable)} superseded run(s), "
            f"{gigabytes:.1f} GB, in {storages}"
        )
        written = [run for run in removable if run.retired_by]
        if written:
            retiring = sorted({run.retired_by for run in written})
            names = retired_recipe_names(dataset.base_dir, retiring)
            recipes = ", ".join(
                f"{names[digest]!r} ({digest})" if names[digest] else digest
                for digest in retiring
            )
            deleted += (
                f", {len(written)} of them written by the retired recipe(s) {recipes}"
            )
        findings.append(ConfirmationFinding("cleanup", deleted))
    duplicates = open_duplicates(
        dataset.base_dir,
        plan.recipe_digest,
        scope,
        computed_runs(plan),
        step_scopes=plan.step_scopes,
    )
    if duplicates:
        findings.append(
            ConfirmationFinding(
                "open_duplicate",
                open_requests_sentence(duplicates),
                duplicates,
            )
        )
    return findings


def _scores_against_labels(planned: PlannedStep) -> bool:
    """Does *planned* score a model's output against labels?

    That is ``evaluate-labels`` and nothing else. A step that merely has a labels
    param is not one: ``id-tag-columns`` copies labels into tag columns, and a
    classifier that applies a model reads none. A fitting step that reads labels
    is the held-out guard's, which refuses it over any held-out group.
    """
    return planned.kind == "feature" and planned.runs == _EVALUATE_LABELS


def _sealed_scoring_findings(plan: Plan) -> list[ConfirmationFinding]:
    """Return a ``score_sealed`` finding for each step scoring a sealed group.

    A step scores against labels (:func:`_scores_against_labels`) and its target
    holds an entry of a group the manifest's ``splits`` give the role ``sealed``.
    A ``test`` group does not count: scoring it is the protocol. The target is
    the tracks entries of the plan, so this fires whether or not the sealed
    group's labels are in the dataset. Keeping those labels outside it is what
    protects the seal, and this finding is the second line.
    """
    from mosaic.core.pipeline.graph.plan import step_target
    from mosaic.core.splits import groups_with_role

    sealed = set(groups_with_role(plan.splits, "sealed"))
    if not sealed:
        return []
    findings: list[ConfirmationFinding] = []
    for planned in plan.steps:
        if not _scores_against_labels(planned):
            continue
        reached = sorted({group for group, _ in step_target(planned, plan)} & sealed)
        if reached:
            groups = ", ".join(repr(group) for group in reached)
            findings.append(
                ConfirmationFinding(
                    "score_sealed",
                    f"step {planned.step_id!r} scores against labels over the "
                    f"sealed group(s) {groups}",
                )
            )
    return findings


def _approved_bytes(plan: Plan, request: Request) -> int:
    """The bytes the runs that *request* approved for removal hold, per *plan*."""
    if request.cleanup is None or plan.cleanup is None:
        return 0
    approved = {(target.storage, target.run_id) for target in request.cleanup.approved}
    return sum(
        run.bytes for run in plan.cleanup.runs if (run.storage, run.run_id) in approved
    )


def _diagnosis(
    base_dir: Path,
    step: str,
    execution_id: str,
    outcome: AttemptOutcome,
    error: RecordedErrorWire | None,
) -> AttemptDiagnosis | None:
    """Diagnose one attempt, ``None`` when it neither failed nor lost an entry."""
    from mosaic.core.pipeline.graph import entry_errors_wire

    if outcome not in ("failed", "partial"):
        return None
    lost = entry_errors_wire(base_dir, execution_id)
    entry_errors = lost if lost is not None and lost.total else None
    refusal = error.refusal if error is not None else None
    shortfall = refusal is not None and refusal.reason == "coverage_shortfall"
    quarantined = entry_errors is not None and entry_errors.held_back_by_quarantine
    return AttemptDiagnosis(
        step=step,
        execution_id=execution_id,
        outcome=outcome,
        error=error,
        entry_errors=entry_errors,
        allow_partial_applies=shortfall or quarantined,
    )


def _run_summary(snapshot: RunLogSnapshot) -> RunSummary:
    """Summarize one attempt's run-log snapshot for a listing, with its liveness."""
    from mosaic.core.pipeline.graph import recorded_error, snapshot_liveness
    from mosaic.runlog import attempt_outcome

    return RunSummary(
        execution_id=snapshot["execution_id"],
        kind=snapshot["kind"],
        target=snapshot["target"],
        run_id=snapshot["run_id"],
        status=snapshot["status"],
        outcome=attempt_outcome(snapshot["status"], snapshot["entries_failed"]),
        owner=snapshot["owner"],
        created_at=snapshot["created_at"],
        started_at=snapshot["started_at"],
        finished_at=snapshot["finished_at"],
        entries_failed=snapshot["entries_failed"],
        entries_written=snapshot["entries_written"],
        cache_hit=snapshot["cache_hit"],
        error=recorded_error(snapshot["error_json"]),
        host=snapshot["host"],
        heartbeat_at=snapshot["heartbeat_at"],
        phase=snapshot["phase"],
        liveness=snapshot_liveness(snapshot),
    )
