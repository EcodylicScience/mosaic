"""The tool backend that answers mosaic's shared tools by calling the library here.

:class:`LocalBackend` serves the datasets of an allowlist, each addressed by its
manifest ``uuid``. It opens every dataset without creating its roots, keeps one
:class:`~mosaic.core.pipeline.inventory.InventoryCache` per dataset behind a
lock, and builds the step-type catalog on a background thread. A submission is
recorded in the dataset and run by a detached driver
(:func:`~mosaic.core.pipeline.graph.start_detached_driver`), which outlives the
server.

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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

from mosaic.core.pipeline.error_envelope import ErrorEnvelope, error_envelope
from mosaic.core.pipeline.error_kind import ClassifiedError

from .spec import (
    LARGE_SCOPE_ENTRIES,
    LISTED_STATUS_NOT,
    LISTING_MISSING_SAMPLE,
    PLAN_MISSING_SAMPLE,
    ActivityPage,
    AttemptDiagnosis,
    CancelRequestInput,
    ConfirmationFinding,
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
    PlanRecipeInput,
    PreviewOutputInput,
    RunSummary,
    SequencePage,
    SequenceRow,
    SubmitPipelineInput,
    SubmittedPipeline,
    SubmittedStep,
    StepTypeDetailLevel,
    SuggestStepsInput,
    ToolName,
    ToolResult,
    ValidateRecipeInput,
    confirmation_required,
    dataset_managed,
    invalid_cursor,
    unknown_dataset,
)

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset
    from mosaic.core.dataset_summary import DatasetSummary
    from mosaic.core.manifest import DatasetManifest
    from mosaic.core.pipeline.cancel import RequestCancel
    from mosaic.core.pipeline.graph import (
        CatalogEntry,
        CatalogIndex,
        DeclarationCatalog,
        DetachedDriver,
        Plan,
        PlanWire,
        Recipe,
        RecordedErrorWire,
        Request,
        RequestStatusWire,
        ValidationReport,
    )
    from mosaic.core.pipeline.inventory import InventoryCache, InventoryWire
    from mosaic.core.pipeline.preview import ArtifactPreview
    from mosaic.core.scope import Scope
    from mosaic.runlog import AttemptOutcome, RunLogSnapshot

    from .outputs import StepSuggestions, StepTypes

__all__ = [
    "POLL_AFTER_SECONDS",
    "POLL_AFTER_TRAINING_SECONDS",
    "DriverStarter",
    "LocalBackend",
    "ToolFailure",
    "answer",
    "discovered_manifests",
]

_LOG = logging.getLogger(__name__)

POLL_AFTER_SECONDS: Final = 10
"""How long a client waits before polling a submitted request."""

POLL_AFTER_TRAINING_SECONDS: Final = 60
"""How long a client waits before polling a request that trains on a GPU."""

type DriverStarter = Callable[[Dataset, Request], DetachedDriver]
"""What starts the process driving a submitted request."""


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
    ) -> None:
        """Serve the datasets of *manifests*.

        Args:
            manifests: The manifest files, or the directories holding them.
            allow_managed: Whether the tools that change a dataset may change one
                that a service manages.
            warm: Whether to open the datasets and build the step-type catalog
                on a background thread now. Otherwise the first call that needs
                each builds it.
            start_driver: What starts the driver of a submitted request.
                :func:`~mosaic.core.pipeline.graph.start_detached_driver` when
                ``None``.
        """
        self._manifests: tuple[Path, ...] = tuple(manifests)
        self._allow_managed: bool = allow_managed
        self._start_driver: DriverStarter | None = start_driver
        self._served: dict[str, _Served] | None = None
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
        """
        manifests = discovered_manifests(root, max_depth=max_depth)
        return cls(
            manifests,
            allow_managed=allow_managed,
            warm=warm,
            start_driver=start_driver,
        )

    def dataset(self, uuid: str) -> Dataset:
        """Return the served dataset *uuid*, as opened without creating roots.

        Raises:
            ToolFailure: The allowlist names no dataset *uuid*.
        """
        return self._dataset(uuid).dataset

    def catalog(self) -> CatalogIndex:
        """Return the step-type catalog index, building it once across threads."""
        return self._catalog_index()

    def wait_until_warm(self, timeout: float | None = None) -> None:
        """Block until the warming thread, if any, has finished."""
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
        return answer(lambda: self._describe_step_types(args))

    def validate_recipe(
        self, args: ValidateRecipeInput
    ) -> ToolResult[ValidationReport]:
        """Answer ``validate_recipe``."""
        return answer(lambda: self._validate_recipe(args))

    def suggest_steps(self, args: SuggestStepsInput) -> ToolResult[StepSuggestions]:
        """Answer ``suggest_steps``."""
        return answer(lambda: self._suggest_steps(args))

    def plan_recipe(self, args: PlanRecipeInput) -> ToolResult[PlanWire]:
        """Answer ``plan_recipe``."""
        return answer(lambda: self._plan_recipe(args))

    def submit_pipeline(
        self, args: SubmitPipelineInput
    ) -> ToolResult[SubmittedPipeline]:
        """Answer ``submit_pipeline``."""
        return answer(lambda: self._submit_pipeline(args))

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

    def preview_output(self, args: PreviewOutputInput) -> ToolResult[ArtifactPreview]:
        """Answer ``preview_output``."""
        return answer(lambda: self._preview_output(args))

    # ------------------------------------------------------------- datasets

    def _list_datasets(self, args: ListDatasetsInput) -> DatasetList:
        from mosaic.core.pipeline.inventory import reportable_universe

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
            open_dataset(held.manifest, ensure_roots=False), found=found
        )

    def _list_sequences(self, args: ListSequencesInput) -> SequencePage:
        from mosaic.core.pipeline.index_csv import index_records
        from mosaic.core.pipeline.tracks_index import (
            count_cell,
            read_frame_extent,
            read_tracks_index,
            select_variant_rows,
        )

        held = self._dataset(args.dataset)
        index = read_tracks_index(held.dataset)
        if args.group is not None:
            index = index[index["group"] == args.group]
        try:
            selected = select_variant_rows(index, args.tracks_variant)
        except ValueError as error:
            raise ToolFailure(
                ErrorEnvelope(
                    kind="invalid",
                    reason="ambiguous_tracks_variant",
                    message=str(error),
                    detail={"field": "tracks_variant"},
                )
            ) from error
        rows: list[SequenceRow] = []
        for record in index_records(selected):
            extent = read_frame_extent(record)
            rows.append(
                SequenceRow(
                    group=record.get("group", ""),
                    sequence=record.get("sequence", ""),
                    tracks_variant=record.get("run_id", ""),
                    producer=record.get("producer", ""),
                    std_format=record.get("std_format", ""),
                    n_rows=count_cell(record.get("n_rows")),
                    n_keypoints=count_cell(record.get("n_keypoints")),
                    frame_min=None if extent is None else extent[0],
                    frame_max=None if extent is None else extent[1],
                )
            )
        page, next_cursor = _page(
            rows, lambda row: (row.group, row.sequence), args.limit, args.cursor
        )
        return SequencePage(items=tuple(page), total=len(rows), next_cursor=next_cursor)

    def _get_inventory(self, args: GetInventoryInput) -> InventoryWire:
        from mosaic.core.pipeline.inventory import MISSING_SAMPLE_WIRE, inventory_wire

        held = self._dataset(args.dataset)
        with held.lock:
            found = self._inventory(held).get()
        return inventory_wire(
            held.dataset,
            found,
            kinds=args.kinds,
            scope=_scope(args.entries),
            tracks_run_id=args.tracks_variant,
            status_not=args.status_not
            if args.status_not is not None or args.ref is not None
            else LISTED_STATUS_NOT,
            limit=args.limit,
            cursor=args.cursor,
            ref=args.ref,
            with_params=args.ref is not None,
            missing_sample=MISSING_SAMPLE_WIRE
            if args.ref is not None
            else LISTING_MISSING_SAMPLE,
        )

    # -------------------------------------------------------------- catalog

    def _describe_step_types(self, args: DescribeStepTypesInput) -> StepTypes:
        from mosaic.core.pipeline.graph import search_catalog

        from .outputs import StepTypes

        index = self._catalog_index()
        level = args.level
        if args.names is not None:
            return self._named_step_types(args.names, level)
        matched = [
            match
            for match in search_catalog(index.entries, args.query)
            if (args.type is None or match.entry.type == args.type)
            and (args.category is None or match.entry.category == args.category)
            and not (args.available_only and match.entry.unavailable)
        ]
        page, next_cursor = _page(
            matched, lambda match: match.key, args.limit, args.cursor
        )
        return StepTypes(
            schema_version=index.schema_version,
            mosaic_version=index.mosaic_version,
            detail=level,
            entries=tuple(match.entry for match in page),
            total=len(matched),
            next_cursor=next_cursor,
        )

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
        from mosaic.core.pipeline.graph import validation_report

        return validation_report(args.recipe, self._declarations())

    def _suggest_steps(self, args: SuggestStepsInput) -> StepSuggestions:
        from mosaic.core.pipeline.graph import (
            attach_candidates,
            input_candidates,
            insert_candidates,
        )

        from .outputs import StepSuggestions

        recipe = _recipe(args.recipe)
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

    # ------------------------------------------------------------- planning

    def _plan_recipe(self, args: PlanRecipeInput) -> PlanWire:
        from mosaic.core.pipeline.graph import plan_pipeline, plan_wire

        held = self._dataset(args.dataset)
        recipe = _recipe(args.recipe)
        catalog = self._declarations()
        with held.lock:
            plan = plan_pipeline(
                held.dataset,
                recipe,
                scope=_scope(args.entries),
                inventory=self._inventory(held),
                catalog=catalog,
                refuse=False,
            )
        return plan_wire(plan, missing_sample=PLAN_MISSING_SAMPLE)

    def _submit_pipeline(self, args: SubmitPipelineInput) -> SubmittedPipeline:
        from mosaic.core.pipeline.graph import (
            GPU_TRAIN_LANE,
            PlanChanged,
            plan_fingerprint,
            plan_pipeline,
            start_detached_driver,
            submit_request,
        )
        from mosaic.core.scope import Scope

        held = self._dataset(args.dataset)
        dataset = self._changeable(held, "submit_pipeline")
        recipe = _recipe(args.recipe)
        catalog = self._declarations()
        narrowing = _scope(args.entries) or Scope()
        with held.lock:
            cache = self._inventory(held)
            plan = plan_pipeline(
                dataset, recipe, scope=narrowing, inventory=cache, catalog=catalog
            )
            found = plan_fingerprint(plan)
            if found != args.plan_fingerprint:
                raise PlanChanged(expected=args.plan_fingerprint, found=found)
            findings = _submission_findings(dataset, recipe, plan, args, catalog)
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
                expect_plan=args.plan_fingerprint,
                refuse_duplicates=not args.confirm,
            )
        request = submitted.request
        start = self._start_driver or start_detached_driver
        try:
            driver = start(dataset, request)
        except ClassifiedError as error:
            envelope = error_envelope(error)
            detail = {**(envelope.detail or {}), "request_id": request.request_id}
            raise ToolFailure(envelope.model_copy(update={"detail": detail})) from error
        lanes = {planned.lane for planned in submitted.plan.steps}
        return SubmittedPipeline(
            request_id=request.request_id,
            recipe_digest=request.recipe_digest,
            plan_fingerprint=plan_fingerprint(submitted.plan),
            steps=tuple(
                SubmittedStep(
                    step=planned.step_id,
                    runs=planned.runs,
                    execution_id=request.execution_of(planned.step_id),
                    lane=planned.lane,
                    run_id=planned.run_id,
                    status=planned.status,
                )
                for planned in submitted.plan.steps
            ),
            duplicates=submitted.duplicates,
            driver_pid=driver.pid,
            driver_log=str(driver.log_path),
            poll_after_seconds=POLL_AFTER_TRAINING_SECONDS
            if GPU_TRAIN_LANE in lanes
            else POLL_AFTER_SECONDS,
        )

    # ------------------------------------------------------------- requests

    def _get_request_status(self, args: GetRequestStatusInput) -> RequestStatusWire:
        from mosaic.core.pipeline.cancel import read_request
        from mosaic.core.pipeline.graph import request_status_wire

        held = self._dataset(args.dataset)
        request = read_request(held.dataset.base_dir, args.request_id)
        return request_status_wire(held.dataset.base_dir, request)

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
        if args.view == "requests":
            requests = list_requests(base, limit=args.limit, after=args.cursor)
            return ActivityPage(
                view=args.view,
                requests=requests.items,
                next_cursor=requests.next_after,
            )
        runs = list_runs(run_log_dir(base), limit=args.limit, after=args.cursor)
        return ActivityPage(
            view=args.view,
            runs=tuple(_run_summary(snapshot) for snapshot in runs["items"]),
            next_cursor=runs["next_after"],
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

    def _preview_output(self, args: PreviewOutputInput) -> ArtifactPreview:
        from mosaic.core.pipeline.preview import preview_artifact, preview_ref

        held = self._dataset(args.dataset)
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
            return self._served

    def _dataset(self, uuid: str) -> _Served:
        """Return the served dataset *uuid*.

        Raises:
            ToolFailure: The allowlist names no dataset *uuid*.
        """
        held = self._datasets().get(uuid)
        if held is None:
            raise ToolFailure(unknown_dataset(uuid))
        return held

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
        """Return the declaration catalog, building it once across threads."""
        from mosaic.core.pipeline.graph import declaration_catalog

        with self._catalog_lock:
            return declaration_catalog()

    def _catalog_index(self) -> CatalogIndex:
        """Return the catalog index, building the editor catalog once across threads."""
        from mosaic.core.pipeline.graph import catalog_index

        with self._catalog_lock:
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
        )

        running = {
            attempt.step_id
            for attempt in request_rollup(dataset.base_dir, request).steps
            if attempt.started and not attempt.terminal
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


def _scope(entries: list[tuple[str, str]] | None) -> Scope | None:
    """Return the scope naming *entries*, ``None`` for the whole dataset."""
    from mosaic.core.scope import Scope

    return None if entries is None else Scope(entries=list(entries))


def _recipe(document: object) -> Recipe:
    """Parse a recipe document, reporting a malformed one as its problems."""
    from mosaic.core.pipeline.graph import read_recipe

    return read_recipe(document)


def _submission_findings(
    dataset: Dataset,
    recipe: Recipe,
    plan: Plan,
    args: SubmitPipelineInput,
    catalog: DeclarationCatalog,
) -> list[ConfirmationFinding]:
    """Return each confirmation condition that submitting *plan* meets."""
    from mosaic.core.pipeline.graph import (
        GPU_INFER_LANE,
        GPU_TRAIN_LANE,
        asked_of,
        open_requests_matching,
        recipe_digest,
    )
    from mosaic.core.pipeline.tracking_roots import TRACKING_ROOTS
    from mosaic.core.scope import Scope

    findings: list[ConfirmationFinding] = []
    for planned in plan.steps:
        if planned.lane in (GPU_TRAIN_LANE, GPU_INFER_LANE):
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
        asked = len(asked_of(planned, plan))
        if asked > LARGE_SCOPE_ENTRIES:
            findings.append(
                ConfirmationFinding(
                    "large_scope",
                    f"step {planned.step_id!r} is asked for {asked} entries",
                )
            )
    if args.allow_partial:
        findings.append(
            ConfirmationFinding("allow_partial", "the submission sets allow_partial")
        )
    if args.bind:
        pinned = ", ".join(sorted(args.bind))
        findings.append(
            ConfirmationFinding("bind", f"the submission pins steps {pinned}")
        )
    if args.entries is None:
        findings.append(
            ConfirmationFinding(
                "whole_dataset",
                f"the submission covers all {len(plan.scope)} entries of the dataset",
            )
        )
    duplicates = open_requests_matching(
        dataset.base_dir,
        recipe_digest(recipe),
        Scope() if args.entries is None else Scope(entries=list(args.entries)),
    )
    if duplicates:
        findings.append(
            ConfirmationFinding(
                "open_duplicate",
                f"request(s) {', '.join(duplicates)} of this recipe have not ended",
            )
        )
    return findings


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
    """Summarize one attempt's run-log snapshot for a listing."""
    from mosaic.core.pipeline.graph import recorded_error
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
    )
