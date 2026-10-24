"""Scoring a finished recipe on held-out data that is added afterwards.

An agent builds a recipe and fits its models on the training groups. The
held-out data stays out of the dataset it can read, and an operator scores the
recipe on it with :func:`score_heldout`. The score means something only if the
data is new to every fit, so the function is built around one check: the run id
of every step of the recipe is compared before and after the held-out data is
added, and a step whose id moved is a step that would be fitted again on
different data. Nothing is run then.

**It works on a copy.** The study is copied to ``out`` and the held-out source,
its conversion, the evaluation steps and the reports all go there. The agent's
dataset is opened read-only and left as it was, and a report written beside the
copy is not served by any resource of the study. A dry run that names a group
already in the dataset (:class:`str` as *heldout*) needs no copy at all, because
nothing is added, and plans the study in place.

**The steps, in order.**

1. Plan the recipe on the copy, which still holds the training data only, and
   record the run id and status of every step (``scoring/state.json``).
2. Declare the held-out file or directory as a tracks source and a labels source,
   scan them and convert them. The conversion calls take the params of the
   recipe's own ``convert-tracks`` and ``convert-labels`` steps, because the
   variant a step names derives from them and a default would mint another. A
   convert *step* is not used for this: the plan counts converted entries as its
   target and would report itself complete without converting the new file.
3. Append one ``evaluate-labels`` step per answer step, scored over the held-out
   entries only, and plan the recipe again.
4. Compare the run ids of the recipe's own steps with step 1.
5. Submit the scoring recipe and run it. Every step that is scope-free computes
   the held-out entries it is missing and nothing else.
6. Read each evaluation's ``metrics.json`` and write ``test_scores.json`` and
   ``test_scores.txt`` beside the copy's manifest.

A group whose role in the manifest's ``splits`` is ``test`` or ``sealed`` can be
named instead of a file. It is already in the dataset, so steps 2 and 4 have
nothing to add or compare, and the planner's held-out guard
(:func:`~.heldout.held_out_fit_problem`) is what stops a fit from covering it.

The module imports nothing from ``behavior`` or ``tracking``. An evaluation's
report is read through :func:`~mosaic.core.pipeline.inventory.read_run_report`
and :class:`~mosaic.core.evaluation.EvaluationReport`.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal

from pydantic import field_validator

from mosaic.core.json_value import JsonValue
from mosaic.core.scope import Scope
from mosaic.core.splits import HELD_OUT_ROLES, role_of
from mosaic.core.strict_model import StrictModel
from mosaic.user_paths import user_path

from ..error_kind import ClassifiedError, ErrorKind
from .digest import recipe_digest
from .model import FeatureStepSpec, OpStepSpec, Recipe, StepRef
from .plan import Plan, plan_pipeline
from .request import PlanChanged, submit_request
from .run import run_pipeline
from .serialize import plan_fingerprint

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mosaic.core.dataset import Dataset
    from mosaic.core.evaluation import EvaluationReport
    from mosaic.core.pipeline.job import CancelToken
    from mosaic.core.pipeline.parallelism import ParallelSettings

__all__ = [
    "HELDOUT_SOURCE_ID",
    "HeldOutFiles",
    "MemberScore",
    "ModelScore",
    "MovedRunId",
    "SCORES_JSON",
    "SCORES_TXT",
    "ScoreAnswers",
    "ScoreProtocol",
    "ScoreRefused",
    "ScoreResult",
    "md5_of_file",
    "score_heldout",
]

HELDOUT_SOURCE_ID: Final = "heldout"
"""The id of the tracks source and of the labels source that hold the held-out data."""

SCORING_DIR: Final = "scoring"
"""Where, under the copy, the scoring recipe, the plans and the state are kept."""

SCORES_JSON: Final = "test_scores.json"
SCORES_TXT: Final = "test_scores.txt"
"""The reports, written into the copy's directory."""

EVALUATION_PREFIX: Final = "test__"
"""What an evaluation step's id starts with, followed by the step it scores."""

_COPY_MARKER: Final = "copy-complete"
"""Written last into ``scoring/`` of a copy, so ``--reuse`` can tell a finished copy
of this study from a directory that merely holds a manifest."""

_STATE_FILE: Final = "state.json"
_EVALUATE: Final = "evaluate-labels"
_DEFAULT_REPORT_FILE: Final = "metrics.json"


class ScoreRefused(ClassifiedError):
    """A scoring that cannot proceed as asked, with the reason in its message."""

    error_kind: ClassVar[ErrorKind] = "invalid"


# --- the inputs -----------------------------------------------------------------


class ScoreAnswers(StrictModel):
    """The steps to score, grouped under the name they are reported by.

    Attributes:
        models: Each name mapped to the ids of the steps that answer it. One name
            with several steps is one model fitted several times (seeds), and its
            row of the table is the mean and standard deviation over them. A step
            listed under two names is evaluated once.
    """

    models: dict[str, list[str]]

    @field_validator("models")
    @classmethod
    def _each_name_has_steps(cls, models: dict[str, list[str]]) -> dict[str, list[str]]:
        if not models:
            raise ValueError("models names no model to score")
        for name, steps in models.items():
            if not name.strip():
                raise ValueError("a model name must not be empty")
            if not steps:
                raise ValueError(f"model {name!r} lists no step")
            if len(set(steps)) != len(steps):
                raise ValueError(f"model {name!r} lists a step twice")
        return models


class ScoreProtocol(StrictModel):
    """How an evaluation scores a step's predictions.

    The fields are those of ``evaluate-labels`` that decide what is counted, and
    each left unset keeps that step's own default.

    Attributes:
        perspective: Which rows of a pair output to score: one individual's
            perspective as an integer, ``"pooled"`` or ``"mean_probabilities"``.
        average_over: The classes the macro F1 averages over.
        classes: The classes to report, in order.
        prediction_column: The column holding each row's predicted class.
    """

    perspective: int | Literal["pooled", "mean_probabilities"] | None = None
    average_over: list[int] | None = None
    classes: list[int] | None = None
    prediction_column: str | None = None

    def evaluation_params(self, labels_step: str) -> dict[str, JsonValue]:
        """The params of one ``evaluate-labels`` step scored under this protocol."""
        params: dict[str, JsonValue] = {"labels": {"step": labels_step}}
        if self.perspective is not None:
            params["perspective"] = self.perspective
        if self.average_over is not None:
            params["average_over"] = list(self.average_over)
        if self.classes is not None:
            params["classes"] = list(self.classes)
        if self.prediction_column is not None:
            params["prediction_column"] = self.prediction_column
        return params


@dataclass(frozen=True, slots=True)
class HeldOutFiles:
    """Held-out raw files to declare as a tracks source and a labels source.

    The converter formats come from the recipe's own ``convert-tracks`` and
    ``convert-labels`` steps, so a format is given here only to check it.

    Attributes:
        tracks: A file, or a directory searched recursively.
        labels: Where the labels are, when they are not beside the tracks.
        src_format: The raw format, which must be the one the recipe converts.
        patterns: Globs selecting files under a directory. Unset takes the
            source's defaults.
        multi_sequences_per_file: One file holds several sequences.
        group_from: ``filename`` or ``parent``, for multi-sequence files.
        group_pattern: A regex extracting the group from a path.
    """

    tracks: Path
    labels: Path | None = None
    src_format: str | None = None
    patterns: tuple[str, ...] | None = None
    multi_sequences_per_file: bool = False
    group_from: str | None = None
    group_pattern: str | None = None


# --- the result -----------------------------------------------------------------


class MovedRunId(StrictModel):
    """A step of the recipe whose run id changed when the held-out data joined."""

    step: str
    before: str | None
    after: str | None

    def __str__(self) -> str:
        return f"RUN IDS CHANGED: {self.step} {self.before} -> {self.after}"


class MemberScore(StrictModel):
    """One answer step's score on the held-out entries.

    Attributes:
        step: The answer step that was scored.
        evaluation: The ``evaluate-labels`` step that scored it.
        run_id: The evaluation's run.
        n_scored: Rows scored.
        accuracy: The share of scored rows predicted correctly.
        macro_f1: Mean F1 over the protocol's classes.
        per_class_f1: The F1 of each class, by the class's name or number.
    """

    step: str
    evaluation: str
    run_id: str
    n_scored: int
    accuracy: float | None
    macro_f1: float | None
    per_class_f1: dict[str, float | None]


class ModelScore(StrictModel):
    """One name of the answers: its members, and their mean and spread.

    ``macro_f1_sd`` and the spreads of ``per_class_sd`` are ``None`` for a name
    with one member.
    """

    name: str
    members: list[MemberScore]
    macro_f1_mean: float | None
    macro_f1_sd: float | None
    per_class_mean: dict[str, float | None]
    per_class_sd: dict[str, float | None]


class ScoreResult(StrictModel):
    """What :func:`score_heldout` found.

    Attributes:
        out: The scored copy, or the study itself for a dry run in place.
        dry_run: Whether it stopped before running anything.
        heldout_scope: The held-out entries as a scope selector.
        heldout_entries: How many entries that is, or ``None`` when the group was
            already in the dataset.
        steps_before: The run id and status of each of the recipe's steps before
            the held-out data joined.
        moved: The steps whose run id differs afterwards. Non-empty means nothing
            was run.
        warnings: Things the operator should read: answer steps that were not
            complete, evaluations that did not finish.
        plan_fingerprint: The fingerprint of the scoring plan, which
            ``--expect-plan`` of a later run compares.
        request_id: The submitted request, empty when nothing ran.
        models: The scores, one entry per name of the answers.
        table: The scores as text, as written to ``test_scores.txt``.
        files: The report files written.
    """

    out: str
    dry_run: bool
    heldout_scope: dict[str, JsonValue]
    heldout_entries: int | None
    steps_before: dict[str, dict[str, str | None]]
    moved: list[MovedRunId]
    warnings: list[str]
    plan_fingerprint: str
    request_id: str = ""
    models: list[ModelScore] = []
    table: str = ""
    files: list[str] = []


class _Record(StrictModel):
    run_id: str | None
    status: str
    fits: bool = True


class _State(StrictModel):
    """What a rerun with ``reuse`` must not recompute: the train-only plan.

    ``entries_before`` is the tracks variant's entries before the held-out source
    was converted, which a rerun after a partly failed conversion cannot read off
    the copy, since the first attempt converted some. ``inputs`` names the held-out
    data and the step scopes the copy was started with.
    """

    recipe: str
    before: dict[str, _Record]
    tracks_variant: str = ""
    entries_before: list[list[str]] | None = None
    inputs: str = ""
    heldout: dict[str, JsonValue] | None = None
    entries: int | None = None


# --- helpers ----------------------------------------------------------------------


def _silent(line: str) -> None:
    """The default *notify*: say nothing."""


def md5_of_file(path: Path) -> str:
    """The MD5 hex digest of *path*, read in blocks."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _only_op(recipe: Recipe, kind: str) -> OpStepSpec | None:
    found = [
        step
        for step in recipe.steps
        if isinstance(step, OpStepSpec) and step.kind == kind
    ]
    if len(found) > 1:
        ids = [step.id for step in found]
        raise ScoreRefused(
            f"the recipe has {len(found)} {kind} steps ({ids}); scoring needs one, "
            "because the variant it converts the held-out data into derives from it"
        )
    return found[0] if found else None


def _check_answers(recipe: Recipe, answers: ScoreAnswers) -> None:
    for name, steps in answers.models.items():
        for step_id in steps:
            try:
                step = recipe.step(step_id)
            except KeyError:
                raise ScoreRefused(
                    f"the answers name step {step_id!r} under {name!r}, which is "
                    f"not a step of the recipe. Its steps: {list(recipe.ids)}"
                ) from None
            if not isinstance(step, FeatureStepSpec):
                raise ScoreRefused(
                    f"the answers name step {step_id!r} under {name!r}, which runs "
                    "an op. An answer is a feature step that writes predictions."
                )


def _copy_study(study: Dataset, out: Path, *, reuse: bool) -> Dataset:
    """Copy the study's directory to *out* and open the copy.

    The driver leases are left behind, because a lease names a process of the
    study. Everything else is copied, so the copy plans as the study does.
    """
    from mosaic.core.dataset import open_dataset

    source = study.base_dir.resolve()
    target = user_path(out).resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ScoreRefused(
            f"--out {target} and the study {source} contain one another; the "
            "scored copy goes in a directory of its own"
        )
    manifest = target / study.manifest_path.name
    marker = target / SCORING_DIR / _COPY_MARKER
    if target.exists() and any(target.iterdir()):
        if not reuse:
            raise ScoreRefused(
                f"{target} exists. Pass --reuse to continue with the copy it holds"
            )
        if not manifest.exists():
            raise ScoreRefused(f"{target} holds no {manifest.name}; it is not a copy")
        if not marker.is_file() or marker.read_text(encoding="utf-8") != _copy_id(
            study
        ):
            raise ScoreRefused(
                f"{target} is not a complete scoring copy of this study, so held-out "
                "data would be written into something else. Use a new --out"
            )
    else:

        def skip(directory: str, names: list[str]) -> list[str]:
            return ["drivers"] if Path(directory).name == ".mosaic" else []

        _ = shutil.copytree(
            source, target, symlinks=True, ignore=skip, dirs_exist_ok=True
        )
        marker.parent.mkdir(exist_ok=True)
        _ = marker.write_text(_copy_id(study), encoding="utf-8")
    return open_dataset(manifest)


def _copy_id(study: Dataset) -> str:
    """What the marker of a finished copy of *study* holds: the study's own uuid."""
    return study.uuid or ""


def _check_heldout_role(ds: Dataset, group: str) -> None:
    """Refuse a ``--heldout`` name that is not a group of role test or sealed."""
    role = role_of(ds, group)
    if role not in HELD_OUT_ROLES:
        raise ScoreRefused(
            f"--heldout {group!r} is neither an existing path nor a group whose "
            f"role is test or sealed (its role: {role}). Give a file or a "
            "directory, or set the role with 'mosaic splits set'"
        )


def _plan(ds: Dataset, recipe: Recipe, step_scopes: Mapping[str, Scope]) -> Plan:
    """The recipe planned over the dataset, its problems kept on the plan."""
    return plan_pipeline(ds, recipe, step_scopes=step_scopes, refuse=False)


def _refuse_problems(plan: Plan, what: str) -> None:
    if plan.problems:
        listed = "; ".join(str(problem) for problem in plan.problems)
        raise ScoreRefused(f"the {what} plan has problems: {listed}")


def _records(plan: Plan) -> dict[str, _Record]:
    return {
        step.step_id: _Record(run_id=step.run_id, status=step.status, fits=step.fits)
        for step in plan.steps
    }


def _inputs_digest(heldout: HeldOutFiles | str, scopes: Mapping[str, Scope]) -> str:
    """Name the held-out data and the step scopes a copy is started with."""
    named: object = heldout
    if isinstance(heldout, HeldOutFiles):
        named = {
            "tracks": str(user_path(heldout.tracks).resolve()),
            "labels": None
            if heldout.labels is None
            else str(user_path(heldout.labels).resolve()),
            "src_format": heldout.src_format,
            "patterns": heldout.patterns,
            "multi_sequences_per_file": heldout.multi_sequences_per_file,
            "group_from": heldout.group_from,
            "group_pattern": heldout.group_pattern,
        }
    document = {
        "heldout": named,
        "step_scopes": {
            step: scope.model_dump(mode="json", exclude_none=True)
            for step, scope in sorted(scopes.items())
        },
    }
    text = json.dumps(document, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_state(work: Path) -> _State | None:
    path = work / _STATE_FILE
    if not path.exists():
        return None
    return _State.model_validate_json(path.read_text(encoding="utf-8"))


def _write_state(work: Path, state: _State) -> None:
    _ = (work / _STATE_FILE).write_text(
        state.model_dump_json(indent=1), encoding="utf-8"
    )


def _declared(ds: Dataset, kind: Literal["tracks", "labels"]) -> bool:
    return any(source.id == HELDOUT_SOURCE_ID for source in ds.scan_sources(kind))


def _declare_sources(
    ds: Dataset,
    files: HeldOutFiles,
    tracks_format: str,
    labels_format: str,
) -> None:
    """Declare the held-out data as a tracks source and a labels source."""
    from mosaic.core.manifest import LabelsScanSource, TracksScanSource

    def declaration(where: Path, src_format: str) -> dict[str, object]:
        where = user_path(where).resolve()
        if not where.exists():
            raise ScoreRefused(f"{where} does not exist")
        found: dict[str, object] = {
            "id": HELDOUT_SOURCE_ID,
            "src_format": src_format,
            "multi_sequences_per_file": files.multi_sequences_per_file,
        }
        if where.is_file():
            # A file source (``files=``) is not walked by the raw scans, which look
            # for globs under a directory, so one file is a directory source
            # that matches its exact name and descends nowhere.
            found["path"] = str(where.parent)
            found["patterns"] = (glob.escape(where.name),)
            found["recursive"] = False
        else:
            found["path"] = str(where)
            if files.patterns is not None:
                found["patterns"] = files.patterns
        if files.group_from is not None:
            found["group_from"] = files.group_from
        if files.group_pattern is not None:
            found["group_pattern"] = files.group_pattern
        return found

    try:
        if not _declared(ds, "tracks"):
            ds.add_scan_source(
                TracksScanSource.model_validate(
                    declaration(files.tracks, tracks_format)
                )
            )
        if not _declared(ds, "labels"):
            ds.add_scan_source(
                LabelsScanSource.model_validate(
                    declaration(files.labels or files.tracks, labels_format)
                )
            )
    except ValueError as error:
        raise ScoreRefused(
            f"the held-out source cannot be declared: {error}"
        ) from error


def _tracks_entries(ds: Dataset, variant: str) -> set[tuple[str, str]]:
    from ..tracks_index import read_tracks_index

    index = read_tracks_index(ds)
    rows = index[index["run_id"] == variant]
    return set(zip(rows["group"], rows["sequence"], strict=True))


def _add_heldout(
    ds: Dataset,
    files: HeldOutFiles,
    tracks_step: OpStepSpec,
    labels_step: OpStepSpec,
    tracks_variant: str,
    before: set[tuple[str, str]],
    say: Callable[[str], None],
) -> tuple[Scope, int]:
    """Declare, scan and convert the held-out data, and say which entries are new.

    *before* is the variant's entries from before the first attempt to convert,
    so that a rerun after a partial failure still counts what that attempt
    converted as new.
    """
    from ..convert_ops import ConvertLabelsParams, ConvertTracksParams

    tracks = ConvertTracksParams.model_validate(tracks_step.params)
    labels = ConvertLabelsParams.model_validate(labels_step.params)
    for given, wanted in (
        (files.src_format, tracks.source_format),
        (files.src_format, labels.source_format),
    ):
        if given is not None and given != wanted:
            raise ScoreRefused(
                f"--src-format {given!r} differs from the format {wanted!r} that the "
                "recipe converts. The held-out data is converted by the recipe's own "
                "steps, so it must be of that format"
            )
    _declare_sources(ds, files, tracks.source_format, labels.source_format)
    say(f"scan the held-out source {HELDOUT_SOURCE_ID!r}")
    _ = ds.scan_tracks(only=[HELDOUT_SOURCE_ID])
    _ = ds.scan_labels(only=[HELDOUT_SOURCE_ID])
    say("convert it with the recipe's own convert-tracks and convert-labels params")
    converted = ds.convert_all_tracks(
        params=dict(tracks.params),
        merge_per_sequence=tracks.merge_per_sequence,
        group_from=tracks.group_from,
        source_format=tracks.source_format,
    )
    if converted.failed:
        raise ScoreRefused(
            f"{converted.failed} held-out raw file(s) did not convert. Fix the files "
            "and run again with --reuse. To change a source option (--patterns, "
            "--group-from, ...), delete --out and start again"
        )
    _ = ds.convert_all_labels(
        kind=labels.kind,
        params=dict(labels.params),
        source_format=labels.source_format,
    )
    after = _tracks_entries(ds, tracks_variant)
    new = sorted(after - before)
    if not new:
        raise ScoreRefused(
            "the held-out source added no entry to the tracks variant "
            f"{tracks_variant}. Check its path, patterns and format"
        )
    groups = sorted({group for group, _ in new})
    added, named = set(new), set(groups)
    whole = all(entry in added for entry in after if entry[0] in named)
    if whole:
        return Scope(groups=groups), len(new)
    return Scope(entries=new), len(new)


def _evaluation_steps(
    recipe: Recipe,
    answers: ScoreAnswers,
    labels_step: str,
    protocol: ScoreProtocol,
) -> tuple[list[FeatureStepSpec], dict[str, str]]:
    """One evaluation per answer step, and the answer step each one scores."""
    steps: list[FeatureStepSpec] = []
    scored: dict[str, str] = {}
    for answer in dict.fromkeys(
        step for listed in answers.models.values() for step in listed
    ):
        identifier = f"{EVALUATION_PREFIX}{answer}"
        if identifier in recipe.ids:
            raise ScoreRefused(
                f"the recipe already has a step {identifier!r}, which is the id the "
                f"evaluation of {answer!r} takes"
            )
        steps.append(
            FeatureStepSpec(
                id=identifier,
                feature=_EVALUATE,
                inputs=[StepRef(step=answer)],
                params=protocol.evaluation_params(labels_step),
            )
        )
        scored[identifier] = answer
    return steps, scored


def _moved(
    before: Mapping[str, _Record], after: Mapping[str, str | None]
) -> tuple[list[MovedRunId], list[str]]:
    """The steps that would be fitted again, and the steps that only re-score.

    A step whose run id changed is fitted again only if it fits. One that learns
    nothing, such as the recipe's own ``evaluate-labels``, moves because it covers
    the held-out entries too and is computed again over them, and the score of a
    model is not disturbed by it.
    """
    refit: list[MovedRunId] = []
    rescored: list[str] = []
    for step, record in before.items():
        if after.get(step) == record.run_id:
            continue
        if record.fits:
            refit.append(
                MovedRunId(step=step, before=record.run_id, after=after.get(step))
            )
        else:
            rescored.append(step)
    return refit, rescored


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _read_report(ds: Dataset, storage: str, run_id: str) -> EvaluationReport | None:
    from mosaic.core.evaluation import EvaluationReport

    from ..index import feature_run_root
    from ..inventory import read_run_report
    from .resolve import feature_class_for_slug

    named: object = getattr(
        feature_class_for_slug(_EVALUATE), "report_file", _DEFAULT_REPORT_FILE
    )
    name = named if isinstance(named, str) else _DEFAULT_REPORT_FILE
    raw = read_run_report(feature_run_root(ds, storage, run_id), name)
    return None if raw is None else EvaluationReport.model_validate(raw)


def _member(
    step: str, evaluation: str, run_id: str, report: EvaluationReport
) -> MemberScore:
    return MemberScore(
        step=step,
        evaluation=evaluation,
        run_id=run_id,
        n_scored=report.n_scored,
        accuracy=_finite(report.accuracy),
        macro_f1=_finite(report.macro_f1),
        per_class_f1={
            row.name or str(row.label): _finite(row.f1) for row in report.per_class
        },
    )


def _mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return statistics.mean(values), (
        statistics.stdev(values) if len(values) > 1 else None
    )


def _model(name: str, members: list[MemberScore]) -> ModelScore:
    classes = list(
        dict.fromkeys(label for member in members for label in member.per_class_f1)
    )
    macro_mean, macro_sd = _mean_sd(
        [m.macro_f1 for m in members if m.macro_f1 is not None]
    )
    means: dict[str, float | None] = {}
    spreads: dict[str, float | None] = {}
    for label in classes:
        means[label], spreads[label] = _mean_sd(
            [v for m in members if (v := m.per_class_f1.get(label)) is not None]
        )
    return ModelScore(
        name=name,
        members=members,
        macro_f1_mean=macro_mean,
        macro_f1_sd=macro_sd,
        per_class_mean=means,
        per_class_sd=spreads,
    )


def _cell(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "-"
    return f"{mean:.4f}" if sd is None else f"{mean:.4f}±{sd:.4f}"


def _table(models: list[ModelScore]) -> str:
    """The scores as one block per name: a row per step, then mean and spread."""
    lines: list[str] = []
    for model in models:
        classes = list(model.per_class_mean)
        lines.append(f"[{model.name}]")
        header = f"{'model step':<24}{'macro F1':>15}"
        header += "".join(f"{label:>15}" for label in classes) + f"{'rows':>10}"
        lines.append(header)
        for member in model.members:
            row = f"{member.step:<24}{_cell(member.macro_f1, None):>15}"
            row += "".join(
                f"{_cell(member.per_class_f1.get(label), None):>15}"
                for label in classes
            )
            lines.append(row + f"{member.n_scored:>10}")
        if len(model.members) > 1:
            row = (
                f"{'mean ± sd':<24}{_cell(model.macro_f1_mean, model.macro_f1_sd):>15}"
            )
            row += "".join(
                f"{_cell(model.per_class_mean[label], model.per_class_sd[label]):>15}"
                for label in classes
            )
            lines.append(row)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --- the entry point -----------------------------------------------------------------


def score_heldout(
    study: Dataset,
    recipe: Recipe,
    *,
    answers: ScoreAnswers,
    heldout: HeldOutFiles | str,
    out: Path,
    protocol: ScoreProtocol | None = None,
    step_scopes: Mapping[str, Scope] | None = None,
    reuse: bool = False,
    dry_run: bool = False,
    expect_plan: str | None = None,
    parallel: ParallelSettings | None = None,
    cancel_token: CancelToken | None = None,
    notify: Callable[[str], None] | None = None,
) -> ScoreResult:
    """Score *recipe*'s answer steps on held-out data added after the fact.

    Args:
        study: The agent's dataset. It is read and never written.
        recipe: The finished recipe. It needs one ``convert-labels`` step, and a
            ``convert-tracks`` step when *heldout* is files.
        answers: The steps to score, by name.
        heldout: Held-out files to add, or the name of a group already in the
            dataset whose role is ``test`` or ``sealed``.
        out: The directory the copy of the study goes to, and where the reports
            are written. It must not exist (or be empty) unless *reuse* is set.
        protocol: How the evaluations score. Unset takes ``evaluate-labels``'s
            defaults.
        step_scopes: The scopes the recipe was run with, which narrow the fitting
            steps to the training groups.
        reuse: Continue with the copy in *out*, after a failure or a dry run. The
            train-only plan recorded in it is the one compared.
        dry_run: Stop after the run id comparison. Nothing is run or scored. A
            group named by *heldout* is planned on the study itself, which is
            then not written at all.
        expect_plan: The fingerprint of the scoring plan that a dry run printed.
            The run is refused if the plan has changed since.
        parallel: How many entries a step computes at once.
        cancel_token: Fired to stop the run.
        notify: Called with a line for each stage.

    Returns:
        The result. Its ``moved`` is non-empty when a step of the recipe would be
        fitted again, and then nothing was run.

    Raises:
        ScoreRefused: The inputs do not fit, the dataset refuses the plan, or the
            held-out source cannot be added.
        RecipeInvalid: The recipe is malformed.
        PlanChanged: *expect_plan* differs from the scoring plan.
        StepRefused: A step declined to run.
    """
    say: Callable[[str], None] = _silent if notify is None else notify
    protocol = protocol or ScoreProtocol()
    scopes = dict(step_scopes or {})
    files = None if isinstance(heldout, str) else heldout
    labels_step = _only_op(recipe, "convert-labels")
    tracks_step = _only_op(recipe, "convert-tracks")
    if labels_step is None:
        raise ScoreRefused(
            "the recipe has no convert-labels step. The evaluations read the "
            "labels variant that step names"
        )
    if files is not None and tracks_step is None:
        raise ScoreRefused(
            "the recipe has no convert-tracks step, so the variant to convert the "
            "held-out files into is not known"
        )
    _check_answers(recipe, answers)
    # What is known without the copy is refused before the study is copied.
    evaluations, scored = _evaluation_steps(recipe, answers, labels_step.id, protocol)
    if isinstance(heldout, str):
        _check_heldout_role(study, heldout)
    if files is not None:
        for where in (files.tracks, files.labels):
            if where is not None and not user_path(where).exists():
                raise ScoreRefused(f"{where} does not exist")

    in_place = dry_run and files is None
    if in_place:
        ds = study
        work: Path | None = None
        say("plan on the study itself: a group in the dataset needs no copy")
    else:
        say(f"copy {study.base_dir} -> {out}")
        ds = _copy_study(study, out, reuse=reuse)
        work = ds.base_dir / SCORING_DIR
        work.mkdir(exist_ok=True)
    state = None if work is None or not reuse else _read_state(work)
    digest = recipe_digest(recipe)
    warnings: list[str] = []

    # 1. the train-only plan
    inputs = _inputs_digest(heldout, scopes)
    if state is not None and state.recipe == digest:
        if state.inputs and state.inputs != inputs:
            raise ScoreRefused(
                f"the copy in {out} was started with other --heldout data or "
                "step scopes than this run names, so the plan recorded in it does "
                "not apply. Give the same ones, or remove --reuse and start again "
                "in a new --out"
            )
        say("use the train-only plan recorded in the copy")
        before = state.before
    else:
        if files is not None and (_declared(ds, "tracks") or _declared(ds, "labels")):
            recorded_for = (
                "holds no record of the plan before it was added"
                if state is None
                else "holds the plan of another recipe, not this one"
            )
            raise ScoreRefused(
                f"the copy already declares the {HELDOUT_SOURCE_ID!r} source but "
                f"{recorded_for}, so a moved run id could not be seen. Remove "
                "--reuse and --out, and start again"
            )
        say("plan the recipe on the data it was built on")
        first = _plan(ds, recipe, scopes)
        _refuse_problems(first, "train-only")
        before = _records(first)
        variant = (
            "" if tracks_step is None else first.step(tracks_step.id).tracks_variant
        )
        state = _State(
            recipe=digest, before=before, tracks_variant=variant, inputs=inputs
        )
        if work is not None:
            _write_state(work, state)
    unfinished = [
        step for step, record in before.items() if record.status != "complete"
    ]
    if unfinished:
        warnings.append(
            f"not complete before scoring: {', '.join(unfinished)}. The recipe was "
            "not fully computed in its own dataset, so scoring computes the rest"
        )

    # 2. the held-out data
    entries: int | None = None
    if isinstance(heldout, str):
        # Again on the copy, which a reused directory may have changed.
        _check_heldout_role(ds, heldout)
        scope = Scope(groups=[heldout])
    elif state.heldout is not None:
        say("the held-out data is already converted in the copy")
        scope = Scope.model_validate(state.heldout)
        entries = state.entries
    else:
        if tracks_step is None or not state.tracks_variant:
            raise ScoreRefused(
                "the recipe's convert-tracks step names no tracks variant, so the "
                "held-out data cannot be located after it is converted"
            )
        recorded = state.entries_before
        if recorded is None:
            # Recorded before the conversion, so that a rerun after some of the
            # files failed still counts the ones that converted as new.
            recorded = [
                list(entry)
                for entry in sorted(_tracks_entries(ds, state.tracks_variant))
            ]
            state = state.model_copy(update={"entries_before": recorded})
            if work is not None:
                _write_state(work, state)
        known = {(group, sequence) for group, sequence in recorded}
        scope, entries = _add_heldout(
            ds, heldout, tracks_step, labels_step, state.tracks_variant, known, say
        )
        if work is not None:
            state = state.model_copy(
                update={
                    "heldout": scope.model_dump(mode="json", exclude_none=True),
                    "entries": entries,
                }
            )
            _write_state(work, state)

    # 3. the scoring recipe and its plan
    scoring = recipe.model_copy(
        update={
            "name": f"{recipe.name or 'recipe'}-heldout",
            "steps": [*recipe.steps, *evaluations],
        }
    )
    all_scopes = {**scopes, **{step.id: scope for step in evaluations}}
    say("plan the recipe with the held-out data and the evaluations")
    after = _plan(ds, scoring, all_scopes)
    _refuse_problems(after, "scoring")
    fingerprint = plan_fingerprint(after)
    if work is not None:
        _ = (work / "scoring-recipe.json").write_text(
            scoring.model_dump_json(indent=1), encoding="utf-8"
        )

    # 4. did any step of the recipe move?
    moved, rescored = _moved(before, after.run_ids)
    if rescored:
        warnings.append(
            f"{', '.join(rescored)} cover(s) the held-out entries and run over "
            "them too. They fit nothing, so no model changed"
        )
    result = ScoreResult(
        out=str(ds.base_dir),
        dry_run=dry_run,
        heldout_scope=scope.model_dump(mode="json", exclude_none=True),
        heldout_entries=entries,
        steps_before={
            step: {"run_id": record.run_id, "status": record.status}
            for step, record in before.items()
        },
        moved=moved,
        warnings=warnings,
        plan_fingerprint=fingerprint,
    )
    if expect_plan is not None and expect_plan != fingerprint and (moved or dry_run):
        # Otherwise ``submit_request`` compares it, below.
        raise PlanChanged(expected=expect_plan, found=fingerprint)
    if moved or dry_run:
        return result

    # 5. run it
    say(
        "run: the held-out entries' features, the answers' predictions, the evaluations"
    )
    submitted = submit_request(
        ds, scoring, step_scopes=all_scopes, expect_plan=expect_plan
    ).request
    _ = run_pipeline(
        ds, scoring, request=submitted, cancel_token=cancel_token, parallel=parallel
    )

    # 6. the scores
    final = _plan(ds, scoring, all_scopes)
    drifted, _ = _moved(before, final.run_ids)
    if drifted:
        return result.model_copy(
            update={"moved": drifted, "request_id": submitted.request_id}
        )
    by_answer: dict[str, MemberScore] = {}
    for identifier, answer in scored.items():
        planned = final.step(identifier)
        report = (
            _read_report(ds, planned.storage_name, planned.run_id)
            if planned.run_id and planned.is_complete
            else None
        )
        if report is None:
            warnings.append(f"{identifier} did not finish ({planned.status}); no score")
            continue
        by_answer[answer] = _member(answer, identifier, planned.run_id or "", report)
    models = [
        _model(name, [by_answer[step] for step in steps if step in by_answer])
        for name, steps in answers.models.items()
        if any(step in by_answer for step in steps)
    ]
    if not models:
        raise ScoreRefused(
            "no evaluation finished, so there is nothing to score: "
            + "; ".join(warnings)
        )
    table = _table(models)
    finished = result.model_copy(
        update={
            "request_id": submitted.request_id,
            "models": models,
            "table": table,
            "warnings": warnings,
            "files": [str(ds.base_dir / SCORES_JSON), str(ds.base_dir / SCORES_TXT)],
        }
    )
    _ = (ds.base_dir / SCORES_TXT).write_text(table, encoding="utf-8")
    _ = (ds.base_dir / SCORES_JSON).write_text(
        finished.model_dump_json(indent=1), encoding="utf-8"
    )
    return finished
