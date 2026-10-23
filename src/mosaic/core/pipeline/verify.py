"""Verifying a finished recipe by fitting it again and comparing with what is stored.

A recipe's answer is a fitted model and the numbers computed from it. Anyone who
doubts the answer, a reviewer or the person who ran it on another machine, has had
one way to check: run the whole recipe again and read two directories side by side.
:func:`verify_plan` does that for the steps whose fit matters.

**Fitting steps.** A step whose feature declares ``fits`` and whose run is
complete is rebuilt from its planned spec and fitted again into a scratch
directory. Its stored per-entry outputs are then compared with the same entries
applied from the new fit, and the state files the fit saves (its model bundle,
its fold table, its reports) are compared with the stored run's own.

**Evaluation steps.** A step whose feature writes a report (``evaluate-labels``)
and whose run is complete is scored again, and its ``metrics.json`` is compared with
the stored one. When it reads a fitting step that was just verified, it scores the
predictions that step's new fit produced, so a changed model shows as a changed
score. Otherwise it scores the stored predictions, which checks the scoring alone.

**What is exact.** Most features are deterministic, and a verification of one of
them must find equal arrays. A feature that trains a network is not so across
thread counts or devices: on CPU at one thread count two fits with one seed are
equal, but on Linux a different thread count moves the head's probabilities by
about 7e-3 on average with about 99% of the rows keeping their argmax. A feature
is tolerant when it has an ``n_threads`` param or a ``device`` param that is not
``cpu``, and only then is a difference up to ``atol`` accepted. Every other
feature is held to equality, and the evaluation of a tolerant fit inherits its
tolerance.

**What it writes.** Nothing under the dataset. The dataset is read, the fit goes
to the scratch directory, and the stored run is never offered to ``load_state``
(an existing bundle there would make it report a cache and skip the fit). The
index of a feature is not touched either: the scratch run is not a run.

The module imports nothing from ``behavior`` or ``tracking``; a feature is reached
through :func:`~mosaic.core.pipeline.graph.resolve.build_step_feature`.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import tempfile
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from mosaic.core.scope import Scope
from mosaic.core.strict_model import StrictModel
from mosaic.user_paths import user_path

from ._utils import ResolvedScope
from .index import feature_index, feature_index_path, feature_run_root
from .loading import CROSS_JOIN_FEATURES, resolve_sequence_identity, stacks_inputs
from .manifest import (
    FilterFactory,
    Manifest,
    build_manifest,
    iter_manifest,
    verify_overlap_supported,
)
from .run import (
    _make_filter_factory,  # pyright: ignore[reportPrivateUsage]
    _resolve_dependencies,  # pyright: ignore[reportPrivateUsage]
    _resolve_pair_filter,  # pyright: ignore[reportPrivateUsage]
    build_output_path,
    compute_run_id,
    fit_input_stream,
    resolve_labels_variants,
)
from .resolve import resolve_references
from .types import InputStream
from .types.data_config import COLUMNS, META_COLS
from .writers import FeatureOutput, trim_feature_output, write_output

if TYPE_CHECKING:
    from mosaic.core.dataset import Dataset

    from .graph.plan import Plan, PlannedStep
    from .types import Feature

__all__ = [
    "DEFAULT_ATOL",
    "EntryDiff",
    "ScratchRefused",
    "StateDiff",
    "StepVerdict",
    "VerifyReport",
    "verify_plan",
]

DEFAULT_ATOL: Final = 1e-5
"""How far a tolerant feature's new fit may sit from the stored one, per value."""

PROBABILITY_PREFIX: Final = "prob_"
"""The prefix of the per-class probability columns a classifier writes."""

SCRATCH_MARKER: Final = ".mosaic-verify"
"""Written into a scratch directory this module created, so it may be reused."""

_MAX_LISTED: Final = 8
"""How many differences a verdict names. The largest gap is reported whatever this is."""

_MAX_DEPTH: Final = 6
"""How deep the comparison of a saved model bundle follows nested objects."""

Status = Literal["ok", "differs", "error", "skipped"]


class ScratchRefused(ValueError):
    """The scratch directory is not one verification may fill and clear."""


class EntryDiff(StrictModel):
    """One entry's stored output against the same entry applied from the new fit.

    Attributes:
        group: The entry's group.
        sequence: The entry's sequence.
        status: ``ok`` when every column is within the tolerance and the shapes
            match, otherwise ``differs``.
        n_rows: Rows in the stored output.
        max_abs_diff: The largest absolute difference over the numeric columns,
            ``None`` when it is not finite (a NaN in one table only).
        n_probability_rows: Rows whose probability columns are all finite in both
            tables.
        n_argmax_agree: Of those, the rows whose most probable class is the same.
        detail: Why the entry differs when the difference is not a number.
    """

    group: str
    sequence: str
    status: Literal["ok", "differs"]
    n_rows: int = 0
    max_abs_diff: float | None = 0.0
    n_probability_rows: int = 0
    n_argmax_agree: int = 0
    detail: str = ""


class StateDiff(StrictModel):
    """One saved state file of the new fit against the stored run's file.

    Attributes:
        file: The file name in the run's directory.
        status: ``ok``, ``differs``, or ``unchecked`` for a file this module
            cannot read or that the stored run does not hold.
        max_abs_diff: The largest absolute difference over its numbers.
        detail: What differs, or why it was not checked.
    """

    file: str
    status: Literal["ok", "differs", "unchecked"]
    max_abs_diff: float | None = 0.0
    detail: str = ""


class StepVerdict(StrictModel):
    """What verifying one step found.

    Attributes:
        step: The step's id in the recipe.
        runs: The feature it runs.
        role: ``fit`` for a fitting step, ``evaluation`` for a scoring step, and
            ``none`` for a step that is not verified.
        status: ``ok``, ``differs``, ``error`` (the fit or the scoring raised), or
            ``skipped``.
        reason: Why it was skipped or raised, or a note on what was compared.
        tolerance: The largest difference accepted. ``0`` is equality.
        max_abs_diff: The largest difference found over outputs, state and scores.
        argmax_agreement: The share of probability rows keeping their most
            probable class, ``None`` when the step writes none.
        entries_checked: How many stored outputs were compared.
        entries: The entries compared, in order.
        state: The state files compared.
        scores: The report values that differ, as ``path: stored -> new``. Empty
            when every score is within the tolerance.
    """

    step: str
    runs: str
    role: Literal["fit", "evaluation", "none"] = "none"
    status: Status
    reason: str = ""
    tolerance: float = 0.0
    max_abs_diff: float | None = None
    argmax_agreement: float | None = None
    entries_checked: int = 0
    entries: list[EntryDiff] = []
    state: list[StateDiff] = []
    scores: list[str] = []


class VerifyReport(StrictModel):
    """The verdict on a plan, step by step.

    Attributes:
        ok: Whether at least one step was verified, every verified step is within
            its tolerance, none raised and no step lacked a complete run to
            compare. A report that compared nothing is not ok.
        atol: The tolerance a tolerant feature was held to.
        n_threads: The thread count a feature with an ``n_threads`` param was fitted
            with.
        verified: How many steps were compared with their stored runs.
        steps: One verdict per step of the plan, in order.
    """

    ok: bool
    atol: float
    n_threads: int
    verified: int
    steps: list[StepVerdict]


# --- Comparison ---------------------------------------------------------------


@dataclass
class _Found:
    """What walking two values side by side found."""

    max_diff: float = 0.0
    mismatches: list[str] = dataclasses.field(default_factory=lambda: list[str]())
    moves: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict[str, float]()
    )
    """The largest gap at each path that has one, which is what a report names."""
    compared: int = 0
    """How many values or tables were actually compared."""
    unchecked: list[str] = dataclasses.field(default_factory=lambda: list[str]())
    """Where a value was left alone because it cannot be read or lies too deep."""

    def skip(self, path: str, why: str) -> None:
        if len(self.unchecked) < _MAX_LISTED:
            self.unchecked.append(f"{path or '.'}: {why}")

    def note(self) -> str:
        """The values left uncompared, as a sentence, or an empty string."""
        return f"not compared: {'; '.join(self.unchecked)}" if self.unchecked else ""

    def widen(self, diff: float, path: str = "") -> None:
        self.max_diff = max(self.max_diff, diff)
        if diff > 0.0 and len(self.moves) < _MAX_LISTED * 8:
            self.moves[path] = max(diff, self.moves.get(path, 0.0))

    def mismatch(self, path: str, what: str) -> None:
        if len(self.mismatches) < _MAX_LISTED:
            self.mismatches.append(f"{path or '.'}: {what}")
        self.max_diff = math.inf

    def within(self, tolerance: float) -> bool:
        return not self.mismatches and self.max_diff <= tolerance

    def report(self, tolerance: float) -> list[str]:
        """The differences past *tolerance*, worst first, as short sentences."""
        past = sorted(
            ((path, diff) for path, diff in self.moves.items() if diff > tolerance),
            key=lambda item: -item[1],
        )
        moved = [f"{path or '.'}: moved by {diff:.3g}" for path, diff in past]
        return [*self.mismatches, *moved][:_MAX_LISTED]


def _array_diff(
    stored: np.ndarray, fresh: np.ndarray, path: str, found: _Found
) -> None:
    """Compare two numeric arrays: equal shapes, NaN where NaN, then the largest gap."""
    found.compared += 1
    if stored.shape != fresh.shape:
        found.mismatch(path, f"shape {stored.shape} against {fresh.shape}")
        return
    if stored.size == 0:
        return
    left = stored.astype(np.float64, copy=False)
    right = fresh.astype(np.float64, copy=False)
    nan_left, nan_right = np.isnan(left), np.isnan(right)
    if not np.array_equal(nan_left, nan_right):
        found.mismatch(path, "NaN in different places")
        return
    both = ~nan_left
    if both.any():
        with np.errstate(invalid="ignore"):
            same = left[both] == right[both]
            gaps = np.abs(left[both] - right[both])
        gaps[same] = 0.0  # equal infinities of one sign are equal, not NaN
        found.widen(float(np.max(gaps)), path)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _as_array(value: object) -> npt.NDArray[np.float64] | None:
    """*value* as a numeric array when it is one (a tensor included), else ``None``."""
    array: object = value
    if not isinstance(array, np.ndarray):
        detach = getattr(array, "detach", None)  # a torch tensor, without torch
        if not callable(detach):
            return None
        try:
            tensor = cast("Any", detach())
            array = np.asarray(tensor.cpu().numpy())
        except Exception:  # noqa: BLE001 - an exotic tensor is simply not compared
            return None
    numeric = cast("npt.NDArray[np.generic]", array)
    if numeric.dtype.kind not in "biuf":
        return None
    return numeric.astype(np.float64, copy=False)


def _sequence(value: object) -> Sequence[object] | None:
    """*value* as a sequence when it is a list or a tuple."""
    return cast("Sequence[object]", value) if isinstance(value, (list, tuple)) else None


def _members(value: object) -> dict[str, object] | None:
    """A mapping's members by text key, an object's attributes, else ``None``."""
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): member for key, member in mapping.items()}
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, dict):
        named = cast("dict[object, object]", attributes)
        return {str(key): member for key, member in named.items()}
    return None


def _walk(
    stored: object,
    fresh: object,
    path: str,
    found: _Found,
    *,
    exact_models: bool,
    seen: frozenset[int] = frozenset(),
    depth: int = 0,
) -> None:
    """Compare two saved values: numbers by gap, containers by member, text by equality.

    A model object is followed through its attributes (a scaler's ``mean_``), a
    gradient-boosted booster is compared by its serialized bytes when
    *exact_models*, and anything else this cannot read is left unchecked.
    """
    if id(stored) in seen:
        return
    if depth > _MAX_DEPTH:
        found.skip(path, f"nested deeper than {_MAX_DEPTH} levels")
        return
    left, right = _as_array(stored), _as_array(fresh)
    if left is not None or right is not None:
        if left is None or right is None:
            found.mismatch(path, "an array against a non-array")
        else:
            _array_diff(left, right, path, found)
        return
    if _is_number(stored) and _is_number(fresh):
        _array_diff(
            np.asarray([stored], dtype=np.float64),
            np.asarray([fresh], dtype=np.float64),
            path,
            found,
        )
        return
    if _is_number(stored) or _is_number(fresh):
        found.mismatch(path, f"{_kind(stored)} against {_kind(fresh)}")
        return
    deeper = seen | {id(stored)}
    items, others = _sequence(stored), _sequence(fresh)
    if (items is None) != (others is None):
        found.mismatch(path, f"{_kind(stored)} against {_kind(fresh)}")
        return
    if items is not None and others is not None:
        if len(items) != len(others):
            found.mismatch(path, f"length {len(items)} against {len(others)}")
            return
        for index, (one, other) in enumerate(zip(items, others, strict=True)):
            _walk(
                one,
                other,
                f"{path}[{index}]",
                found,
                exact_models=exact_models,
                seen=deeper,
                depth=depth + 1,
            )
        return
    if isinstance(stored, (str, bytes, bool)) or stored is None:
        found.compared += 1
        if stored != fresh:
            found.mismatch(path, f"{stored!r} against {fresh!r}")
        return
    if isinstance(fresh, (str, bytes, bool)) or fresh is None:
        found.mismatch(path, f"{_kind(stored)} against {fresh!r}")
        return
    if callable(getattr(stored, "save_raw", None)):
        # A gradient-boosted booster. Its serialized bytes are equal when the fit
        # was, and not otherwise, so a tolerant feature is judged on its outputs.
        if exact_models:
            found.compared += 1
            if not _same_booster(stored, fresh):
                found.mismatch(path, "the serialized model differs")
        else:
            found.skip(path, "a model is judged on its outputs")
        return
    members, fresh_members = _members(stored), _members(fresh)
    if (members is None) != (fresh_members is None):
        found.mismatch(path, f"{_kind(stored)} against {_kind(fresh)}")
        return
    if members is None or fresh_members is None:
        if type(stored) is not type(fresh):
            found.mismatch(path, f"{_kind(stored)} against {_kind(fresh)}")
        else:
            found.skip(path, f"a {type(stored).__name__} holds nothing readable")
        return
    if isinstance(stored, Mapping) != isinstance(fresh, Mapping):
        found.mismatch(path, f"{_kind(stored)} against {_kind(fresh)}")
        return
    if not isinstance(stored, Mapping) and type(stored) is not type(fresh):
        found.mismatch(path, f"{type(stored).__name__} against {type(fresh).__name__}")
        return
    if members.keys() != fresh_members.keys():
        found.mismatch(path, "keys differ")
        return
    for name, member in members.items():
        _walk(
            member,
            fresh_members[name],
            f"{path}.{name}",
            found,
            exact_models=exact_models,
            seen=deeper,
            depth=depth + 1,
        )


def _kind(value: object) -> str:
    """A short name for what *value* is, for a mismatch between two kinds."""
    return "nothing" if value is None else type(value).__name__


def _same_booster(stored: object, fresh: object) -> bool:
    """Whether two boosters serialize alike. One that will not serialize is not a finding."""
    try:
        return bool(
            cast("Any", stored).save_raw(raw_format="ubj")
            == cast("Any", fresh).save_raw(raw_format="ubj")
        )
    except Exception:  # noqa: BLE001
        return True


def _compare_frames(
    stored: pd.DataFrame, fresh: pd.DataFrame
) -> tuple[_Found, int, int]:
    """Compare two tables column by column; also count the rows keeping their argmax.

    Returns:
        What was found, the rows whose probability columns are finite in both,
        and the share of those rows that keep their most probable class.
    """
    found = _Found()
    if list(stored.columns) != list(fresh.columns):
        found.mismatch(
            "", f"columns {list(stored.columns)} against {list(fresh.columns)}"
        )
        return found, 0, 0
    if len(stored) != len(fresh):
        found.mismatch("", f"{len(stored)} rows against {len(fresh)}")
        return found, 0, 0
    for column in stored.columns:
        left, right = stored[column], fresh[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            _array_diff(left.to_numpy(), right.to_numpy(), str(column), found)
        else:
            found.compared += 1
            if not left.astype(str).equals(right.astype(str)):
                found.mismatch(str(column), "values differ")
    probability = [c for c in stored.columns if str(c).startswith(PROBABILITY_PREFIX)]
    if not probability or not len(stored):
        return found, 0, 0
    left = stored[probability].to_numpy(dtype=np.float64)
    right = fresh[probability].to_numpy(dtype=np.float64)
    finite = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
    agree = int((left[finite].argmax(axis=1) == right[finite].argmax(axis=1)).sum())
    return found, int(finite.sum()), agree


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


# --- Scratch ------------------------------------------------------------------


@contextmanager
def _scratch_directory(requested: Path | None, ds: Dataset) -> Generator[Path]:
    """A directory outside the dataset that this verification may fill.

    A directory it was given is kept for inspection and must be absent, empty, or
    one an earlier verification marked, because each step's folder is cleared
    before it is used and a bundle left in one would be read as a cache.
    """
    if requested is None:
        with tempfile.TemporaryDirectory(prefix="mosaic-verify-") as made:
            yield Path(made)
        return
    root = user_path(requested).resolve()
    if root == ds.base_dir.resolve() or ds.base_dir.resolve() in root.parents:
        raise ScratchRefused(
            f"--scratch {root} is inside the dataset, and a verification writes "
            f"nothing there. Name a directory outside {ds.base_dir}."
        )
    if root.exists() and not (root / SCRATCH_MARKER).exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ScratchRefused(
                f"--scratch {root} exists and holds other files, and each step's "
                f"folder in it is cleared first. Name an empty or absent directory."
            )
    root.mkdir(parents=True, exist_ok=True)
    _ = (root / SCRATCH_MARKER).write_text("created by mosaic pipeline verify\n")
    yield root


def _fresh_directory(path: Path) -> Path:
    """*path*, empty and present."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


# --- The run being rebuilt -----------------------------------------------------


@dataclass
class _Rebuilt:
    """A step's feature with everything ``run_feature`` resolves before it fits."""

    feature: Feature
    manifest: Manifest
    resolved: ResolvedScope
    stream: InputStream
    filter_factory: FilterFactory | None
    cross_join: bool
    artifact_paths: dict[str, Path]
    dependency_lookups: dict[str, dict[tuple[str, str], Path]]
    run_id: str


def _tolerant(feature: Feature) -> bool:
    """Whether the feature's fit may move with the machine, so a tolerance applies."""
    fields = type(feature.params).model_fields
    if "n_threads" in fields:
        return True
    device = getattr(feature.params, "device", "cpu") if "device" in fields else "cpu"
    return str(device) != "cpu"


def _build(ds: Dataset, planned: PlannedStep, plan: Plan, n_threads: int) -> _Rebuilt:
    """Rebuild a step's feature as the engine would before fitting it.

    Raises:
        RuntimeError: The rebuilt feature is not the planned run.
    """
    from .graph.plan import asked_of
    from .graph.resolve import build_step_feature

    spec = planned.spec
    feature = build_step_feature(spec)
    if "n_threads" in type(feature.params).model_fields:
        # A throughput knob, outside the run id, that the thread-count promise
        # of a network fit depends on. Pinned so the new fit matches a stored one
        # made at the recipe's own (default one) thread count.
        feature = build_step_feature(
            dataclasses.replace(spec, params={**spec.params, "n_threads": n_threads})
        )
    _ = resolve_references(ds, feature)
    asked = asked_of(planned, plan)
    selector = Scope(entries=sorted(asked)) if asked else Scope()
    if feature.inputs.is_empty:
        manifest: Manifest = {}
        resolved = ResolvedScope(selector=selector)
    else:
        manifest, resolved = build_manifest(
            ds,
            feature.inputs,
            selector,
            tracks_run_id=spec.tracks_run_id,
            stack=stacks_inputs(feature),
        )
    resolved.labels_variants = resolve_labels_variants(ds, feature, spec.labels_run_id)
    from mosaic.core.helpers import resolve_frame_range

    frame_start, frame_end = resolve_frame_range(
        ds.meta_float("fps_default"),
        spec.filter_start_frame,
        spec.filter_end_frame,
        spec.filter_start_time,
        spec.filter_end_time,
    )
    run_id, _ = compute_run_id(
        feature, frame_start, frame_end, resolved, overlap_frames=spec.overlap_frames
    )
    if planned.run_id is not None and run_id != planned.run_id:
        raise RuntimeError(
            f"the rebuilt step is run {run_id}, not the planned run "
            f"{planned.run_id}, so the stored run was not fitted as this recipe says."
        )
    if spec.overlap_frames > 0:
        verify_overlap_supported(manifest, spec.overlap_frames)
    artifact_paths, dependency_lookups = _resolve_dependencies(
        ds, feature, spec.labels_run_id
    )
    filter_factory = _make_filter_factory(
        ds, resolved, _resolve_pair_filter(feature.params), frame_start, frame_end
    )
    cross_join = feature.name in CROSS_JOIN_FEATURES
    bind = getattr(feature, "bind_dataset", None)
    if callable(bind):
        _ = bind(ds)
    narrow = getattr(feature, "set_scope", None)
    if callable(narrow):
        _ = narrow(resolved)
    return _Rebuilt(
        feature=feature,
        manifest=manifest,
        resolved=resolved,
        stream=fit_input_stream(feature, manifest, filter_factory, cross_join),
        filter_factory=filter_factory,
        cross_join=cross_join,
        artifact_paths=artifact_paths,
        dependency_lookups=dependency_lookups,
        run_id=run_id,
    )


def _stored_outputs(
    ds: Dataset, storage: str, run_id: str
) -> tuple[dict[tuple[str, str], Path], set[tuple[str, str]]]:
    """The per-entry parquet outputs the index records for a run.

    Returns:
        The outputs whose file is on disk, and the entries the index records whose
        file is not.
    """
    index = feature_index_path(ds, storage)
    if not index.exists():
        return {}, set()
    rows = feature_index(index).read(
        run_id=run_id, filter_ext=".parquet", validate_paths=False
    )
    found: dict[tuple[str, str], Path] = {}
    missing: set[tuple[str, str]] = set()
    for _, row in rows.iterrows():
        path = ds.resolve_path(str(row.get("abs_path", "")))
        entry = (str(row.get("group", "")), str(row.get("sequence", "")))
        if path.exists():
            found[entry] = path
        else:
            missing.add(entry)
    return found, missing


def _entries_of(rebuilt: _Rebuilt) -> Iterator[tuple[tuple[str, str], pd.DataFrame]]:
    """Each manifest entry's frame, with the frames' identity, in the engine's order."""
    for key, frame in iter_manifest(
        rebuilt.manifest,
        filter_factory=rebuilt.filter_factory,
        cross_join=rebuilt.cross_join,
    ):
        yield resolve_sequence_identity(key, rebuilt.resolved.entry_map), frame


# --- Fitting steps --------------------------------------------------------------


@dataclass
class _Applied:
    """A verified fitting step's re-applied outputs, kept for the scoring after it."""

    outputs: Path
    tolerant: bool


def _compare_state(
    state_dir: Path, stored_root: Path, tolerance: float, *, exact_models: bool
) -> list[StateDiff]:
    """Compare each file the new fit saved with the stored run's file of that name."""
    found: list[StateDiff] = []
    for path in sorted(state_dir.iterdir()):
        if not path.is_file() or path.name == SCRATCH_MARKER:
            continue
        other = stored_root / path.name
        if not other.exists():
            found.append(
                StateDiff(
                    file=path.name,
                    status="unchecked",
                    detail="the stored run holds no file of this name",
                )
            )
            continue
        result = _Found()
        try:
            match path.suffix:
                case ".parquet":
                    result, _, _ = _compare_frames(
                        pd.read_parquet(other), pd.read_parquet(path)
                    )
                case ".csv":
                    result, _, _ = _compare_frames(
                        pd.read_csv(other), pd.read_csv(path)
                    )
                case ".json":
                    _walk(
                        json.loads(other.read_text()),
                        json.loads(path.read_text()),
                        "",
                        result,
                        exact_models=exact_models,
                    )
                case ".npz":
                    with np.load(other) as left, np.load(path) as right:
                        _walk(
                            {name: left[name] for name in left.files},
                            {name: right[name] for name in right.files},
                            "",
                            result,
                            exact_models=exact_models,
                        )
                case ".npy":
                    _walk(
                        np.load(other),
                        np.load(path),
                        "",
                        result,
                        exact_models=exact_models,
                    )
                case ".joblib":
                    import joblib

                    _walk(
                        joblib.load(other),
                        joblib.load(path),
                        "",
                        result,
                        exact_models=exact_models,
                    )
                case _:
                    found.append(
                        StateDiff(
                            file=path.name,
                            status="unchecked",
                            detail=f"a {path.suffix or 'plain'} file is not compared",
                        )
                    )
                    continue
        except Exception as exc:  # noqa: BLE001 - a file that will not open is a finding
            found.append(
                StateDiff(
                    file=path.name,
                    status="unchecked",
                    detail=f"could not be read: {type(exc).__name__}: {exc}",
                )
            )
            continue
        if not result.within(tolerance):
            status: Literal["ok", "differs", "unchecked"] = "differs"
        elif result.compared == 0:
            status = "unchecked"
        else:
            status = "ok"
        found.append(
            StateDiff(
                file=path.name,
                status=status,
                max_abs_diff=_finite(result.max_diff),
                detail="; ".join(
                    filter(None, [*result.report(tolerance), result.note()])
                ),
            )
        )
    return found


def _verify_fit(
    ds: Dataset,
    planned: PlannedStep,
    plan: Plan,
    work: Path,
    *,
    atol: float,
    n_threads: int,
    applied: dict[tuple[str, str], _Applied],
) -> StepVerdict:
    rebuilt = _build(ds, planned, plan, n_threads)
    feature = rebuilt.feature
    tolerant = _tolerant(feature)
    tolerance = atol if tolerant else 0.0
    storage, run_id = planned.storage_name, planned.run_id or ""
    verdict = StepVerdict(
        step=planned.step_id,
        runs=planned.runs,
        role="fit",
        status="ok",
        tolerance=tolerance,
    )
    state_dir = _fresh_directory(work / "state")
    ready = feature.load_state(
        state_dir, rebuilt.artifact_paths, rebuilt.dependency_lookups
    )
    if ready:
        return verdict.model_copy(
            update={
                "status": "skipped",
                "reason": "it loads a model that was given and fits nothing",
            }
        )
    feature.fit(rebuilt.stream)
    feature.save_state(state_dir)
    stored_root = feature_run_root(ds, storage, run_id)
    verdict.state = _compare_state(
        state_dir, stored_root, tolerance, exact_models=not tolerant
    )

    outputs = _fresh_directory(work / "outputs")
    stored, missing = _stored_outputs(ds, storage, run_id)
    overlap = planned.spec.overlap_frames
    worst = 0.0
    n_rows = n_agree = n_compared = 0
    entries: list[EntryDiff] = []
    if overlap > 0:
        stream = (
            (resolve_sequence_identity(key, rebuilt.resolved.entry_map), frame, sel)
            for key, frame, sel in iter_manifest(
                rebuilt.manifest,
                filter_factory=rebuilt.filter_factory,
                overlap_frames=overlap,
                cross_join=rebuilt.cross_join,
            )
        )
    else:
        stream = ((entry, frame, None) for entry, frame in _entries_of(rebuilt))
    for entry, frame, selector in stream:
        stored_path = stored.get(entry)
        if stored_path is None:
            # The plan called this run complete for the entries asked, so an
            # entry the fit writes that the run lacks is a finding, not a skip.
            entries.append(
                EntryDiff(
                    group=entry[0],
                    sequence=entry[1],
                    status="differs",
                    max_abs_diff=None,
                    detail=(
                        "the index records this output but its file is missing"
                        if entry in missing
                        else "the stored run holds no output for this entry"
                    ),
                )
            )
            worst = math.inf
            continue
        fresh = feature.apply(frame)
        if selector is not None:
            fresh = trim_feature_output(fresh, selector)
        _ = _write_entry(outputs, entry, fresh)
        found, rows, agree = _compare_frames(
            pd.read_parquet(stored_path),
            pd.read_parquet(build_output_path(*entry, outputs)),
        )
        worst = max(worst, found.max_diff)
        n_compared += 1
        n_rows += rows
        n_agree += agree
        entries.append(
            EntryDiff(
                group=entry[0],
                sequence=entry[1],
                status="ok" if found.within(tolerance) else "differs",
                n_rows=len(pd.read_parquet(stored_path)),
                max_abs_diff=_finite(found.max_diff),
                n_probability_rows=rows,
                n_argmax_agree=agree,
                detail="; ".join(found.report(tolerance)),
            )
        )
    verdict.entries = entries
    verdict.entries_checked = n_compared
    for diff in verdict.state:
        if diff.max_abs_diff is None:
            worst = math.inf
        else:
            worst = max(worst, diff.max_abs_diff)
    verdict.max_abs_diff = _finite(worst)
    verdict.argmax_agreement = n_agree / n_rows if n_rows else None
    differing = [e for e in entries if e.status == "differs"] + [
        s for s in verdict.state if s.status == "differs"
    ]
    if differing:
        verdict.status = "differs"
    if (
        not differing
        and not n_compared
        and not any(s.status != "unchecked" for s in verdict.state)
    ):
        verdict.status = "skipped"
        verdict.reason = "the stored run holds nothing this fit writes again"
    applied[(storage, run_id)] = _Applied(outputs=outputs, tolerant=tolerant)
    return verdict


def _write_entry(outputs: Path, entry: tuple[str, str], frame: FeatureOutput) -> int:
    """Write a re-applied entry beside its siblings, through the one parquet writer."""
    from ._utils import FeatureMeta

    meta = FeatureMeta(
        group=entry[0], sequence=entry[1], out_path=build_output_path(*entry, outputs)
    )
    return write_output(meta, frame)


# --- Evaluation steps -------------------------------------------------------------


def _overlay(frame: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame | None:
    """*frame* with the columns of *fresh* it shares, or ``None`` when rows do not align."""
    if len(frame) != len(fresh):
        return None
    for column in META_COLS & set(frame.columns) & set(fresh.columns):
        if column == COLUMNS.frame_col and not np.array_equal(
            frame[column].to_numpy(), fresh[column].to_numpy()
        ):
            return None
    shared = [c for c in fresh.columns if c in frame.columns]
    merged = frame.copy()
    for column in shared:
        merged[column] = fresh[column].to_numpy()
    return merged


def _verify_scores(
    ds: Dataset,
    planned: PlannedStep,
    plan: Plan,
    work: Path,
    *,
    atol: float,
    n_threads: int,
    applied: dict[tuple[str, str], _Applied],
) -> StepVerdict:
    rebuilt = _build(ds, planned, plan, n_threads)
    feature = rebuilt.feature
    report_file = str(getattr(type(feature), "report_file", ""))
    sources = [
        applied[(result.feature, result.run_id or "")]
        for result in feature.inputs.feature_inputs
        if (result.feature, result.run_id or "") in applied
    ]
    # One verified input can be laid over the frames; several cannot be told apart.
    source = sources[0] if len(sources) == 1 and len(feature.inputs.root) == 1 else None
    tolerance = atol if source is not None and source.tolerant else 0.0
    verdict = StepVerdict(
        step=planned.step_id,
        runs=planned.runs,
        role="evaluation",
        status="ok",
        tolerance=tolerance,
    )
    counts = {"overlaid": 0, "stored": 0}
    stream = rebuilt.stream
    if source is not None:
        base = stream
        overlay_from = source.outputs

        def overlaid() -> Iterator[tuple[str, pd.DataFrame]]:
            for key, frame in base():
                entry = resolve_sequence_identity(key, rebuilt.resolved.entry_map)
                path = build_output_path(*entry, overlay_from)
                merged = (
                    _overlay(frame, pd.read_parquet(path)) if path.exists() else None
                )
                counts["overlaid" if merged is not None else "stored"] += 1
                yield key, frame if merged is None else merged

        stream = InputStream(overlaid, n_entries=base.n_entries)
    state_dir = _fresh_directory(work / "state")
    if feature.load_state(
        state_dir, rebuilt.artifact_paths, rebuilt.dependency_lookups
    ):
        return verdict.model_copy(
            update={"status": "skipped", "reason": "it loads a report that was given"}
        )
    feature.fit(stream)
    feature.save_state(state_dir)
    stored_root = feature_run_root(ds, planned.storage_name, planned.run_id or "")
    new, old = state_dir / report_file, stored_root / report_file
    if not report_file or not new.exists() or not old.exists():
        return verdict.model_copy(
            update={
                "status": "error",
                "reason": f"no {report_file or 'report'} to compare in one of the runs",
            }
        )
    found = _Found()
    _walk(
        json.loads(old.read_text()),
        json.loads(new.read_text()),
        "",
        found,
        exact_models=True,
    )
    if found.compared == 0 and found.within(tolerance):
        return verdict.model_copy(
            update={
                "status": "error",
                "reason": f"nothing in {report_file} could be compared. {found.note()}".strip(),
            }
        )
    verdict.scores = found.report(tolerance)
    verdict.max_abs_diff = _finite(found.max_diff)
    verdict.entries_checked = counts["overlaid"] + counts["stored"]
    verdict.state = [
        StateDiff(
            file=report_file,
            status="ok" if found.within(tolerance) else "differs",
            max_abs_diff=_finite(found.max_diff),
            detail="; ".join(filter(None, [*found.report(tolerance), found.note()])),
        )
    ]
    if source is not None and counts["overlaid"]:
        verdict.reason = (
            "scored the predictions of the new fit"
            if not counts["stored"]
            else f"scored the predictions of the new fit for {counts['overlaid']} "
            f"entries and the stored predictions for {counts['stored']}"
        )
    else:
        verdict.reason = "scored the stored predictions"
    if found.unchecked:
        verdict.reason = f"{verdict.reason}; {found.note()}"
    if not found.within(tolerance):
        verdict.status = "differs"
    return verdict


# --- The plan ----------------------------------------------------------------------


def verify_plan(
    ds: Dataset,
    plan: Plan,
    *,
    atol: float = DEFAULT_ATOL,
    n_threads: int = 1,
    scratch: Path | None = None,
) -> VerifyReport:
    """Fit each fitting step of *plan* again and compare it with its stored run.

    Args:
        ds: The dataset, opened read-only. Nothing is written under it.
        plan: The plan of the request, as ``plan_pipeline`` resolves it. A step
            is verified when its run is complete.
        atol: The difference accepted from a tolerant feature, whose fit moves with
            the thread count or the device. Every other feature must match
            exactly.
        n_threads: The thread count a feature with an ``n_threads`` param is fitted
            with. It is outside the run id, so this changes nothing the stored run
            is named by.
        scratch: Where the new fits go. ``None`` takes a temporary directory that
            is removed afterwards.

    Returns:
        One verdict per step. A step that cannot be verified is ``skipped`` with the
        reason, and a fit or a scoring that raised is ``error``.

    Raises:
        ScratchRefused: *scratch* is inside the dataset or holds other files.
    """
    verdicts: list[StepVerdict] = []
    unfinished = 0
    applied: dict[tuple[str, str], _Applied] = {}
    with _scratch_directory(scratch, ds) as root:
        for planned in plan.steps:
            skipped = StepVerdict(
                step=planned.step_id, runs=planned.runs, status="skipped"
            )
            if planned.kind != "feature":
                verdicts.append(
                    skipped.model_copy(
                        update={"reason": "an op step is not fitted here"}
                    )
                )
                continue
            evaluation = _is_evaluation(planned)
            if not planned.fits and not evaluation:
                verdicts.append(
                    skipped.model_copy(update={"reason": "it fits nothing"})
                )
                continue
            if not planned.is_complete or planned.run_id is None:
                unfinished += 1
                verdicts.append(
                    skipped.model_copy(
                        update={
                            "reason": f"it has no complete run to compare ({planned.status})"
                        }
                    )
                )
                continue
            work = root / planned.step_id
            check = _verify_scores if evaluation and not planned.fits else _verify_fit
            try:
                verdicts.append(
                    check(
                        ds,
                        planned,
                        plan,
                        work,
                        atol=atol,
                        n_threads=n_threads,
                        applied=applied,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - the verdict names what broke
                verdicts.append(
                    StepVerdict(
                        step=planned.step_id,
                        runs=planned.runs,
                        role="evaluation" if evaluation else "fit",
                        status="error",
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
    verified = sum(1 for v in verdicts if v.status in ("ok", "differs"))
    return VerifyReport(
        ok=verified > 0
        and not unfinished
        and all(v.status in ("ok", "skipped") for v in verdicts),
        atol=atol,
        n_threads=n_threads,
        verified=verified,
        steps=verdicts,
    )


def _is_evaluation(planned: PlannedStep) -> bool:
    """Whether the step's feature writes a report that scores another run."""
    from .graph.resolve import feature_class_for_slug

    cls = feature_class_for_slug(planned.runs)
    return cls is not None and isinstance(getattr(cls, "report_file", None), str)
