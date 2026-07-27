"""Rule-level checks for the hashing and data-consistency contract.

One test per independently checkable rule from the reference document. These are
cheap invariants that hold (or should hold) across the whole registry, as
distinct from the literal-identifier pinning in ``test_identity_golden.py`` and
the end-to-end scenarios in ``test_hashing_workflows.py``.

Assertions describing target state rather than current behaviour are marked
``xfail(strict=True)`` with the implementation item that closes them. Strict
means the marker is self-clearing: when the item lands, the test reports XPASS
and fails until the marker is removed.
"""

from __future__ import annotations

import ast
import csv
import inspect
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest

from mosaic.behavior.feature_library import FEATURES
from mosaic.cli._features import build_feature
from mosaic.core.dataset import Dataset
from mosaic.core.pipeline._utils import (
    Scope,
    hash_params,
    identity_ready,
    json_ready,
)
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase
from mosaic.core.pipeline.manifest import build_manifest
from mosaic.core.pipeline.pipeline import FeatureStep, Pipeline
from mosaic.core.pipeline.run import MissingScopeDeclaration, compute_run_id

PIPELINE_DIR = Path(inspect.getfile(compute_run_id)).parent


# --- P2f: a global feature's fit source is reachable from its identity --------


def _fit_reads_its_stream(cls: type) -> bool | None:
    """Does ``cls.fit`` load its ``inputs`` parameter anywhere in its body?

    Returns None when there is no inspectable ``fit`` (no source, or no
    parameter beyond ``self``). A feature whose ``fit`` merely accepts the
    stream to satisfy the protocol -- ``xgboost``, ``feral`` -- reads its
    training data from params instead and correctly answers False.
    """
    fit = getattr(cls, "fit", None)
    if fit is None:
        return None
    try:
        source = textwrap.dedent(inspect.getsource(fit))
    except (OSError, TypeError):
        return None
    node = ast.parse(source).body[0]
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return None
    positional = [a.arg for a in node.args.args if a.arg != "self"]
    if not positional:
        return False
    stream = positional[0]
    return any(
        isinstance(sub, ast.Name) and sub.id == stream and isinstance(sub.ctx, ast.Load)
        for sub in ast.walk(node)
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "global-identity-megadescriptor and global-identity-dinov2-temporal fit from "
        "the stream while declaring scope_dependent = False. Closed by implementation "
        "item 1.4, which adds the pre-fitted model artifact branch and then flips the "
        "flag."
    ),
)
def test_stream_fitting_features_declare_scope_dependent() -> None:
    """P2f: if ``fit()`` consumes its stream, the scope is the training set.

    A feature that fits from the ambient ``InputStream`` and does not declare
    ``scope_dependent`` gets one identifier for every training set, so a run over
    a wider scope silently reuses a model fitted on the narrower one.
    """
    violations: list[str] = []
    for cls in FEATURES.values():
        if _fit_reads_its_stream(cls) is not True:
            continue
        if getattr(cls, "scope_dependent", False) is not True:
            violations.append(str(getattr(cls, "name", cls.__name__)))
    assert not violations, (
        f"features fit from their input stream but do not declare "
        f"scope_dependent = True: {sorted(violations)}"
    )


def test_every_feature_declares_scope_dependent() -> None:
    """``compute_run_id`` reads the attribute directly; absence is a hard crash."""
    missing = [
        str(getattr(cls, "name", cls.__name__))
        for cls in FEATURES.values()
        if not hasattr(cls, "scope_dependent")
    ]
    assert not missing, (
        f"features with no scope_dependent declaration: {sorted(missing)}"
    )


def test_missing_scope_declaration_names_the_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The absence is reported as a named error, not a bare AttributeError.

    Exercised against a real registered feature with the declaration removed,
    rather than a stand-in, so the message is the one a genuinely undeclared
    feature would produce. The message has to identify the feature from a
    traceback alone.
    """
    feature = build_feature("speed-angvel", None, None)
    monkeypatch.delattr(type(feature), "scope_dependent")

    with pytest.raises(MissingScopeDeclaration, match="speed-angvel"):
        _ = compute_run_id(feature, None, None, Scope())


# --- P2d: scope enters identity only for global features ----------------------


def test_scope_free_feature_identity_ignores_scope() -> None:
    """A per-frame feature computes S from S, so a subset and the full set match."""
    feature = build_feature("speed-angvel", None, None)
    narrow, _ = compute_run_id(feature, None, None, Scope(entries={("", "a")}))
    wide, _ = compute_run_id(feature, None, None, Scope(entries={("", "a"), ("", "b")}))
    assert narrow == wide


def test_scope_dependent_feature_identity_tracks_scope() -> None:
    """A scope-dependent feature fitted on more sequences is a different artifact."""
    feature = build_feature("arhmm", [{"feature": "pair-wavelet"}], None)
    narrow, _ = compute_run_id(feature, None, None, Scope(entries={("", "a")}))
    wide, _ = compute_run_id(feature, None, None, Scope(entries={("", "a"), ("", "b")}))
    assert narrow != wide


# --- P2: the digest itself ----------------------------------------------------


_SET_PAYLOAD_PROBE = (
    "from mosaic.core.pipeline._utils import hash_params; "
    "print(hash_params({'k': {'alpha','beta','gamma','delta','epsilon','zeta'}}))"
)


def _digest_under_hash_seed(seed: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", _SET_PAYLOAD_PROBE],
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
    )
    return completed.stdout.strip()


def test_set_valued_identity_term_is_process_stable() -> None:
    """A collection in identity must be ordered before hashing.

    Latent today -- no ``Params`` field is set-typed and ``compute_run_id`` sorts
    ``scope.entries`` explicitly -- but the composition hashes and consumed-root
    records add exactly this shape of term.
    """
    digests = {_digest_under_hash_seed(seed) for seed in ("0", "1", "2")}
    assert len(digests) == 1, f"digest varies with PYTHONHASHSEED: {sorted(digests)}"


def test_sequence_order_is_preserved_not_sorted() -> None:
    """Ordering sets must not become ordering everything.

    Sequence order is semantic wherever identity hashes it -- ``Inputs`` is a
    tuple, ``_frame_range`` is ``[start, end]`` -- so two differently ordered
    lists are two different recipes and must not collapse to one digest.
    """
    assert hash_params({"k": [1, 2, 3]}) != hash_params({"k": [3, 2, 1]})
    assert hash_params({"k": (1, 2)}) != hash_params({"k": (2, 1)})
    # ...while two spellings of one set are one recipe.
    assert hash_params({"k": {1, 2, 3}}) == hash_params({"k": {3, 2, 1}})


def test_unrecognized_type_in_identity_raises() -> None:
    """A uid or composition hash carried as an object must not hash to a constant.

    The object must be a plain class. A dataclass is *recognized* -- converted
    through ``dataclasses.asdict`` before the fallback is reached -- and that
    conversion is lossless and deterministic, so it is correct behaviour rather
    than the defect. ``Scope`` and ``FeatureMeta`` are themselves dataclasses.
    """

    class Opaque:
        def __init__(self, value: int) -> None:
            self.value = value

    with pytest.raises(TypeError):
        _ = identity_ready(Opaque(1))


def test_a_dataclass_is_converted_rather_than_rejected() -> None:
    """The strict serializer rejects the unrepresentable, not the unfamiliar."""

    @dataclass
    class Point:
        x: int
        y: int

    assert identity_ready(Point(1, 2)) == {"x": 1, "y": 2}


def test_provenance_serialization_still_degrades() -> None:
    """``json_ready`` keeps the lossy behaviour its callers depend on.

    ``params.json`` and the two ``run_params.json`` writers record what ran on a
    best-effort basis inside ``try/except``. Raising there would turn a lossy
    record into a missing one, which is worse: nothing reads these to make a
    decision, and 0.4's scheme marker is about to live beside one.
    """

    class Opaque:
        pass

    assert json_ready(Opaque()) == "<Opaque>"


def test_distinct_payloads_have_distinct_digests() -> None:
    """The baseline the two xfails above are protecting."""
    assert hash_params({"a": 1}) != hash_params({"a": 2})


# --- P2e: the identity payload is built in exactly one place ------------------


def _files_building_an_identity_payload() -> set[str]:
    """Pipeline modules containing a dict literal keyed ``_params`` + ``_inputs``."""
    found: set[str] = set()
    for path in sorted(PIPELINE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
            if {"_params", "_inputs"} <= keys:
                found.add(path.name)
    return found


def test_identity_payload_is_built_in_one_module() -> None:
    """P2e: one function builds the payload; no caller reconstructs it.

    ``run.py`` legitimately holds two such literals -- the hash payload and the
    ``params.json`` save payload -- so this checks the set of *modules*, not the
    count of literals.
    """
    assert _files_building_an_identity_payload() == {"run.py"}


def test_chain_runner_predicts_what_run_feature_computes(
    scenario_dataset: Dataset,
) -> None:
    """P2e, behaviourally: the prediction equals the identifier, not just the code.

    ``test_identity_payload_is_built_in_one_module`` is a structural check --
    it passes the moment the duplicated dict literal is deleted, whether or not
    the surviving call site resolves its scope the same way. This asserts the
    value, over a scope-dependent feature, where getting the scope term wrong
    is what would show.

    The prediction is load-bearing for execution: when the predicted directory
    reads complete, ``Pipeline.run`` skips ``run_feature`` and pins the
    predicted identifier into the next step's inputs.
    """
    pipeline = Pipeline()
    _ = pipeline.add(FeatureStep("pca", FEATURES["PairPoseDistancePCA"], None))
    predicted = pipeline._resolve_step_cache(scenario_dataset)[0]["expected_run_id"]

    feature = build_feature("pair-posedistance-pca", None, None)
    _, scope = build_manifest(scenario_dataset, feature.inputs, None, None, None)
    computed, _ = compute_run_id(feature, None, None, scope)

    assert feature.scope_dependent, "fixture must exercise the scope term"
    assert scope.entries, "fixture must resolve a non-empty scope"
    assert predicted == computed


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A cold multi-step chain still predicts a different identifier than it "
        "executes: the preview hashes Result(run_id=None) for an uncached upstream "
        "while execution hashes the concrete run_id. Collapsing the payload (0.1) "
        "does not close this -- the divergence is in the inputs object, not the "
        "payload construction. Closed by implementation item 1.1, which resolves "
        "inputs before hashing and writes the resolution back."
    ),
)
def test_cold_chain_predicts_what_it_executes(scenario_dataset: Dataset) -> None:
    """The remaining half of P2e, owned by 1.1 rather than 0.1."""
    pipeline = Pipeline()
    _ = pipeline.add(FeatureStep("speed", FEATURES["SpeedAngvel"], None))
    _ = pipeline.add(
        FeatureStep("stack", FEATURES["TemporalStack"], None, input_names=["speed"])
    )

    predicted = pipeline._resolve_step_cache(scenario_dataset)[1]["expected_run_id"]

    # What the same step computes once its upstream is a concrete reference.
    upstream_feature = build_feature("speed-angvel", None, None)
    upstream_id, _ = compute_run_id(upstream_feature, None, None, Scope())
    executed_feature = build_feature(
        "temporal-stack",
        [{"feature": "speed-angvel__from__tracks", "run_id": upstream_id}],
        None,
    )
    executed, _ = compute_run_id(executed_feature, None, None, Scope())

    assert predicted == executed


# --- P7: index writes are serialized ------------------------------------------


@dataclass(frozen=True)
class _Row(IndexRowBase):
    key: str


@pytest.mark.xfail(
    strict=True,
    reason=(
        "IndexCSV.append is a full-file read-modify-write with no lock: atomic "
        "replacement prevents a torn read, not a lost update. Closed by "
        "implementation item 0.2."
    ),
)
def test_concurrent_index_appends_do_not_lose_rows(tmp_path: Path) -> None:
    """Two writers whose reads interleave must not silently drop one's write."""
    index: IndexCSV[_Row] = IndexCSV(tmp_path / "index.csv", _Row)
    index.ensure()

    ready = threading.Barrier(2)

    def writer(name: str) -> None:
        ready.wait(timeout=5)
        index.append([_Row(abs_path=Path(f"{name}.parquet"), key=name)])

    threads = [threading.Thread(target=writer, args=(n,)) for n in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # Read back through csv rather than the pandas accessor: the assertion is
    # about which rows survived, and stdlib csv keeps it typed.
    with (tmp_path / "index.csv").open(newline="") as handle:
        written = {row["key"] for row in csv.DictReader(handle)}
    assert written == {"first", "second"}, f"a concurrent append was lost: {written}"
