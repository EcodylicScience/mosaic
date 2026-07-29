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
import json
import inspect
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
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
from mosaic.core.pipeline.identity_scheme import (
    FEATURE_IDENTITY_SCHEME,
    MARKER_NAME,
    read_identity_scheme,
)
from mosaic.core.pipeline.index import (
    FeatureIndexRow,
    feature_index,
    feature_index_path,
    feature_run_root,
)
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase
from mosaic.core.pipeline.index_lock import IndexLockTimeout, index_lock
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


def test_chain_runner_prediction_honours_the_tracks_selector(tmp_path: Path) -> None:
    """The selector must reach *both* halves of the chain runner or neither.

    Execution splats ``run_kwargs`` into ``run_feature``, so a ``tracks_run_id``
    there takes effect whether or not prediction knows about it. Prediction reads
    keys by name. Left unread, the two would hash different ``_tracks`` terms:
    ``status()`` would report a step permanently not-cached, and
    ``clean(dry_run=False)`` -- whose keep set is the identifiers it predicted --
    would delete the directory ``run()`` had just written.
    """
    from mosaic.core.dataset import new_dataset_manifest

    from .conftest import add_tracks_variant

    manifest = new_dataset_manifest(name="pinned", base_dir=tmp_path / "dataset")
    dataset = Dataset(manifest_path=manifest).load(ensure_roots=True)
    variant = "convert-trex_npz.0.1-aaaaaaaaaa"
    add_tracks_variant(dataset, variant, "seq_a", "seq_b")
    add_tracks_variant(dataset, "trex.0.1-bbbbbbbbbb", "seq_a", "seq_b")

    pipeline = Pipeline()
    _ = pipeline.add(
        FeatureStep(
            "pca",
            FEATURES["PairPoseDistancePCA"],
            None,
            run_kwargs={"tracks_run_id": variant},
        )
    )
    predicted = pipeline._resolve_step_cache(dataset)[0]["expected_run_id"]

    executed = dataset.run_feature(
        build_feature("pair-posedistance-pca", None, None), tracks_run_id=variant
    )
    assert predicted == executed.run_id

    # And the second run reads cached, which is what proves the *universe* was
    # narrowed to the pinned variant too: measured against the whole index the
    # step would look permanently incomplete.
    assert pipeline._resolve_step_cache(dataset)[0]["cached"]


def test_cold_chain_predicts_what_it_executes(scenario_dataset: Dataset) -> None:
    """The remaining half of P2e, owned by 1.1 rather than 0.1.

    The preview used to hash ``Result(run_id=None)`` for an uncached upstream
    while execution hashed the concrete identifier, so a cold chain predicted
    directories it then did not produce. Item 1.1 pins the prediction into the
    downstream step's inputs, which is what execution does.

    ``TemporalStackingFeature`` is the registry key; the earlier spelling of
    this test asked for ``"TemporalStack"`` and died on a ``KeyError`` before
    reaching an assertion -- which its ``xfail`` then reported as expected
    failure. A strict xfail proves a test fails, not that it fails for the
    stated reason.
    """
    pipeline = Pipeline()
    _ = pipeline.add(FeatureStep("speed", FEATURES["SpeedAngvel"], None))
    _ = pipeline.add(
        FeatureStep(
            "stack", FEATURES["TemporalStackingFeature"], None, input_names=["speed"]
        )
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


def test_a_prediction_survives_an_upstream_that_holds_only_older_runs(
    scenario_dataset: Dataset,
) -> None:
    """``status()`` must not raise on the ordinary post-params-change state.

    Pinning the prediction unconditionally means a downstream step resolves an
    upstream ``run_id`` the index does not hold, which is what an upstream whose
    params changed since it last ran always looks like. ``IndexCSV.read`` raises
    for an absent run and ``_resolve_step_cache`` is deliberately unguarded past
    construction, so without the ``on_missing_run="empty"`` policy this would
    propagate out of a preview.

    The upstream index is written directly rather than produced by a run: the
    state under test is "the index exists and holds some *other* run", and
    building it by hand says so without depending on a feature that computes on
    the synthetic fixture.
    """
    upstream = "speed-angvel__from__tracks"
    index = feature_index(feature_index_path(scenario_dataset, upstream))
    index.append(
        [
            FeatureIndexRow(
                abs_path=Path("seq_a.parquet"),
                run_id="0.1-0000000000",
                feature=upstream,
                version="0.1",
                group="",
                sequence="seq_a",
                params_hash="0000000000",
                n_rows=3,
            )
        ]
    )

    pipeline = Pipeline()
    _ = pipeline.add(FeatureStep("speed", FEATURES["SpeedAngvel"], None))
    _ = pipeline.add(
        FeatureStep(
            "stack", FEATURES["TemporalStackingFeature"], None, input_names=["speed"]
        )
    )

    resolved = pipeline._resolve_step_cache(scenario_dataset)

    assert resolved[0]["expected_run_id"] != "0.1-0000000000"
    assert resolved[1]["expected_run_id"] is not None
    assert resolved[1]["cached"] is False


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A cold scope_dependent step predicts an empty scope term: its scope "
        "comes from build_manifest over an upstream index that does not exist "
        "yet, while execution hashes the real entries. Item 1.1 does not reach "
        "this -- the divergence is in the scope term, not _inputs. Closing it "
        "means predicting the entry set of a run that has not happened, which "
        "belongs with the chain-runner work (implementation item 9.6). It "
        "cannot cause a wrong execution: an uncached upstream forces "
        "cached=False, so the predicted identifier gates no skip."
    ),
)
def test_a_cold_scope_dependent_step_predicts_its_scope(
    scenario_dataset: Dataset,
) -> None:
    """The residual M1 leaves behind, recorded rather than remembered.

    Asserted against what execution *would* hash rather than by running the
    chain, because the synthetic fixture carries no pose columns -- and a test
    that dies before its assertion is an xfail that proves nothing.
    """
    pipeline = Pipeline()
    _ = pipeline.add(FeatureStep("speed", FEATURES["SpeedAngvel"], None))
    _ = pipeline.add(
        FeatureStep("pca", FEATURES["PairPoseDistancePCA"], None, input_names=["speed"])
    )
    resolved = pipeline._resolve_step_cache(scenario_dataset)

    downstream = build_feature(
        "pair-posedistance-pca",
        [
            {
                "feature": "speed-angvel__from__tracks",
                "run_id": resolved[0]["expected_run_id"],
            }
        ],
        None,
    )
    assert downstream.scope_dependent, "fixture must exercise the scope term"
    executed, _ = compute_run_id(
        downstream, None, None, Scope(entries={("", "seq_a"), ("", "seq_b")})
    )

    assert resolved[1]["expected_run_id"] == executed


# --- P7: index writes are serialized ------------------------------------------


@dataclass(frozen=True)
class _Row(IndexRowBase):
    key: str


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


_APPEND_PROBE = """
import sys
from pathlib import Path
from dataclasses import dataclass
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase

@dataclass(frozen=True)
class Row(IndexRowBase):
    key: str

path, name, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
index = IndexCSV(path, Row)
# Read the whole file first and only then append, which is the interleaving a
# lock has to prevent -- without one, both writers compute a merged frame from
# the same starting state and the second erases the first.
_ = index.read()
barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
index.append([Row(abs_path=Path(name + ".parquet"), key=name)])
"""


def test_concurrent_index_appends_across_processes_do_not_lose_rows(
    tmp_path: Path,
) -> None:
    """The real contention is between processes, not threads.

    The thread test above is satisfied by a plain ``threading.Lock``, which
    leaves the case that actually happens -- two queue workers, or two
    ``mosaic run`` invocations, in separate interpreters -- completely
    unprotected. Only a file lock covers this one.
    """
    index_path = tmp_path / "index.csv"
    index: IndexCSV[_Row] = IndexCSV(index_path, _Row)
    index.ensure()

    gate = tmp_path / "gate"
    gate.mkdir()

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _APPEND_PROBE,
                str(index_path),
                name,
                str(gate / f"{name}.ready"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for name in ("first", "second")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err.decode()[-800:]

    with index_path.open(newline="") as handle:
        written = {row["key"] for row in csv.DictReader(handle)}
    assert written == {"first", "second"}, f"a concurrent append was lost: {written}"


def test_a_failed_lock_acquisition_raises_rather_than_writing_unlocked(
    tmp_path: Path,
) -> None:
    """Contention must fail loudly, never degrade to an unlocked write.

    Degrading is what the lock exists to prevent, and it would degrade exactly
    when the system is busiest.
    """
    index_path = tmp_path / "index.csv"
    index: IndexCSV[_Row] = IndexCSV(index_path, _Row)
    index.ensure()

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time\n"
            "from pathlib import Path\n"
            "from mosaic.core.pipeline.index_lock import index_lock\n"
            "with index_lock(Path(sys.argv[1])):\n"
            "    Path(sys.argv[2]).write_text('held')\n"
            "    time.sleep(30)\n",
            str(index_path),
            str(tmp_path / "held"),
        ]
    )
    try:
        deadline = time.monotonic() + 30
        while not (tmp_path / "held").exists():
            assert time.monotonic() < deadline, "holder never acquired the lock"
            time.sleep(0.02)

        with pytest.raises(IndexLockTimeout):
            with index_lock(index_path, timeout=0.5):
                pass  # pragma: no cover - the raise is the assertion
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_the_index_lock_is_reentrant_within_a_thread() -> None:
    """``IndexCSV.append`` calls ``ensure``; both take the lock.

    A non-re-entrant file lock would deadlock against itself here, with no
    error and no timeout distinguishable from real contention.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.csv"
        with index_lock(path, timeout=5):
            with index_lock(path, timeout=5):
                pass


# --- 0.3 / 0.4: persisted scope and the identity-scheme marker ----------------


def test_a_run_records_its_scope_and_scheme(scenario_dataset: Dataset) -> None:
    """Both are written, and the marker never reaches the identifier."""
    feature = build_feature("speed-angvel", None, None)
    result = scenario_dataset.run_feature(feature)
    run_root = feature_run_root(
        scenario_dataset, "speed-angvel__from__tracks", str(result.run_id)
    )

    saved = json.loads((run_root / "params.json").read_text())
    # The flag mirrors the feature's declaration, so a reader can tell a fit
    # scope from an apply scope without knowing the feature. Exercised here for
    # a scope-free feature; a scope-dependent one needs pose columns the
    # synthetic fixture does not carry.
    assert saved["_scope"]["scope_dependent"] is feature.scope_dependent
    assert saved["_scope"]["scope_dependent"] is False
    assert sorted(tuple(e) for e in saved["_scope"]["entries"]) == [
        ("", "seq_a"),
        ("", "seq_b"),
    ]

    assert read_identity_scheme(run_root) == FEATURE_IDENTITY_SCHEME
    # The marker is provenance, so it reaches no path segment. That it reaches
    # no *hash* either is proved by the golden corpus staying byte-identical
    # across the commit that introduced it, which is a stronger check than
    # anything assertable here.
    assert MARKER_NAME not in str(run_root)


def test_an_unmarked_run_root_reads_as_predating_the_scheme(tmp_path: Path) -> None:
    """An honest empty beats a confident wrong answer (migration M3)."""
    assert read_identity_scheme(tmp_path) == ""


def test_the_scheme_column_is_back_compatible(tmp_path: Path) -> None:
    """Adding the column must not break an index written before it existed.

    Old rows keep an empty cell, meaning "predates the scheme"; new rows carry
    the current one. Both must survive in one file.
    """
    path = tmp_path / "index.csv"
    legacy = (
        "abs_path,run_id,started_at,finished_at,feature,version,group,"
        "sequence,params_hash,n_rows\n"
        "a.parquet,0.1-aaaaaaaaaa,t,,speed,0.1,,seq_a,aaaaaaaaaa,3\n"
    )
    _ = path.write_text(legacy)

    index = feature_index(path)
    index.append(
        [
            FeatureIndexRow(
                abs_path=Path("b.parquet"),
                run_id="0.1-bbbbbbbbbb",
                feature="speed",
                version="0.1",
                group="",
                sequence="seq_b",
                params_hash="bbbbbbbbbb",
                n_rows=3,
                identity_scheme=FEATURE_IDENTITY_SCHEME,
            )
        ]
    )

    with path.open(newline="") as handle:
        rows = {
            r["sequence"]: r.get("identity_scheme", "") for r in csv.DictReader(handle)
        }
    assert rows == {"seq_a": "", "seq_b": FEATURE_IDENTITY_SCHEME}


_REPLACE_PROBE = """
import sys
from pathlib import Path
from dataclasses import dataclass
from mosaic.core.pipeline.index_csv import IndexCSV, SchemaRowBase

@dataclass(frozen=True)
class Row(SchemaRowBase):
    key: str = ""
    value: str = ""

path, name, barrier = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
index = IndexCSV(path, Row)
index.ensure()
barrier.write_text("ready")
while len(list(barrier.parent.glob("*.ready"))) < 2:
    pass
# Each writer declares a whole set of its own, several rows wide, so a torn or
# interleaved write shows up as a file mixing the two.
index.replace([Row(key=name, value=name) for _ in range(200)])
"""


def test_concurrent_index_replaces_never_tear_the_file(tmp_path: Path) -> None:
    """``replace`` serializes, so a reader always sees one writer's whole set.

    Deliberately NOT a lost-update test. ``rows`` is computed by the caller
    before the lock is taken, so two processes each declaring a different whole
    set have the last one win, entire -- which is what a projection means. What
    the lock guarantees is that the file is never a mixture of the two, and never
    half-written: every surviving row belongs to one writer, and the file parses.
    """
    index_path = tmp_path / "index.csv"
    gate = tmp_path / "gate"
    gate.mkdir()

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _REPLACE_PROBE,
                str(index_path),
                name,
                str(gate / f"{name}.ready"),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for name in ("first", "second")
    ]
    for proc in procs:
        _, err = proc.communicate(timeout=60)
        assert proc.returncode == 0, err.decode()[-800:]

    with index_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Every row carries the writer that produced it, so a mixture is detectable.
    writers = {row["value"] for row in rows}
    assert len(writers) == 1, f"the file mixes two writers' sets: {writers}"
    assert {row["key"] for row in rows} == writers
    assert len(rows) == 200, "a partial set survived"
