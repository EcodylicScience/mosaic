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
from mosaic.core.pipeline._utils import Scope, hash_params, json_ready
from mosaic.core.pipeline.index_csv import IndexCSV, IndexRowBase
from mosaic.core.pipeline.run import compute_run_id

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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TimelinePlot and the global-colored plot declare no scope_dependent, so "
        "compute_run_id raises AttributeError and they cannot run at all. The correct "
        "value for each is a judgment call about the visualization, not a mechanical "
        "fix."
    ),
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "json_ready serializes a set to a list without sorting and sort_keys only "
        "orders dict keys, so a set-valued identity term hashes differently per "
        "process. Closed by implementation item 0.5."
    ),
)
def test_set_valued_identity_term_is_process_stable() -> None:
    """A collection in identity must be ordered before hashing.

    Latent today -- no ``Params`` field is set-typed and ``compute_run_id`` sorts
    ``scope.entries`` explicitly -- but the composition hashes and consumed-root
    records add exactly this shape of term.
    """
    digests = {_digest_under_hash_seed(seed) for seed in ("0", "1", "2")}
    assert len(digests) == 1, f"digest varies with PYTHONHASHSEED: {sorted(digests)}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "json_ready collapses an unrecognized type to f'<{type(obj).__name__}>', a "
        "constant, so every value of that type hashes alike. Closed by implementation "
        "item 0.5."
    ),
)
def test_unrecognized_type_in_identity_raises() -> None:
    """A uid or composition hash carried as an object must not hash to a constant."""

    @dataclass
    class Opaque:
        value: int

    with pytest.raises(TypeError):
        _ = json_ready(Opaque(1))


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pipeline._resolve_step_cache rebuilds the hashable payload inline instead of "
        "calling compute_run_id. Closed by implementation item 0.1."
    ),
)
def test_identity_payload_is_built_in_one_module() -> None:
    """P2e: one function builds the payload; no caller reconstructs it.

    ``run.py`` legitimately holds two such literals -- the hash payload and the
    ``params.json`` save payload -- so this checks the set of *modules*, not the
    count of literals.
    """
    assert _files_building_an_identity_payload() == {"run.py"}


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
