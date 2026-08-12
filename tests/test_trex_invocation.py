"""Unit tests for the env-aware TRex CLI invocation resolution.

Covers :func:`mosaic.tracking.trex.run._trex_invocation` (how ``trex`` is
launched: in a conda env, via an explicit binary, or from ``$PATH``) and the
``DISPLAY`` overlay, without invoking the real ``trex`` binary.
"""

from __future__ import annotations

import pytest

from mosaic.tracking.common import toolenv

from mosaic.tracking.trex import run as trex_run
from mosaic.tracking.trex.run import (
    TRexNotFoundError,
    _resolve_display,
    _trex_invocation,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove TREX env vars and make ``which`` resolve fake trex/conda paths."""
    for var in (
        "MOSAIC_TREX_CONDA_ENV",
        "MOSAIC_TREX_BIN",
        "MOSAIC_TREX_DISPLAY",
        "CONDA_EXE",
        "CONDA_ENVS_DIRS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        toolenv.shutil,
        "which",
        lambda name: {"trex": "/p/trex", "conda": "/p/bin/conda"}.get(name),
    )


# --- precedence: param conda env > param bin > env conda > env bin > which ---


def test_param_conda_env_wins():
    assert _trex_invocation(trex_conda_env="track") == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "track",
        "trex",
    ]


def test_param_bin():
    assert _trex_invocation(trex_bin="/x/trex") == ["/x/trex"]


def test_param_conda_beats_param_bin():
    assert (
        _trex_invocation(trex_conda_env="track", trex_bin="/x/trex")[0]
        == "/p/bin/conda"
    )


def test_env_conda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_CONDA_ENV", "envc")
    assert _trex_invocation() == [
        "/p/bin/conda",
        "run",
        "--no-capture-output",
        "-n",
        "envc",
        "trex",
    ]


def test_param_beats_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_CONDA_ENV", "envc")
    assert _trex_invocation(trex_bin="/x/trex") == ["/x/trex"]


def test_env_bin(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_BIN", "/y/trex")
    assert _trex_invocation() == ["/y/trex"]


def test_default_path_lookup():
    assert _trex_invocation() == ["/p/trex"]


# --- error paths ---


def test_default_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    with pytest.raises(TRexNotFoundError):
        _trex_invocation()


def test_conda_missing_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(toolenv.shutil, "which", lambda name: None)
    with pytest.raises(TRexNotFoundError):
        _trex_invocation(trex_conda_env="track")


def test_conda_uses_conda_exe_fallback(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        toolenv.shutil,
        "which",
        lambda name: "/p/trex" if name == "trex" else None,
    )
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    assert _trex_invocation(trex_conda_env="track")[0] == "/opt/conda/bin/conda"


# --- DISPLAY overlay ---


def test_resolve_display_explicit():
    assert _resolve_display(":99") == {"DISPLAY": ":99"}


def test_resolve_display_none():
    assert _resolve_display(None) is None


def test_resolve_display_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MOSAIC_TREX_DISPLAY", ":7")
    assert _resolve_display(None) == {"DISPLAY": ":7"}


# --- _run_trex wires invocation + env into the supervised subprocess call ---


def test_run_trex_passes_invocation_and_env(monkeypatch: pytest.MonkeyPatch):
    """_run_trex now runs via the killable ``run_supervised`` helper; verify it
    still threads the resolved invocation prefix, args, and env overlay through."""
    captured: dict = {}

    def fake_supervised(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env")
        captured["cancel_check"] = kwargs.get("cancel_check")
        return ("ok", "", 0)  # (stdout, stderr, returncode)

    monkeypatch.setattr(trex_run, "run_supervised", fake_supervised)
    out, err = trex_run._run_trex(
        ["-task", "convert"],
        idle_timeout=5,
        invocation=["/p/bin/conda", "run", "-n", "track", "trex"],
        env={"DISPLAY": ":99"},
    )
    assert captured["cmd"][:5] == ["/p/bin/conda", "run", "-n", "track", "trex"]
    assert captured["cmd"][-2:] == ["-task", "convert"]
    assert captured["env"]["DISPLAY"] == ":99"
    assert out == "ok"


# --- argv for TREx's array parameters ---------------------------------------


def test_a_flat_array_keeps_the_form_it_has_always_had():
    """analysis_range and its kind must not move."""
    from mosaic.tracking.trex.run import _build_args

    assert _build_args({"analysis_range": [0, 1000]}) == [
        "-analysis_range",
        "[0,1000]",
    ]


def test_a_nested_array_is_json_because_a_python_repr_is_not_accepted():
    """What made output_fields unreachable.

    TREx's output_fields is a list of ``[name, [sources]]`` pairs and is how a
    user asks for tracklet_id or blobid, neither of which is in TREx's default
    export. Written with str() it came out as a Python repr -- single quotes and
    spaces -- which TREx's parameter parser rejects, so passing it through
    track_extra_settings could not work.
    """
    from mosaic.tracking.trex.run import _build_args

    fields = [["X", ["RAW", "WCENTROID"]], ["tracklet_id", []]]

    assert _build_args({"output_fields": fields}) == [
        "-output_fields",
        '[["X",["RAW","WCENTROID"]],["tracklet_id",[]]]',
    ]


def test_a_mapping_is_json_too():
    from mosaic.tracking.trex.run import _build_args

    assert _build_args({"opts": {"a": 1}}) == ["-opts", '{"a":1}']


# --- mosaic sends no default of its own -------------------------------------

# The nine parameters that used to carry a mosaic default. Each is named with
# the phase that sends it, because they are split across two argv builds and a
# parameter that quietly moved phase would otherwise still pass.
UNSENT_WHEN_UNSET: dict[str, tuple[str, ...]] = {
    "convert": (
        "detect_type",
        "detect_conf_threshold",
        "detect_iou_threshold",
        "cm_per_pixel",
        "meta_encoding",
        "track_max_individuals",
    ),
    "track": (
        "track_max_individuals",
        "track_max_speed",
        "track_max_reassign_time",
        "track_trusted_probability",
    ),
}


class _Captured(Exception):
    """Stops a phase the moment its argv exists, so nothing after it matters."""


def _captured_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path, phase: str, **kwargs: object
) -> list[str]:
    """Run one phase against a stubbed subprocess and return the argv it built.

    The stub raises rather than returning, so neither phase reaches its
    output-location step. Returning a success would make the two phases differ
    in whether they then raise -- convert refuses a missing ``.pv``, track
    tolerates missing results -- and a test whose control flow depends on that
    is a test that can pass because it went down the wrong branch.
    """
    captured: list[str] = []

    def fake_supervised(cmd, **_kwargs):
        captured.extend(cmd)
        raise _Captured

    monkeypatch.setattr(trex_run, "run_supervised", fake_supervised)
    out = tmp_path / phase
    with pytest.raises(_Captured):
        if phase == "convert":
            trex_run.run_trex_convert(tmp_path / "v.mp4", out, **kwargs)
        else:
            trex_run.run_trex_track(tmp_path / "v.pv", out, **kwargs)
    # Without this the argv-absence assertions below pass on an empty list, which
    # is exactly how the first version of this helper was silently vacuous.
    assert captured, "the phase never reached the subprocess, so no argv was built"
    return captured


@pytest.mark.parametrize("phase", sorted(UNSENT_WHEN_UNSET))
def test_an_unset_parameter_is_not_on_the_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path, phase: str
):
    """mosaic declares no default, so an unset parameter reaches TREx as absent.

    Every one of these used to carry a mosaic default, and not one of them
    matched TREx's -- ``detect_conf_threshold`` was five times stricter,
    ``meta_encoding`` forced a grayscale ``.pv`` where TREx writes ``rgb8``,
    ``track_max_individuals`` tracked one animal against TREx's 1024. Since the
    wrappers put every parameter on the argv unconditionally, a caller who set
    nothing still got mosaic's opinion and had no way to decline it.

    This is the test that fails if a default creeps back in, which is easy to do
    by writing ``= 0.5`` in a signature and easy to miss in review, because
    nothing else in the suite reads the argv of an all-defaults call.
    """
    argv = _captured_argv(monkeypatch, tmp_path, phase)
    sent = {token[1:] for token in argv if token.startswith("-")}
    assert not sent & set(UNSENT_WHEN_UNSET[phase]), (
        f"the {phase} phase put an unset parameter on the argv: "
        f"{sorted(sent & set(UNSENT_WHEN_UNSET[phase]))}"
    )


def test_a_set_parameter_is_still_sent(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The other half: unset means absent, it does not mean unreachable."""
    argv = _captured_argv(
        monkeypatch,
        tmp_path,
        "track",
        track_max_individuals=4,
        track_trusted_probability=0.25,
    )
    assert "-track_max_individuals" in argv
    assert argv[argv.index("-track_max_individuals") + 1] == "4"
    assert "-track_trusted_probability" in argv
    assert argv[argv.index("-track_trusted_probability") + 1] == "0.25"
    # Its neighbours stay unset -- setting one parameter must not resurrect the
    # rest, which is what a "fill in the defaults when any is given" fix would do.
    assert "-track_max_speed" not in argv


def test_extra_settings_can_unset_a_parameter(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """A ``None`` in the pass-through dict removes a parameter the caller set.

    The escape hatch for the case where the two layers disagree, and the
    mechanism the whole change rests on: ``_build_args`` skips a ``None``, and
    ``extra_settings`` is merged over the assembled params, so this is the last
    word on any single flag.
    """
    argv = _captured_argv(
        monkeypatch,
        tmp_path,
        "convert",
        detect_conf_threshold=0.5,
        extra_settings={"detect_conf_threshold": None},
    )
    assert "-detect_conf_threshold" not in argv
