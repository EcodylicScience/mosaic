"""Unit tests for the keypoint-MoSeq license gate and the ladder ``KPMS_ENV`` uses.

Covers :func:`~mosaic.behavior.feature_library.kpms.check_license_accepted`,
:data:`~mosaic.behavior.feature_library.kpms.KPMS_ENV`, and the order in which
``KpmsFeature`` applies them. It also covers ``KpmsLicense``, the pipeline
catalog's prerequisite, which gives the same answer.
``tests/behavior/feature_library/test_feature_prerequisites.py`` covers kpms's
``RunnerInterpreter`` with the other interpreter prerequisites. These tests do not
need the external environment and run in the default (non-slow) suite.
``test_kpms_integration.py`` needs a keypoint-moseq install.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from mosaic.behavior.feature_library.kpms import (
    KPMS_CONDA_ENV_VAR,
    KPMS_ENV,
    KPMS_LICENSE_ENV,
    KPMS_PYTHON_ENV,
    KpmsFeature,
    KpmsLicense,
    KpmsNotFoundError,
    check_license_accepted,
)
from mosaic.core.pipeline.toolenv import tool_invocation
from mosaic.core.pipeline.types import InputStream
from tests.helpers import (
    fake_path_lookup,
    isolate_tool_location,
    keypoint_rows,
    kpms_params,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, scratch: Path) -> None:
    """Unset every variable the ladder reads for kpms, and fake ``$PATH``.

    A developer entitled to run keypoint-MoSeq will have the acceptance
    exported and a real environment built, and every refusal test below would
    pass vacuously on their machine without this. A feature creates its
    exchange directory in *scratch*.
    """
    monkeypatch.delenv(KPMS_LICENSE_ENV, raising=False)
    isolate_tool_location(monkeypatch, KPMS_ENV, found={})


def _make_feature(**overrides: object) -> KpmsFeature:
    return KpmsFeature(
        inputs=KpmsFeature.Inputs(("tracks",)), params=kpms_params(**overrides)
    )


def _one_track() -> InputStream:
    """One entry with the three keypoints that :func:`_make_feature` names.

    The license and the interpreter are asked for at the first request, which
    sends a track.
    """
    rows = keypoint_rows(range(5), n_keypoints=3)
    return InputStream(lambda: iter([("g__s", rows)]), 1)


# --- acceptance ------------------------------------------------------------


def test_unset_refuses_and_says_why() -> None:
    with pytest.raises(RuntimeError) as exc:
        check_license_accepted()
    message = str(exc.value)
    assert KPMS_LICENSE_ENV in message
    assert "non-commercial" in message
    assert "commercial use is expressly prohibited" in message
    # The terms themselves, so the reader can decide rather than take our word.
    assert "keypoint-moseq/blob/main/LICENSE.md" in message


def test_accepts_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    check_license_accepted()


@pytest.mark.parametrize("value", [" 1 ", "1\n", "\t1"])
def test_surrounding_whitespace_is_ignored(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A value carried in from a ``.env`` file or a YAML ``env:`` block."""
    monkeypatch.setenv(KPMS_LICENSE_ENV, value)
    check_license_accepted()


@pytest.mark.parametrize("value", ["", "  ", "0", "true", "True", "yes", "on"])
def test_nothing_but_one_accepts(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Deliberate strictness: widening this widens accidental acceptance.

    A job matrix that sets every flag to ``true`` must not assert an
    institution's entitlement on its behalf.
    """
    monkeypatch.setenv(KPMS_LICENSE_ENV, value)
    with pytest.raises(RuntimeError, match="non-commercial"):
        check_license_accepted()


# --- interpreter resolution, on the shared ladder ---------------------------


def test_a_named_conda_env_beats_the_bin_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conda = tmp_path / "bin" / "conda"
    conda.parent.mkdir(parents=True)
    conda.touch()
    python = tmp_path / "envs" / "kpms" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()
    fake_path_lookup(monkeypatch, {"conda": str(conda)})
    monkeypatch.setenv(KPMS_CONDA_ENV_VAR, "kpms")
    monkeypatch.setenv(KPMS_PYTHON_ENV, "/should-not-be-used")

    got = tool_invocation(KPMS_ENV, executable="python")

    assert got == [str(conda), "run", "--no-capture-output", "-n", "kpms", str(python)]


def test_the_bin_var_beats_the_in_repository_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(interpreter))

    assert tool_invocation(KPMS_ENV, executable="python") == [str(interpreter)]


def test_the_param_beats_the_bin_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from_param = tmp_path / "param-python"
    from_param.touch()
    (tmp_path / "env-python").touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(tmp_path / "env-python"))

    placed = KPMS_ENV.placed(bin_path=str(from_param))

    assert tool_invocation(placed, executable="python") == [str(from_param)]


# --- order -----------------------------------------------------------------


def test_license_is_checked_before_the_filesystem(scratch: Path) -> None:
    """With neither acceptance nor an environment, the license answer wins.

    The refusal must not depend on whether an interpreter happens to be
    installed, so this asserts the license error rather than
    :class:`KpmsNotFoundError`.
    """
    with (
        closing(_make_feature()) as feature,
        pytest.raises(RuntimeError, match="non-commercial"),
    ):
        feature.fit(_one_track())

    assert list(scratch.iterdir()) == [], "close left the exchange directory"


def test_accepted_then_missing_interpreter(
    monkeypatch: pytest.MonkeyPatch, scratch: Path
) -> None:
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    monkeypatch.setenv(KPMS_PYTHON_ENV, "/nonexistent/kpms/bin/python")
    with (
        closing(_make_feature()) as feature,
        pytest.raises(KpmsNotFoundError, match="/nonexistent/kpms/bin/python"),
    ):
        feature.fit(_one_track())

    assert list(scratch.iterdir()) == [], "close left the exchange directory"


def test_the_kpms_python_param_places_the_feature_on_the_interpreter_it_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The feature places its runner with ``params.kpms_python``.

    ``MOSAIC_KPMS_PYTHON`` names an interpreter that exists. A refusal can then
    only come from the parameter. Without ``.placed(bin_path=...)`` from the
    parameter, the environment variable would resolve instead, and this test
    would pass for the wrong reason.
    """
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    existing = tmp_path / "env-python"
    existing.touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(existing))
    with (
        closing(_make_feature(kpms_python="/nonexistent/param/python")) as feature,
        pytest.raises(KpmsNotFoundError) as exc,
    ):
        feature.fit(_one_track())

    message = str(exc.value)
    assert "/nonexistent/param/python" in message
    assert "the binary this run was placed with" in message


def test_interpreter_path_is_not_part_of_run_identity() -> None:
    """Two machines, two venv paths, one identifier for identical output."""
    base = _make_feature().params
    relocated = base.model_copy(update={"kpms_python": "/somewhere/else/python"})
    assert relocated.identity_dump() == base.identity_dump()
    # It still reaches params.json and any parallel worker.
    assert relocated.model_dump()["kpms_python"] == "/somewhere/else/python"


# --- the prerequisites the catalog reads ------------------------------------


def test_the_license_prerequisite_is_met_once_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unaccepted = KpmsLicense().missing()

    assert unaccepted is not None
    assert "non-commercial" in unaccepted
    assert f"{KPMS_LICENSE_ENV}=1" in unaccepted
    assert "keypoint-moseq/blob/main/LICENSE.md" in unaccepted
    monkeypatch.setenv(KPMS_LICENSE_ENV, " 1 ")
    assert KpmsLicense().missing() is None
