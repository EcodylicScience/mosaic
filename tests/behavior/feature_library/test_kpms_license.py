"""Unit tests for the keypoint-MoSeq license gate and the ladder ``KPMS_ENV`` uses.

Covers :func:`~mosaic.behavior.feature_library.kpms.check_license_accepted`,
:data:`~mosaic.behavior.feature_library.kpms.KPMS_ENV`, and the order in which
``KpmsFeature`` applies them. It also covers ``KpmsLicense`` and
``KpmsEnvironment``, the prerequisites of the pipeline catalog, which give the
same answers. None of this needs the external environment, so these run in the
default (non-slow) suite -- unlike ``test_kpms_integration.py``, which needs a
real keypoint-moseq install.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.behavior.feature_library.kpms import (
    KPMS_CONDA_ENV_VAR,
    KPMS_ENV,
    KPMS_LICENSE_ENV,
    KPMS_PYTHON_ENV,
    KpmsEnvironment,
    KpmsFeature,
    KpmsLicense,
    KpmsNotFoundError,
    check_license_accepted,
)
from mosaic.core.pipeline.toolenv import tool_invocation
from mosaic.core.pipeline.types import InputStream
from tests.helpers import fake_path_lookup, isolate_tool_location


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset every variable the ladder reads for kpms, and fake ``$PATH``.

    A developer entitled to run keypoint-MoSeq will have the acceptance
    exported and a real environment built, and every refusal test below would
    pass vacuously on their machine without this.
    """
    monkeypatch.delenv(KPMS_LICENSE_ENV, raising=False)
    isolate_tool_location(monkeypatch, KPMS_ENV, found={})


def _make_feature(kpms_python: str | None = None) -> KpmsFeature:
    params: dict[str, object] = {
        "pose": {"pose_n": 3, "keypoint_names": ["nose", "left", "right"]},
        "anterior_bodyparts": ["nose"],
        "posterior_bodyparts": ["left", "right"],
        "latent_dim": 4,
    }
    if kpms_python is not None:
        params["kpms_python"] = kpms_python
    return KpmsFeature(inputs=KpmsFeature.Inputs(("tracks",)), params=params)


def _empty_stream() -> InputStream:
    return InputStream(lambda: iter([]), 0)


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


def test_repo_venv_is_the_external_directory() -> None:
    assert KPMS_ENV.repo_venv is not None
    assert (KPMS_ENV.repo_venv / "kpms_server.py").is_file()
    assert (KPMS_ENV.repo_venv / "pyproject.toml").is_file()


def test_nothing_configured_names_the_unbuilt_interpreter_and_how_to_get_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_tool_location(monkeypatch, KPMS_ENV, found={}, repo_venv=tmp_path)

    with pytest.raises(KpmsNotFoundError) as exc:
        tool_invocation(KPMS_ENV, executable="python")

    message = str(exc.value)
    assert str(tmp_path / ".venv" / "bin" / "python") in message
    assert "never bundled" in message
    assert KPMS_PYTHON_ENV in message
    assert KPMS_CONDA_ENV_VAR in message
    assert "external/README.md" in message


def test_a_named_conda_env_beats_the_bin_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conda = tmp_path / "bin" / "conda"
    conda.parent.mkdir(parents=True)
    conda.touch()
    fake_path_lookup(monkeypatch, {"conda": str(conda)})
    monkeypatch.setenv(KPMS_CONDA_ENV_VAR, "kpms")
    monkeypatch.setenv(KPMS_PYTHON_ENV, "/should-not-be-used")

    got = tool_invocation(KPMS_ENV, executable="python")

    assert got == [str(conda), "run", "--no-capture-output", "-n", "kpms", "python"]


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


def test_license_is_checked_before_the_filesystem() -> None:
    """With neither acceptance nor an environment, the license answer wins.

    The refusal must not depend on whether an interpreter happens to be
    installed, so this asserts the license error rather than
    :class:`KpmsNotFoundError`.
    """
    with pytest.raises(RuntimeError, match="non-commercial"):
        _make_feature().fit(_empty_stream())


def test_accepted_then_missing_interpreter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    monkeypatch.setenv(KPMS_PYTHON_ENV, "/nonexistent/kpms/bin/python")
    with pytest.raises(KpmsNotFoundError, match="/nonexistent/kpms/bin/python"):
        _make_feature().fit(_empty_stream())


def test_the_kpms_python_param_places_the_feature_on_the_interpreter_it_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_start_server`` places the feature with ``params.kpms_python``.

    ``MOSAIC_KPMS_PYTHON`` names an interpreter that exists, so a refusal can
    only come from the parameter: if ``_start_server`` dropped
    ``.placed(bin_path=self.params.kpms_python)``, the environment variable
    would resolve instead and this test would pass for the wrong reason.
    """
    monkeypatch.setenv(KPMS_LICENSE_ENV, "1")
    existing = tmp_path / "env-python"
    existing.touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(existing))
    feature = _make_feature(kpms_python="/nonexistent/param/python")

    with pytest.raises(KpmsNotFoundError) as exc:
        feature.fit(_empty_stream())

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


def test_the_environment_prerequisite_is_met_by_an_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolved as the spawn resolves it, without starting the interpreter."""
    absent = KpmsEnvironment().missing()

    assert absent is not None
    assert KPMS_PYTHON_ENV in absent
    assert "external/README.md" in absent
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(interpreter))
    assert KpmsEnvironment().missing() is None
