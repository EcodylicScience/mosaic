"""Unit tests for the keypoint-MoSeq license gate and interpreter resolution.

Covers :func:`~mosaic.behavior.feature_library.kpms.check_license_accepted`,
:func:`~mosaic.behavior.feature_library.kpms.resolve_kpms_python`, and the order
in which ``KpmsFeature`` applies them. None of this needs the external
environment, so these run in the default (non-slow) suite -- unlike
``test_kpms_integration.py``, which needs a real keypoint-moseq install.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaic.behavior.feature_library.kpms import (
    KPMS_LICENSE_ENV,
    KPMS_PYTHON_ENV,
    KpmsFeature,
    KpmsNotFoundError,
    check_license_accepted,
    resolve_kpms_python,
)
from mosaic.core.pipeline.types import InputStream


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset both kpms variables.

    A developer entitled to run keypoint-MoSeq will have the acceptance
    exported, and every refusal test below would pass vacuously on their
    machine.
    """
    monkeypatch.delenv(KPMS_LICENSE_ENV, raising=False)
    monkeypatch.delenv(KPMS_PYTHON_ENV, raising=False)


def _make_feature() -> KpmsFeature:
    return KpmsFeature(
        inputs=KpmsFeature.Inputs(("tracks",)),
        params={
            "pose": {"pose_n": 3, "keypoint_names": ["nose", "left", "right"]},
            "anterior_bodyparts": ["nose"],
            "posterior_bodyparts": ["left", "right"],
            "latent_dim": 4,
        },
    )


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


# --- interpreter resolution ------------------------------------------------


def test_missing_interpreter_says_how_to_get_one() -> None:
    with pytest.raises(KpmsNotFoundError) as exc:
        resolve_kpms_python()
    message = str(exc.value)
    assert "never bundled" in message
    assert KPMS_PYTHON_ENV in message
    assert "external/README.md" in message


def test_env_var_beats_the_bundled_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    interpreter = tmp_path / "python"
    interpreter.touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(interpreter))
    assert resolve_kpms_python() == interpreter


def test_param_beats_the_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from_param = tmp_path / "param-python"
    from_param.touch()
    (tmp_path / "env-python").touch()
    monkeypatch.setenv(KPMS_PYTHON_ENV, str(tmp_path / "env-python"))
    assert resolve_kpms_python(str(from_param)) == from_param


# --- order ------------------------------------------------------------------


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


def test_interpreter_path_is_not_part_of_run_identity() -> None:
    """Two machines, two venv paths, one identifier for identical output."""
    base = _make_feature().params
    relocated = base.model_copy(update={"kpms_python": "/somewhere/else/python"})
    assert relocated.identity_dump() == base.identity_dump()
    # It still reaches params.json and any parallel worker.
    assert relocated.model_dump()["kpms_python"] == "/somewhere/else/python"
