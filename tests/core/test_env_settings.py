"""Environment variables read as integers and as exact-match flags."""

from __future__ import annotations

import pytest

from mosaic.env_settings import flag_env, int_env

NAME = "MOSAIC_TEST_ENV_SETTING"


def test_an_unset_integer_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(NAME, raising=False)
    assert int_env(NAME, 7) == 7


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_a_blank_integer_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(NAME, raw)
    assert int_env(NAME, 7) == 7


@pytest.mark.parametrize(("raw", "expected"), [("42", 42), (" 42 ", 42), ("-3", -3)])
def test_an_integer_value_is_parsed(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int
) -> None:
    monkeypatch.setenv(NAME, raw)
    assert int_env(NAME, 7) == expected


@pytest.mark.parametrize("raw", ["many", "1.5", "0x10"])
def test_a_malformed_integer_raises_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(NAME, raw)
    with pytest.raises(ValueError, match=NAME):
        _ = int_env(NAME, 7)


def test_an_unset_flag_is_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NAME, raising=False)
    assert flag_env(NAME) is False


@pytest.mark.parametrize("raw", ["1", " 1", "1 ", "\t1\n"])
def test_a_flag_set_to_one_is_true_whitespace_aside(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(NAME, raw)
    assert flag_env(NAME) is True


@pytest.mark.parametrize("raw", ["", "  ", "0", "true", "yes", "11", "maybe"])
def test_a_flag_set_to_anything_else_is_false(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv(NAME, raw)
    assert flag_env(NAME) is False
