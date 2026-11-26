"""The ``claude_desktop_config.json`` entry that starts mosaic's server.

Claude Desktop starts the server without the shell's environment, so the entry
carries a fixed ``PATH`` and the variables the shell set. Every test passes a fake
environment, prefix, platform and conda lookup, so the result never depends on
the machine that runs the suite.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

from mosaic.agent import SITE_URL, Installation
from mosaic.agent.desktop import (
    DesktopConfigRefused,
    DesktopEntry,
    desktop_config_path,
    desktop_server_entry,
    install_desktop_server,
    path_checked_tools,
)
from mosaic.core.json_value import JsonValue
from tests.helpers import SOURCE_ROOT, source_tree

SKIPPED_ENVS: Final = frozenset(
    {
        "MOSAIC_KPMS_CONDA_ENV",
        "MOSAIC_VAME_CONDA_ENV",
        "MOSAIC_LMT_CONDA_ENV",
        "MOSAIC_LITPOSE_CONDA_ENV",
    }
)
"""The ``ToolEnv`` declarations that are not checked, by their conda variable."""

SYSTEM_DIRECTORIES: Final = ("/usr/bin", "/bin", "/usr/sbin", "/sbin")


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _installation(prefix: Path) -> Installation:
    return Installation(
        executable=prefix / "bin" / "mosaic",
        interpreter=prefix / "bin" / "python",
        docs=None,
        manifest=None,
    )


def _prefix(tmp_path: Path, *, ffmpeg: bool = True) -> Path:
    prefix = tmp_path / "miniforge3" / "envs" / "mosaic"
    _ = _executable(prefix / "bin" / "mosaic")
    if ffmpeg:
        _ = _executable(prefix / "bin" / "ffmpeg")
        _ = _executable(prefix / "bin" / "ffprobe")
    return prefix


def _build(
    tmp_path: Path,
    *,
    environ: Mapping[str, str] | None = None,
    prefix: Path | None = None,
    conda: Path | None = None,
    platform: str = "linux",
    trust_pickled_files: bool = False,
) -> DesktopEntry:
    chosen = _prefix(tmp_path) if prefix is None else prefix
    return desktop_server_entry(
        _installation(chosen),
        [tmp_path / "data" / "dataset"],
        trust_pickled_files=trust_pickled_files,
        environ={} if environ is None else environ,
        prefix=chosen,
        platform=platform,
        conda=lambda: conda,
    )


def _env(result: DesktopEntry) -> dict[str, JsonValue]:
    env = result.entry["env"]
    assert isinstance(env, dict)
    return env


def _path(result: DesktopEntry, separator: str = os.pathsep) -> list[str]:
    path = _env(result)["PATH"]
    assert isinstance(path, str)
    return path.split(separator)


def _existing_system_directories() -> list[str]:
    return [name for name in SYSTEM_DIRECTORIES if Path(name).is_dir()]


def test_entry_has_the_servers_command_without_a_type(tmp_path: Path) -> None:
    prefix = _prefix(tmp_path)
    result = _build(tmp_path, prefix=prefix)

    assert result.entry["command"] == str(prefix / "bin" / "mosaic")
    assert result.entry["args"] == [
        "agent",
        "serve",
        "--dataset",
        str(tmp_path / "data" / "dataset"),
    ]
    assert "type" not in result.entry


def test_path_lists_the_environment_conda_and_the_system_in_order(
    tmp_path: Path,
) -> None:
    prefix = _prefix(tmp_path)
    install = tmp_path / "miniforge3"
    conda = _executable(install / "condabin" / "conda")

    result = _build(tmp_path, prefix=prefix, conda=conda)

    assert _path(result) == [
        str(prefix / "bin"),
        str(install / "condabin"),
        *_existing_system_directories(),
    ]


def test_path_omits_directories_that_do_not_exist(tmp_path: Path) -> None:
    result = _build(tmp_path)

    assert all(Path(entry).is_dir() for entry in _path(result))
    assert str(tmp_path / "miniforge3" / "condabin") not in _path(result)


def test_conda_found_in_bin_adds_condabin_and_never_bin(tmp_path: Path) -> None:
    prefix = _prefix(tmp_path)
    install = tmp_path / "miniforge3"
    conda = _executable(install / "bin" / "conda")
    (install / "condabin").mkdir()

    result = _build(tmp_path, prefix=prefix, conda=conda)

    assert str(install / "condabin") in _path(result)
    assert str(install / "bin") not in _path(result)
    assert result.warnings == []


def test_conda_without_condabin_warns_and_stays_off_the_path(tmp_path: Path) -> None:
    install = tmp_path / "miniforge3"
    conda = _executable(install / "bin" / "conda")

    result = _build(tmp_path, conda=conda)

    assert str(install / "bin") not in _path(result)
    assert any("condabin" in warning for warning in result.warnings)


def test_conda_found_in_scripts_adds_condabin(tmp_path: Path) -> None:
    install = tmp_path / "miniconda3"
    conda = _executable(install / "Scripts" / "conda.exe")
    (install / "condabin").mkdir()

    result = _build(tmp_path, conda=conda)

    assert str(install / "condabin") in _path(result)
    assert str(install / "Scripts") not in _path(result)
    assert result.warnings == []


def test_conda_reached_through_a_symlink_adds_the_real_condabin(
    tmp_path: Path,
) -> None:
    install = tmp_path / "miniforge3"
    real = _executable(install / "condabin" / "conda")
    link = tmp_path / "links" / "conda"
    link.parent.mkdir()
    link.symlink_to(real)

    result = _build(tmp_path, conda=link)

    assert str(install / "condabin") in _path(result)
    assert str(link.parent) not in _path(result)
    assert result.warnings == []


def test_conda_in_a_directory_of_another_name_warns_and_adds_nothing(
    tmp_path: Path,
) -> None:
    conda = _executable(tmp_path / "odd" / "conda")
    without = _build(tmp_path)

    result = _build(tmp_path, conda=conda)

    assert _path(result) == _path(without)
    [warning] = result.warnings
    assert str(conda) in warning
    assert "leaves conda out" in warning


def test_empty_copied_variables_are_treated_as_unset(tmp_path: Path) -> None:
    result = _build(
        tmp_path, environ={"MOSAIC_TREX_CONDA_ENV": "", "CONDA_ENVS_DIRS": ""}
    )

    assert list(_env(result)) == ["PATH"]
    assert result.warnings == []
    assert result.notes == []


def test_no_conda_adds_nothing_and_does_not_warn(tmp_path: Path) -> None:
    result = _build(tmp_path)

    assert result.warnings == []


def test_windows_path_lists_the_conda_activate_directories(tmp_path: Path) -> None:
    prefix = tmp_path / "miniconda3" / "envs" / "mosaic"
    for name in ("ffmpeg", "ffprobe"):
        _ = _executable(prefix / "Library" / "bin" / name)
        _ = _executable(prefix / "Library" / "bin" / f"{name}.exe")
    for name in ("Library/mingw-w64/bin", "Scripts", "bin"):
        (prefix / name).mkdir(parents=True)
    root = tmp_path / "Windows"
    (root / "System32" / "Wbem").mkdir(parents=True)
    install = tmp_path / "miniconda3"
    (install / "condabin").mkdir()
    conda = _executable(install / "Scripts" / "conda.exe")

    result = _build(
        tmp_path,
        prefix=prefix,
        conda=conda,
        platform="win32",
        environ={"SystemRoot": str(root)},
    )

    assert _path(result, ";") == [
        str(prefix),
        str(prefix / "Library" / "mingw-w64" / "bin"),
        str(prefix / "Library" / "bin"),
        str(prefix / "Scripts"),
        str(prefix / "bin"),
        str(install / "condabin"),
        str(root / "System32"),
        str(root),
        str(root / "System32" / "Wbem"),
    ]
    assert result.warnings == []


def test_missing_ffmpeg_or_ffprobe_warns_with_the_installation_page(
    tmp_path: Path,
) -> None:
    prefix = _prefix(tmp_path, ffmpeg=False)
    _ = _executable(prefix / "bin" / "ffmpeg")

    result = _build(tmp_path, prefix=prefix)

    [warning] = result.warnings
    assert "ffprobe" in warning
    assert f"{SITE_URL}installation/#one-ffmpeg-build-per-environment" in warning


def test_ffmpeg_in_the_environment_does_not_warn(tmp_path: Path) -> None:
    assert _build(tmp_path).warnings == []


def test_ffmpeg_outside_the_environment_directories_is_not_found(
    tmp_path: Path,
) -> None:
    elsewhere = tmp_path / "elsewhere"
    _ = _executable(elsewhere / "ffmpeg")
    _ = _executable(elsewhere / "ffprobe")

    result = _build(
        tmp_path,
        prefix=_prefix(tmp_path, ffmpeg=False),
        environ={"PATH": str(elsewhere)},
    )

    assert any("ffmpeg" in warning for warning in result.warnings)


def test_copied_variables_are_written_as_given(tmp_path: Path) -> None:
    shell = {
        "MOSAIC_TREX_CONDA_ENV": "trex",
        "MOSAIC_SLEAP_PYTHON": "/opt/sleap/bin/python",
        "MOSAIC_KPMS_LICENSE_ACCEPTED": "1",
        "CONDA_ENVS_DIRS": "/data/envs",
        "MOSAIC_ALLOW_TOOL_CODECS": "1",
        "MOSAIC_HIDRA_MODELS_ROOT": "~/hidra",
        "MOSAIC_TREX_DISPLAY": ":99",
        "MOSAIC_RAW_ARRAY_LIMIT_BYTES": "1000",
        "DISPLAY": ":0",
        "MOSAIC_UNRELATED": "x",
        "HOME": "/home/me",
    }

    env = _env(_build(tmp_path, environ=shell))

    assert {key: value for key, value in env.items() if key != "PATH"} == {
        "MOSAIC_TREX_CONDA_ENV": "trex",
        "MOSAIC_SLEAP_PYTHON": "/opt/sleap/bin/python",
        "MOSAIC_KPMS_LICENSE_ACCEPTED": "1",
        "CONDA_ENVS_DIRS": "/data/envs",
        "MOSAIC_ALLOW_TOOL_CODECS": "1",
        "MOSAIC_HIDRA_MODELS_ROOT": "~/hidra",
        "MOSAIC_TREX_DISPLAY": ":99",
    }


def test_a_tilde_value_is_not_expanded(tmp_path: Path) -> None:
    env = _env(_build(tmp_path, environ={"MOSAIC_HIDRA_MODELS_ROOT": "~/hidra"}))

    assert env["MOSAIC_HIDRA_MODELS_ROOT"] == "~/hidra"


def test_bare_bin_value_resolves_on_the_shell_path_and_prints_both(
    tmp_path: Path,
) -> None:
    trex = _executable(tmp_path / "shellbin" / "trex")

    result = _build(
        tmp_path,
        environ={"MOSAIC_TREX_BIN": "trex", "PATH": str(trex.parent)},
    )

    assert _env(result)["MOSAIC_TREX_BIN"] == str(trex)
    [note] = result.notes
    assert "MOSAIC_TREX_BIN" in note
    assert "trex" in note
    assert str(trex) in note


def test_bare_bin_value_not_found_is_written_as_given_with_a_warning(
    tmp_path: Path,
) -> None:
    result = _build(
        tmp_path,
        environ={"MOSAIC_TREX_BIN": "trex", "PATH": str(tmp_path / "nowhere")},
    )

    assert _env(result)["MOSAIC_TREX_BIN"] == "trex"
    assert any("MOSAIC_TREX_BIN" in warning for warning in result.warnings)


def test_bin_value_with_a_separator_is_written_as_given(tmp_path: Path) -> None:
    result = _build(tmp_path, environ={"MOSAIC_TREX_BIN": "/opt/trex/trex"})

    assert _env(result)["MOSAIC_TREX_BIN"] == "/opt/trex/trex"
    assert result.notes == []
    assert not any("MOSAIC_TREX_BIN" in warning for warning in result.warnings)


def test_tool_on_the_shell_path_only_is_warned_about(tmp_path: Path) -> None:
    sleap_nn = _executable(tmp_path / "shellbin" / "sleap-nn")

    result = _build(tmp_path, environ={"PATH": str(sleap_nn.parent)})

    [warning] = result.warnings
    assert warning == (
        f"SLEAP is on your shell's PATH at {sleap_nn}, which Claude Desktop's "
        f"server does not search; set MOSAIC_SLEAP_BIN={sleap_nn} or "
        "MOSAIC_SLEAP_CONDA_ENV and run again."
    )


def test_yolo_on_the_shell_path_warns_for_both_of_its_tools(tmp_path: Path) -> None:
    yolo = _executable(tmp_path / "shellbin" / "yolo")

    result = _build(tmp_path, environ={"PATH": str(yolo.parent)})

    assert sorted(warning.split(" ")[0] for warning in result.warnings) == [
        "POLO",
        "Ultralytics",
    ]


def test_tool_with_its_conda_env_or_bin_set_is_not_warned_about(
    tmp_path: Path,
) -> None:
    _ = _executable(tmp_path / "shellbin" / "trex")
    _ = _executable(tmp_path / "shellbin" / "sleap-nn")
    _ = _executable(tmp_path / "shellbin" / "hidra-predict")

    result = _build(
        tmp_path,
        environ={
            "PATH": str(tmp_path / "shellbin"),
            "MOSAIC_TREX_CONDA_ENV": "trex",
            "MOSAIC_SLEAP_BIN": "/opt/sleap/bin/sleap-convert",
            "MOSAIC_HIDRA_CONDA_ENV": "hidra",
        },
    )

    assert result.warnings == []


def test_tool_in_a_directory_of_the_entry_path_is_not_warned_about(
    tmp_path: Path,
) -> None:
    prefix = _prefix(tmp_path)
    trex = _executable(prefix / "bin" / "trex")

    result = _build(tmp_path, prefix=prefix, environ={"PATH": str(trex.parent)})

    assert result.warnings == []


def test_the_checked_tools_are_the_declared_envs_less_the_skipped() -> None:
    declared: dict[str, str] = {}
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        relative = path.relative_to(SOURCE_ROOT)
        if ".venv" in relative.parts or "external" in relative.parts[-2:-1]:
            continue
        for node in ast.walk(source_tree(path)):
            if not isinstance(node, ast.Assign | ast.AnnAssign):
                continue
            value = node.value
            if not (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "ToolEnv"
            ):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            assert isinstance(target, ast.Name)
            module = importlib.import_module(
                ".".join(("mosaic", *relative.with_suffix("").parts))
            )
            declared[target.id] = getattr(module, target.id).conda_env_var

    checked = {env.conda_env_var for env, _ in path_checked_tools()}
    assert set(declared.values()) - SKIPPED_ENVS == checked
    assert len(declared) == len(checked) + len(SKIPPED_ENVS)


EXECUTABLE_HOMES: Final = {
    "TREX_EXECUTABLE": SOURCE_ROOT / "tracking" / "trex" / "run.py",
    "SLEAP_NN_EXECUTABLE": SOURCE_ROOT / "tracking" / "sleap" / "training.py",
    "YOLO_SCRIPT": SOURCE_ROOT / "tracking" / "common" / "ultralytics_env.py",
    "HIDRA_PREDICT_SCRIPT": SOURCE_ROOT
    / "behavior"
    / "feature_library"
    / "hidra_feature.py",
}
"""The public constant of each checked executable, and the module that declares it."""


def _module_of(path: Path) -> ModuleType:
    relative = path.relative_to(SOURCE_ROOT).with_suffix("")
    return importlib.import_module(".".join(("mosaic", *relative.parts)))


def _passed_by_runner(path: Path, constant: str) -> bool:
    """Whether the module at *path* passes *constant* where a runner locates a tool."""
    for node in ast.walk(source_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        callee = node.func.id if isinstance(node.func, ast.Name) else ""
        if callee not in {"tool_invocation", "locate_tool", "ToolEnv"}:
            continue
        for keyword in node.keywords:
            if (
                keyword.arg in {"executable", "locator"}
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == constant
            ):
                return True
    return False


@pytest.mark.parametrize("constant", sorted(EXECUTABLE_HOMES))
def test_each_executable_name_is_the_constant_its_runner_passes(
    constant: str,
) -> None:
    home = EXECUTABLE_HOMES[constant]
    value = getattr(_module_of(home), constant)

    assert _passed_by_runner(home, constant)
    assert value in {executable for _, executable in path_checked_tools()}


def test_the_checked_table_names_each_tool_and_executable() -> None:
    table = {env.tool: executable for env, executable in path_checked_tools()}

    assert table == {
        "T-Rex": "trex",
        "SLEAP": "sleap-nn",
        "Ultralytics": "yolo",
        "POLO": "yolo",
        "HiDRA": "hidra-predict",
    }


@pytest.mark.parametrize("tool", ["KPMS", "VAME", "LITPOSE"])
def test_skipped_tools_are_never_looked_up(tmp_path: Path, tool: str) -> None:
    shellbin = tmp_path / "shellbin"
    for name in ("python", "kpms", "vame", "litpose"):
        _ = _executable(shellbin / name)

    result = _build(tmp_path, environ={"PATH": str(shellbin)})

    assert not any(tool in warning.upper() for warning in result.warnings)


def test_pickle_trust_is_absent_without_the_flag_or_the_variable(
    tmp_path: Path,
) -> None:
    result = _build(tmp_path)

    assert "MOSAIC_TRUST_PICKLED_FILES" not in _env(result)
    assert result.warnings == []


def test_the_flag_writes_pickle_trust_and_warns(tmp_path: Path) -> None:
    result = _build(tmp_path, trust_pickled_files=True)

    assert _env(result)["MOSAIC_TRUST_PICKLED_FILES"] == "1"
    assert _trust_warnings(result) == [TRUST_WARNING]


def test_a_shell_value_of_one_writes_pickle_trust_and_warns(tmp_path: Path) -> None:
    result = _build(tmp_path, environ={"MOSAIC_TRUST_PICKLED_FILES": " 1 "})

    assert _env(result)["MOSAIC_TRUST_PICKLED_FILES"] == "1"
    assert _trust_warnings(result) == [TRUST_WARNING]


@pytest.mark.parametrize("value", ["true", "0", "yes", ""])
def test_another_shell_value_writes_no_pickle_trust(tmp_path: Path, value: str) -> None:
    result = _build(tmp_path, environ={"MOSAIC_TRUST_PICKLED_FILES": value})

    assert "MOSAIC_TRUST_PICKLED_FILES" not in _env(result)
    assert result.warnings == []


TRUST_WARNING: Final = (
    "MOSAIC_TRUST_PICKLED_FILES=1 lets this server unpickle raw files in every "
    "dataset it serves, and unpickling can run code in the file. Use it only "
    "for datasets from sources you trust."
)


def _trust_warnings(result: DesktopEntry) -> list[str]:
    return [warning for warning in result.warnings if "UNPICKLE" in warning.upper()]


CONFIG_NAME: Final = "claude_desktop_config.json"


def _home_config(home: Path, platform: str) -> Path:
    if platform == "darwin":
        return home / "Library" / "Application Support" / "Claude" / CONFIG_NAME
    return home / ".config" / "Claude" / CONFIG_NAME


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_unix_config_path_is_in_the_home_directory(
    tmp_path: Path, platform: str
) -> None:
    expected = _home_config(tmp_path, platform)
    expected.parent.mkdir(parents=True)

    assert desktop_config_path(platform=platform, home=tmp_path) == expected


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_unix_config_path_refuses_a_missing_directory(
    tmp_path: Path, platform: str
) -> None:
    directory = _home_config(tmp_path, platform).parent

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = desktop_config_path(platform=platform, home=tmp_path)

    message = str(refused.value)
    assert (
        f"Claude Desktop's configuration directory {directory} does not exist. "
        "Install Claude Desktop and open it once, or pass --desktop-config."
    ) in message


def test_the_override_skips_the_search(tmp_path: Path) -> None:
    chosen = tmp_path / "elsewhere" / "config.json"
    chosen.parent.mkdir()

    assert desktop_config_path(chosen, platform="darwin", home=tmp_path) == chosen


def test_the_override_needs_its_directory(tmp_path: Path) -> None:
    chosen = tmp_path / "missing" / "config.json"

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = desktop_config_path(chosen, platform="linux", home=tmp_path)

    assert str(tmp_path / "missing") in str(refused.value)


def _windows(tmp_path: Path) -> dict[str, str]:
    return {
        "APPDATA": str(tmp_path / "AppData" / "Roaming"),
        "LOCALAPPDATA": str(tmp_path / "AppData" / "Local"),
    }


def _documented(tmp_path: Path) -> Path:
    return tmp_path / "AppData" / "Roaming" / "Claude" / CONFIG_NAME


def _packaged(tmp_path: Path, package: str = "Claude_abc") -> Path:
    return (
        tmp_path
        / "AppData"
        / "Local"
        / "Packages"
        / package
        / "LocalCache"
        / "Roaming"
        / "Claude"
        / CONFIG_NAME
    )


def _data(tmp_path: Path, package: str = "Claude_abc") -> Path:
    return tmp_path / "AppData" / "Local" / "Claude-Data" / package / CONFIG_NAME


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text("{}", encoding="utf-8")
    return path


def _windows_path(tmp_path: Path) -> Path:
    return desktop_config_path(
        platform="win32", home=tmp_path, environ=_windows(tmp_path)
    )


@pytest.mark.parametrize("which", ["documented", "packaged", "data"])
def test_windows_one_existing_file_is_the_one_written(
    tmp_path: Path, which: str
) -> None:
    chosen = _touch(
        {
            "documented": _documented,
            "packaged": _packaged,
            "data": _data,
        }[which](tmp_path)
    )

    assert _windows_path(tmp_path) == chosen


def test_windows_several_existing_files_are_refused_and_listed(
    tmp_path: Path,
) -> None:
    found = [_touch(_documented(tmp_path)), _touch(_data(tmp_path))]

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = _windows_path(tmp_path)

    message = str(refused.value)
    assert all(str(path) in message for path in found)
    assert "--desktop-config" in message


def test_windows_two_matches_of_one_glob_are_several(tmp_path: Path) -> None:
    found = [
        _touch(_packaged(tmp_path, "Claude_a")),
        _touch(_packaged(tmp_path, "Claude_b")),
    ]

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = _windows_path(tmp_path)

    assert all(str(path) in str(refused.value) for path in found)


def test_windows_none_writes_the_documented_path_when_its_directory_exists(
    tmp_path: Path,
) -> None:
    _documented(tmp_path).parent.mkdir(parents=True)

    assert _windows_path(tmp_path) == _documented(tmp_path)


def test_windows_none_without_the_directory_names_the_three_candidates(
    tmp_path: Path,
) -> None:
    with pytest.raises(DesktopConfigRefused) as refused:
        _ = _windows_path(tmp_path)

    message = str(refused.value)
    assert str(_documented(tmp_path)) in message
    assert str(tmp_path / "AppData" / "Local" / "Packages" / "Claude_*") in message
    assert str(tmp_path / "AppData" / "Local" / "Claude-Data" / "Claude_*") in message
    assert "--desktop-config" in message


def test_windows_without_appdata_still_searches_localappdata(tmp_path: Path) -> None:
    chosen = _touch(_data(tmp_path))
    environ = {"LOCALAPPDATA": str(tmp_path / "AppData" / "Local")}

    assert (
        desktop_config_path(platform="win32", home=tmp_path, environ=environ) == chosen
    )


def _config(tmp_path: Path) -> Path:
    directory = tmp_path / "Claude"
    directory.mkdir(exist_ok=True)
    return directory / CONFIG_NAME


def _entry(tmp_path: Path) -> DesktopEntry:
    return _build(tmp_path)


def _servers(path: Path) -> dict[str, JsonValue]:
    document: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    servers = document["mcpServers"]
    assert isinstance(servers, dict)
    return servers


def test_a_missing_file_is_created(tmp_path: Path) -> None:
    path = _config(tmp_path)
    result = _entry(tmp_path)

    done = install_desktop_server(path, result.entry)

    assert (done.path, done.action, done.dropped_env_keys) == (path, "created", [])
    assert _servers(path) == {"mosaic": result.entry}


def test_an_empty_file_starts_a_new_document(tmp_path: Path) -> None:
    path = _config(tmp_path)
    _ = path.write_text("  \n", encoding="utf-8")
    result = _entry(tmp_path)

    done = install_desktop_server(path, result.entry)

    assert done.action == "updated"
    assert _servers(path) == {"mosaic": result.entry}


def test_the_merge_keeps_other_servers_and_preferences(tmp_path: Path) -> None:
    path = _config(tmp_path)
    other: dict[str, JsonValue] = {"command": "other", "args": ["x"]}
    document: dict[str, JsonValue] = {
        "preferences": {"sidebarMode": "chat"},
        "mcpServers": {"other": other},
    }
    _ = path.write_text(json.dumps(document), encoding="utf-8")
    result = _entry(tmp_path)

    done = install_desktop_server(path, result.entry)

    assert done.action == "updated"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["preferences"] == {"sidebarMode": "chat"}
    assert saved["mcpServers"] == {"other": other, "mosaic": result.entry}


def test_a_second_run_is_unchanged_and_writes_nothing(tmp_path: Path) -> None:
    path = _config(tmp_path)
    result = _entry(tmp_path)
    _ = install_desktop_server(path, result.entry)
    before = path.stat().st_mtime_ns
    inode = path.stat().st_ino

    done = install_desktop_server(path, result.entry)

    assert done.action == "unchanged"
    assert (path.stat().st_mtime_ns, path.stat().st_ino) == (before, inode)


def test_the_files_mode_is_kept(tmp_path: Path) -> None:
    path = _config(tmp_path)
    _ = path.write_text("{}", encoding="utf-8")
    path.chmod(0o640)

    _ = install_desktop_server(path, _entry(tmp_path).entry)

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_a_rerun_drops_a_hand_added_env_key_and_reports_it(tmp_path: Path) -> None:
    path = _config(tmp_path)
    result = _entry(tmp_path)
    _ = install_desktop_server(path, result.entry)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["mcpServers"]["mosaic"]["env"]["MOSAIC_TRUST_PICKLED_FILES"] = "1"
    document["mcpServers"]["mosaic"]["env"]["ANOTHER"] = "x"
    _ = path.write_text(json.dumps(document), encoding="utf-8")

    done = install_desktop_server(path, result.entry)

    assert done.action == "updated"
    assert done.dropped_env_keys == ["ANOTHER", "MOSAIC_TRUST_PICKLED_FILES"]
    assert _servers(path) == {"mosaic": result.entry}


def test_an_earlier_entry_without_an_env_drops_nothing(tmp_path: Path) -> None:
    path = _config(tmp_path)
    old: dict[str, JsonValue] = {"mosaic": {"command": "mosaic"}}
    _ = path.write_text(json.dumps({"mcpServers": old}), encoding="utf-8")

    done = install_desktop_server(path, _entry(tmp_path).entry)

    assert done.dropped_env_keys == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("{not json", CONFIG_NAME),
        ("[1, 2]", "JSON object"),
        ('{"mcpServers": []}', "mcpServers"),
    ],
)
def test_an_unusable_file_is_refused_and_left_unchanged(
    tmp_path: Path, text: str, fragment: str
) -> None:
    path = _config(tmp_path)
    _ = path.write_text(text, encoding="utf-8")

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = install_desktop_server(path, _entry(tmp_path).entry)

    assert fragment in str(refused.value)
    assert str(path) in str(refused.value)
    assert path.read_text(encoding="utf-8") == text
    assert sorted(item.name for item in path.parent.iterdir()) == [CONFIG_NAME]


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    path = _config(tmp_path)
    _ = path.write_bytes(b"\xff\xfe\x00bad")

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = install_desktop_server(path, _entry(tmp_path).entry)

    assert str(path) in str(refused.value)


def test_a_missing_directory_is_refused_and_not_created(tmp_path: Path) -> None:
    path = tmp_path / "Claude" / CONFIG_NAME

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = install_desktop_server(path, _entry(tmp_path).entry)

    assert str(path.parent) in str(refused.value)
    assert not path.parent.exists()


@pytest.mark.parametrize(
    ("indent", "ensure_ascii", "newline"),
    [(2, True, ""), (4, False, "\n"), (None, False, "")],
)
def test_a_reformatted_file_with_the_same_content_is_unchanged(
    tmp_path: Path, indent: int | None, ensure_ascii: bool, newline: str
) -> None:
    path = _config(tmp_path)
    result = _entry(tmp_path)
    _ = install_desktop_server(path, result.entry)
    document: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    document["preferences"] = {"name": "café"}
    _ = path.write_text(
        json.dumps(document, indent=indent, ensure_ascii=ensure_ascii) + newline,
        encoding="utf-8",
    )
    before = (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino)

    done = install_desktop_server(path, result.entry)

    assert done.action == "unchanged"
    assert (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) == before


def test_a_missing_directory_remedy_does_not_advise_installing_desktop(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Claude" / CONFIG_NAME

    with pytest.raises(DesktopConfigRefused) as refused:
        _ = install_desktop_server(path, _entry(tmp_path).entry)

    assert "Install Claude Desktop" not in str(refused.value)
    assert "does not exist" in str(refused.value)


def test_a_symlinked_config_stays_a_link_and_its_target_holds_the_entry(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dotfiles" / CONFIG_NAME
    target.parent.mkdir()
    _ = target.write_text('{"preferences": {"a": 1}}', encoding="utf-8")
    target.chmod(0o640)
    path = _config(tmp_path)
    path.symlink_to(target)
    result = _entry(tmp_path)

    done = install_desktop_server(path, result.entry)

    assert done.action == "updated"
    assert path.is_symlink()
    assert path.resolve() == target.resolve()
    assert _servers(target) == {"mosaic": result.entry}
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert sorted(item.name for item in target.parent.iterdir()) == [CONFIG_NAME]
