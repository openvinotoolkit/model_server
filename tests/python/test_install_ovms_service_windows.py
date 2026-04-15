#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only test")


def _normalize_win_path(path_value: str) -> str:
    stripped = path_value.strip().strip('"')
    if not stripped:
        return ""
    return os.path.normcase(os.path.normpath(stripped))


def _split_path_entries(path_value: str) -> set[str]:
    return {_normalize_win_path(entry) for entry in path_value.split(";") if entry.strip()}


def _run_cmd(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["cmd", "/v:on", "/c", command], capture_output=True, text=True, check=False)


def _run_powershell(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )


def _get_user_env(var_name: str) -> str:
    result = _run_powershell(f"[Environment]::GetEnvironmentVariable('{var_name}','User')")
    assert result.returncode == 0, f"Failed to read user env {var_name}: {result.stderr}"
    return result.stdout.strip()


def _set_user_env(var_name: str, value: str | None) -> None:
    if value is None:
        value_literal = "$null"
    else:
        escaped = value.replace("'", "''")
        value_literal = f"'{escaped}'"
    result = _run_powershell(
        f"[Environment]::SetEnvironmentVariable('{var_name}',{value_literal},'User')"
    )
    assert result.returncode == 0, f"Failed to write user env {var_name}: {result.stderr}"


@pytest.fixture()
def preserve_user_env():
    original = {
        "OVMS_MODEL_REPOSITORY_PATH": _get_user_env("OVMS_MODEL_REPOSITORY_PATH"),
        "PYTHONHOME": _get_user_env("PYTHONHOME"),
        "Path": _get_user_env("Path"),
    }
    try:
        yield
    finally:
        _set_user_env("OVMS_MODEL_REPOSITORY_PATH", original["OVMS_MODEL_REPOSITORY_PATH"])
        _set_user_env("PYTHONHOME", original["PYTHONHOME"])
        _set_user_env("Path", original["Path"])


def _prepare_test_copy(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = Path(__file__).resolve().parents[2] / "install_ovms_service.bat"
    ovms_dir = tmp_path / "ovms install"
    models_dir = tmp_path / "models dir"
    ovms_dir.mkdir(parents=True, exist_ok=True)
    script_copy = ovms_dir / "install_ovms_service.bat"

    script_text = source.read_text(encoding="utf-8")
    # Skip service registration and ovms service-install call in unit tests.
    script_text = script_text.replace(
        "sc create ovms binPath= \"!binPath_cmd!\" DisplayName= \"OpenVino Model Server\"",
        "set \"SC_CREATE_ERROR=0\"\nREM [TEST] skipped sc create",
    )
    script_text = script_text.replace(
        "set \"SC_CREATE_ERROR=!errorlevel!\"",
        "REM [TEST] skipped SC_CREATE_ERROR from sc create",
    )
    script_text = script_text.replace(
        '"!OVMS_DIR!\\ovms.exe" install',
        "echo [TEST] skipped ovms.exe install",
    )
    script_copy.write_text(script_text, encoding="utf-8")

    return script_copy, ovms_dir, models_dir


def _parse_markers(output: str) -> dict[str, str]:
    markers = {}
    for line in output.splitlines():
        if line.startswith("__") and "=" in line:
            key, value = line.split("=", 1)
            markers[key.strip()] = value.strip()
    return markers


def test_install_script_creates_repo_config_and_sets_current_session_env(tmp_path: Path, preserve_user_env):
    script_copy, ovms_dir, models_dir = _prepare_test_copy(tmp_path)

    cmd = (
        f'call "{script_copy}" "{models_dir}" '
        "& echo __CUR_OVMS_REPO=!OVMS_MODEL_REPOSITORY_PATH! "
        "& echo __CUR_PYTHONHOME=!PYTHONHOME! "
        "& echo __CUR_PATH=!PATH!"
    )
    result = _run_cmd(cmd)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    config_path = models_dir / "config.json"
    assert models_dir.is_dir(), f"Model repository was not created: {models_dir}"
    assert config_path.is_file(), f"Config file was not created: {config_path}"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model_config_list": []}

    markers = _parse_markers(result.stdout)
    assert _normalize_win_path(markers["__CUR_OVMS_REPO"]) == _normalize_win_path(str(models_dir))

    expected_pythonhome = _normalize_win_path(str(ovms_dir / "python"))
    assert _normalize_win_path(markers["__CUR_PYTHONHOME"]) == expected_pythonhome

    current_path_entries = _split_path_entries(markers["__CUR_PATH"])
    assert _normalize_win_path(str(ovms_dir)) in current_path_entries
    assert expected_pythonhome in current_path_entries
    assert _normalize_win_path(str(ovms_dir / "python" / "Scripts")) in current_path_entries


def test_install_script_persists_env_for_new_terminals(tmp_path: Path, preserve_user_env):
    script_copy, ovms_dir, models_dir = _prepare_test_copy(tmp_path)

    result = _run_cmd(f'call "{script_copy}" "{models_dir}"')
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    user_repo = _normalize_win_path(_get_user_env("OVMS_MODEL_REPOSITORY_PATH"))
    user_pythonhome = _normalize_win_path(_get_user_env("PYTHONHOME"))
    user_path_entries = _split_path_entries(_get_user_env("Path"))

    assert user_repo == _normalize_win_path(str(models_dir))

    expected_pythonhome = _normalize_win_path(str(ovms_dir / "python"))
    assert user_pythonhome == expected_pythonhome

    assert _normalize_win_path(str(ovms_dir)) in user_path_entries
    assert expected_pythonhome in user_path_entries
    assert _normalize_win_path(str(ovms_dir / "python" / "Scripts")) in user_path_entries


def test_install_script_call_and_argument_paths_with_spaces(tmp_path: Path, preserve_user_env):
    script_copy, ovms_dir, models_dir = _prepare_test_copy(tmp_path / "workspace with spaces")

    command = (
        f'call "{script_copy}" "{models_dir}" '
        "& echo __SPACE_OVMS_REPO=!OVMS_MODEL_REPOSITORY_PATH! "
        "& echo __SPACE_PYTHONHOME=!PYTHONHOME! "
        "& echo __SPACE_PATH=!PATH!"
    )
    result = _run_cmd(command)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    markers = _parse_markers(result.stdout)

    # OVMS_MODEL_REPOSITORY_PATH resolved correctly with spaces
    assert _normalize_win_path(markers["__SPACE_OVMS_REPO"]) == _normalize_win_path(str(models_dir))

    # Current-session PYTHONHOME points into the spaced OVMS dir
    expected_pythonhome = _normalize_win_path(str(ovms_dir / "python"))
    assert _normalize_win_path(markers["__SPACE_PYTHONHOME"]) == expected_pythonhome

    # Current-session PATH contains all three spaced entries
    current_path_entries = _split_path_entries(markers["__SPACE_PATH"])
    assert _normalize_win_path(str(ovms_dir)) in current_path_entries
    assert expected_pythonhome in current_path_entries
    assert _normalize_win_path(str(ovms_dir / "python" / "Scripts")) in current_path_entries

    # User PATH (new terminal) also contains the spaced entries
    user_path_entries = _split_path_entries(_get_user_env("Path"))
    assert _normalize_win_path(str(ovms_dir)) in user_path_entries
    assert expected_pythonhome in user_path_entries
    assert _normalize_win_path(str(ovms_dir / "python" / "Scripts")) in user_path_entries


def test_install_script_preserves_existing_path_with_special_characters(tmp_path: Path, preserve_user_env):
    """Verify that existing user PATH entries containing parentheses and percent-style tokens
    are not corrupted or dropped when the installer prepends the OVMS entries."""
    script_copy, ovms_dir, _ = _prepare_test_copy(tmp_path)

    # Seed user PATH with entries that contain parentheses and a percent-style token —
    # the original crash trigger: C:\Program Files (x86)\... and %USERPROFILE%\AppData\...
    seed_entries = [
        r"C:\Program Files (x86)\NVIDIA Corporation\PhysX\Common",
        r"C:\Program Files (x86)\Intel\OpenVINO",
        r"%USERPROFILE%\AppData\Local\Microsoft\WindowsApps",
        r"C:\Some (other) tool\bin",
    ]
    seed_path = ";".join(seed_entries)
    _set_user_env("Path", seed_path)

    models_dir = tmp_path / "models"
    result = _run_cmd(f'call "{script_copy}" "{models_dir}"')
    assert result.returncode == 0, (
        f"Script failed — likely a parsing error caused by special chars in PATH.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    user_path_after = _get_user_env("Path")
    user_path_entries = _split_path_entries(user_path_after)

    # All original seed entries must still be present, untouched
    for entry in seed_entries:
        assert _normalize_win_path(entry) in user_path_entries, (
            f"Existing PATH entry was lost or corrupted: {entry}\n"
            f"User PATH after install: {user_path_after}"
        )

    # OVMS entries must have been prepended
    expected_pythonhome = _normalize_win_path(str(ovms_dir / "python"))
    assert _normalize_win_path(str(ovms_dir)) in user_path_entries
    assert expected_pythonhome in user_path_entries
    assert _normalize_win_path(str(ovms_dir / "python" / "Scripts")) in user_path_entries


def test_install_script_does_not_duplicate_path_entries(tmp_path: Path, preserve_user_env):
    """Running the installer twice must not add duplicate entries to user PATH."""
    script_copy, ovms_dir, models_dir = _prepare_test_copy(tmp_path)

    # First run
    result = _run_cmd(f'call "{script_copy}" "{models_dir}"')
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    user_path_after_first = _get_user_env("Path")

    # Second run — installer must detect paths already present and not re-add them
    result = _run_cmd(f'call "{script_copy}" "{models_dir}"')
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    user_path_after_second = _get_user_env("Path")

    # The raw user PATH string must be identical — no duplicates appended
    assert user_path_after_first == user_path_after_second, (
        "User PATH changed after second install run — duplicate entries were added.\n"
        f"After 1st run: {user_path_after_first}\n"
        f"After 2nd run: {user_path_after_second}"
    )

    # Verify each expected entry appears exactly once
    ovms_dir_norm = _normalize_win_path(str(ovms_dir))
    pythonhome_norm = _normalize_win_path(str(ovms_dir / "python"))
    scripts_norm = _normalize_win_path(str(ovms_dir / "python" / "Scripts"))

    all_entries = [_normalize_win_path(e) for e in user_path_after_second.split(";") if e.strip()]
    assert all_entries.count(ovms_dir_norm) == 1, f"OVMS_DIR appears {all_entries.count(ovms_dir_norm)} times in PATH"
    assert all_entries.count(pythonhome_norm) == 1, f"PYTHONHOME appears {all_entries.count(pythonhome_norm)} times in PATH"
    assert all_entries.count(scripts_norm) == 1, f"PYTHONHOME\\Scripts appears {all_entries.count(scripts_norm)} times in PATH"
