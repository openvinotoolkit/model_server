::
:: Copyright (c) 2026 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::      http:::www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::

@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Usage: windows_test_functional.bat <ovms_release_artifacts_path> [pytest_params] [xdist_workers]
::   arg1 - absolute path(s) to OVMS Windows release artifacts (e.g. C:\ovms\ovms.zip), comma-separated
::   arg2 - pytest target(s) and options, quote when it contains spaces. Default: tests/functional
::   arg3 - number of pytest-xdist workers. Default: 4
:: Other TT_* variables (e.g. TT_TARGET_DEVICE, TT_LOGGING_LEVEL_OVMS) are taken from the current environment.

IF "%~1"=="" (
    echo [ERROR] Path to OVMS release artifacts is required.
    echo Usage: windows_test_functional.bat ^<ovms_release_artifacts_path^> [pytest_params] [xdist_workers]
    exit /b 1
)

set "TT_OVMS_C_RELEASE_ARTIFACTS_PATH=%~1"

IF "%~2"=="" (
    set "pytestParams=tests/functional"
) ELSE (
    set "pytestParams=%~2"
)

IF "%~3"=="" (
    set "xdistWorkers=4"
) ELSE (
    set "xdistWorkers=%~3"
)

:: Remove stale results from previous runs
del /f /q pytest-functional.xml 2>nul

:: Settings expected by tests/functional/config.py on Windows
set "TT_BASE_OS=windows"
set "TT_OVMS_TYPE=BINARY"
set "TT_DISABLE_DMESG_LOG_MONITOR=True"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
IF "%TT_TARGET_DEVICE%"=="" set "TT_TARGET_DEVICE=CPU"
IF "%TT_LOGGING_LEVEL_OVMS%"=="" set "TT_LOGGING_LEVEL_OVMS=INFO"
IF "%TT_ON_COMMIT_TESTS%"=="" set "TT_ON_COMMIT_TESTS=True"
IF "%TT_RUN_REGRESSION_TESTS%"=="" set "TT_RUN_REGRESSION_TESTS=False"

:: Short basetemp keeps paths under the Windows path length limit
set "baseTemp=C:\tmp\pytest-functional"
if exist "%baseTemp%" rmdir /s /q "%baseTemp%"

:: Prefer the Python installed by windows_install_build_dependencies.bat
IF EXIST "C:\opt\Python312\python.exe" (
    set "PATH=C:\opt\Python312\;C:\opt\Python312\Scripts\;%PATH%"
)

where python >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Python was not found in PATH.
    exit /b 1
)

if not exist .venv (
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Creating virtual environment failed with error code !errorlevel!.
        exit /b !errorlevel!
    )
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
if !errorlevel! neq 0 (
    echo [ERROR] pip upgrade failed with error code !errorlevel!.
    exit /b !errorlevel!
)
python -m pip install -r tests\requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Installing tests\requirements.txt failed with error code !errorlevel!.
    exit /b !errorlevel!
)

echo Running: pytest !pytestParams! -v -n !xdistWorkers! --junitxml=pytest-functional.xml --basetemp=!baseTemp!
pytest !pytestParams! -v -n !xdistWorkers! --junitxml=pytest-functional.xml --basetemp=!baseTemp!
set "pytestExitCode=!errorlevel!"

if exist "%baseTemp%" rmdir /s /q "%baseTemp%"

if !pytestExitCode! neq 0 (
    echo [ERROR] Functional tests failed with exit code !pytestExitCode!. See pytest-functional.xml.
    exit /b !pytestExitCode!
)

echo [INFO] Functional tests finished with no failures.
exit /b 0
