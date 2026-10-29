::
:: Copyright (c) 2024 Intel Corporation
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
@echo on
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
for %%I in ("%ROOT%.") do set "ROOT=%%~fI"
set "LOCAL_TOOLS_DIR=%ROOT%\.windows_make_tools"
:: Bazel / Python startup can inherit stale PYTHONHOME/PYTHONPATH values from a different
:: service install; clear them before configuring the expected runtime.
if defined PYTHONHOME set "PYTHONHOME="
if defined PYTHONPATH set "PYTHONPATH="
set "SETUP_EXIT_CODE=0"
:: Load chosen dependency versions from versions.mk
for /f "usebackq eol=# tokens=1,3" %%A in ("%cd%\versions.mk") do (
    if "%%A"=="OPENCV_VERSION" if "!opencv_version!"=="" set "opencv_version=%%B"
    if "%%A"=="CURL_VERSION" if "!curl_version!"=="" set "curl_version=%%B"
)

if not exist "%LOCAL_TOOLS_DIR%" mkdir "%LOCAL_TOOLS_DIR%"
if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo [INFO] Creating local .venv for repo Python dependencies...
    "C:\opt\Python312\python.exe" -m venv "%ROOT%\.venv"
)
if not exist "%ROOT%\.venv-style\Scripts\python.exe" (
    echo [INFO] Creating local .venv-style for style tools...
    "C:\opt\Python312\python.exe" -m venv "%ROOT%\.venv-style"
)
if not exist "%ROOT%\.venv-style\Scripts\clang-format.exe" (
    echo [INFO] Installing style dependencies into .venv-style...
    "%ROOT%\.venv-style\Scripts\python.exe" -m pip install --disable-pip-version-check --no-input -r "%ROOT%\ci\style_requirements.txt"
)

"C:\opt\Python312\python.exe" -c "from pathlib import Path; d = Path(r'%LOCAL_TOOLS_DIR%'); d.mkdir(exist_ok=True); d.joinpath('python3').write_text('''#!/usr/bin/env bash\nif [ -x \"/c/opt/Python312/python.exe\" ]; then\n    exec \"/c/opt/Python312/python.exe\" \"$@\"\nfi\nif command -v python >/dev/null 2>&1; then\n    exec python \"$@\"\nfi\necho \"[ERROR] python3 shim could not find a Python 3 interpreter.\" >&2\nexit 127\n''', encoding='utf-8'); d.joinpath('lscpu').write_text('''#!/usr/bin/env bash\nif command -v /usr/bin/lscpu >/dev/null 2>&1; then\n    exec /usr/bin/lscpu \"$@\"\nfi\ncore_count=${NUMBER_OF_PROCESSORS:-1}\necho \"Core(s) per socket: $core_count\"\necho \"Socket(s): 1\"\n''', encoding='utf-8'); d.joinpath('git').write_text('''#!/usr/bin/env bash\nif [ -x \"/c/Program Files/Git/bin/git.exe\" ]; then\n    exec \"/c/Program Files/Git/bin/git.exe\" \"$@\"\nfi\nif [ -x \"/c/Program Files/Git/cmd/git.exe\" ]; then\n    exec \"/c/Program Files/Git/cmd/git.exe\" \"$@\"\nfi\nif command -v git >/dev/null 2>&1; then\n    exec git \"$@\"\nfi\necho \"[ERROR] Git for Windows was not found in the expected installation paths.\" >&2\nexit 127\n''', encoding='utf-8')"

:: Setting up default OVMS compilation environment variables
set "setPath=C:\opt;C:\opt\Python312\;C:\opt\Python312\Scripts\;C:\opt\msys64\usr\bin\;C:\opt\curl-!curl_version!-win64-mingw\bin;%ROOT%\.venv-style\Scripts;%LOCAL_TOOLS_DIR%;%PATH%;"
set "PYTHONHOME=C:\opt\Python312"
set "PYTHONPATH=C:\opt\Python312;%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"
set "setPythonPath=%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"
:: Bazel normally exports TEST_SRCDIR/TEST_WORKSPACE when a test binary is launched under Bazel.
:: For direct Windows invocation, seed them only if they are not already present.
if not defined TEST_SRCDIR set "TEST_SRCDIR=%cd%\bazel-bin"
if not defined TEST_WORKSPACE set "TEST_WORKSPACE=."
set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Bazel compilation settings
set VS_2022_BT="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
IF /I EXIST %VS_2022_BT% goto :msvc_bt ELSE goto :msvc_error

:msvc_error
echo [ERROR] Required MSVC compiler not installed
set "SETUP_EXIT_CODE=1"
goto :exit_build_error
:msvc_bt
echo [INFO] Using MSVC %VS_2022_BT%
set BAZEL_VS=%VS_2022_BT%

:: Bazel compilation settings end
set "BAZEL_VC=%BAZEL_VS:"=%\VC"
set "BAZEL_VC_FULL_VERSION=14.44.35207"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set "PATH=%setPath%"
set "PYTHONHOME=C:\opt\Python312"
set "PYTHONPATH=C:\opt\Python312;%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"
set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call C:\opt\openvino\setupvars.bat"
set "opencvBatch=call C:\opt\opencv_!opencv_version!\setup_vars_opencv4.cmd"

:: Set required libraries paths
%openvinoBatch%
if !errorlevel! neq 0 (
    set "SETUP_EXIT_CODE=!errorlevel!"
    goto :exit_build_error
)
%opencvBatch%
if !errorlevel! neq 0 (
    set "SETUP_EXIT_CODE=!errorlevel!"
    goto :exit_build_error
)
set "PYTHONPATH=%PYTHONPATH:C:\opt\opencv_!opencv_version!\python_loader_is_not_installed;=%"
set "PYTHONPATH=%PYTHONPATH:;C:\opt\opencv_!opencv_version!\python_loader_is_not_installed=%"
set "PYTHONPATH=%PYTHONPATH:C:\opt\opencv_!opencv_version!\python_loader_is_not_installed=%"
set "PYTHONHOME=C:\opt\Python312"
if defined PYTHONPATH set "PYTHONPATH=C:\opt\Python312;%PYTHONPATH%;%setPythonPath%"
if not defined PYTHONPATH set "PYTHONPATH=C:\opt\Python312;%setPythonPath%"

:exit_build
echo [INFO] Setup finished
goto :propagate_env

:exit_build_error
echo [ERROR] Setup finished with error
if "!SETUP_EXIT_CODE!"=="0" set "SETUP_EXIT_CODE=!errorlevel!"

:propagate_env
endlocal & set "PATH=%PATH%" & set "PYTHONPATH=%PYTHONPATH%" & set "PYTHONHOME=C:\opt\Python312" & set "TEST_SRCDIR=%TEST_SRCDIR%" & set "TEST_WORKSPACE=%TEST_WORKSPACE%" & set "BAZEL_VS=%BAZEL_VS%" & set "BAZEL_VC=%BAZEL_VC%" & set "BAZEL_SH=%BAZEL_SH%" & set "BAZEL_VC_FULL_VERSION=%BAZEL_VC_FULL_VERSION%" & exit /b %SETUP_EXIT_CODE%