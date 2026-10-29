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

set "ROOT=%~dp0"
for %%I in ("%ROOT%.") do set "ROOT=%%~fI"
set "PYTHONHOME=C:\opt\Python312"
set "PYTHONPATH=C:\opt\Python312"
set "BASH_EXE=C:\opt\msys64\usr\bin\bash.exe"
set "LOCAL_TOOLS_DIR=%ROOT%\.windows_make_tools"

if not exist "%BASH_EXE%" (
    echo [ERROR] MSYS2 Bash was not found at %BASH_EXE%
    echo [ERROR] Run windows_install_build_dependencies.bat first.
    exit /b 1
)

if not exist "%LOCAL_TOOLS_DIR%" mkdir "%LOCAL_TOOLS_DIR%"

if not exist "%ROOT%\.venv\Scripts\python.exe" (
    echo [INFO] Creating local .venv for repo Python dependencies...
    "%PYTHONHOME%\python.exe" -m venv "%ROOT%\.venv"
)
if not exist "%ROOT%\.venv-style\Scripts\python.exe" (
    echo [INFO] Creating local .venv-style for style tools...
    "%PYTHONHOME%\python.exe" -m venv "%ROOT%\.venv-style"
)
if not exist "%ROOT%\.venv-style\Scripts\clang-format.exe" (
    echo [INFO] Installing style dependencies into .venv-style...
    "%ROOT%\.venv-style\Scripts\python.exe" -m pip install --disable-pip-version-check --no-input -r "%ROOT%\ci\style_requirements.txt"
)

"C:\opt\Python312\python.exe" -c "from pathlib import Path; d = Path(r'%LOCAL_TOOLS_DIR%'); d.mkdir(exist_ok=True); d.joinpath('python3').write_text('''#!/usr/bin/env bash\nif [ -x \"/c/opt/Python312/python.exe\" ]; then\n    exec \"/c/opt/Python312/python.exe\" \"$@\"\nfi\nif command -v python >/dev/null 2>&1; then\n    exec python \"$@\"\nfi\necho \"[ERROR] python3 shim could not find a Python 3 interpreter.\" >&2\nexit 127\n''', encoding='utf-8'); d.joinpath('lscpu').write_text('''#!/usr/bin/env bash\nif command -v /usr/bin/lscpu >/dev/null 2>&1; then\n    exec /usr/bin/lscpu \"$@\"\nfi\ncore_count=${NUMBER_OF_PROCESSORS:-1}\necho \"Core(s) per socket: $core_count\"\necho \"Socket(s): 1\"\n''', encoding='utf-8'); d.joinpath('git').write_text('''#!/usr/bin/env bash\nif [ -x \"/c/Program Files/Git/bin/git.exe\" ]; then\n    exec \"/c/Program Files/Git/bin/git.exe\" \"$@\"\nfi\nif [ -x \"/c/Program Files/Git/cmd/git.exe\" ]; then\n    exec \"/c/Program Files/Git/cmd/git.exe\" \"$@\"\nfi\nif command -v git >/dev/null 2>&1; then\n    exec git \"$@\"\nfi\necho \"[ERROR] Git for Windows was not found in the expected installation paths.\" >&2\nexit 127\n''', encoding='utf-8')"

set "ROOT_MINGW=%ROOT:\=/%"
set "ROOT_MINGW=/c%ROOT_MINGW:~2%"
set "LOCAL_TOOLS_MINGW=%LOCAL_TOOLS_DIR:\=/%"
set "LOCAL_TOOLS_MINGW=/c%LOCAL_TOOLS_MINGW:~2%"

"%BASH_EXE%" -lc "chmod +x %LOCAL_TOOLS_MINGW%/python3 %LOCAL_TOOLS_MINGW%/lscpu %LOCAL_TOOLS_MINGW%/git" >nul 2>&1

set "MSYS_PATH=%LOCAL_TOOLS_MINGW%:/c/Program\ Files/Git/bin:/c/Program\ Files/Git/cmd:/c/Program\ Files/Git/usr/bin:/c/opt/Python312:/usr/local/bin:/usr/bin:/bin:/c/opt/msys64/usr/bin:/c/opt/Python312/Scripts:/c/opt"
set "HTTP_PROXY=%http_proxy%"
set "HTTPS_PROXY=%https_proxy%"
set "NO_PROXY=%no_proxy%"

set "MAKE_ARGS=%*"

echo [INFO] Running make target(s) in MSYS2 environment: %MAKE_ARGS%
"%BASH_EXE%" -lc "unset PYTHONHOME PYTHONPATH; export PATH='%MSYS_PATH%:$PATH'; export http_proxy='%HTTP_PROXY%'; export https_proxy='%HTTPS_PROXY%'; export no_proxy='%NO_PROXY%'; cd '%ROOT_MINGW%'; make %MAKE_ARGS%"
set "MAKE_EXIT=%ERRORLEVEL%"

echo [INFO] Finished make invocation. Exit code: %MAKE_EXIT%
exit /b %MAKE_EXIT%
