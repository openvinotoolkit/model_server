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
call "%~dp0windows_setupvars.bat"
if errorlevel 1 exit /b %errorlevel%

set "ROOT=%~dp0"
for %%I in ("%ROOT%.") do set "ROOT=%%~fI"

set "BASH_EXE=C:\opt\msys64\usr\bin\bash.exe"
set "LOCAL_TOOLS_DIR=%ROOT%\.windows_make_tools"
set "ROOT_MINGW=%ROOT:\=/%"
set "ROOT_MINGW=/c%ROOT_MINGW:~2%"
set "LOCAL_TOOLS_MINGW=%LOCAL_TOOLS_DIR:\=/%"
set "LOCAL_TOOLS_MINGW=/c%LOCAL_TOOLS_MINGW:~2%"

set "MSYS_PATH=%LOCAL_TOOLS_MINGW%:/c/Program\ Files/Git/bin:/c/Program\ Files/Git/cmd:/c/Program\ Files/Git/usr/bin:/c/opt/Python312:/usr/local/bin:/usr/bin:/bin:/c/opt/msys64/usr/bin:/c/opt/Python312/Scripts:/c/opt"
set "HTTP_PROXY=%http_proxy%"
set "HTTPS_PROXY=%https_proxy%"
set "NO_PROXY=%no_proxy%"

if "%*"=="" (
    set "MAKE_ARGS=style"
) else (
    set "MAKE_ARGS=%*"
)

echo [INFO] Running make target(s) in MSYS2 environment: %MAKE_ARGS%
"%BASH_EXE%" -lc "unset PYTHONHOME PYTHONPATH; export PATH='%MSYS_PATH%:$PATH'; export http_proxy='%HTTP_PROXY%'; export https_proxy='%HTTPS_PROXY%'; export no_proxy='%NO_PROXY%'; cd '%ROOT_MINGW%'; make %MAKE_ARGS%"
set "MAKE_EXIT=%ERRORLEVEL%"

echo [INFO] Finished make invocation. Exit code: %MAKE_EXIT%
exit /b %MAKE_EXIT%
