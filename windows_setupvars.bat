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
:: Setting up default OVMS compilation environment variables
set "setPath=C:\opt\Python39\;C:\opt\Python39\Scripts\;C:\opt\msys64\usr\bin\;C:\opt;%PATH%;"
set "setPythonPath=%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"
set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Bazel compilation settings
set VS_2019_PRO="C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
set VS_2022_BT="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
IF /I EXIST %VS_2019_PRO% goto :msvc_pro
IF /I EXIST %VS_2022_BT% goto :msvc_bt ELSE goto :mscv_error

:mscv_error
echo [ERROR] Required MSVC compiler not installed
goto :exit_build_error
:msvc_pro
echo [INFO] Using MSVC %VS_2019_PRO%
set BAZEL_VS=%VS_2019_PRO%
goto :msvc_end
:msvc_bt
echo [INFO] Using MSVC %VS_2022_BT%
set BAZEL_VS=%VS_2022_BT%

:: Bazel compilation settings end
:msvc_end
set "BAZEL_VC=%BAZEL_VS:"=%\VC"
set "BAZEL_VC_FULL_VERSION=14.29.30133"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set "PATH=%setPath%"
set "PYTHONPATH=%PYTHONPATH%;%setPythonPath%"

set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call C:\opt\openvino\setupvars.bat"
set "opencvBatch=call C:\opt\opencv\setup_vars_opencv4.cmd"

:: Set required libraries paths
%openvinoBatch%
if %errorlevel% neq 0 exit /b %errorlevel%
%opencvBatch%
if %errorlevel% neq 0 exit /b %errorlevel%

:exit_build
echo [INFO] Setup finished
exit /b 0
:exit_build_error
echo [ERROR] Setup finished with error
exit /b 1
