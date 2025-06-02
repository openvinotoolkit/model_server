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
@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Need to set shorter build paths for bazel cache for too long commands in mediapipe compilation
:: We expect a first script argument to be "PR-1234" number passed here from jenkins so that a tmp directory will be created
IF "%~1"=="" (
    echo No argument provided. Using default opt path
    set "BAZEL_SHORT_PATH=C:\opt"
) ELSE (
    echo Argument provided: Using install path %1
    set "BAZEL_SHORT_PATH=C:\%1"
)

IF "%~2"=="--with_python" (
    echo Building model server with Python
    set "bazelBuildArgs=--config=win_mp_on_py_on"
) ELSE (
    echo Building model server without Python 
    set "bazelBuildArgs=--config=win_mp_on_py_off"
)

IF "%~3"=="--with_tests" (
    echo Building model server with tests
    set "buildTargets=//src:ovms //src:ovms_test"
) ELSE (
    echo Building model server without tests
    set "buildTargets=//src:ovms"
)

set "bazelStartupCmd=--output_user_root=!BAZEL_SHORT_PATH!"
set "openvino_dir=!BAZEL_SHORT_PATH!/openvino/runtime/cmake"

set "buildCommand=bazel %bazelStartupCmd% build  %bazelBuildArgs% --action_env OpenVINO_DIR=%openvino_dir% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures %buildTargets% 2>&1 | tee win_build.log"
set "setOvmsVersionCmd=python windows_set_ovms_version.py"

:: Setting PATH environment variable based on default windows node settings: Added ovms_windows specific python settings and c:/opt and removed unused Nvidia and OCL specific tools.
:: When changing the values here you can print the node default PATH value and base your changes on it.
set "setPath=C:\opt;C:\opt\Python312\;C:\opt\Python312\Scripts\;C:\opt\msys64\usr\bin\;%PATH%;"
set "PYTHONHOME=C:\opt\Python312"
set "envPath=win_environment.log"
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

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call !BAZEL_SHORT_PATH!\openvino\setupvars.bat"
set "opencvBatch=call C:\opt\opencv\setup_vars_opencv4.cmd"
set "PYTHONPATH=%PYTHONPATH%;%setPythonPath%"

:: Set required libraries paths
%openvinoBatch%
if !errorlevel! neq 0 exit /b !errorlevel!
%opencvBatch%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Log all environment variables
set > %envPath%
if !errorlevel! neq 0 exit /b !errorlevel!
:: WA
set "genai_dir=openvino_genai_windows_2025.2.0.0rc1_x86_64"
rmdir /S /Q %BAZEL_SHORT_PATH%\%genai_dir%\runtime\include\openvino\genai\image_generation

:: Set ovms.exe --version parameters
%setOvmsVersionCmd% "%bazelBuildArgs%" !BAZEL_SHORT_PATH!
:: Start bazel build
%buildCommand%
if !errorlevel! neq 0 exit /b !errorlevel!

endlocal
