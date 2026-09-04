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

set "bazelStartupCmd=--output_user_root=!BAZEL_SHORT_PATH!"
set "openvino_dir=!BAZEL_SHORT_PATH!/openvino/runtime/cmake"
set "OVMS_MEDIA_URL_ALLOW_REDIRECTS=1"

IF "%~2"=="--with_python" (
    set "bazelBuildArgs=--config=win_mp_on_py_on --action_env OpenVINO_DIR=%openvino_dir%"
) ELSE (
    set "bazelBuildArgs=--config=win_mp_on_py_off --action_env OpenVINO_DIR=%openvino_dir%"
)

IF "%~3"=="" (
    set "gtestFilter=*"
) ELSE (
    set "gtestFilter=%3"
)

set "buildTestCommand=bazel %bazelStartupCmd% build %bazelBuildArgs% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures //src:ovms_test"
set "changeConfigsCmd=python windows_change_test_configs.py"
set "runTest=%cd%\bazel-bin\src\ovms_test.exe --gtest_filter=!gtestFilter! > win_full_test.log 2>&1"

:: Load chosen dependency versions from versions.mk
for /f "usebackq eol=# tokens=1,3" %%A in ("%cd%\versions.mk") do (
    if "%%A"=="OPENCV_VERSION" if "!opencv_version!"=="" set "opencv_version=%%B"
    if "%%A"=="CURL_VERSION" if "!curl_version!"=="" set "curl_version=%%B"
)

:: Setting PATH environment variable based on default windows node settings: Added ovms_windows specific python settings and c:/opt and removed unused Nvidia and OCL specific tools.
:: When changing the values here you can print the node default PATH value and base your changes on it.
set "setPath=C:\opt;C:\opt\Python312\;C:\opt\Python312\Scripts\;C:\opt\msys64\usr\bin\;C:\opt\curl-!curl_version!-win64-mingw\bin;%PATH%;"
set "setPythonPath=%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"
set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Bazel compilation settings
:: Auto-detect Visual Studio (BuildTools/Community/Pro/Enterprise; VS2019/2022/2026+) via vswhere.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto :msvc_error
set "VS_DETECTED="
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_DETECTED=%%i"
if not defined VS_DETECTED goto :msvc_error
set VS_2022_BT="%VS_DETECTED%"
IF /I EXIST %VS_2022_BT% goto :msvc_bt ELSE goto :msvc_error
:msvc_error
echo [ERROR] Required MSVC compiler not installed (need Visual Studio 2019/2022/2026 with the C++ x64 toolset)
goto :exit_build_error
:msvc_bt
echo [INFO] Using MSVC %VS_2022_BT%
set BAZEL_VS=%VS_2022_BT%

:: Bazel compilation settings end
set "BAZEL_VC=%BAZEL_VS:"=%\VC"
:: Auto-detect the latest installed MSVC toolset version (was hardcoded 14.44.35207)
set "BAZEL_VC_FULL_VERSION="
for /f "delims=" %%v in ('dir /b /ad /o-n "%BAZEL_VC%\Tools\MSVC" 2^>nul') do if not defined BAZEL_VC_FULL_VERSION set "BAZEL_VC_FULL_VERSION=%%v"
if not defined BAZEL_VC_FULL_VERSION (
    echo [ERROR] Could not detect an MSVC toolset under "%BAZEL_VC%\Tools\MSVC" - install the C++ x64 build tools for the detected Visual Studio
    exit /b 1
)
echo [INFO] Using MSVC toolset %BAZEL_VC_FULL_VERSION%

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set "PATH=%setPath%"
set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call !BAZEL_SHORT_PATH!\openvino\setupvars.bat"

set "opencvBatch=call C:\opt\opencv_!opencv_version!\setup_vars_opencv4.cmd"
set "PYTHONHOME=C:\opt\Python312"
set "PYTHONPATH=%PYTHONPATH%;%setPythonPath%"

:: Set required libraries paths
%openvinoBatch%
if !errorlevel! neq 0 exit /b !errorlevel!
%opencvBatch%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Start bazel build test
%buildTestCommand% 2>&1 | tee win_build_test.log
set "bazelExitCode=!errorlevel!"
:: Check the exit code and exit if it's not 0
if !bazelExitCode! neq 0 exit /b !bazelExitCode!


:: Change tests configs to windows paths
%changeConfigsCmd%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Download LLMs
call %cd%\windows_prepare_llm_models.bat %cd%\src\test\llm_testing
if !errorlevel! neq 0 exit /b !errorlevel!

:: Run install_ovms_service.bat unit tests
echo Running install_ovms_service.bat unit tests...
python -m pytest tests\python\test_install_ovms_service_windows.py -v > win_install_service_test.log 2>&1
set "pytestExitCode=!errorlevel!"
type win_install_service_test.log
if !pytestExitCode! neq 0 (
    echo [ERROR] install_ovms_service.bat unit tests failed. See win_install_service_test.log.
    exit /b !pytestExitCode!
)
echo [INFO] install_ovms_service.bat unit tests passed.

:: Start unit test
echo Running: %runTest%
%runTest%

:: Cut tests log to results
set regex="\[  .* ms"
set sed_clean="s/ (.* ms)//g"
C:\Windows\System32\tar.exe -a -c -f win_test_log.zip win_full_test.log

:: Create summary log with filtered results, always create the file even if grep finds no matches
 grep -a %regex% win_full_test.log > win_test_summary.tmp
 if !errorlevel! equ 0 (
     sed %sed_clean% win_test_summary.tmp > win_test_summary.log 2>&1
 ) else (
     echo No matching test results found > win_test_summary.log
 )
 if exist win_test_summary.tmp del /f /q win_test_summary.tmp

:: Parse logs and decide final test status using dedicated parser script
:: Skip expensive parsing only if PASSED summary exists AND no FAILED markers
grep -a -q "\[  PASSED  \] " win_full_test.log
set "hasPassed=!errorlevel!"
grep -a -q "\[  FAILED  \] " win_full_test.log
set "hasFailed=!errorlevel!"
if !hasPassed! equ 0 if !hasFailed! neq 0 (
    echo [INFO] Tests finished with no failures. Check the summary in win_test_summary.log.
    exit /b 0
)
call %cd%\windows_parse_tests.bat win_full_test.log win_test_summary.log
set "parseExitCode=!errorlevel!"
if !parseExitCode! neq 0 exit /b !parseExitCode!

echo [INFO] Tests finished with no failures. Check the summary in win_test_summary.log.
exit /b 0

:exit_build_error
echo [ERROR] windows_test.bat failed before test parsing stage.
exit /b 1
endlocal
