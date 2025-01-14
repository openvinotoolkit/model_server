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
:: Need to set shorter build paths for bazel cache for too long commands in mediapipe compilation
:: We expect a first script argument to be "PR-1234" number passed here from jenkins so that a tmp directory will be created
IF "%~1"=="" (
    echo No argument provided. Using default opt path
    set "BAZEL_SHORT_PATH=C:\opt"
) ELSE (
    echo Argument provided: Using install path %1
    set "BAZEL_SHORT_PATH=C:\%1"
)

set "bazelStartupCmd=--output_user_root=%BAZEL_SHORT_PATH%"
set "openvino_dir=C:/%1/openvino/runtime/cmake"

set "bazelBuildArgs=--config=windows --action_env OpenVINO_DIR=%openvino_dir%"
set "buildCommand=bazel %bazelStartupCmd% build  %bazelBuildArgs% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures //src:ovms 2>&1 | tee win_build.log"
set "buildTestCommand=bazel %bazelStartupCmd% build %bazelBuildArgs% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures //src:ovms_test 2>&1 | tee win_build_test.log"
set "changeConfigsCmd=windows_change_test_configs.py"
set "setOvmsVersionCmd=windows_set_ovms_version.py"
set "runTest=%cd%\bazel-bin\src\ovms_test.exe --gtest_filter=* 2>&1 | tee win_full_test.log"

:: Setting PATH environment variable based on default windows node settings: Added ovms_windows specific python settings and c:/opt and removed unused Nvidia and OCL specific tools.
:: When changing the values here you can print the node default PATH value and base your changes on it.
set "setPath=C:\opt;C:\opt\Python39\;C:\opt\Python39\Scripts\;C:\opt\msys64\usr\bin\;%PATH%;"
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
set "PYTHONPATH=%PYTHONPATH%;%setPythonPath%"

set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call %BAZEL_SHORT_PATH%\openvino\setupvars.bat"
set "opencvBatch=call C:\opt\opencv\setup_vars_opencv4.cmd"

:: Set required libraries paths
%openvinoBatch%
if !errorlevel! neq 0 exit /b !errorlevel!
%opencvBatch%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Log all environment variables
set > %envPath%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Set ovms.exe --version parameters
%setOvmsVersionCmd% "%bazelBuildArgs%" %BAZEL_SHORT_PATH%
:: Start bazel build
%buildCommand%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Start bazel build test
%buildTestCommand%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Change tests configs to windows paths
%changeConfigsCmd%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Copy OpenVINO GenAI and tokenizers libs
:: TODO this is a hack to be improved after bazel llm windows integration
copy %cd%\bazel-out\x64_windows-opt\bin\external\llm_engine\copy_openvino_genai\openvino_genai\runtime\bin\Release\*.dll %cd%\bazel-bin\src\

ls %cd%\bazel-bin\src

:: Install Jinja in Python for chat templates to work
set PYTHONHOME=C:\opt\Python39
call C:\opt\Python39\python.exe -m pip install "Jinja2==3.1.4" "MarkupSafe==3.0.2"

:: Download LLMs
call %cd%\windows_prepare_llm_models.bat %cd%\src\test\llm_testing

:: Start unit test
%runTest%

:: Cut tests log to results
set regex="\[  .* ms"
set sed_clean="s/ (.* ms)//g"
grep -a %regex% win_full_test.log | sed %sed_clean% | tee win_test.log
:exit_build
echo [INFO] Build finished
exit /b 0
:exit_build_error
echo [ERROR] Build finished with error
exit /b 1
endlocal
