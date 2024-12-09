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
setlocal EnableExtensions DisableDelayedExpansion
:: Need to set shorter build paths for bazel cache for too long commands in mediapipe compilation
:: We expect a first script argument to be "PR-1234" number passed here from jenkins so that a tmp directory will be created
set "BAZEL_SHORT_PATH=C:\%1"
set "bazelStartupCmd=--output_user_root=%BAZEL_SHORT_PATH%"
set "openvino_dir=C:/%1/openvino/runtime/cmake"

set "buildCommand=bazel %bazelStartupCmd% build --config=windows --action_env OpenVINO_DIR=%openvino_dir% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures //src:ovms 2>&1 | tee win_build.log"
set "buildTestCommand=bazel %bazelStartupCmd% build --config=windows --action_env OpenVINO_DIR=%openvino_dir% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures //src:ovms_test 2>&1 | tee win_build_test.log"
set "copyPyovms=cp %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.so %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.pyd"
set "changeConfigsCmd=windows_change_test_configs.py"
set "runTest=%cd%\bazel-bin\src\ovms_test.exe --gtest_filter=* 2>&1 | tee win_full_test.log"

:: Setting PATH environment variable based on default windows node settings: Added ovms_windows specific python settings and c:/opt and removed unused Nvidia and OCL specific tools.
:: When changing the values here you can print the node default PATH value and base your changes on it.
set "setPath=%PATH%;c:\opt"
set "envPath=win_environment.log"
set "setPythonPath=%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"

:: Bazel compilation settings
set "VS_2019_PRO=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional"
set "VS_2022_BT=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
IF /I EXIST %VS_2019_PRO% (
    echo [INFO] Using MSVC %VS_2019_PRO%
    set "BAZEL_VS=%VS_2019_PRO%"
) ELSE (
    IF /I EXIST %VS_2022_BT% (
        echo [INFO] Using MSVC %VS_2022_BT%
        set "BAZEL_VS=%VS_2022_BT%"
    ) ELSE (
        echo [ERROR] Required MSVC compiler not installed
        goto exit
    )
)

set "BAZEL_VC=%BAZEL_VS%\VC"
set "BAZEL_VC_FULL_VERSION=14.29.30133"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set "PATH=%setPath%"
set "PYTHONPATH=%setPythonPath%"

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call %BAZEL_SHORT_PATH%\openvino\setupvars.bat"
set "opencvBatch=call C:\opt\opencv\setup_vars_opencv4.cmd"

:: Set required libraries paths
%openvinoBatch%
%opencvBatch%

:: Log all environment variables
set > %envPath%

:: Start bazel build
%buildCommand%

:: Copy pyovms.so -> pyovms.pyd
%copyPyovms%

:: Start bazel build test
%buildTestCommand%

:: Change tests configs to windows paths
%changeConfigsCmd%

:: Start unit test
%runTest%

:: Cut tests log to results
set regex="\[  .* ms"
set sed_clean="s/ (.* ms)//g"
grep -a %regex% win_full_test.log | sed %sed_clean% | tee win_test.log
:exit
endlocal
