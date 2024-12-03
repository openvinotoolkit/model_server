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

:: OpenVINO
set "openvino_dir=w_openvino_toolkit_windows_2024.5.0.17288.7975fa5da0c_x86_64"
set "openvino_ver=w_openvino_toolkit_windows_2024.5.0.17288.7975fa5da0c_x86_64.zip"
set "openvino_http=https://storage.openvinotoolkit.org/repositories/openvino/packages/2024.5/windows/"
set "openvino_wget=wget -P %BAZEL_SHORT_PATH%\ %openvino_http%%openvino_ver%"
set "openvino_zip=%BAZEL_SHORT_PATH%\%openvino_ver%"
set "openvino_workspace=C:\\\\opt\\\\intel\\\\openvino\\\\runtime"
set "openvino_new_worksapce=C:\\%1\\openvino\\runtime"

:: Download OpenVINO
IF /I EXIST %openvino_zip% (
    echo [INFO] file exists %openvino_zip%
) ELSE (
    %openvino_wget%
)
:: Extract OpenVINO
IF /I EXIST %BAZEL_SHORT_PATH%\%openvino_dir% (
    echo [INFO] directory exists %BAZEL_SHORT_PATH%%openvino_dir%
) ELSE (
    tar -xf %openvino_zip% -C %BAZEL_SHORT_PATH%
)
:: Create OpenVINO link
IF /I EXIST %BAZEL_SHORT_PATH%\openvino (
    echo [INFO] link exists %BAZEL_SHORT_PATH%\openvino
) ELSE (
    mklink /d %BAZEL_SHORT_PATH%\openvino %BAZEL_SHORT_PATH%\%openvino_dir%
)

powershell -Command "(gc -Path WORKSPACE -Raw) -replace '%openvino_workspace%', '%openvino_new_worksapce%' | Set-Content -Path WORKSPACE"

endlocal