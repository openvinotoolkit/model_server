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
set "setPath=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;c:\opt;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\bin\Roslyn;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\devinit;C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x86;C:\Program Files (x86)\Windows Kits\10\bin\x86;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Utils\;C:\Program Files\Git\cmd;C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\usr\bin;C:\Ninja;C:\Program Files\CMake\bin;C:\Program Files\7-zip;C:\opt\Python39\Scripts\;C:\opt\Python39\;C:\opencl\install\;C:\opencl\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel, wget to PATH
set "PATH=%setPath%"

:: OpenVINO
set "openvino_dir=w_openvino_toolkit_windows_2025.0.0.dev20241203_x86_64"
set "openvino_ver=w_openvino_toolkit_windows_2025.0.0.dev20241203_x86_64.zip"
set "openvino_http=https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.0.0-17513-963b1be951b/"
set "openvino_zip=%BAZEL_SHORT_PATH%\%openvino_ver%"
set "openvino_workspace=C:\\\\opt\\\\intel\\\\openvino\\\\runtime"
set "openvino_new_worksapce=C:\\%1\\openvino\\runtime"

:: Download OpenVINO
IF /I EXIST %openvino_zip% (
    echo [INFO] file exists %openvino_zip%
) ELSE (
    wget -P %BAZEL_SHORT_PATH%\ %openvino_http%%openvino_ver%
)
:: Extract OpenVINO
IF /I EXIST %BAZEL_SHORT_PATH%\%openvino_dir% (
    echo [INFO] directory exists %BAZEL_SHORT_PATH%%openvino_dir%
) ELSE (
    tar -xf %openvino_zip% -C %BAZEL_SHORT_PATH%
)
:: Create OpenVINO link - always to make sure it points to latest version
IF /I EXIST %BAZEL_SHORT_PATH%\openvino (
    rm %BAZEL_SHORT_PATH%\openvino
)
mklink /d %BAZEL_SHORT_PATH%\openvino %BAZEL_SHORT_PATH%\%openvino_dir%

powershell -Command "(gc -Path WORKSPACE -Raw) -replace '%openvino_workspace%', '%openvino_new_worksapce%' | Set-Content -Path WORKSPACE"

endlocal