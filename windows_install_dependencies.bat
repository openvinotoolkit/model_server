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

:: %1 First parameter is the --output_user_root value on c:\ drive - bazel uses this, we want to install dependencies per build there
:: %2 Second parameter is the --expunge flag - when set to 1 we will force reinstall c:\opt dependencies - default 0
@echo on
setlocal EnableExtensions DisableDelayedExpansion
:: Need to set shorter build paths for bazel cache for too long commands in mediapipe compilation
:: We expect a first script argument to be "PR-XXXX" number passed here from jenkins so that a tmp directory will be created
if "%1"=="" or "%1"=="null" (
  echo No argument provided. Using default opt path
  set "output_user_root=opt"
) else (
  echo Argument provided: Using install path %1
  set "output_user_root=%1"
)
if "%2"=="" or "%2"=="null" (
  echo No argument provided. Using default expunge = 0
  set "expunge=0"
) else (
  echo Argument provided: Using expunge = %2
  set "expunge=1"
)
set "BAZEL_SHORT_PATH=C:\%output_user_root%"
set "setPath=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;c:\opt;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\bin\Roslyn;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\devinit;C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x86;C:\Program Files (x86)\Windows Kits\10\bin\x86;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Utils\;C:\Program Files\Git\cmd;C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\usr\bin;C:\Ninja;C:\Program Files\CMake\bin;C:\Program Files\7-zip;C:\opt\Python39\Scripts\;C:\opt\Python39\;C:\opencl\install\;C:\opencl\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel, wget to PATH
set "PATH=%setPath%"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install in c:\PR-XXXX\ section started - once per build, reinstalled only with expunge clean ::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenVINO - reinstalled per build trigger
set "openvino_dir=w_openvino_toolkit_windows_2025.0.0.dev20241203_x86_64"
set "openvino_ver=w_openvino_toolkit_windows_2025.0.0.dev20241203_x86_64.zip"
set "openvino_http=https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.0.0-17513-963b1be951b/"
set "openvino_zip=%BAZEL_SHORT_PATH%\%openvino_ver%"
set "openvino_workspace=C:\\\\opt\\\\openvino\\\\runtime"
set "openvino_new_workspace=C:\\%output_user_root%\\openvino\\runtime"

IF /I EXIST %BAZEL_SHORT_PATH% (
    echo [INFO] directory exists %BAZEL_SHORT_PATH%
) ELSE (
    mkdir %BAZEL_SHORT_PATH%
)

echo [INFO] ::::::::::::::::::::::: OpenVino: %openvino_dir%
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

:: Replace path to openvino in ovms WORKSPACE file
powershell -Command "(gc -Path WORKSPACE -Raw) -replace '%openvino_workspace%', '%openvino_new_workspace%' | Set-Content -Path WORKSPACE"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCL headers
set "opencl_git=https://github.com/KhronosGroup/OpenCL-SDK"
set "opencl_ver=v2024.10.24"
set "opencl_dir=%BAZEL_SHORT_PATH%\opencl"

:: Clone OpenCL
IF /I EXIST %opencl_dir% (
    if %expunge% EQU 1 (
        rm -rf %opencl_dir%
        git clone --depth 1 --branch %opencl_ver% %opencl_git% %opencl_dir%
    ) else (
        echo [INFO] ::::::::::::::::::::::: OpenCL is already installed in: %opencl_dir%
    )
) ELSE (
    git clone --depth 1 --branch %opencl_ver% %opencl_git% %opencl_dir%
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install in c:\opt\ section started - ONE per system, not per BUILD, reinstalled only with expunge clean :::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
set "opt_install_dir=c:\opt"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: BoringSSL 
:: defined in .bazelrc build:windows --override_repository="boringssl=C:\\opt\\boringSSL-SwiftPM"
set "bringssl_git=https://github.com/firebase/boringSSL-SwiftPM/"
set "bringssl_ver=0.32.1"
set "boringssl_dir=%opt_install_dir%\boringSSL-SwiftPM"

echo "[INFO] BoringSSL: "%bringssl_ver%
:: Clone BoringSSL
IF /I EXIST %boringssl_dir% (
    if %expunge% EQU 1 (
        rm -rf %boringssl_dir%
        git clone --depth 1 --branch %bringssl_ver% %bringssl_git% %boringssl_dir%
    ) else ( echo [INFO] ::::::::::::::::::::::: BoringSSL already installed in %boringssl_dir% )
) ELSE (
    git clone --depth 1 --branch %bringssl_ver% %bringssl_git% %boringssl_dir%
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCV
call windows_opencv.bat opt %expunge%

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install wget
set "wget_path=%opt_install_dir%\wget.exe"
IF /I EXIST %wget_path% (
    if %expunge% EQU 1 (
        rm -rf %wget_path%
        curl https://eternallybored.org/misc/wget/1.21.4/64/wget.exe > %wget_path%
    ) else ( echo [INFO] ::::::::::::::::::::::: wget installed already in %wget_path% )
) ELSE (
    curl https://eternallybored.org/misc/wget/1.21.4/64/wget.exe > %wget_path%
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install bazel
set "bazel_path=%opt_install_dir%\bazel.exe"
IF /I EXIST %bazel_path% (
    if %expunge% EQU 1 (
        rm -rf %bazel_path%
        wget -O %bazel_path% https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
    ) else (
        echo [INFO] ::::::::::::::::::::::: bazel already installed
    )
) ELSE (
    wget -O %bazel_path% https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
)

endlocal