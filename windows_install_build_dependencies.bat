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
setlocal EnableExtensions EnableDelayedExpansion
:: Need to set shorter build paths for bazel cache for too long commands in mediapipe compilation
:: We expect a first script argument to be "PR-XXXX" number passed here from jenkins so that a tmp directory will be created
IF "%~1"=="" (
    echo No argument provided. Using default opt path
    set "output_user_root=opt"
) ELSE (
    echo Argument provided: Using install path %1
    set "output_user_root=%1"
)
IF "%2"=="1" (
    echo Argument provided: Using expunge = %2
    set "expunge=1"
) ELSE (
    echo No argument provided. Using default expunge = 0
    set "expunge=0"
)

set "BAZEL_SHORT_PATH=C:\%output_user_root%"
set "opt_install_dir=C:\opt"

:: Python 39 needs to be first in the windows path, as well as MSYS tools
set "setPath=C:\opt\Python39\;C:\opt\Python39\Scripts\;C:\opt\msys64\usr\bin\;C:\opt;%PATH%;"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel, wget to PATH
set "PATH=%setPath%"

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Check directories
IF /I EXIST %BAZEL_SHORT_PATH% (
    echo [INFO] directory exists %BAZEL_SHORT_PATH%
) ELSE (
    mkdir %BAZEL_SHORT_PATH%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

IF /I EXIST %opt_install_dir% (
    echo [INFO] directory exists %opt_install_dir%
) ELSE (
    mkdir %opt_install_dir%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install wget
set "wget_path=%opt_install_dir%\wget.exe"
IF /I EXIST %wget_path% (
    if %expunge% EQU 1 (
        rmdir /S /Q %wget_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
        curl -vf --ca-native -o %wget_path% https://eternallybored.org/misc/wget/1.21.4/64/wget.exe
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] ::::::::::::::::::::::: wget installed already in %wget_path% )
) ELSE (
    curl -vf --ca-native -o %wget_path% https://eternallybored.org/misc/wget/1.21.4/64/wget.exe
    if !errorlevel! neq 0 exit /b !errorlevel!
)

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Msys bash
set "bash_path=%opt_install_dir%\msys64\usr\bin\bash.exe"
set "msys_path=%opt_install_dir%\msys64\"
set "msys_url=https://github.com/msys2/msys2-installer/releases/download/2024-07-27/msys2-x86_64-20240727.exe"
set "msys_exe=msys2-x86_64-20240727.exe"
set "msys_install=%opt_install_dir%\%msys_exe%"
IF /I EXIST %bash_path% (
    if %expunge% EQU 1 (goto :install_msys) else (
        echo [INFO] ::::::::::::::::::::::: Msys bash already installed in: %bash_path%
    )
) ELSE (
    :install_msys
    IF /I EXIST %msys_path% (
        rmdir /S /Q %msys_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
    )
    IF /I NOT EXIST %msys_install% (
        wget -P %opt_install_dir%\ %msys_url%
        if !errorlevel! neq 0 exit /b !errorlevel!
    )

    start "Installing_msys" %msys_install% in --confirm-command --accept-messages --root %msys_path%
    for /l %%i in (1,1,90) do (
        echo Timeout iteration: %%i
        tasklist /NH | find /i "%msys_exe%" > nul
        if !errorlevel! neq 0 (
            Process "%msys_exe%" finished
            goto install_finished
        ) else (
            echo Process "%msys_exe%" in progress ...
            ping 1.1.1.1 -n 1 > nul
        )
    )

    taskkill /f /t /im %msys_exe%
    if !errorlevel! neq 0 exit /b !errorlevel!
    :install_finished
    echo [INFO] ::::::::::::::::::::::: Msys installed in: %msys_path%
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install in c:\PR-XXXX\ section started - once per build, reinstalled only with expunge clean ::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenVINO - reinstalled per build trigger
set "openvino_dir=w_openvino_toolkit_windows_2025.0.0.dev20241217_x86_64"
set "openvino_ver=w_openvino_toolkit_windows_2025.0.0.dev20241217_x86_64.zip"
set "openvino_http=https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/2025.0.0-17638-c5137ff870d/"
set "openvino_zip=%BAZEL_SHORT_PATH%\%openvino_ver%"
set "openvino_workspace=C:\\\\opt\\\\openvino\\\\runtime"
set "openvino_new_workspace=C:\\%output_user_root%\\openvino\\runtime"

echo [INFO] ::::::::::::::::::::::: OpenVino: %openvino_dir%
:: Download OpenVINO
IF /I EXIST %openvino_zip% (
    if %expunge% EQU 1 (
        del /S /Q %openvino_zip%
        if !errorlevel! neq 0 exit /b !errorlevel!
        wget -P %BAZEL_SHORT_PATH%\ %openvino_http%%openvino_ver%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] file exists %openvino_zip% )
    
) ELSE (
    wget -P %BAZEL_SHORT_PATH%\ %openvino_http%%openvino_ver%
    if !errorlevel! neq 0 exit /b !errorlevel!
)
:: Extract OpenVINO
IF /I EXIST %BAZEL_SHORT_PATH%\%openvino_dir% (
     if %expunge% EQU 1 (
        rmdir /S /Q %BAZEL_SHORT_PATH%\%openvino_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
        C:\Windows\System32\tar.exe -xf "%openvino_zip%" -C %BAZEL_SHORT_PATH%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] directory exists %BAZEL_SHORT_PATH%%openvino_dir% )
    
) ELSE (
    C:\Windows\System32\tar.exe -xf "%openvino_zip%" -C %BAZEL_SHORT_PATH%
    if !errorlevel! neq 0 exit /b !errorlevel!
)
:: Create OpenVINO link - always to make sure it points to latest version
IF /I EXIST %BAZEL_SHORT_PATH%\openvino (
    rmdir /S /Q %BAZEL_SHORT_PATH%\openvino
    if !errorlevel! neq 0 exit /b !errorlevel!
)
mklink /d %BAZEL_SHORT_PATH%\openvino %BAZEL_SHORT_PATH%\%openvino_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Replace path to openvino in ovms WORKSPACE file
powershell -Command "(gc -Path WORKSPACE) -replace '%openvino_workspace%', '%openvino_new_workspace%' | Set-Content -Path WORKSPACE"
if !errorlevel! neq 0 exit /b !errorlevel!

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCL headers
set "opencl_git=https://github.com/KhronosGroup/OpenCL-SDK"
set "opencl_ver=v2024.10.24"
set "opencl_dir=%BAZEL_SHORT_PATH%\opencl"

:: Clone OpenCL
IF /I EXIST %opencl_dir% (
    if %expunge% EQU 1 (
        rmdir /S /Q %opencl_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
        git clone --depth 1 --branch %opencl_ver% %opencl_git% %opencl_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else (
        echo [INFO] ::::::::::::::::::::::: OpenCL is already installed in: %opencl_dir%
    )
) ELSE (
    git clone --depth 1 --branch %opencl_ver% %opencl_git% %opencl_dir%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install in c:\opt\ section started - ONE per system, not per BUILD, reinstalled only with expunge clean :::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
        rmdir /S /Q %boringssl_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
        git clone --depth 1 --branch %bringssl_ver% %bringssl_git% %boringssl_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] ::::::::::::::::::::::: BoringSSL already installed in %boringssl_dir% )
) ELSE (
    git clone --depth 1 --branch %bringssl_ver% %bringssl_git% %boringssl_dir%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install bazel
set "bazel_path=%opt_install_dir%\bazel.exe"
set "bazel_file=bazel-6.4.0-windows-x86_64.exe"
IF /I EXIST %bazel_path% (
    if %expunge% EQU 1 (
        del /S /Q %bazel_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
        wget -P %opt_install_dir%\ https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
        if !errorlevel! neq 0 exit /b !errorlevel!
        xcopy /Y /D /I %opt_install_dir%\%bazel_file% %bazel_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else (
        echo [INFO] ::::::::::::::::::::::: bazel already installed
    )
) ELSE (
    wget -P %opt_install_dir%\ https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
    if !errorlevel! neq 0 exit /b !errorlevel!
    xcopy /Y /D /I %opt_install_dir%\%bazel_file% %bazel_path%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Python39
set "python39_path=%opt_install_dir%\Python39\"
set "python39_system=C:\Program Files\Python39\"
IF /I EXIST %python39_path% (
    IF %expunge% EQU 1 (
        rmdir /S /Q %python39_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
        IF /I EXIST "%python39_system%" (
            :: Copy system Python
            xcopy /s /e /q /y "%python39_system%" %python39_path%
            if !errorlevel! neq 0 exit /b !errorlevel!
            pip install numpy==1.23
            if !errorlevel! neq 0 exit /b !errorlevel!
        ) ELSE (
            echo [ERROR] ::::::::::::::::::::::: Python39 not found
            goto :exit_dependencies_error
        )
    ) ELSE (
        echo [INFO] ::::::::::::::::::::::: Python39 already installed
    )
) ELSE (
    IF /I EXIST "%python39_system%" (
        :: Copy system Python
        xcopy /s /e /q /y "%python39_system%" %python39_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
        %python39_path%python.exe -m pip install numpy==1.23
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) ELSE (
        echo [ERROR] ::::::::::::::::::::::: Python39 not found
        goto :exit_dependencies_error
    )
)
python --version
if !errorlevel! neq 0 exit /b !errorlevel!

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCV

set "opencv_git=https://github.com/opencv/opencv.git"
set "opencv_contrib=https://github.com/opencv/opencv_contrib.git"
set "opencv_ver=4.10.0"
set "opencv_dir=%opt_install_dir%\opencv_git"
set "opencv_contrib_dir=%opt_install_dir%\opencv_contrib_git"
set "opencv_install=%opt_install_dir%\opencv"
set "opencv_flags=-D BUILD_LIST=core,improc,imgcodecs,calib3d,features2d,highgui,imgproc,video,videoio,optflow -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_aruco=OFF -D BUILD_opencv_bgsegm=OFF -D BUILD_opencv_bioinspired=OFF -D BUILD_opencv_ccalib=OFF -D BUILD_opencv_datasets=OFF -D BUILD_opencv_dnn=OFF -D BUILD_opencv_dnn_objdetect=OFF -D BUILD_opencv_dpm=OFF -D BUILD_opencv_face=OFF -D BUILD_opencv_fuzzy=OFF -D BUILD_opencv_hfs=OFF -D BUILD_opencv_img_hash=OFF -D BUILD_opencv_js=OFF -D BUILD_opencv_line_descriptor=OFF -D BUILD_opencv_phase_unwrapping=OFF -D BUILD_opencv_plot=OFF -D BUILD_opencv_quality=OFF -D BUILD_opencv_reg=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_saliency=OFF -D BUILD_opencv_shape=OFF -D BUILD_opencv_structured_light=OFF -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_world=ON -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_xphoto=OFF -D CV_ENABLE_INTRINSICS=ON -D WITH_EIGEN=ON -D WITH_PTHREADS=ON -D WITH_PTHREADS_PF=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_TIFF=ON "

IF /I EXIST %opencv_install% (
    if %expunge% EQU 1 (rmdir /S /Q %opencv_install%) else (
        echo "[INFO] OpenCV installed in: "%opencv_install%
        goto :exit_dependencies
    )
)

echo [INFO] Installing OpenCV: %opencv_ver%
:: Clone OpenCL
IF /I EXIST %opencv_dir% (
    rmdir /S /Q %opencv_dir%
    if !errorlevel! neq 0 exit /b !errorlevel!
)
IF /I EXIST %opencv_contrib_dir% (
    rmdir /S /Q %opencv_contrib_dir%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

git clone --depth 1 --branch %opencv_ver% %opencv_git% %opencv_dir%
if !errorlevel! neq 0 exit /b !errorlevel!
git clone --depth 1 --branch %opencv_ver% %opencv_contrib% %opencv_contrib_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

cd %opencv_dir%
if !errorlevel! neq 0 exit /b !errorlevel!
mkdir build
if !errorlevel! neq 0 exit /b !errorlevel!
cd build
if !errorlevel! neq 0 exit /b !errorlevel!
:: Expected compilers in CI - -G "Visual Studio 16 2019", local -G "Visual Studio 17 2022" as default
cmake -T v142 .. -D CMAKE_INSTALL_PREFIX=%opencv_install% -D OPENCV_EXTRA_MODULES_PATH=%opencv_contrib_dir%\modules %opencv_flags%
if !errorlevel! neq 0 exit /b !errorlevel!
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
if !errorlevel! neq 0 exit /b !errorlevel!
cmake --install .
if !errorlevel! neq 0 exit /b !errorlevel!

:exit_dependencies
echo [INFO] Dependencies installed
exit /b 0
:exit_dependencies_error
echo [ERROR] Some dependencies not installed
exit /b 1
endlocal