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
IF "%1"=="1" (
    echo Argument provided: Using install path %1
    set "output_user_root=%1"
) ELSE (
    echo No argument provided. Using default opt path
    set "output_user_root=opt"
)
IF "%2"=="1" (
    echo Argument provided: Using expunge = %2
    set "expunge=1"
) ELSE (
    echo No argument provided. Using default expunge = 0
    set "expunge=0"
)

set "BAZEL_SHORT_PATH=C:\%output_user_root%"
set "setPath=%PATH%;c:\opt"

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

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Python39
set "python39_path=%opt_install_dir%\Python39"
set "python39_system=C:\Program Files\Python39"
IF /I EXIST %python39_path% (
    IF %expunge% EQU 1 (
        rm -rf %python39_path%
        IF /I EXIST "%python39_system%" (
            :: Link system path
            mklink /d %python39_path% "%python39_system%"
            pip install numpy==1.23
        ) ELSE (
            echo [ERROR] ::::::::::::::::::::::: Python39 not found
            goto :exit_dependencies_error
        )
    ) ELSE (
        echo [INFO] ::::::::::::::::::::::: Python39 already installed
    )
) ELSE (
    IF /I EXIST "%python39_system%" (
        :: Link system path
        mklink /d %python39_path% "%python39_system%"
        pip install numpy==1.23
    ) ELSE (
        echo [ERROR] ::::::::::::::::::::::: Python39 not found
        goto :exit_dependencies_error
    )
)
python --version

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCV

set "opencv_git=https://github.com/opencv/opencv.git"
set "opencv_contrib=https://github.com/opencv/opencv_contrib.git"
set "opencv_ver=4.10.0"
set "opencv_dir=%BAZEL_SHORT_PATH%\opencv_git"
set "opencv_contrib_dir=%BAZEL_SHORT_PATH%\opencv_contrib_git"
set "opencv_install=%BAZEL_SHORT_PATH%\opencv"
set "opencv_flags=-D BUILD_LIST=core,improc,imgcodecs,calib3d,features2d,highgui,imgproc,video,videoio,optflow -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_aruco=OFF -D BUILD_opencv_bgsegm=OFF -D BUILD_opencv_bioinspired=OFF -D BUILD_opencv_ccalib=OFF -D BUILD_opencv_datasets=OFF -D BUILD_opencv_dnn=OFF -D BUILD_opencv_dnn_objdetect=OFF -D BUILD_opencv_dpm=OFF -D BUILD_opencv_face=OFF -D BUILD_opencv_fuzzy=OFF -D BUILD_opencv_hfs=OFF -D BUILD_opencv_img_hash=OFF -D BUILD_opencv_js=OFF -D BUILD_opencv_line_descriptor=OFF -D BUILD_opencv_phase_unwrapping=OFF -D BUILD_opencv_plot=OFF -D BUILD_opencv_quality=OFF -D BUILD_opencv_reg=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_saliency=OFF -D BUILD_opencv_shape=OFF -D BUILD_opencv_structured_light=OFF -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_world=ON -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_xphoto=OFF -D CV_ENABLE_INTRINSICS=ON -D WITH_EIGEN=ON -D WITH_PTHREADS=ON -D WITH_PTHREADS_PF=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_TIFF=ON "

IF /I EXIST %opencv_install% (
    if %expunge% EQU 1 (rm -rf %opencv_install%) else (
        echo "[INFO] OpenCV installed in: "%opencv_install%
        goto :exit_dependencies
    )
)

echo [INFO] Installing OpenCV: %opencv_ver%
:: Clone OpenCL
IF /I EXIST %opencv_dir% (
    rm -rf %opencv_dir%
)
IF /I EXIST %opencv_contrib_dir% (
    rm -rf %opencv_contrib_dir%
)

git clone --depth 1 --branch %opencv_ver% %opencv_git% %opencv_dir%
git clone --depth 1 --branch %opencv_ver% %opencv_contrib% %opencv_contrib_dir%

cd %opencv_dir%
mkdir build
cd build
:: -D CMAKE_INSTALL_PREFIX=C:\opt\opencv - Add if installation needed
cmake .. -D CMAKE_INSTALL_PREFIX=%opencv_install% -D OPENCV_EXTRA_MODULES_PATH=%opencv_contrib_dir%\modules %opencv_flags%
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
cmake --install .

:exit_dependencies
echo [INFO] Dependencies installed
exit /b 0
:exit_dependencies_error
echo [ERROR] Some dependencies not installed
exit /b 1
endlocal