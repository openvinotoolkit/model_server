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
:: We expect a first script argument to be "PR-XXXX" number passed here from jenkins so that a tmp directory will be created
:: %2 Second parameter is the --expunge flag - when set to 1 we will force reinstall c:\opt dependencies - default 0
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
set "setPath=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;c:\opt;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\bin\Roslyn;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\devinit;C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x86;C:\Program Files (x86)\Windows Kits\10\bin\x86;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Utils\;C:\Program Files\Git\cmd;C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\usr\bin;C:\Ninja;C:\Program Files\CMake\bin;C:\Program Files\7-zip;C:\opt\Python39\Scripts\;C:\opt\Python39\;C:\opencl\install\;C:\opencl\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel, wget to PATH
set "PATH=%setPath%"

:: OpenCV

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
        goto :eof
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
endlocal