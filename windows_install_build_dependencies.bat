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
@echo off
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
set "setPath=C:\opt;C:\opt\Python311\;C:\opt\Python311\Scripts\;C:\opt\msys64\usr\bin\;c:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\;%PATH%;"

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
::::::::::::::::::::::: Installing wget
echo [INFO] Installing wget ...
set "wget_path=%opt_install_dir%\wget.exe"
IF /I EXIST %wget_path% (
    if %expunge% EQU 1 (
        rmdir /S /Q %wget_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
        curl -k -o %wget_path% https://eternallybored.org/misc/wget/1.21.4/64/wget.exe
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] ::::::::::::::::::::::: wget installed already in %wget_path% )
) ELSE (
    curl -k -o %wget_path% https://eternallybored.org/misc/wget/1.21.4/64/wget.exe
    if !errorlevel! neq 0 exit /b !errorlevel!
)
echo [INFO] Wget installed in %wget_path%
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Msys bash
echo Installing msys ...
set "bash_path=%opt_install_dir%\msys64\usr\bin\bash.exe"
set "msys_path=%opt_install_dir%\msys64\"
set "msys_url=https://github.com/msys2/msys2-installer/releases/download/2024-07-27/msys2-x86_64-20240727.exe"
set "msys_exe=msys2-x86_64-20240727.exe"
set "msys_install=%opt_install_dir%\%msys_exe%"
IF /I EXIST %bash_path% (
    if %expunge% EQU 1 (goto :install_msys) else (
        echo [INFO] Msys bash already installed in: %bash_path%
    )
) ELSE (
    :install_msys
    IF /I EXIST %msys_path% (
        rmdir /S /Q %msys_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
    )
    IF /I NOT EXIST %msys_install% (
        %wget_path% -P %opt_install_dir%\ %msys_url%
        if !errorlevel! neq 0 exit /b !errorlevel!
    )

    start "Installing_msys" %msys_install% in --confirm-command --accept-messages --root %msys_path%
    for /l %%i in (1,1,200) do (
        echo Timeout iteration: %%i
        tasklist /NH | c:\Windows\SysWOW64\find.exe /i "%msys_exe%" > nul
        if !errorlevel! neq 0 (
            echo Process "%msys_exe%" finished
            goto msys_install_finished
        ) else (
            echo Process "%msys_exe%" in progress ...
            ping 1.1.1.1 -n 3 > nul
        )
    )

    taskkill /f /t /im %msys_exe%
    if !errorlevel! neq 0 exit /b !errorlevel!
    :msys_install_finished
    echo [INFO] Msys installed in: %msys_path%
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install in c:\PR-XXXX\ section started - once per build, reinstalled only with expunge clean ::::::::::::::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: GENAI/OPENVINO - reinstalled per build trigger
set "genai_dir=openvino_genai_windows_2025.2.0.0.dev20250411_x86_64"
set "genai_ver=openvino_genai_windows_2025.2.0.0.dev20250411_x86_64.zip"
set "genai_http=https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/2025.2.0.0.dev20250411/"

set "genai_zip=%BAZEL_SHORT_PATH%\%genai_ver%"
set "genai_workspace=C:\\\\opt\\\\openvino\\\\runtime"
set "genai_new_workspace=C:\\%output_user_root%\\openvino\\runtime"

echo [INFO] Installing GenAI: %genai_dir% ...
:: Download GenAi
IF /I EXIST %genai_zip% (
    if %expunge% EQU 1 (
        del /S /Q %genai_zip%
        if !errorlevel! neq 0 exit /b !errorlevel!
        %wget_path% -P %BAZEL_SHORT_PATH%\ %genai_http%%genai_ver%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] file exists %genai_zip% )
    
) ELSE (
    %wget_path% -P %BAZEL_SHORT_PATH%\ %genai_http%%genai_ver%
    if !errorlevel! neq 0 exit /b !errorlevel!
)
:: Extract GenAi
IF /I EXIST %BAZEL_SHORT_PATH%\%genai_dir% (
     if %expunge% EQU 1 (
        rmdir /S /Q %BAZEL_SHORT_PATH%\%genai_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
        C:\Windows\System32\tar.exe -xf "%genai_zip%" -C %BAZEL_SHORT_PATH%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] directory exists %BAZEL_SHORT_PATH%\%genai_dir% )
    
) ELSE (
    C:\Windows\System32\tar.exe -xf "%genai_zip%" -C %BAZEL_SHORT_PATH%
    if !errorlevel! neq 0 exit /b !errorlevel!
)
:: Create GenAi link - always to make sure it points to latest version
IF /I EXIST %BAZEL_SHORT_PATH%\openvino (
    rmdir /S /Q %BAZEL_SHORT_PATH%\openvino
)
mklink /d %BAZEL_SHORT_PATH%\openvino %BAZEL_SHORT_PATH%\%genai_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Replace path to GenAi in ovms WORKSPACE file
if "!output_user_root!" neq "opt" (
    powershell -Command "(gc -Path WORKSPACE) -replace '%genai_workspace%', '%genai_new_workspace%' | Set-Content -Path WORKSPACE"
    if !errorlevel! neq 0 exit /b !errorlevel!
)
echo [INFO] GenAi installed: %BAZEL_SHORT_PATH%\%genai_dir%

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCL headers
echo [INFO] Installing OpenCL headers ...
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
echo [INFO] OpenCL headers installed: %opencl_dir%
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install in c:\opt\ section started - ONE per system, not per BUILD, reinstalled only with expunge clean :::::::::::::::::::::::
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: BoringSSL 
:: defined in .bazelrc build:windows --override_repository="boringssl=C:\\opt\\boringSSL-SwiftPM"
set "bringssl_git=https://github.com/firebase/boringSSL-SwiftPM/"
set "bringssl_ver=0.32.1"
set "boringssl_dir=%opt_install_dir%\boringSSL-SwiftPM"
echo [INFO] Installing BoringSSL ...
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
echo [INFO] BoringSSL installed: %boringssl_dir%
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install bazel
echo [INFO] Installing bazel ...
set "bazel_path=%opt_install_dir%\bazel.exe"
set "bazel_file=bazel-6.4.0-windows-x86_64.exe"
IF /I EXIST %bazel_path% (
    if %expunge% EQU 1 (
        del /S /Q %bazel_path%
        if !errorlevel! neq 0 exit /b !errorlevel!
        %wget_path% -P %opt_install_dir%\ https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
        if !errorlevel! neq 0 exit /b !errorlevel!
        xcopy /Y /D /I %opt_install_dir%\%bazel_file% %bazel_path%*
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else (
        echo [INFO] ::::::::::::::::::::::: bazel already installed
    )
) ELSE (
	IF /I EXIST %bazel_file% (
		echo %bazel_file% exists
	) ELSE (
		%wget_path% -P %opt_install_dir%\ https://github.com/bazelbuild/bazel/releases/download/6.4.0/bazel-6.4.0-windows-x86_64.exe
	)
    if !errorlevel! neq 0 exit /b !errorlevel!
    xcopy /Y /D /I %opt_install_dir%\%bazel_file% %bazel_path%*
    if !errorlevel! neq 0 exit /b !errorlevel!
)
echo [INFO] Bazel installed: %bazel_file%
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Python
set "python_version=3.11.9"
echo [INFO] Installing python %python_version% ...
for /f "tokens=1,2 delims=." %%a in ("%python_version%") do (
        set MAJOR_VER=%%a
        set MINOR_VER=%%b
    )
set "python_dir=python%MAJOR_VER%%MINOR_VER%"
set "python_path=%opt_install_dir%\%python_dir%"
set "python_full_name=python-%python_version%-amd64"
::https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
set "python_url=https://www.python.org/ftp/python/%python_version%/%python_full_name%.exe"

IF /I EXIST %python_path%\python.exe (
    IF %expunge% EQU 1 (
        if !errorlevel! neq 0 exit /b !errorlevel!
        :: Uninstall Python
        IF /I EXIST "%opt_install_dir%\%python_full_name%.exe" (
            call :UninstallPython
            rm %opt_install_dir%\%python_full_name%.exe
            if !errorlevel! neq 0 exit /b !errorlevel!
        )
        rmdir /S /Q %python_path%
        if !errorlevel! neq 0 exit /b !errorlevel!

        :: Install python
        curl -k %python_url% -o %opt_install_dir%\%python_full_name%.exe
        if !errorlevel! neq 0 exit /b !errorlevel!
        call :UninstallPython
        if !errorlevel! neq 0 exit /b !errorlevel!
        call :InstallPython
        IF /I not EXIST %python_path%\python.exe (
            echo "Python installation failed. Errorlevel: !errorlevel!"
            echo "Please review the installation log: python/install.log"
            echo "To fix the installation run %opt_install_dir%\%python_full_name%.exe in GUI and press repair."
            echo "Rerun the windows_install_build_dependencies.bat, once the installation is fixed."
            exit /b !errorlevel!
        )
    ) ELSE (
        echo [INFO] ::::::::::::::::::::::: %python_path% already installed
    )
) ELSE (
    :: Uninstall Python
    IF /I EXIST "%opt_install_dir%\%python_full_name%.exe" (
        call :UninstallPython
        rm %opt_install_dir%\%python_full_name%.exe
        if !errorlevel! neq 0 exit /b !errorlevel!
    )
    :: Install python
    curl -k %python_url% -o %opt_install_dir%\%python_full_name%.exe
    if !errorlevel! neq 0 exit /b !errorlevel!
    call :UninstallPython
    if !errorlevel! neq 0 exit /b !errorlevel!
    call :InstallPython
    IF /I not EXIST %python_path%\python.exe (
        echo "Python installation failed."
        echo "Please review the installation log: python/install.log"
        echo "To fix the installation run %opt_install_dir%\%python_full_name%.exe in GUI and press repair."
        echo "Rerun the windows_install_build_dependencies.bat, once the installation is fixed."
        exit /b !errorlevel!
    )
)
python --version
if !errorlevel! neq 0 exit /b !errorlevel!
%python_path%\python.exe -m ensurepip --upgrade
if !errorlevel! neq 0 exit /b !errorlevel!
:: setuptools<60.0 required for numpy1.23 on python311 to install
%python_path%\python.exe -m pip install "setuptools<60.0" "numpy==1.23" "Jinja2==3.1.6" "MarkupSafe==3.0.2"
if !errorlevel! neq 0 exit /b !errorlevel!
echo [INFO] Python %python_version% installed: %python_path%
goto install_curl
:::::::::::::::::::::: Uninstall function
:UninstallPython
start "Unstalling_python" %opt_install_dir%\%python_full_name%.exe /quiet /uninstall /log python/uninstall.log
echo [INFO] Uninstalling python
for /l %%i in (1,1,300) do (
    echo Timeout iteration: %%i
    tasklist /NH | c:\Windows\SysWOW64\find.exe /i "%python_full_name%.exe" > nul
    if !errorlevel! neq 0 (
        echo Process "%python_full_name%.exe" finished
        goto python_uninstall_finished
    ) else (
        echo Process "%python_full_name%.exe" in progress ...
        ping 1.1.1.1 -n 3 > nul
    )
)

:python_uninstall_finished
exit /b 0
:::::::::::::::::::::: Uninstall python function end
:::::::::::::::::::::: Install python function
:InstallPython
echo [INFO] Running python installer
start "Installing_python" %opt_install_dir%\%python_full_name%.exe /passive /quiet /simple /InstallAllUsers TargetDir=%python_path% /log python/install.log
if !errorlevel! neq 0 exit /b !errorlevel!

for /l %%i in (1,1,300) do (
    echo Timeout iteration: %%i
    tasklist /NH | c:\Windows\SysWOW64\find.exe /i "%python_full_name%.exe" > nul
    if !errorlevel! neq 0 (
        echo Process "%python_full_name%.exe" finished
        goto python_install_finished
    ) else (
        echo Process "%python_full_name%.exe" in progress ...
        ping 1.1.1.1 -n 3 > nul
    )
)
:python_install_finished
exit /b 0
:::::::::::::::::::::: Uninstall function end

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: Install curl
:install_curl
echo [INFO] Installing curl ...

set "curl_dir=curl-8.13.0_1-win64-mingw"
set "curl_ver=curl-8.13.0_1-win64-mingw.zip"
set "curl_http=https://curl.se/windows/dl-8.13.0_1/"

set "curl_zip=%opt_install_dir%\%curl_ver%"

:: Download curl
IF /I EXIST %curl_zip% (
    if %expunge% EQU 1 (
        del /S /Q %curl_zip%
        if !errorlevel! neq 0 exit /b !errorlevel!
        %wget_path% -P %opt_install_dir%\ %curl_http%%curl_ver%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] file exists %curl_zip% )
    
) ELSE (
    %wget_path% -P %opt_install_dir%\ %curl_http%%curl_ver%
    if !errorlevel! neq 0 exit /b !errorlevel!
)
:: Extract curl
IF /I EXIST %opt_install_dir%\%curl_dir% (
     if %expunge% EQU 1 (
        rmdir /S /Q %opt_install_dir%\%curl_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
        C:\Windows\System32\tar.exe -xf "%curl_zip%" -C %opt_install_dir%
        if !errorlevel! neq 0 exit /b !errorlevel!
    ) else ( echo [INFO] directory exists %opt_install_dir%\%curl_dir% )
    
) ELSE (
    C:\Windows\System32\tar.exe -xf "%curl_zip%" -C %opt_install_dir%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::::::::: OpenCV
:install_opencv
set "opencv_git=https://github.com/opencv/opencv.git"
set "opencv_contrib=https://github.com/opencv/opencv_contrib.git"
set "opencv_ver=4.10.0"
set "opencv_dir=%opt_install_dir%\opencv_git"
set "opencv_contrib_dir=%opt_install_dir%\opencv_contrib_git"
set "opencv_install=%opt_install_dir%\opencv"
set "opencv_flags=-D BUILD_LIST=core,improc,imgcodecs,calib3d,features2d,highgui,imgproc,video,videoio,optflow -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_opencv_ts=OFF -D BUILD_opencv_aruco=OFF -D BUILD_opencv_bgsegm=OFF -D BUILD_opencv_bioinspired=OFF -D BUILD_opencv_ccalib=OFF -D BUILD_opencv_datasets=OFF -D BUILD_opencv_dnn=OFF -D BUILD_opencv_dnn_objdetect=OFF -D BUILD_opencv_dpm=OFF -D BUILD_opencv_face=OFF -D BUILD_opencv_fuzzy=OFF -D BUILD_opencv_hfs=OFF -D BUILD_opencv_img_hash=OFF -D BUILD_opencv_js=OFF -D BUILD_opencv_line_descriptor=OFF -D BUILD_opencv_phase_unwrapping=OFF -D BUILD_opencv_plot=OFF -D BUILD_opencv_quality=OFF -D BUILD_opencv_reg=OFF -D BUILD_opencv_rgbd=OFF -D BUILD_opencv_saliency=OFF -D BUILD_opencv_shape=OFF -D BUILD_opencv_structured_light=OFF -D BUILD_opencv_surface_matching=OFF -D BUILD_opencv_world=ON -D BUILD_opencv_xobjdetect=OFF -D BUILD_opencv_xphoto=OFF -D CV_ENABLE_INTRINSICS=ON -D WITH_EIGEN=ON -D WITH_PTHREADS=ON -D WITH_PTHREADS_PF=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_TIFF=OFF -D WITH_OPENEXR=OFF"

echo [INFO] Installing OpenCV %opencv_ver% ...
IF /I EXIST %opencv_install% (
    if %expunge% EQU 1 (rmdir /S /Q %opencv_install%) else (
        echo "[INFO] OpenCV installed in: "%opencv_install%
        goto :exit_dependencies
    )
)

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
echo [INFO] OpenCV %opencv_ver% installed: %opencv_install%
:exit_dependencies
echo [INFO] Dependencies installed 
exit /b 0
:exit_dependencies_error
echo [ERROR] Some dependencies not installed
exit /b 1
endlocal
