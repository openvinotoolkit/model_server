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
echo off
setlocal EnableExtensions EnableDelayedExpansion
set "setPath=C:\opt;C:\opt\msys64\usr\bin\;%PATH%;"
set "PATH=%setPath%"

:: Load chosen dependency versions from versions.mk
for /f "usebackq eol=# tokens=1,3" %%A in ("%cd%\versions.mk") do (
    if "%%A"=="OPENCV_VERSION" if "!opencv_version!"=="" set "opencv_version=%%B"
    if "%%A"=="CURL_VERSION" if "!curl_version!"=="" set "curl_version=%%B"
)
:: Build DLL suffix by removing dots (e.g. 4.13.0 -> 4130)
set "opencv_dll_ver=!opencv_version:.=!"
IF "%~1"=="" (
    echo No argument provided. Using default opt path
    set "output_user_root=opt"
) ELSE (
    echo Argument provided: Using install path %1
    set "output_user_root=%1"
)

IF "%~2"=="--with_python" (
    echo Self contained Python will be included in the package
    set "with_python=true"
) ELSE (
    echo Self contained Python will not be included in the package
    set "with_python=false"
)

:: Set default USE_OV_BINARY if not set
if "%OV_USE_BINARY%"=="" (
    set "OV_USE_BINARY=1"
)

if exist dist\windows\ovms (
    rmdir /s /q dist\windows\ovms
    if !errorlevel! neq 0 exit /b !errorlevel!
)

md dist\windows\ovms
copy bazel-bin\src\ovms.exe dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy C:\%output_user_root%\openvino\runtime\bin\intel64\Release\*.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

set "dest_dir=C:\opt"

if /i "%with_python%"=="true" (
    :: Copy pyovms module
    md dist\windows\ovms\python
    copy %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.pyd dist\windows\ovms\python
    if !errorlevel! neq 0 exit /b !errorlevel!

    :: Prepare self-contained python
    set "python_version=3.12.10"

    call %cd%\windows_prepare_python.bat %dest_dir% !python_version!
    if !errorlevel! neq 0 (
        echo Error occurred when creating Python environment for the distribution.
        exit /b !errorlevel!
    )
    :: Copy whole catalog to dist folder and install dependencies required by LLM pipelines
    xcopy %dest_dir%\python-!python_version!-embed-amd64 dist\windows\ovms\python /E /I /H
    if !errorlevel! neq 0 (
        echo Error occurred when creating Python environment for the distribution.
        exit /b !errorlevel!
    )
    .\dist\windows\ovms\python\python.exe -m pip install "setuptools==80.9.0" "Jinja2==3.1.6" "MarkupSafe==3.0.2"
    if !errorlevel! neq 0 (
        echo Error during Python dependencies for LLM installation. The package will not be fully functional.
    )
)

copy C:\%output_user_root%\openvino\runtime\3rdparty\tbb\bin\tbb12.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

:: Copy from bazel-out if the genai is from sources
copy %cd%\bazel-out\x64_windows-opt\bin\src\opencv_world!opencv_dll_ver!.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\openvino_genai.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\openvino_tokenizers.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\libcurl-x64.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\git2.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
:: Old package had core_tokenizers
if exist %cd%\bazel-out\x64_windows-opt\bin\src\core_tokenizers.dll (
    copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\core_tokenizers.dll dist\windows\ovms
    if !errorlevel! neq 0 exit /b !errorlevel!
)

:: Bundle espeak-ng DLL + data when it was built from source by Bazel
:: (--//:espeak=on). Picked up from the rules_foreign_cc cmake output tree.
for /f "delims=" %%D in ('dir /b /s /a:-d "%cd%\bazel-out\x64_windows-opt\bin\external\espeak_ng\espeak-ng.dll" 2^>nul') do (
    copy /Y "%%D" dist\windows\ovms
    if !errorlevel! neq 0 exit /b !errorlevel!
)
for /f "delims=" %%D in ('dir /b /s /a:d "%cd%\bazel-out\x64_windows-opt\bin\external\espeak_ng" 2^>nul ^| findstr /e "espeak-ng-data"') do (
    xcopy "%%D" dist\windows\ovms\espeak-ng-data /E /I /H /Y
    if !errorlevel! neq 0 exit /b !errorlevel!
)

copy %cd%\setupvars.* dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy %cd%\install_ovms_service.bat dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

:: Adding licenses
set "license_dest=%cd%\dist\windows\ovms\thirdparty-licenses\"
md %license_dest%
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\opencv_!opencv_version!\etc\licenses\* %license_dest%
if !errorlevel! neq 0 exit /b !errorlevel!
IF "%OV_USE_BINARY%"=="1" (
    copy C:\%output_user_root%\openvino\docs\licensing\LICENSE %license_dest%openvino.LICENSE.txt
    if !errorlevel! neq 0 exit /b !errorlevel!
    copy C:\%output_user_root%\openvino\docs\licensing\LICENSE-GENAI %license_dest%LICENSE-GENAI.txt
    if !errorlevel! neq 0 exit /b !errorlevel!
) ELSE (
    copy C:\%output_user_root%\openvino\licenses %license_dest%
    if !errorlevel! neq 0 exit /b !errorlevel!
)

copy %cd%\release_files\LICENSE %cd%\dist\windows\ovms\
if !errorlevel! neq 0 exit /b !errorlevel!
copy %cd%\release_files\thirdparty-licenses\* %license_dest%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Bundle eSpeak-ng license text when eSpeak artifacts are included.
set "espeak_license_src="
for /f "delims=" %%F in ('dir /b /s /a:-d "%cd%\bazel-out\x64_windows-opt\bin\external\espeak_ng\COPYING*" 2^>nul') do (
    set "espeak_license_src=%%F"
    goto :copy_espeak_license
)
for /f "delims=" %%F in ('dir /b /s /a:-d "%cd%\bazel-out\x64_windows-opt\bin\external\espeak_ng\LICENSE*" 2^>nul') do (
    set "espeak_license_src=%%F"
    goto :copy_espeak_license
)
:copy_espeak_license
if defined espeak_license_src (
    copy /Y "!espeak_license_src!" "%license_dest%espeak-ng.LICENSE.txt"
    if !errorlevel! neq 0 exit /b !errorlevel!
)

set "curl_dir=curl-!curl_version!-win64-mingw"
echo Adding curl licenses from !curl_dir!...
copy C:\opt\!curl_dir!\COPYING.txt %license_dest%LICENSE-CURL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\brotli\LICENSE.txt %license_dest%LICENSE-BROTLI.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\certdata\LICENSE.url %license_dest%LICENSE-CERTDATA.url
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\libpsl\COPYING.txt %license_dest%LICENSE-LIBPSL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\libressl\COPYING.txt %license_dest%LICENSE-LIBRESSL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\libssh2\COPYING.txt %license_dest%LICENSE-LIBSSH2.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\nghttp2\COPYING.txt %license_dest%LICENSE-NGHTTP2.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\nghttp3\COPYING.txt %license_dest%LICENSE-NGHTTP3.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\ngtcp2\COPYING.txt %license_dest%LICENSE-NGTCP2.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\zlibng\LICENSE.md %license_dest%LICENSE-ZLIBNG.md
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\!curl_dir!\dep\zstd\LICENSE.txt %license_dest%LICENSE-ZSTD.txt

:: Add when CAPI enabled and tested
::mkdir -vp /ovms_release/include && cp /ovms/src/ovms.h /ovms_release/include

:: Testing package
call dist\windows\ovms\setupvars.bat
if !errorlevel! neq 0 exit /b !errorlevel!

dist\windows\ovms\ovms.exe --version
if !errorlevel! neq 0 exit /b !errorlevel!

dist\windows\ovms\ovms.exe --help
if !errorlevel! neq 0 exit /b !errorlevel!

cd dist\windows
C:\Windows\System32\tar.exe -a -c -f ovms.zip ovms
if !errorlevel! neq 0 exit /b !errorlevel!
cd ..\..
dir dist\windows\ovms.zip
echo [INFO] Package created
