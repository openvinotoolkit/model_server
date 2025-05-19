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
    set "python_version=3.12.9"

    call %cd%\windows_prepare_python.bat %dest_dir% %python_version%
    if !errorlevel! neq 0 (
        echo Error occurred when creating Python environment for the distribution.
        exit /b !errorlevel!
    )
    :: Copy whole catalog to dist folder and install dependencies required by LLM pipelines
    xcopy %dest_dir%\python-%python_version%-embed-amd64 dist\windows\ovms\python /E /I /H
    if !errorlevel! neq 0 (
        echo Error occurred when creating Python environment for the distribution.
        exit /b !errorlevel!
    )
    .\dist\windows\ovms\python\python.exe -m pip install "Jinja2==3.1.6" "MarkupSafe==3.0.2"
    if !errorlevel! neq 0 (
        echo Error during Python dependencies for LLM installation. The package will not be fully functional.
    )
)

copy C:\%output_user_root%\openvino\runtime\3rdparty\tbb\bin\tbb12.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

:: Copy from bazel-out if the genai is from sources
copy %cd%\bazel-out\x64_windows-opt\bin\src\opencv_world4100.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\icudt70.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\icuuc70.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\openvino_genai.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\openvino_tokenizers.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\git2.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy /Y %dest_dir%\git-lfs.exe dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\bin\libcurl-x64.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
:: Old package had core_tokenizers
if exist %cd%\bazel-out\x64_windows-opt\bin\src\core_tokenizers.dll (
    copy /Y %cd%\bazel-out\x64_windows-opt\bin\src\core_tokenizers.dll dist\windows\ovms
    if !errorlevel! neq 0 exit /b !errorlevel!
)

copy %cd%\setupvars.* dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

:: Adding licenses
set "license_dest=%cd%\dist\windows\ovms\thirdparty-licenses\"
md %license_dest%
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\opencv\etc\licenses\* %license_dest%
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\%output_user_root%\openvino\docs\licensing\LICENSE %license_dest%openvino.LICENSE.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\%output_user_root%\openvino\docs\licensing\LICENSE-GENAI %license_dest%LICENSE-GENAI.txt
if !errorlevel! neq 0 exit /b !errorlevel!

copy %cd%\release_files\LICENSE %cd%\dist\windows\ovms\
if !errorlevel! neq 0 exit /b !errorlevel!
copy %cd%\release_files\thirdparty-licenses\* %license_dest%
if !errorlevel! neq 0 exit /b !errorlevel!

if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\COPYING.txt %license_dest%LICENSE-CURL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\brotli\LICENSE.txt %license_dest%LICENSE-BROTIL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\cacert\LICENSE.url %license_dest%LICENSE-CACERT.url
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\libpsl\COPYING.txt %license_dest%LICENSE-LIBPSL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\libressl\COPYING.txt %license_dest%LICENSE-LIBRESSL.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\libssh2\COPYING.txt %license_dest%LICENSE-LIBSSH2.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\nghttp2\COPYING.txt %license_dest%LICENSE-NGHTTP2.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\nghttp3\COPYING.txt %license_dest%LICENSE-NGHTTP3.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\zlib\LICENSE.txt %license_dest%LICENSE-ZLIB.txt
if !errorlevel! neq 0 exit /b !errorlevel!
copy C:\opt\curl-8.13.0_1-win64-mingw\dep\zstd\LICENSE.txt %license_dest%LICENSE-ZSTD.txt

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
