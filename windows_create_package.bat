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

set "ovms_exe_src="
set "ovms_runtime_shared_src="
set "libovmspython_src="
set "libpython_calculators_src="

IF "%~2"=="--with_python" (
    echo Self contained Python will be included in the package
    set "with_python=true"
) ELSE (
    echo Self contained Python will not be included in the package
    set "with_python=false"
)

:: Resolve the expected Bazel outputs before copying. This allows packaging to fail
:: with a specific, actionable error when the server or Python ABI artifacts were
:: not built yet, instead of a generic "The system cannot find the file specified."
if exist %cd%\bazel-bin\src\ovms.exe (
    set "ovms_exe_src=%cd%\bazel-bin\src\ovms.exe"
) else if exist %cd%\bazel-out\x64_windows-opt\bin\src\ovms.exe (
    set "ovms_exe_src=%cd%\bazel-out\x64_windows-opt\bin\src\ovms.exe"
)
if not defined ovms_exe_src (
    echo Packaging validation failed: ovms.exe is missing from the Bazel outputs. Build the server first, e.g. with: bazel build //src:ovms
    exit /b 1
)

:: Prefer the newest runtime-shared artifact from the staged Python ABI addon when
:: a Python build was performed. This avoids packaging a stale default-output DLL
:: left behind by a previous build or a separate output_user_root.
if /i "%with_python%"=="true" (
    if exist "%cd%\dist\windows\python_abi_addons\cp312\ovms_mediapipe_runtime_shared-cp312.dll" (
        set "ovms_runtime_shared_src=%cd%\dist\windows\python_abi_addons\cp312\ovms_mediapipe_runtime_shared-cp312.dll"
    )
)
if not defined ovms_runtime_shared_src if exist %cd%\bazel-bin\src\ovms_mediapipe_runtime_shared.dll (
    set "ovms_runtime_shared_src=%cd%\bazel-bin\src\ovms_mediapipe_runtime_shared.dll"
) else if exist %cd%\bazel-out\x64_windows-opt\bin\src\ovms_mediapipe_runtime_shared.dll (
    set "ovms_runtime_shared_src=%cd%\bazel-out\x64_windows-opt\bin\src\ovms_mediapipe_runtime_shared.dll"
)
if not defined ovms_runtime_shared_src (
    echo Packaging validation failed: ovms_mediapipe_runtime_shared.dll is missing from the Bazel outputs. Build the runtime shared library first, e.g. with: bazel build //src:ovms_mediapipe_runtime_shared
    exit /b 1
)

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

:: Set default USE_OV_BINARY if not set
if "%OV_USE_BINARY%"=="" (
    set "OV_USE_BINARY=1"
)

if exist dist\windows\ovms (
    rmdir /s /q dist\windows\ovms
    if !errorlevel! neq 0 exit /b !errorlevel!
)

md dist\windows\ovms
copy "!ovms_exe_src!" dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy "!ovms_runtime_shared_src!" "dist\windows\ovms\ovms_mediapipe_runtime_shared.dll"
if !errorlevel! neq 0 exit /b !errorlevel!

copy C:\%output_user_root%\openvino\runtime\bin\intel64\Release\*.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

set "dest_dir=C:\opt"

:: NOTE: this whole section intentionally avoids wrapping everything in a
:: single outer parenthesized IF (...) block, and avoids "::" comments and
:: "else if" chains inside nested blocks. cmd.exe's block pre-parser can
:: fail with errors like "'e.g.' was unexpected at this time." when deeply
:: nested parenthesized blocks contain "::" style comments or long else-if
:: chains - the error only surfaces once execution actually reaches the
:: block, not at initial parse time. Using goto plus flag variables instead
:: of nested parens sidesteps this entirely (see windows_build.bat for the
:: same fix applied to its additional-Python-ABI section).
if /i not "%with_python%"=="true" goto :skip_python_packaging

if exist "%cd%\dist\windows\python_abi_addons\cp312\libovmspython-cp312.dll" (
    set "libovmspython_src=%cd%\dist\windows\python_abi_addons\cp312\libovmspython-cp312.dll"
) else if exist %cd%\bazel-bin\src\python\libovmspython.dll (
    set "libovmspython_src=%cd%\bazel-bin\src\python\libovmspython.dll"
) else if exist %cd%\bazel-out\x64_windows-opt\bin\src\python\libovmspython.dll (
    set "libovmspython_src=%cd%\bazel-out\x64_windows-opt\bin\src\python\libovmspython.dll"
)
if not defined libovmspython_src (
    echo Missing libovmspython.dll in bazel output. Ensure //src/python:libovmspython is built.
    exit /b 1
)

if exist "%cd%\dist\windows\python_abi_addons\cp312\libpython_calculators-cp312.dll" (
    set "libpython_calculators_src=%cd%\dist\windows\python_abi_addons\cp312\libpython_calculators-cp312.dll"
) else if exist %cd%\bazel-bin\src\python\libpython_calculators.dll (
    set "libpython_calculators_src=%cd%\bazel-bin\src\python\libpython_calculators.dll"
) else if exist %cd%\bazel-out\x64_windows-opt\bin\src\python\libpython_calculators.dll (
    set "libpython_calculators_src=%cd%\bazel-out\x64_windows-opt\bin\src\python\libpython_calculators.dll"
)
if not defined libpython_calculators_src (
    echo Missing libpython_calculators.dll in bazel output. Ensure //src/python:libpython_calculators is built.
    exit /b 1
)

:: Derive the default ABI tag from the embedded Python version (e.g. 3.12.10 -> cp312)
set "python_version=3.12.10"
for /f "tokens=1,2 delims=." %%a in ("!python_version!") do (
    set "DEFAULT_MAJOR_VER=%%a"
    set "DEFAULT_MINOR_VER=%%b"
)
set "defaultAbiTag=cp!DEFAULT_MAJOR_VER!!DEFAULT_MINOR_VER!"

:: Copy pyovms module into root python/ (default import path) and into python\cp<tag>\ for
:: symmetry with the additional-ABI addon layout (e.g. python\cp313\pyovms.pyd).
md dist\windows\ovms\python
copy %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.pyd dist\windows\ovms\python
if !errorlevel! neq 0 exit /b !errorlevel!
if exist dist\windows\ovms\python\!defaultAbiTag! (
    rmdir /s /q dist\windows\ovms\python\!defaultAbiTag!
    if !errorlevel! neq 0 exit /b !errorlevel!
)
md dist\windows\ovms\python\!defaultAbiTag!
if !errorlevel! neq 0 exit /b !errorlevel!
copy %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.pyd "dist\windows\ovms\python\!defaultAbiTag!\pyovms.pyd"
if !errorlevel! neq 0 exit /b !errorlevel!

:: Copy shared OVMS Python runtime libraries.
:: Unversioned names (e.g. libovmspython.dll) are the loader fallback used when ABI detection
:: returns empty (standard packaged deployment via setupvars.bat).
:: Versioned names (e.g. libovmspython-cp312.dll) are symmetric with the additional-ABI addons
:: and are used as the primary candidate when PYTHONHOME encodes the version.
copy "!libovmspython_src!" "dist\windows\ovms\libovmspython.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!libovmspython_src!" "dist\windows\ovms\libovmspython-!defaultAbiTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!libpython_calculators_src!" "dist\windows\ovms\libpython_calculators.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!libpython_calculators_src!" "dist\windows\ovms\libpython_calculators-!defaultAbiTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!

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
if not exist dist\windows\ovms\python\python312.zip (
    echo Packaging validation failed: embedded stdlib python312.zip is missing from dist\windows\ovms\python.
    exit /b 1
)
.\dist\windows\ovms\python\python.exe -m pip install "setuptools==80.9.0" "Jinja2==3.1.6" "MarkupSafe==3.0.2"
if !errorlevel! neq 0 (
    echo Error during Python dependencies for LLM installation. The package will not be fully functional.
)

:: Package any additional Python ABI runtime libraries staged by
:: windows_build.bat under dist\windows\python_abi_addons\cp<tag>\, for
:: example a cp313 build produced alongside the default cp312 build above.
set "abiAddonsDir=%cd%\dist\windows\python_abi_addons"
if not exist "!abiAddonsDir!" goto :skip_python_packaging

if /i "%with_python%"=="true" (
    if not exist "!abiAddonsDir!\cp*" (
        echo Packaging validation failed: no staged Python ABI directories were found under !abiAddonsDir!.
        echo Build the default runtime and any additional ABI runtimes first, e.g. with: windows_build.bat opt --with_python 3.13.1
        exit /b 1
    )
)

for /d %%V in ("!abiAddonsDir!\cp*") do (
    set "abiTagDir=%%~nxV"
    set "abiTag=!abiTagDir:cp=!"
    echo Packaging additional Python ABI: !abiTag!

    set "abiFilesOk=1"
    if not exist "%%V\libovmspython-!abiTagDir!.dll" set "abiFilesOk=0"
    if not exist "%%V\libpython_calculators-!abiTagDir!.dll" set "abiFilesOk=0"
    if not exist "%%V\ovms_mediapipe_runtime_shared-!abiTagDir!.dll" set "abiFilesOk=0"
    if not exist "%%V\pyovms.pyd" set "abiFilesOk=0"

    if "!abiFilesOk!"=="0" (
        echo Missing required staged files for ABI !abiTag! in %%V. Skipping.
    ) else (
        copy "%%V\libovmspython-!abiTagDir!.dll" dist\windows\ovms
        if !errorlevel! neq 0 exit /b !errorlevel!
        copy "%%V\libpython_calculators-!abiTagDir!.dll" dist\windows\ovms
        if !errorlevel! neq 0 exit /b !errorlevel!
        copy "%%V\ovms_mediapipe_runtime_shared-!abiTagDir!.dll" dist\windows\ovms
        if !errorlevel! neq 0 exit /b !errorlevel!

        if exist dist\windows\ovms\python\!abiTagDir! (
            rmdir /s /q dist\windows\ovms\python\!abiTagDir!
            if !errorlevel! neq 0 exit /b !errorlevel!
        )
        md dist\windows\ovms\python\!abiTagDir!
        if !errorlevel! neq 0 exit /b !errorlevel!
        copy "%%V\pyovms.pyd" dist\windows\ovms\python\!abiTagDir!\pyovms.pyd
        if !errorlevel! neq 0 exit /b !errorlevel!
    )
)

:skip_python_packaging

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

if /i "%with_python%"=="true" (
    if not exist dist\windows\ovms\libovmspython.dll (
        echo Packaging validation failed: libovmspython.dll is missing from dist\windows\ovms.
        exit /b 1
    )
    if not exist dist\windows\ovms\libpython_calculators.dll (
        echo Packaging validation failed: libpython_calculators.dll is missing from dist\windows\ovms.
        exit /b 1
    )
)

cd dist\windows
C:\Windows\System32\tar.exe -a -c -f ovms.zip ovms
if !errorlevel! neq 0 exit /b !errorlevel!
cd ..\..
dir dist\windows\ovms.zip
echo [INFO] Package created
