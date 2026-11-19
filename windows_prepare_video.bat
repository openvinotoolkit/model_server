::
:: Copyright 2026 Intel Corporation
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
::     http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.
::

:: Windows counterpart of prepare_mpi_video.sh.
:: Downloads (and SHA-256 verifies) a test video used by the GStreamer/D3D11
:: video tests. Arguments (all optional): %1 destination path, %2 source URL,
:: %3 expected SHA-256. Defaults target the MPI test video.

@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: Defaults target the MPI test video; override via arguments for other assets.
set "DEFAULT_VIDEO_URL=https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/CEO%%20Pat%%20Gelsinger%%20on%%20Leading%%20Intel.mp4"
set "DEFAULT_VIDEO_SHA256=8b50542436ead023522f7cb42cc4fe576b8dc752e695e034edaba44a60c244bc"
set "DEFAULT_VIDEO_PATH=models\mpi\ceo_pat_gelsinger.mp4"

IF "%~1"=="" (
    set "video_path=%DEFAULT_VIDEO_PATH%"
) ELSE (
    set "video_path=%~1"
)
IF "%~2"=="" (
    set "VIDEO_URL=%DEFAULT_VIDEO_URL%"
) ELSE (
    set "VIDEO_URL=%~2"
)
IF "%~3"=="" (
    set "VIDEO_SHA256=%DEFAULT_VIDEO_SHA256%"
) ELSE (
    set "VIDEO_SHA256=%~3"
)
for %%F in ("!video_path!") do set "video_directory=%%~dpF"

set "wget_path=C:\opt\wget.exe"

IF /I EXIST "!video_path!" (
    call :validate_video "!video_path!"
    if !errorlevel! equ 0 (
        echo [INFO] Verified cached video at !video_path!
        exit /b 0
    )
    echo [WARN] Cached video failed verification, re-downloading ...
    del /f /q "!video_path!"
)

IF /I NOT EXIST "!video_directory!" (
    mkdir "!video_directory!"
    if !errorlevel! neq 0 exit /b !errorlevel!
)

echo [INFO] Downloading video to !video_path! ...
IF /I EXIST "!wget_path!" (
    "!wget_path!" -O "!video_path!" "!VIDEO_URL!"
) ELSE (
    curl --fail --location --output "!video_path!" "!VIDEO_URL!"
)
if !errorlevel! neq 0 (
    echo [ERROR] Failed to download video from !VIDEO_URL!
    exit /b 1
)

call :validate_video "!video_path!"
if !errorlevel! neq 0 (
    echo [ERROR] Downloaded video failed SHA-256 verification
    del /f /q "!video_path!"
    exit /b 1
)
echo [INFO] Prepared video at !video_path!
exit /b 0

:validate_video
set "check_path=%~1"
set "actual_sha256="
for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile "!check_path!" SHA256') do (
    if not defined actual_sha256 set "actual_sha256=%%H"
)
:: certutil prints the hash with spaces on some locales; strip them
set "actual_sha256=!actual_sha256: =!"
if /I "!actual_sha256!"=="%VIDEO_SHA256%" (
    exit /b 0
) else (
    echo [ERROR] Unexpected SHA-256 for !check_path!: !actual_sha256!
    exit /b 1
)
