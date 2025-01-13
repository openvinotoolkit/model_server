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
setlocal EnableExtensions EnableDelayedExpansion
IF "%~1"=="" (
    echo No argument provided. Using default opt path
    set "output_user_root=opt"
) ELSE (
    echo Argument provided: Using install path %1
    set "output_user_root=%1"
)

echo Sth1
if exist dist\windows\ovms (
    rmdir /s /q dist\windows\ovms
    if !errorlevel! neq 0 exit /b !errorlevel!
)

md dist\windows\ovms
copy bazel-bin\src\ovms.exe dist\windows\ovms
echo Sth2
if !errorlevel! neq 0 exit /b !errorlevel!

copy C:\%output_user_root%\openvino\runtime\bin\intel64\Release\*.dll dist\windows\ovms
echo Sth3
if !errorlevel! neq 0 exit /b !errorlevel!

:: Copy pyovms module
md dist\windows\ovms\python
echo Sth4
copy  %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.pyd dist\windows\ovms\python
if !errorlevel! neq 0 exit /b !errorlevel!
echo Sth5

:: Prepare self-contained python
set "dest_dir=C:\opt"
set "python_version=3.9.13"
echo Sth6
call %cd%\windows_prepare_python.bat %dest_dir% %python_version%
:: Copy whole catalog to dist folder and install dependencies required by LLM pipelines
xcopy %dest_dir%\python-%python_version%-embed-amd64 dist\windows\ovms\python /E /I /H
echo Sth7
.\dist\windows\ovms\python\python.exe -m pip install "Jinja2==3.1.4" "MarkupSafe==3.0.2"
echo Sth8
if !errorlevel! neq 0 (
    echo Error copying python into the distribution location. The package will not contain self-contained python.
)
echo Sth9

:: Below includes OpenVINO tokenizers
:: copy %cd%\bazel-bin\external\llm_engine\openvino_genai\runtime\bin\Release\*.dll dist\windows\ovms
:: TODO Better manage dependency declarationw with llm_engine & bazel
copy %cd%\bazel-out\x64_windows-opt\bin\external\llm_engine\copy_openvino_genai\openvino_genai\runtime\bin\Release\*.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
echo Sth10
copy C:\%output_user_root%\openvino\runtime\3rdparty\tbb\bin\tbb12.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

echo Sth11
copy  %cd%\bazel-out\x64_windows-opt\bin\src\opencv_world4100.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!
echo Sth12

dist\windows\ovms\ovms.exe --version
echo Sth13
if !errorlevel! neq 0 exit /b !errorlevel!
echo Sth14

dist\windows\ovms\ovms.exe --help
if !errorlevel! neq 0 exit /b !errorlevel!
echo Sth15

cd dist\windows
tar -a -c -f ovms.zip ovms
if !errorlevel! neq 0 exit /b !errorlevel!
echo Sth16
cd ..\..
echo Sth17
dir dist\windows\ovms.zip
