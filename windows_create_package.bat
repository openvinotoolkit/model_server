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

md dist\windows\ovms
copy bazel-bin\src\ovms.exe dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy  %cd%\bazel-out\x64_windows-opt\bin\src\python39.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy  %cd%\bazel-out\x64_windows-opt\bin\src\python\binding\pyovms.pyd dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy C:\%output_user_root%\openvino\runtime\bin\intel64\Release\*.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

:: Prepare self-contained python
call %cd%\windows_prepare_python.bat
:: Copy whole catalog to dist folder and install dependencies required by LLM pipelines
xcopy C:\opt\ovms-python-3.9.13-embed dist\windows\ovms\python /E /I /H
.\dist\windows\ovms\python\python.exe -m pip install "Jinja2==3.1.4" "MarkupSafe==3.0.2"
if !errorlevel! neq 0 (
    echo Error copying python into the distribution location. The package will not contain self-contained python.
)

:: Below includes OpenVINO tokenizers
copy %cd%\bazel-bin\external\llm_engine\openvino_genai\runtime\bin\Release\*.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy C:\%output_user_root%\openvino\runtime\3rdparty\tbb\bin\tbb12.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

copy  %cd%\bazel-out\x64_windows-opt\bin\src\opencv_world4100.dll dist\windows\ovms
if !errorlevel! neq 0 exit /b !errorlevel!

dist\windows\ovms\ovms.exe --version
if !errorlevel! neq 0 exit /b !errorlevel!

dist\windows\ovms\ovms.exe --help
if !errorlevel! neq 0 exit /b !errorlevel!

cd dist\windows
tar -a -c -f ovms.zip ovms
if !errorlevel! neq 0 exit /b !errorlevel!
cd ..\..
dir dist\windows\ovms.zip
