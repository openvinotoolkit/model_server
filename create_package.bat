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

md dist\windows
copy bazel-bin\src\ovms.exe dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy  %cd%/bazel-out/x64_windows-opt/bin/src/python39.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy c:\opt\openvino\runtime\bin\intel64\Release\*.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy c:\opt\openvino\runtime\3rdparty\tbb\bin\tbb12.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
copy  %cd%\bazel-out\x64_windows-opt\bin\src\opencv_world4100.dll dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%
tar -czf dist\ovms.zip dist\windows
if %errorlevel% neq 0 exit /b %errorlevel%