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

:: %1 First parameter is the --output_user_root value on c:\ drive
:: %2 Second parameter is the --expunge flag - bazel clean expunge when set to 1 - default 0
@echo on
setlocal EnableExtensions DisableDelayedExpansion
set "BAZEL_SHORT_PATH=C:\%1"
set "bazelStartupCmd=--output_user_root=%BAZEL_SHORT_PATH%"
set "setPath=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\HostX86\x86;c:\opt;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\VC\VCPackages;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TestWindow;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\MSBuild\Current\bin\Roslyn;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Team Tools\Performance Tools;C:\Program Files (x86)\Microsoft Visual Studio\Shared\Common\VSPerfCollectionTools\vs2019\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\devinit;C:\Program Files (x86)\Windows Kits\10\bin\10.0.19041.0\x86;C:\Program Files (x86)\Windows Kits\10\bin\x86;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\\MSBuild\Current\Bin;C:\Windows\Microsoft.NET\Framework\v4.0.30319;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\Tools\;C:\Program Files\Common Files\Oracle\Java\javapath;C:\Windows\system32;C:\Windows;C:\Windows\System32\Wbem;C:\Windows\System32\WindowsPowerShell\v1.0\;C:\Utils\;C:\Program Files\Git\cmd;C:\Program Files\Git\mingw64\bin;C:\Program Files\Git\usr\bin;C:\Ninja;C:\Program Files\CMake\bin;C:\Program Files\7-zip;C:\opt\Python39\Scripts\;C:\opt\Python39\;C:\opencl\install\;C:\opencl\;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if %2 EQU 1 (set "cleanCmd=bazel %bazelStartupCmd% clean --expunge") else ( set "cleanCmd=bazel %bazelStartupCmd% clean" )


:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set "PATH=%setPath%"

:: Log all environment variables
%cleanCmd%
rm -rf win_environment.log
rm -rf win_build.log
rm -rf win_build_test.log
rm -rf win_full_test.log
rm -rf win_test.log
endlocal