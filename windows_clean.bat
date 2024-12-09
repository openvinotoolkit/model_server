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
set "setPath=%PATH%;c:\opt"
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