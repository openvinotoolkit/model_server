::
:: Copyright (c) 2026 Intel Corporation
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
@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "fullLog=%~1"
set "summaryLog=%~2"
if "%fullLog%"=="" set "fullLog=win_full_test.log"
if "%summaryLog%"=="" set "summaryLog=win_test_summary.log"

if not exist "%fullLog%" (
    echo [ERROR] Full test log not found: %fullLog%
    exit /b 1
)
if not exist "%summaryLog%" (
    echo [WARN] Summary log not found: %summaryLog%
)

:: Check for FAILED markers first - do not allow PASSED text to mask test failures
grep -a -q "\[  FAILED  \]\| FAILED " "%fullLog%"
if !errorlevel! equ 0 goto :exit_build_error

:: Also check for segmentation faults or crashes
grep -a -q -i "segmentation fault\|segfault\|crashed\|abnormal termination\|access violation\|exception\|sigsegv" "%fullLog%"
if !errorlevel! equ 0 goto :exit_build_error

:: Consider the run successful only if PASSED summary is present near the end of the log
tail -50 "%fullLog%" | grep -a -q "\[  PASSED  \]"
if !errorlevel! equ 0 exit /b 0

:: If we reach here, tests did not complete correctly (no FAILED/crash marker but no final PASSED summary)
goto :exit_build_error

:exit_build_error
echo.
echo [ERROR] FAILED TESTS OR CRASHES DETECTED:
echo.
echo === Failed Tests (from summary/full log) ===
grep -a "^\[  FAILED  \]" "%summaryLog%"
grep -a "^\[  FAILED  \]" "%fullLog%"
echo.
echo === Last Successful Test ===
grep -a " OK ]" "%fullLog%" | tail -1
echo.
echo === Last Running Test (likely the one that failed) ===
set "lastRunEntry="
for /F "delims=" %%A in ('grep -a "\[ RUN" "%fullLog%" ^| tail -1') do (
    set "lastRunEntry=%%A"
)
if defined lastRunEntry (
    echo !lastRunEntry!
) else (
    echo [WARN] No gtest RUN marker found in %fullLog%.
)
echo.
echo === Output from Last Running Test to End of Log ===
set "lastRunLine="
for /F "tokens=1 delims=:" %%A in ('grep -a -n "\[ RUN" "%fullLog%" ^| tail -1') do (
    set "lastRunLine=%%A"
)
echo !lastRunLine! | findstr /R "^[0-9][0-9]*$" > nul
if !errorlevel! equ 0 (
    sed -n "!lastRunLine!,$p" "%fullLog%" | head -200
) else (
    echo [WARN] Could not determine last RUN line. Showing recent RUN markers and log tail.
    grep -a -n "\[ RUN" "%fullLog%" | tail -20
    echo.
    tail -200 "%fullLog%"
)
echo.
echo === Context Around First FAILED Test ===
set "firstFailedLine="
set "firstFailedRunLine="
for /F "tokens=1 delims=:" %%A in ('grep -a -n "^\[  FAILED  \].*(" "%fullLog%" ^| head -1') do (
    set "firstFailedLine=%%A"
)
echo !firstFailedLine! | findstr /R "^[0-9][0-9]*$" > nul
if !errorlevel! equ 0 (
    for /F "tokens=1 delims=:" %%B in ('sed -n "1,!firstFailedLine!p" "%fullLog%" ^| grep -a -n "\[ RUN" ^| tail -1') do (
        set "firstFailedRunLine=%%B"
    )
    echo !firstFailedRunLine! | findstr /R "^[0-9][0-9]*$" > nul
    if !errorlevel! equ 0 (
        sed -n "!firstFailedRunLine!,$p" "%fullLog%" | head -160
    ) else (
        echo [WARN] Could not determine RUN line for first FAILED test.
    )
) else (
    echo [INFO] No per-test FAILED entry with timing found.
)
echo.
echo === SEH/Access Violation Context ===
set "firstSehLine="
set "firstSehRunLine="
for /F "tokens=1 delims=:" %%A in ('grep -a -n -i "unknown file: error: SEH exception\|0xc0000005\|access violation\|SEH exception" "%fullLog%" ^| head -1') do (
    set "firstSehLine=%%A"
)
echo !firstSehLine! | findstr /R "^[0-9][0-9]*$" > nul
if !errorlevel! equ 0 (
    for /F "tokens=1 delims=:" %%B in ('sed -n "1,!firstSehLine!p" "%fullLog%" ^| grep -a -n "\[ RUN" ^| tail -1') do (
        set "firstSehRunLine=%%B"
    )
    echo !firstSehRunLine! | findstr /R "^[0-9][0-9]*$" > nul
    if !errorlevel! equ 0 (
        sed -n "!firstSehRunLine!,$p" "%fullLog%" | head -160
    ) else (
        echo [WARN] Could not determine RUN line for SEH exception entry.
    )
) else (
    echo [INFO] No SEH/Access Violation entry found.
)
echo.
echo === Segfault/Crash Messages (if any) ===
grep -a -i "segmentation fault\|segfault\|crashed\|abnormal termination\|access violation\|exception\|stack trace\|sigsegv" "%fullLog%" || echo (none found)
echo.
echo [ERROR] Check tests summary in '%summaryLog%' and tests logs in '%fullLog%'. Rerun failed test with: windows_setupvars.bat and %cd%\bazel-bin\src\ovms_test.exe --gtest_filter='*.*'
exit /b 1
