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
if not defined fullLog set "fullLog=win_full_test.log"
if not defined summaryLog set "summaryLog=win_test_summary.log"

if not exist "!fullLog!" (
    echo [ERROR] Full test log not found: !fullLog!
    exit /b 1
)
if not exist "!summaryLog!" (
    echo [WARN] Summary log not found: !summaryLog!
)

set "parserOutputTmp=!summaryLog!.parse.tmp"
set "summaryBackupTmp=!summaryLog!.orig.tmp"

set "CRASH_PATTERN=segmentation fault\|segfault\|abnormal termination\|access violation\|sigsegv\|seh exception\|0xc0000005\|unknown file: error:"
set "FAILED_TEST_PATTERN=^\[  FAILED  \].*( [0-9][0-9]* ms)$"

:: Check for per-test FAILED markers first - do not allow PASSED text to mask test failures
:: Match only timed gtest FAILED lines to avoid duplicates from the final "listed below" section.
grep -a -q "%FAILED_TEST_PATTERN%" "%fullLog%"
if !errorlevel! equ 0 goto :exit_build_error

:: Also check for segmentation faults or crashes
grep -a -q -i "%CRASH_PATTERN%" "%fullLog%"
if !errorlevel! equ 0 goto :exit_build_error

:: Consider the run successful only if the global PASSED summary is present anywhere in the log
:: (it can appear hundreds of lines before EOF due to SKIPPED and debug trace lines following it)
grep -a -q "\[  PASSED  \] [0-9]" "%fullLog%"
if !errorlevel! equ 0 exit /b 0

:: If we reach here, tests did not complete correctly (no FAILED/crash marker but no final PASSED summary)
goto :exit_build_error

:exit_build_error
if exist "%parserOutputTmp%" del /f /q "%parserOutputTmp%"
if exist "%summaryBackupTmp%" del /f /q "%summaryBackupTmp%"

set "segfaultDetected=0"
grep -a -q -i "%CRASH_PATTERN%" "!fullLog!"
if !errorlevel! equ 0 set "segfaultDetected=1"

(
echo.
echo [ERROR] FAILED TESTS OR CRASHES DETECTED:
echo.
echo === Failed Tests ^(from summary/full log^) ===
grep -a "!FAILED_TEST_PATTERN!" "!fullLog!"
echo.
echo === Last Successful Test ===
grep -a " OK ]" "!fullLog!" | tail -1
echo.
if "!segfaultDetected!"=="1" (
    echo === Last Running Test ^(likely the one that failed^) ===
    set "lastRunEntry="
    for /F "delims=" %%A in ('grep -a "\[ RUN" "!fullLog!" ^| tail -1') do (
        set "lastRunEntry=%%A"
    )
    if defined lastRunEntry (
        echo !lastRunEntry!
    ) else (
        echo [WARN] No gtest RUN marker found in !fullLog!.
    )
    echo.
    echo === Output from Last Running Test to End of Log ===
    set "lastRunLine="
    for /F "tokens=1 delims=:" %%A in ('grep -a -n "\[ RUN" "!fullLog!" ^| tail -1') do (
        set "lastRunLine=%%A"
    )
    if defined lastRunLine (
        sed -n "!lastRunLine!,$p" "!fullLog!" | head -120
    ) else (
        echo [WARN] Could not determine last RUN line. Showing recent RUN markers and log tail.
        grep -a -n "\[ RUN" "!fullLog!" | tail -3
        echo.
        tail -20 "!fullLog!"
    )
    echo.
)
echo === Context Around First FAILED Test ===
set "firstFailedLine="
set "firstFailedRunLine="
for /F "tokens=1 delims=:" %%A in ('grep -a -n "!FAILED_TEST_PATTERN!" "!fullLog!" ^| head -1') do (
    set "firstFailedLine=%%A"
)
if defined firstFailedLine (
    for /F "tokens=1 delims=:" %%B in ('sed -n "1,!firstFailedLine!p" "!fullLog!" ^| grep -a -n "\[ RUN" ^| tail -1') do (
        set "firstFailedRunLine=%%B"
    )
    if defined firstFailedRunLine (
        sed -n "!firstFailedRunLine!,$p" "!fullLog!" | head -160
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
set "firstSehRunEntry="
for /F "tokens=1 delims=:" %%A in ('grep -a -n -i "unknown file: error: SEH exception\|0xc0000005\|access violation\|SEH exception" "!fullLog!" ^| head -1') do (
    set "firstSehLine=%%A"
)
if defined firstSehLine (
    for /F "tokens=1 delims=:" %%B in ('sed -n "1,!firstSehLine!p" "!fullLog!" ^| grep -a -n "\[ RUN" ^| tail -1') do (
        set "firstSehRunLine=%%B"
    )
    for /F "delims=" %%C in ('sed -n "1,!firstSehLine!p" "!fullLog!" ^| grep -a "\[ RUN" ^| tail -1') do (
        set "firstSehRunEntry=%%C"
    )
    if defined firstSehRunEntry (
        echo [INFO] Unit test at first SEH marker: !firstSehRunEntry!
    ) else (
        echo [WARN] Could not determine unit test name at first SEH marker.
    )
    if defined firstSehRunLine (
        sed -n "!firstSehRunLine!,$p" "!fullLog!" | head -160
    ) else (
        echo [WARN] Could not determine RUN line for SEH exception entry.
    )
) else (
    echo [INFO] No SEH/Access Violation entry found.
)
echo.
echo === Segfault/Crash Messages ^(if any^) ===
grep -a -i "%CRASH_PATTERN%\|stack trace" "!fullLog!" || echo ^(none found^)
echo.
echo [ERROR] Check tests summary in '!summaryLog!' and tests logs in '!fullLog!'. Rerun failed test with: windows_setupvars.bat and %cd%\bazel-bin\src\ovms_test.exe --gtest_filter='*.*'
) > "!parserOutputTmp!" 2>&1

if exist "!summaryLog!" (
    copy /Y "!summaryLog!" "!summaryBackupTmp!" > nul
) else (
    type nul > "!summaryBackupTmp!"
)

(
    type "%parserOutputTmp%"
    echo.
    type "%summaryBackupTmp%"
) > "%summaryLog%"

if exist "!parserOutputTmp!" del /f /q "!parserOutputTmp!"
if exist "!summaryBackupTmp!" del /f /q "!summaryBackupTmp!"

exit /b 1
