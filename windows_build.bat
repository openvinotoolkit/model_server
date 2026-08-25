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
@echo off
setlocal EnableExtensions EnableDelayedExpansion
:: Need to set shorter build paths for bazel cache for too long commands in mediapipe compilation
:: We expect a first script argument to be "PR-1234" number passed here from jenkins so that a tmp directory will be created
IF "%~1"=="" (
    echo No argument provided. Using default opt path
    set "BAZEL_SHORT_PATH=C:\opt"
) ELSE (
    echo Argument provided: Using install path %1
    set "BAZEL_SHORT_PATH=C:\%1"
)

IF "%~2"=="--with_python" (
    echo Building model server with Python
    set "bazelBuildArgs=--config=win_mp_on_py_on"
    set "pythonRuntimeTargets=//src/python:libpython_calculators //src/python:libovmspython"
    set "additionalPythonAbi="

    :: Accept a Python ABI in the direct form: windows_build.bat opt --with_python 3.13.1
    :: and in the legacy form: windows_build.bat opt --with_python --with_tests 3.13.1
    if not "%~3"=="" if /I not "%~3"=="--with_tests" if /I not "%~3"=="--integrity" set "additionalPythonAbi=%~3"
    if not "%~4"=="" if /I not "%~4"=="--with_tests" if /I not "%~4"=="--integrity" if "!additionalPythonAbi!"=="" set "additionalPythonAbi=%~4"
    if not "%~5"=="" if /I not "%~5"=="--with_tests" if /I not "%~5"=="--integrity" if "!additionalPythonAbi!"=="" set "additionalPythonAbi=%~5"
) ELSE (
    echo Building model server without Python 
    set "bazelBuildArgs=--config=win_mp_on_py_off"
    set "pythonRuntimeTargets="
    set "additionalPythonAbi="
)

set "withTests=false"
if "%~3"=="--with_tests" set "withTests=true"
if "%~4"=="--with_tests" set "withTests=true"
if "%~5"=="--with_tests" set "withTests=true"

set "buildTargets=//src:ovms"
if "!withTests!"=="true" set "buildTargets=!buildTargets! //src:ovms_test"
set "buildTargets=!buildTargets! //src:ovms_mediapipe_runtime_shared //third_party:espeak_ng //third_party:espeak_ng_data !pythonRuntimeTargets!"
if "!withTests!"=="true" (
    echo Building model server with tests
) else (
    echo Building model server without tests
)

IF "%~4"=="--integrity" (
    echo Building model server with integrity checks
    set "buildWithIntegrity=--config=win_integritycheck"
) ELSE (
    echo Building model server without integrity checks
    set "buildWithIntegrity="
)

:: Optional Python ABI override: e.g. 3.13.1 or 3.13. Build the corresponding
:: libovmspython/libpython_calculators/pyovms + runtime shared library against a
:: dev Python install at C:\opt\Python<MAJOR><MINOR>.
set "additionalPythonAbi=!additionalPythonAbi!"

:: Keep the default Python-enabled build isolated from any later extra-ABI build.
:: This avoids reusing a cached local_config_python repository from a different
:: Python minor version when the same workspace has been built for cp313 before.
set "mainBazelStartupCmd=--output_user_root=!BAZEL_SHORT_PATH!"
if not "!pythonRuntimeTargets!"=="" set "mainBazelStartupCmd=--output_user_root=!BAZEL_SHORT_PATH!_py312"
set "openvino_dir=!BAZEL_SHORT_PATH!/openvino/runtime/cmake"

set "buildCommand=bazel !mainBazelStartupCmd! build  %buildWithIntegrity% %bazelBuildArgs% --action_env OpenVINO_DIR=%openvino_dir% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures %buildTargets% 2>&1 | tee win_build.log"
set "setOvmsVersionCmd=python windows_set_ovms_version.py"

:: Setting PATH environment variable based on default windows node settings: Added ovms_windows specific python settings and c:/opt and removed unused Nvidia and OCL specific tools.
:: When changing the values here you can print the node default PATH value and base your changes on it.
set "setPath=C:\opt;C:\opt\Python312\;C:\opt\Python312\Scripts\;C:\opt\msys64\usr\bin\;%PATH%;"
set "PYTHONHOME=C:\opt\Python312"
set "envPath=win_environment.log"
set "setPythonPath=%cd%\bazel-out\x64_windows-opt\bin\src\python\binding"
set "BAZEL_SH=C:\opt\msys64\usr\bin\bash.exe"

:: Remove stale MediaPipe runtime outputs before a build so a previously generated
:: default-output DLL cannot survive and win the DLL search order when the newer
:: Python ABI build is staged to the package directory.
for %%F in (
    "%cd%\bazel-bin\src\ovms_mediapipe_runtime_shared.dll"
    "%cd%\bazel-bin\src\ovms_mediapipe_runtime_shared-cp313.dll"
    "%cd%\bazel-out\x64_windows-opt\bin\src\ovms_mediapipe_runtime_shared.dll"
    "%cd%\dist\windows\ovms\ovms_mediapipe_runtime_shared.dll"
    "%cd%\dist\windows\ovms\ovms_mediapipe_runtime_shared-cp313.dll"
) do (
    if exist "%%~F" del /F /Q "%%~F"
)

:: Load chosen dependency versions from versions.mk
for /f "usebackq eol=# tokens=1,3" %%A in ("%cd%\versions.mk") do (
    if "%%A"=="OPENCV_VERSION" if "!opencv_version!"=="" set "opencv_version=%%B"
)

:: Bazel compilation settings
set VS_2022_BT="C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
IF /I EXIST %VS_2022_BT% goto :msvc_bt ELSE goto :msvc_error

:msvc_error
echo [ERROR] Required MSVC compiler not installed
goto :exit_build_error
:msvc_bt
echo [INFO] Using MSVC %VS_2022_BT%
set BAZEL_VS=%VS_2022_BT%

:: Bazel compilation settings end
:msvc_end
set "BAZEL_VC=%BAZEL_VS:"=%\VC"
set "BAZEL_VC_FULL_VERSION=14.44.35207"

:: Set proper PATH environment variable: Remove other python paths and add c:\opt with bazel to PATH
set "PATH=%setPath%"

:: Set paths with libs for execution - affects PATH
set "openvinoBatch=call !BAZEL_SHORT_PATH!\openvino\setupvars.bat"
set "opencvBatch=call C:\opt\opencv_!opencv_version!\setup_vars_opencv4.cmd"

:: Set required libraries paths
%openvinoBatch%
if !errorlevel! neq 0 exit /b !errorlevel!
%opencvBatch%
set "PYTHONPATH=%PYTHONPATH%;%setPythonPath%"

:: Log all environment variables
set > %envPath%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Set ovms.exe --version parameters
%setOvmsVersionCmd% "%bazelBuildArgs%" !BAZEL_SHORT_PATH!
:: Start bazel build. Keep the root OVMS target explicit so the direct ABI form
:: (`windows_build.bat opt --with_python 3.13.1`) still rebuilds ovms.exe in addition
:: to the Python runtime artifacts.
echo [INFO] Building OVMS targets: !buildTargets!
%buildCommand%
if !errorlevel! neq 0 exit /b !errorlevel!

IF "!pythonRuntimeTargets!"=="" goto :skip_default_python_abi_stage

:: Stage the default Python ABI outputs immediately so packaging can rely on the
:: copied artifacts instead of the mutable bazel-bin symlink, which may later
:: point at an extra-ABI build.
set "defaultPythonTag=312"
set "defaultStageDir=%cd%\dist\windows\python_abi_addons\cp!defaultPythonTag!"
if exist "!defaultStageDir!" rmdir /S /Q "!defaultStageDir!"
md "!defaultStageDir!"
if !errorlevel! neq 0 exit /b !errorlevel!

set "defaultOutputBase=!BAZEL_SHORT_PATH!_py!defaultPythonTag!"
set "defaultBinDir="
for /d %%d in ("!defaultOutputBase!\*") do (
    if exist "%%d\execroot\ovms\bazel-out\x64_windows-opt\bin\src\python\libovmspython.dll" (
        set "defaultBinDir=%%d\execroot\ovms\bazel-out\x64_windows-opt\bin"
    )
)
if "!defaultBinDir!"=="" (
    echo [ERROR] Could not locate default-ABI build artifacts under !defaultOutputBase!
    exit /b 1
)

set "defaultLibOvmspython=!defaultBinDir!\src\python\libovmspython.dll"
set "defaultLibPyCalc=!defaultBinDir!\src\python\libpython_calculators.dll"
set "defaultPyovms=!defaultBinDir!\src\python\binding\pyovms.pyd"
set "defaultMpRuntime=!defaultBinDir!\src\ovms_mediapipe_runtime_shared.dll"

copy "!defaultLibOvmspython!" "!defaultStageDir!\libovmspython-cp!defaultPythonTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!defaultLibPyCalc!" "!defaultStageDir!\libpython_calculators-cp!defaultPythonTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!defaultPyovms!" "!defaultStageDir!\pyovms.pyd"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!defaultMpRuntime!" "!defaultStageDir!\ovms_mediapipe_runtime_shared-cp!defaultPythonTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!

echo [INFO] Staged default Python ABI cp!defaultPythonTag! runtime libraries in !defaultStageDir!

:: NOTE: this section intentionally avoids wrapping everything in a single
:: parenthesized IF (...) block. Combining a nested "for /f ... do (...)"
:: loop and a piped "| tee" command invocation inside an outer parenthesized
:: block trips up cmd.exe's block pre-parser ("'.' was unexpected at this
:: time." / similar errors appearing only once execution reaches the block).
:: Using goto instead of nested parens sidesteps this entirely.
IF "!additionalPythonAbi!"=="" goto :skip_extra_python_abi
IF "!pythonRuntimeTargets!"=="" goto :skip_extra_python_abi

echo Building additional Python runtime libraries for Python ABI: !additionalPythonAbi!
for /f "tokens=1,2 delims=." %%a in ("!additionalPythonAbi!") do (
    set "EXTRA_MAJOR_VER=%%a"
    set "EXTRA_MINOR_VER=%%b"
)
set "extraPythonTag=!EXTRA_MAJOR_VER!!EXTRA_MINOR_VER!"
set "extraPythonHome=C:\opt\Python!extraPythonTag!"

IF NOT EXIST "!extraPythonHome!\python!extraPythonTag!.dll" (
    echo [ERROR] Missing dev Python install for ABI !extraPythonTag!: !extraPythonHome!\python!extraPythonTag!.dll not found.
    echo [ERROR] Install a full Python !additionalPythonAbi! development build to !extraPythonHome! before requesting this additional ABI build.
    exit /b 1
)

:: IMPORTANT: two distinct Python configurations must both be overridden, or
:: pybind11-based targets will silently keep linking the primary ABI's DLL:
::  1. --repo_env=OVMS_WINDOWS_PYTHON_VERSION selects our custom
::     @python3_windows repo (third_party/python/python_repo_win.bzl).
::  2. --repo_env/--action_env=PYTHON_BIN_PATH + --python_path select the
::     @local_config_python repo (pybind11_bazel's python_configure), which
::     is what @pybind11//:pybind11_embed (and therefore libovmspython,
::     libpython_calculators, pyovms) actually links against for the
::     embedded interpreter. Skipping this leaves those targets linked to
::     the primary ABI's pythonXY.dll even though the build "succeeds".
:: PYTHONHOME/PATH must also be pointed at the extra Python install for
:: this invocation: local_config_python's repository rule spawns
:: "!extraPythonHome!\python.exe" to probe its version, and that process
:: inherits PYTHONHOME from the environment. windows_setupvars.bat already
:: forced PYTHONHOME to the primary ABI's install, which would make the
:: extra interpreter load the wrong stdlib and crash ("SRE module
:: mismatch"). Restore the original PATH/PYTHONHOME after this build so
:: later steps in this script are unaffected.
:: Finally, target the concrete copy_* (shared-library-producing) Bazel
:: targets, not the wrapping cc_library targets (libovmspython,
:: libpython_calculators) - requesting only the wrapper does not
:: necessarily force Bazel to relink the underlying shared library if one
:: already exists on disk, which can silently leave a stale, wrong-ABI
:: artifact in place.
set "SAVED_PATH_FOR_EXTRA_ABI=%PATH%"
set "SAVED_PYTHONHOME_FOR_EXTRA_ABI=%PYTHONHOME%"
set "PYTHONHOME=!extraPythonHome!"
set "PATH=!extraPythonHome!\;!extraPythonHome!\Scripts\;%PATH%"

set "extraBuildTargets=//src:ovms //src/python:copy_libovmspython //src/python:copy_libpython_calculators //src/python/binding:copy_pyovms //src:ovms_mediapipe_runtime_shared"
:: Use a separate output_user_root for the extra-ABI build. This is critical:
:: pybind11_bazel's python_configure repository rule generates @local_config_python
:: with hardcoded paths to the Python include dir and .lib file. If the extra-ABI
:: build reuses the same output base as the default build, Bazel may serve the
:: cached Python312 @local_config_python even when PYTHON_BIN_PATH is overridden,
:: causing libovmspython-cpXYZ.dll to link against BOTH python312.dll (from the
:: cached @local_config_python) and pythonXYZ.dll (from @python3_windows which
:: does correctly respond to OVMS_WINDOWS_PYTHON_VERSION). The dual-runtime linkage
:: causes a fatal crash when the DLL is loaded. A separate output base forces a
:: clean @local_config_python fetch for the extra ABI.
set "extraBazelStartupCmd=--output_user_root=!BAZEL_SHORT_PATH!_py!extraPythonTag!"
set "extraBuildCommand=bazel !extraBazelStartupCmd! build %buildWithIntegrity% %bazelBuildArgs% --repo_env=OVMS_WINDOWS_PYTHON_VERSION=!extraPythonTag! --repo_env=PYTHON_BIN_PATH=!extraPythonHome:\=/!/python.exe --action_env=PYTHON_BIN_PATH=!extraPythonHome:\=/!/python.exe --action_env=PYTHON_LIB_PATH=!extraPythonHome:\=/!/lib/site-packages --python_path=!extraPythonHome:\=/!/python.exe --action_env OpenVINO_DIR=%openvino_dir% --jobs=%NUMBER_OF_PROCESSORS% --verbose_failures !extraBuildTargets! 2>&1 | tee win_build_py!extraPythonTag!.log"
:: Invoke via %var% (percent expansion), NOT !var! (delayed expansion): percent
:: expansion happens during cmd.exe's initial line parsing, before pipe/redirect
:: tokenization, so the embedded "| tee" is correctly recognized as a pipeline
:: (matching how %buildCommand% is invoked above for the main build). Delayed
:: expansion (!var!) happens AFTER that parsing step, so an embedded "|" would
:: instead be passed as a literal argument to bazel.exe, causing errors like
:: "Illegal char <|> ... " from Bazel trying to resolve "|" as a path/target.
%extraBuildCommand%
set "EXTRA_BUILD_RESULT=!errorlevel!"

set "PATH=%SAVED_PATH_FOR_EXTRA_ABI%"
set "PYTHONHOME=%SAVED_PYTHONHOME_FOR_EXTRA_ABI%"
if !EXTRA_BUILD_RESULT! neq 0 exit /b !EXTRA_BUILD_RESULT!

:: Stage ABI-suffixed artifacts so a subsequent default-ABI build cannot
:: overwrite them, and so windows_create_package.bat can package them.
set "extraStageDir=%cd%\dist\windows\python_abi_addons\cp!extraPythonTag!"
if exist "!extraStageDir!" rmdir /S /Q "!extraStageDir!"
md "!extraStageDir!"
if !errorlevel! neq 0 exit /b !errorlevel!

:: The extra-ABI build used a separate --output_user_root, so bazel-bin/ symlinks
:: still point to the default output base. Locate the artifacts via the extra
:: output base execroot path directly.
set "extraOutputBase=!BAZEL_SHORT_PATH!_py!extraPythonTag!"
set "extraExecroot=!extraOutputBase!\*\execroot\ovms"
:: Resolve the glob to a concrete path (the subdir contains a hash)
set "extraBinDir="
for /d %%d in ("!extraOutputBase!\*") do (
    if exist "%%d\execroot\ovms\bazel-out\x64_windows-opt\bin\src\python\libovmspython.dll" (
        set "extraBinDir=%%d\execroot\ovms\bazel-out\x64_windows-opt\bin"
    )
)
if "!extraBinDir!"=="" (
    echo [ERROR] Could not locate extra-ABI build artifacts under !extraOutputBase!
    exit /b 1
)

set "extraLibOvmspython=!extraBinDir!\src\python\libovmspython.dll"
set "extraLibPyCalc=!extraBinDir!\src\python\libpython_calculators.dll"
set "extraPyovms=!extraBinDir!\src\python\binding\pyovms.pyd"
set "extraMpRuntime=!extraBinDir!\src\ovms_mediapipe_runtime_shared.dll"

copy "!extraLibOvmspython!" "!extraStageDir!\libovmspython-cp!extraPythonTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!extraLibPyCalc!" "!extraStageDir!\libpython_calculators-cp!extraPythonTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!extraPyovms!" "!extraStageDir!\pyovms.pyd"
if !errorlevel! neq 0 exit /b !errorlevel!
copy "!extraMpRuntime!" "!extraStageDir!\ovms_mediapipe_runtime_shared-cp!extraPythonTag!.dll"
if !errorlevel! neq 0 exit /b !errorlevel!

echo [INFO] Staged additional Python ABI cp!extraPythonTag! runtime libraries in !extraStageDir!

:: With a separate output base for the extra-ABI build, the default output base
:: (bazel-bin/) was never touched. No restore build is needed.

:skip_extra_python_abi

:skip_default_python_abi_stage

endlocal
exit /b 0

:exit_build_error
echo Build failed.
endlocal
exit /b 1
