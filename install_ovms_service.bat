::
:: Copyright (c) 2025 Intel Corporation
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
if "%~1"=="-h" (
    echo Usage: install_ovms_service.bat [model_repository_path]
    exit /b 0
)
if "%~1"=="/?" (
    echo Usage: install_ovms_service.bat [model_repository_path]
    exit /b 0
)

IF "%~1"=="" (
    set /p "OVMS_MODEL_REPOSITORY_PATH=Enter the model repository absolute path: "
) ELSE (
    set "OVMS_MODEL_REPOSITORY_PATH=%~1"
)

::::::::::::::::::::::: Check directory
pushd "%CD%"
cd /d "!OVMS_MODEL_REPOSITORY_PATH!" 2>nul
if errorlevel 1 (
    echo [INFO] Creating model repository path !OVMS_MODEL_REPOSITORY_PATH!
    mkdir !OVMS_MODEL_REPOSITORY_PATH!
    if !errorlevel! neq 0 exit /b !errorlevel!
)
set "OVMS_MODEL_REPOSITORY_PATH=%CD%"
set "config_path=!OVMS_MODEL_REPOSITORY_PATH!\config.json"
IF /I EXIST !config_path! (
    echo [INFO] config exists !config_path!
) ELSE (
    echo {"model_config_list":[]} > !config_path!
    if !errorlevel! neq 0 exit /b !errorlevel!
    echo [INFO] created empty server config !config_path!
)
popd

echo Using model repository path !OVMS_MODEL_REPOSITORY_PATH!

set "OVMS_DIR=%~dp0"
::::::::::::::::::::::: Add persistent OVMS_DIR to PATH
setx "PATH" "%OVMS_DIR%;%PATH%"
setx "OVMS_MODEL_REPOSITORY_PATH" !OVMS_MODEL_REPOSITORY_PATH!

::::::::::::::::::::::: Create the service
sc create ovms binPath= "%OVMS_DIR%\ovms.exe --rest_port 8000 --config_path !config_path! --log_level INFO --log_path !OVMS_DIR!\ovms_server.log" DisplayName= "OpenVino Model Server"
if !errorlevel! neq 0 (
    echo [ERROR] sc create ovms failed !errorlevel!
    exit /b !errorlevel!
)
set "PYTHONHOME=%OVMS_DIR%\python"
set "PATH=%PYTHONHOME%;%PYTHONHOME%\Scripts;%PATH%"
::::::::::::::::::::::: Install the service by adding required environment variables to registry
!OVMS_DIR!\ovms.exe install
if !errorlevel! neq 0 (
    echo [ERROR] ovms.exe install failed !errorlevel!
    exit /b !errorlevel!
)
echo OpenVINO Model Server Service Installed