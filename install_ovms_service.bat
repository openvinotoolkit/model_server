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
if not exist "!OVMS_MODEL_REPOSITORY_PATH!\" (
    echo [INFO] Creating model repository path !OVMS_MODEL_REPOSITORY_PATH!
    mkdir "!OVMS_MODEL_REPOSITORY_PATH!"
    if !errorlevel! neq 0 exit /b !errorlevel!
)
pushd "!OVMS_MODEL_REPOSITORY_PATH!" 2>nul
if errorlevel 1 (
    echo [ERROR] Model repository path is invalid: !OVMS_MODEL_REPOSITORY_PATH!
    exit /b !errorlevel!
)
set "OVMS_MODEL_REPOSITORY_PATH=%CD%"
set "config_path=!OVMS_MODEL_REPOSITORY_PATH!\config.json"
IF /I EXIST "!config_path!" (
    echo [INFO] config exists !config_path!
) ELSE (
    echo {"model_config_list":[]} > "!config_path!"
    if !errorlevel! neq 0 exit /b !errorlevel!
    echo [INFO] created empty server config !config_path!
)
popd

echo Using model repository path !OVMS_MODEL_REPOSITORY_PATH!

set "OVMS_DIR=%~dp0"

::::::::::::::::::::::: Create the service
REM Build binPath carefully to avoid quoting issues with delayed expansion
set "binPath_cmd=\"!OVMS_DIR!ovms.exe\" --rest_port 8000 --config_path \"!config_path!\" --log_level INFO --log_path \"!OVMS_DIR!ovms_server.log\""
sc create ovms binPath= "!binPath_cmd!" DisplayName= "OpenVino Model Server"
set "SC_CREATE_ERROR=!errorlevel!"
if "!SC_CREATE_ERROR!"=="1073" (
    echo [INFO] Service ovms already exists. Updating service configuration.
    sc config ovms binPath= "!binPath_cmd!" DisplayName= "OpenVino Model Server"
    if !errorlevel! neq 0 (
        echo [ERROR] sc config ovms failed !errorlevel!
        exit /b !errorlevel!
    )
) else if !SC_CREATE_ERROR! neq 0 (
    echo [ERROR] sc create ovms failed !SC_CREATE_ERROR!
    exit /b !SC_CREATE_ERROR!
)
set "PYTHONHOME=%OVMS_DIR%\python"
set "PATH=%PYTHONHOME%;%PYTHONHOME%\Scripts;%PATH%"
::::::::::::::::::::::: Install the service by adding required environment variables to registry
"!OVMS_DIR!\ovms.exe" install
if !errorlevel! neq 0 (
    echo [ERROR] ovms.exe install failed !errorlevel!
    exit /b !errorlevel!
)
endlocal & (
    set "OVMS_DIR=%OVMS_DIR%"
    set "OVMS_MODEL_REPOSITORY_PATH=%OVMS_MODEL_REPOSITORY_PATH%"
)
set "OVMS_DIR_NORM=%OVMS_DIR%"
if "%OVMS_DIR_NORM:~-1%"=="\" set "OVMS_DIR_NORM=%OVMS_DIR_NORM:~0,-1%"

::::::::::::::::::::::: Add persistent variables for future cmd.exe sessions
set "PYTHONHOME=%OVMS_DIR_NORM%\python"
set "OVMS_PERSIST_PS=$ovmsDir=$env:OVMS_DIR_NORM.TrimEnd('\\');"
set "OVMS_PERSIST_PS_PART= $modelRepo=$env:OVMS_MODEL_REPOSITORY_PATH;"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= $pythonHome=Join-Path $ovmsDir 'python';"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= [Environment]::SetEnvironmentVariable('OVMS_MODEL_REPOSITORY_PATH',$modelRepo,'User');"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= [Environment]::SetEnvironmentVariable('PYTHONHOME',$pythonHome,'User');"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= $userPath=[Environment]::GetEnvironmentVariable('Path','User');"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= $parts=@(); if (-not [string]::IsNullOrWhiteSpace($userPath)) { $parts=@($userPath -split ';' | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }) };"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= $pythonScripts=Join-Path $pythonHome 'Scripts';"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= $required=@($ovmsDir,$pythonHome,$pythonScripts);"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= foreach ($req in $required) { $exists=$false; foreach ($p in $parts) { if ($p.TrimEnd('\\') -ieq $req.TrimEnd('\\')) { $exists=$true; break } }; if (-not $exists) { $parts=@($req)+$parts } };"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= $newPath=($parts -join ';');"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
set "OVMS_PERSIST_PS_PART= [Environment]::SetEnvironmentVariable('Path',$newPath,'User')"
set "OVMS_PERSIST_PS=%OVMS_PERSIST_PS%%OVMS_PERSIST_PS_PART%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "%OVMS_PERSIST_PS%"
if errorlevel 1 (
    echo [ERROR] Failed to persist user environment variables
    exit /b %errorlevel%
)

::::::::::::::::::::::: Update current cmd.exe session
echo ;%PATH%; | findstr /i /c:";%OVMS_DIR_NORM%;" /c:";%OVMS_DIR_NORM%\;" >nul
if errorlevel 1 set "PATH=%OVMS_DIR_NORM%;%PATH%"
echo ;%PATH%; | findstr /i /c:";%PYTHONHOME%;" >nul
if errorlevel 1 set "PATH=%PYTHONHOME%;%PATH%"
echo ;%PATH%; | findstr /i /c:";%PYTHONHOME%\Scripts;" >nul
if errorlevel 1 set "PATH=%PYTHONHOME%\Scripts;%PATH%"
echo OpenVINO Model Server Service Installed