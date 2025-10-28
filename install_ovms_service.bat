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
set "OVMS_DIR=%~dp0"
set "PYTHONHOME=%OVMS_DIR%\python"
set "PATH=%OVMS_DIR%;%PYTHONHOME%;%PYTHONHOME%\Scripts;%PATH%"
sc create ovms binPath= "!OVMS_DIR!\ovms.exe --rest_port 8000 --config_path c:\models\config.json --log_level INFO --log_path !OVMS_DIR!\ovms_server.log" DisplayName= "OpenVino Model Server"
if !errorlevel! neq 0 exit /b !errorlevel!
!OVMS_DIR!\ovms.exe install
if !errorlevel! neq 0 exit /b !errorlevel!
echo "OpenVINO Model Server Service Installed"