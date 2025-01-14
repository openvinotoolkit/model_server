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
:: Prepares embedded python installation for the purpose of ovms building and creating the final ovms distribution. 
setlocal EnableExtensions EnableDelayedExpansion
@echo off
if "%~1"=="" (
    set "dest_dir=C:\opt"
    echo Destination directory not specified. Using: C:\opt
) else (
    set "dest_dir=%~1"
    echo User specified destination directory: %1
)

if "%~2"=="" (
    set "python_version=3.9.13"
    echo Python version not specified. Using: 3.9.13
) else (
    set "python_version=%~2"
    echo User specified Python version: %2
)

if "%~3"=="" (
    set "python_dir=python39"
    echo Python directory not specified. Using: python39
) else (
    set "python_dir=%~3"
    echo User specified Python directory: %3
)

set "python_full_name=python-%python_version%-embed-amd64"
set "embeddable_python_url=https://www.python.org/ftp/python/%python_version%/%python_full_name%.zip"

:: Download and unpack everything
rmdir /S /Q %dest_dir%\%python_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

md %dest_dir%\%python_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

if exist %dest_dir%\%python_dir%.zip (
    echo Python zip already downloaded. Will unpack existing file.
) else (
    curl -k %embeddable_python_url% -o %dest_dir%\%python_dir%.zip
    if !errorlevel! neq 0 exit /b !errorlevel!
)

C:\Windows\System32\tar.exe -xf %dest_dir%\%python_dir%.zip -C %dest_dir%\%python_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

cd %dest_dir%\%python_dir%
md %python_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

C:\Windows\System32\tar.exe -xf %python_dir%.zip -C %python_dir%
if !errorlevel! neq 0 exit /b !errorlevel!

:: Adjust paths so everything is accessible
(
echo .\%python_dir%
echo .
echo .\Scripts
echo .\Lib\site-packages
) > %python_dir%._pth
if !errorlevel! neq 0 exit /b !errorlevel!

:: Install pip
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
if !errorlevel! neq 0 exit /b !errorlevel!
.\python.exe get-pip.py
if !errorlevel! neq 0 exit /b !errorlevel!
