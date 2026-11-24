#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

param(
    [string]$InstallDir
)

$ErrorActionPreference = "Stop"

if (-not $InstallDir) {
    $sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
    $InstallDir = Join-Path $sourceRoot "dist\windows\ovms"
}

$required = @(
    "ovms.exe",
    "setupvars.bat",
    "setupvars.ps1",
    "install_ovms_service.bat"
)

$missing = @()
foreach ($file in $required) {
    $path = Join-Path $InstallDir $file
    if (-not (Test-Path $path)) {
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    throw "Missing required files in ${InstallDir}: $($missing -join ', ')"
}

& (Join-Path $InstallDir "ovms.exe") --version
if ($LASTEXITCODE -ne 0) {
    throw "ovms.exe --version failed."
}

Write-Host "[INFO] Install folder validation passed: $InstallDir"
