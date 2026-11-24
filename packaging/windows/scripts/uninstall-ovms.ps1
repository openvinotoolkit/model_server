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
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [ValidateSet("PreserveAll", "RemoveSettingsKeepModels", "RemoveAll")][string]$DataMode = "PreserveAll"
)

$ErrorActionPreference = "Stop"

$runtimePath = Join-Path $DataDir "runtime.json"
if (Test-Path $runtimePath) {
    try {
        $runtime = Get-Content $runtimePath -Raw | ConvertFrom-Json
        if ($runtime.pid) {
            $process = Get-Process -Id ([int]$runtime.pid) -ErrorAction SilentlyContinue
            if ($process -and $process.ProcessName -ieq "ovms") {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Warning "Could not read runtime state: $_"
    }
    Remove-Item -Force -Path $runtimePath -ErrorAction SilentlyContinue
}

& (Join-Path $PSScriptRoot "uninstall-service.ps1")
& (Join-Path $PSScriptRoot "set-path.ps1") -InstallDir $InstallDir -Action Remove
[Environment]::SetEnvironmentVariable("OVMS_DIR", $null, "User")
[Environment]::SetEnvironmentVariable("PYTHONHOME", $null, "User")
[Environment]::SetEnvironmentVariable("ESPEAK_DATA_PATH", $null, "User")

switch ($DataMode) {
    "PreserveAll" {
        # Keep everything under $DataDir.
    }
    "RemoveSettingsKeepModels" {
        Remove-Item -Force -Path (Join-Path $DataDir "settings.json") -ErrorAction SilentlyContinue
        Remove-Item -Force -Path (Join-Path $DataDir "install.json") -ErrorAction SilentlyContinue
        Remove-Item -Force -Path (Join-Path $DataDir "runtime.json") -ErrorAction SilentlyContinue

        $logsDir = Join-Path $DataDir "logs"
        if (Test-Path $logsDir) {
            Remove-Item -LiteralPath $logsDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        $packagesDir = Join-Path $DataDir "packages"
        if (Test-Path $packagesDir) {
            Remove-Item -LiteralPath $packagesDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        $downloadsDir = Join-Path $DataDir "downloads"
        if (Test-Path $downloadsDir) {
            Remove-Item -LiteralPath $downloadsDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        $diagnosticsDir = Join-Path $DataDir "diagnostics"
        if (Test-Path $diagnosticsDir) {
            Remove-Item -LiteralPath $diagnosticsDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    "RemoveAll" {
        if (Test-Path $DataDir) {
            Remove-Item -LiteralPath $DataDir -Recurse -Force
        }
    }
}

Write-Host "[INFO] OVMS uninstall cleanup complete."
