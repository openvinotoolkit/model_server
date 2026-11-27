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

function Remove-EnvironmentVariableIfMatches {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$TargetValue,
        [Parameter(Mandatory = $true)][ValidateSet('User', 'Machine')][string]$Scope
    )

    $currentValue = [Environment]::GetEnvironmentVariable($Name, $Scope)
    if ([string]::IsNullOrWhiteSpace($currentValue)) {
        return
    }

    $matches = $false
    if ($Name -eq 'OVMS_DIR' -or $Name -eq 'PYTHONHOME' -or $Name -eq 'ESPEAK_DATA_PATH') {
        $normalizedCurrent = [System.IO.Path]::GetFullPath($currentValue).TrimEnd('\')
        $normalizedTarget = [System.IO.Path]::GetFullPath($TargetValue).TrimEnd('\')
        $matches = $normalizedCurrent.Equals($normalizedTarget, [System.StringComparison]::OrdinalIgnoreCase)
    }

    if ($matches) {
        [Environment]::SetEnvironmentVariable($Name, $null, $Scope)
        Write-Host "[INFO] Cleared $Scope $Name because it pointed to this OVMS installation."
    }
}

$installDirFull = [System.IO.Path]::GetFullPath($InstallDir).TrimEnd('\')
$pythonHome = Join-Path $installDirFull 'python'
$espeakData = Join-Path $installDirFull 'espeak-ng-data'

foreach ($scope in @('User', 'Machine')) {
    Remove-EnvironmentVariableIfMatches -Name 'OVMS_DIR' -TargetValue $installDirFull -Scope $scope
    Remove-EnvironmentVariableIfMatches -Name 'PYTHONHOME' -TargetValue $pythonHome -Scope $scope
    Remove-EnvironmentVariableIfMatches -Name 'ESPEAK_DATA_PATH' -TargetValue $espeakData -Scope $scope
}

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
