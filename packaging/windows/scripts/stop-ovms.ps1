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
    [string]$DataDir = "$env:LOCALAPPDATA\OVMS"
)

$ErrorActionPreference = "Stop"

$settingsPath = Join-Path $DataDir "settings.json"
$runtimePath = Join-Path $DataDir "runtime.json"

if (Test-Path $settingsPath) {
    $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
    if ($settings.runMode -eq "service") {
        $service = Get-Service ovms -ErrorAction SilentlyContinue
        if ($service -and $service.Status -ne "Stopped") {
            Stop-Service ovms -ErrorAction Stop
            Write-Host "[INFO] Stopped OVMS service."
            exit 0
        }
    }
}

if (Test-Path $runtimePath) {
    try {
        $runtime = Get-Content $runtimePath -Raw | ConvertFrom-Json
        if ($runtime.pid) {
            $process = Get-Process -Id ([int]$runtime.pid) -ErrorAction SilentlyContinue
            if ($process -and $process.ProcessName -ieq "ovms") {
                Stop-Process -Id $process.Id -Force
                Write-Host "[INFO] Stopped OVMS process $($process.Id)."
            }
        }
    } finally {
        Remove-Item -Force -Path $runtimePath -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "[INFO] No OVMS runtime state found."
}

