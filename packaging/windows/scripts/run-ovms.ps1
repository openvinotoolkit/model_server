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
    [Parameter(Mandatory = $true)][string]$DataDir
)

$ErrorActionPreference = "Stop"

$settingsPath = Join-Path $DataDir "settings.json"
if (-not (Test-Path -LiteralPath $settingsPath)) {
    throw "Settings file not found: $settingsPath"
}

$settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
$runtimePath = Join-Path $DataDir "runtime.json"
$ovmsExe = Join-Path ([string]$settings.installDir) "ovms.exe"
if (-not (Test-Path -LiteralPath $ovmsExe)) {
    throw "ovms.exe not found: $ovmsExe"
}

. (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir ([string]$settings.installDir)

$arguments = @(
    "--rest_port", [string]$settings.restPort,
    "--rest_bind_address", [string]$settings.bindAddress,
    "--config_path", [string]$settings.configPath,
    "--log_level", [string]$settings.logLevel,
    "--log_path", [string]$settings.logPath
)
if ($settings.grpcPort -and [int]$settings.grpcPort -gt 0) {
    $arguments += @("--port", [string]$settings.grpcPort, "--grpc_bind_address", [string]$settings.bindAddress)
}

$argumentText = ($arguments | ForEach-Object {
    if ($_ -match '^[A-Za-z0-9_./:-]+$') {
        $_
    } else {
        '"' + ($_ -replace '"', '\"') + '"'
    }
}) -join ' '

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$process = $null
try {
    $process = Start-Process -FilePath $ovmsExe -ArgumentList $argumentText -WorkingDirectory ([string]$settings.installDir) -PassThru

    $runtime = [ordered]@{
        pid = $process.Id
        processName = "ovms"
        installDir = [string]$settings.installDir
        dataDir = $DataDir
        startedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    }
    $runtimeTmp = "$runtimePath.tmp"
    [System.IO.File]::WriteAllText($runtimeTmp, ($runtime | ConvertTo-Json -Depth 10), $utf8NoBom)
    Move-Item -Force -LiteralPath $runtimeTmp -Destination $runtimePath

    $process.WaitForExit()
    exit $process.ExitCode
} finally {
    if (Test-Path -LiteralPath $runtimePath) {
        try {
            $currentRuntime = Get-Content -LiteralPath $runtimePath -Raw | ConvertFrom-Json
            if ($process -and [int]$currentRuntime.pid -eq $process.Id) {
                Remove-Item -LiteralPath $runtimePath -Force -ErrorAction SilentlyContinue
            }
        } catch {
            # Do not clear runtime state that we could not verify.
        }
    }
}
