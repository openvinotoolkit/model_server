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
if (-not (Test-Path $settingsPath)) {
    throw "Settings file not found: $settingsPath"
}

$settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
$runtimePath = Join-Path $DataDir "runtime.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

$service = Get-Service ovms -ErrorAction SilentlyContinue
if ($service -and $settings.runMode -eq "service") {
    if ($service.Status -ne "Running") {
        Start-Service ovms
    }
    Write-Host "[INFO] OVMS service is running."
    exit 0
}

$restPort = [int]$settings.restPort
$grpcPort = [int]$settings.grpcPort
$ports = @($restPort, $grpcPort) | Where-Object { $_ -gt 0 } | Select-Object -Unique
foreach ($port in $ports) {
    $listeners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        if ($listener.OwningProcess -ne $PID) {
            throw "Port $port is already in use by process id $($listener.OwningProcess)."
        }
    }
}

if (Test-Path $runtimePath) {
    try {
        $runtime = Get-Content $runtimePath -Raw | ConvertFrom-Json
        if ($runtime.pid) {
            $existing = Get-Process -Id ([int]$runtime.pid) -ErrorAction SilentlyContinue
            if ($existing -and $existing.ProcessName -ieq "ovms") {
                Write-Host "[INFO] OVMS is already running with pid $($runtime.pid)."
                exit 0
            }
        }
    } catch {
        Write-Warning "Ignoring invalid runtime state: $runtimePath"
    }
}

$ovmsExe = Join-Path $settings.installDir "ovms.exe"
if (-not (Test-Path $ovmsExe)) {
    throw "ovms.exe not found: $ovmsExe"
}

. (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir $settings.installDir

$args = @(
    "--rest_port", $settings.restPort,
    "--rest_bind_address", $settings.bindAddress,
    "--config_path", $settings.configPath,
    "--log_level", $settings.logLevel,
    "--log_path", $settings.logPath
)

if ($settings.grpcPort -and [int]$settings.grpcPort -gt 0) {
    $args += @("--port", $settings.grpcPort, "--grpc_bind_address", $settings.bindAddress)
}

$commandText = "ovms " + (($args | ForEach-Object {
    if ($_ -match '^[A-Za-z0-9_./:-]+$') {
        $_
    } else {
        '"' + ($_ -replace '"', '\"') + '"'
    }
}) -join ' ')

$escapedInstallDir = $settings.installDir.Replace("'", "''").TrimEnd('\')
$escapedCommandText = $commandText.Replace("'", "''")

$setupVarsPath = Join-Path $settings.installDir "setupvars.ps1"
$runOvmsPath = Join-Path $settings.installDir "scripts\run-ovms.ps1"
$escapedDataDir = ([string]$settings.dataDir).Replace("'", "''")
$escapedSetupVarsPath = $setupVarsPath.Replace("'", "''")
$escapedRunOvmsPath = $runOvmsPath.Replace("'", "''")

$terminalCommand = @(
    "Set-Location -LiteralPath '$escapedInstallDir'",
    "try {",
    "  . '$escapedSetupVarsPath'",
    "  Write-Host 'You can start ovms with:'",
    "  Write-Host '  $escapedCommandText'",
    "  & '$escapedRunOvmsPath' -DataDir '$escapedDataDir'",
    "  exit `$LASTEXITCODE",
    "} catch {",
    "  Write-Error $_",
    "  Write-Host 'Failed to initialize or run OVMS.'",
    "  Read-Host 'Press Enter to close this window'",
    "  exit 1",
    "}"
) -join '; '

Start-Process -FilePath "powershell.exe" -ArgumentList @(
    '-NoExit',
    '-NoLogo',
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-Command',
    $terminalCommand
) -WindowStyle Normal

Write-Host "[INFO] Displayed OVMS startup command in a new terminal."
exit 0
