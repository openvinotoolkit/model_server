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
    [string]$SetupLogPath = (Join-Path (Split-Path -Parent $PSScriptRoot) "setup.log")
)

$ErrorActionPreference = "Stop"

function Write-SetupLog {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [Parameter(Mandatory = $true)][string]$LogPath
    )

    if ([string]::IsNullOrWhiteSpace($LogPath)) {
        return
    }

    $scriptName = [System.IO.Path]::GetFileName($MyInvocation.ScriptName)
    if ([string]::IsNullOrWhiteSpace($scriptName)) {
        $scriptName = 'unknown-script'
    }

    $logDir = Split-Path -Parent $LogPath
    if (-not [string]::IsNullOrWhiteSpace($logDir) -and -not (Test-Path -LiteralPath $logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }

    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp] [$scriptName] $Message"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::AppendAllText($LogPath, $line + [Environment]::NewLine, $utf8NoBom)
    Write-Host $line
}

try {
    Write-SetupLog -Message "[INFO] Removing existing OVMS service if present." -LogPath $SetupLogPath

    $service = Get-Service ovms -ErrorAction SilentlyContinue
    if ($service) {
        if ($service.Status -ne "Stopped") {
            Write-SetupLog -Message "[INFO] Stopping existing OVMS service before deletion." -LogPath $SetupLogPath
            Stop-Service ovms -ErrorAction SilentlyContinue
        }
        $deleteOutput = & sc.exe delete ovms 2>&1
        foreach ($line in $deleteOutput) {
            Write-SetupLog -Message $line.ToString() -LogPath $SetupLogPath
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to delete ovms service."
        }
    }

    $serviceRegistryPath = "SYSTEM\CurrentControlSet\Services\ovms"
    $serviceRegistryKey = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($serviceRegistryPath, $true)
    if ($null -ne $serviceRegistryKey) {
        $serviceRegistryKey.DeleteValue("Environment", $false)
        $serviceRegistryKey.Close()
        Write-SetupLog -Message "[INFO] Removed OVMS service-specific environment registry values." -LogPath $SetupLogPath
    }

    Write-SetupLog -Message "[INFO] OVMS service cleanup completed." -LogPath $SetupLogPath
} catch {
    Write-SetupLog -Message "[ERROR] OVMS service cleanup failed: $($_.Exception.Message)" -LogPath $SetupLogPath
    throw
}

