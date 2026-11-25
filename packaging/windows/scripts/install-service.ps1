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

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdministrator)) {
    try {
        Write-SetupLog -Message "[INFO] Requesting elevation for OVMS service setup." -LogPath $SetupLogPath

        $argumentList = @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", ('"{0}"' -f $PSCommandPath),
            "-InstallDir", ('"{0}"' -f $InstallDir),
            "-DataDir", ('"{0}"' -f $DataDir),
            "-SetupLogPath", ('"{0}"' -f $SetupLogPath)
        )

        $process = Start-Process -FilePath "powershell.exe" -ArgumentList $argumentList -Verb RunAs -PassThru -Wait
        if ($null -eq $process) {
            throw "Elevation process did not start."
        }

        exit $process.ExitCode
    } catch {
        Write-SetupLog -Message "[ERROR] Elevation request failed or was canceled: $($_.Exception.Message)" -LogPath $SetupLogPath
        exit 1223
    }
}

function Test-ProtectedServiceLocation {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Description
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\\')
    if (-not (Test-Path -LiteralPath $fullPath)) {
        throw "$Description does not exist: $fullPath"
    }

    $userProfileRoot = [System.IO.Path]::GetFullPath([Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)).TrimEnd('\')
    $localAppDataRoot = [System.IO.Path]::GetFullPath([Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)).TrimEnd('\')
    $publicRoot = [System.IO.Path]::GetFullPath([Environment]::GetFolderPath([Environment+SpecialFolder]::CommonDocuments)).TrimEnd('\')

    $profileRoots = @($userProfileRoot, $localAppDataRoot)
    $usersRoot = Join-Path $env:SystemDrive "Users"
    if (-not [string]::IsNullOrWhiteSpace($usersRoot)) {
        $profileRoots += [System.IO.Path]::GetFullPath($usersRoot).TrimEnd('\')
    }

    foreach ($root in $profileRoots) {
        if (-not [string]::IsNullOrWhiteSpace($root) -and
            ($fullPath.Equals($root, [System.StringComparison]::OrdinalIgnoreCase) -or
             $fullPath.StartsWith($root + '\\', [System.StringComparison]::OrdinalIgnoreCase))) {
            throw "$Description is under a user-writable profile location and is unsafe for the restricted OVMS service account: $fullPath"
        }
    }

    if (-not [string]::IsNullOrWhiteSpace($publicRoot) -and
        ($fullPath.Equals($publicRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
         $fullPath.StartsWith($publicRoot + '\\', [System.StringComparison]::OrdinalIgnoreCase))) {
        throw "$Description is under Public Documents and is unsafe for the restricted OVMS service account: $fullPath"
    }

    $writableRights = [System.Security.AccessControl.FileSystemRights]::Write -bor
        [System.Security.AccessControl.FileSystemRights]::Modify -bor
        [System.Security.AccessControl.FileSystemRights]::FullControl -bor
        [System.Security.AccessControl.FileSystemRights]::GenericWrite -bor
        [System.Security.AccessControl.FileSystemRights]::GenericAll -bor
        [System.Security.AccessControl.FileSystemRights]::Delete -bor
        [System.Security.AccessControl.FileSystemRights]::ChangePermissions -bor
        [System.Security.AccessControl.FileSystemRights]::TakeOwnership

    $pathsToCheck = @($fullPath)
    if (Test-Path -LiteralPath $fullPath -PathType Container) {
        $pathsToCheck += Get-ChildItem -LiteralPath $fullPath -Force -Recurse -ErrorAction Stop |
            ForEach-Object { $_.FullName }
    }

    foreach ($pathToCheck in $pathsToCheck) {
        if (-not (Test-Path -LiteralPath $pathToCheck)) {
            continue
        }

        foreach ($rule in (Get-Acl -LiteralPath $pathToCheck).Access) {
            if ($rule.AccessControlType -ne [System.Security.AccessControl.AccessControlType]::Allow) {
                continue
            }
            if (($rule.FileSystemRights -band $writableRights) -eq 0) {
                continue
            }

            $sid = $null
            try {
                $sid = $rule.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
            } catch {
                $sid = $rule.IdentityReference.Value
            }

            $trusted = $sid -eq "S-1-5-19" -or
                $sid -eq "S-1-5-18" -or
                $sid -eq "S-1-5-32-544" -or
                $sid -like "S-1-5-80-*"

            if (-not $trusted) {
                throw "$Description grants writable access to untrusted principal '$($rule.IdentityReference.Value)' (SID $sid) at '$pathToCheck'."
            }
        }
    }
}

try {
    Write-SetupLog -Message "[INFO] Starting OVMS service installation for $InstallDir" -LogPath $SetupLogPath

    $settingsPath = Join-Path $DataDir "settings.json"
    if (-not (Test-Path $settingsPath)) {
        throw "Settings file not found: $settingsPath"
    }

    $settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
    $ovmsExe = Join-Path $InstallDir "ovms.exe"
    if (-not (Test-Path $ovmsExe)) {
        throw "ovms.exe not found: $ovmsExe"
    }

    Test-ProtectedServiceLocation -Path $InstallDir -Description "OVMS install directory"
    Test-ProtectedServiceLocation -Path $DataDir -Description "OVMS service data directory"

    if (-not [string]::IsNullOrWhiteSpace([string]$settings.configPath)) {
        $configPath = [string]$settings.configPath
        Test-ProtectedServiceLocation -Path (Split-Path -Parent $configPath) -Description "OVMS model configuration directory"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$settings.modelRepositoryPath)) {
        Test-ProtectedServiceLocation -Path ([string]$settings.modelRepositoryPath) -Description "OVMS model repository"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$settings.logPath)) {
        $logParent = Split-Path -Parent ([string]$settings.logPath)
        if (-not [string]::IsNullOrWhiteSpace($logParent)) {
            if (-not (Test-Path -LiteralPath $logParent)) {
                New-Item -ItemType Directory -Force -Path $logParent | Out-Null
            }
            Test-ProtectedServiceLocation -Path $logParent -Description "OVMS log directory"
        }
    }
    Write-SetupLog -Message "[INFO] Verified protected paths for the restricted OVMS service account." -LogPath $SetupLogPath

    . (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir $InstallDir

    $binPath = "`"$ovmsExe`" --rest_port $($settings.restPort) --rest_bind_address `"$($settings.bindAddress)`" --config_path `"$($settings.configPath)`" --log_level $($settings.logLevel) --log_path `"$($settings.logPath)`""
    if ($settings.grpcPort -and [int]$settings.grpcPort -gt 0) {
        $binPath += " --port $($settings.grpcPort) --grpc_bind_address `"$($settings.bindAddress)`""
    }

    $existing = Get-Service ovms -ErrorAction SilentlyContinue
    if ($existing) {
        Write-SetupLog -Message "[INFO] Existing OVMS service is present; will stop and reinstall service." -LogPath $SetupLogPath
        if ($existing.Status -ne "Stopped") {
            Write-SetupLog -Message "[INFO] Stopping running OVMS service before reinstall." -LogPath $SetupLogPath
            Stop-Service ovms -ErrorAction SilentlyContinue
        }
        & (Join-Path $PSScriptRoot "uninstall-service.ps1") -SetupLogPath $SetupLogPath
        $existing = $null
    }

    Write-SetupLog -Message "[INFO] Creating Windows service with binPath: $binPath" -LogPath $SetupLogPath
    $createOutput = & sc.exe create ovms binPath= $binPath obj= "NT AUTHORITY\LocalService" DisplayName= "OpenVINO Model Server" start= demand 2>&1
    foreach ($line in $createOutput) {
        Write-SetupLog -Message $line.ToString() -LogPath $SetupLogPath
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create ovms service."
    }

    $service = Get-Service ovms -ErrorAction SilentlyContinue
    if (-not $service) {
        throw "OVMS service was not created successfully; Get-Service ovms returned no service object."
    }

    try {
        Write-SetupLog -Message "[INFO] Running ovms.exe install step." -LogPath $SetupLogPath
        $installOutput = & $ovmsExe install 2>&1
        foreach ($line in $installOutput) {
            Write-SetupLog -Message $line.ToString() -LogPath $SetupLogPath
        }
    } catch {
        Write-SetupLog -Message "[WARN] ovms.exe install returned an error: $($_.Exception.Message)" -LogPath $SetupLogPath
    }

    $settings.runMode = "service"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($settingsPath, ($settings | ConvertTo-Json -Depth 10), $utf8NoBom)

    $service = Get-Service ovms -ErrorAction SilentlyContinue
    if (-not $service) {
        throw "OVMS service disappeared after installation."
    }

    if ($service.Status -ne "Running") {
        Write-SetupLog -Message "[INFO] Starting OVMS service." -LogPath $SetupLogPath
        Start-Service ovms -ErrorAction Stop
    }

    $service = Get-Service ovms -ErrorAction SilentlyContinue
    if (-not $service -or $service.Status -ne "Running") {
        throw "OVMS service did not reach Running state after startup attempt."
    }

    Write-SetupLog -Message "[INFO] OVMS service installed and started." -LogPath $SetupLogPath

    $environmentValues = @("OVMS_DIR=$env:OVMS_DIR")
    if (Test-Path -LiteralPath (Join-Path $InstallDir "python\python.exe")) {
        $environmentValues += "PYTHONHOME=$env:PYTHONHOME"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$env:PATH)) {
        $environmentValues += "PATH=$env:PATH"
    }
    if (-not [string]::IsNullOrWhiteSpace([string]$env:ESPEAK_DATA_PATH)) {
        $environmentValues += "ESPEAK_DATA_PATH=$env:ESPEAK_DATA_PATH"
    }

    $serviceRegistryPath = "SYSTEM\CurrentControlSet\Services\ovms"
    $serviceRegistryKey = [Microsoft.Win32.Registry]::LocalMachine.CreateSubKey($serviceRegistryPath)
    if ($null -eq $serviceRegistryKey) {
        throw "Failed to open OVMS service registry key: HKLM\\$serviceRegistryPath"
    }
    $serviceRegistryKey.SetValue("Environment", [string[]]$environmentValues, [Microsoft.Win32.RegistryValueKind]::MultiString)
    $serviceRegistryKey.Close()
    Write-SetupLog -Message "[INFO] Configured OVMS service-specific environment; PYTHONHOME is not stored as a machine environment variable." -LogPath $SetupLogPath
} catch {
    Write-SetupLog -Message "[ERROR] OVMS service installation failed: $($_.Exception.Message)" -LogPath $SetupLogPath
    try {
        Write-SetupLog -Message "[INFO] Attempting OVMS service cleanup after failure." -LogPath $SetupLogPath
        & (Join-Path $PSScriptRoot "uninstall-service.ps1") -SetupLogPath $SetupLogPath
    } catch {
        Write-SetupLog -Message "[WARN] OVMS service cleanup also failed: $($_.Exception.Message)" -LogPath $SetupLogPath
    }
    throw
}
