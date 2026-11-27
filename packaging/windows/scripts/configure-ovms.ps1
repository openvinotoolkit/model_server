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
    [string]$ModelRepositoryPath = "",
    [ValidateSet("system", "embedded", "none")][string]$PythonVariant = "none",
    [string]$RunMode = "user-login",
    [int]$RestPort = 8000,
    [int]$GrpcPort = 0,
    [string]$BindAddress = "127.0.0.1",
    [string]$LogLevel = "INFO",
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

trap {
    $message = $_.Exception.Message
    Write-SetupLog -Message "[ERROR] OVMS configuration failed: $message" -LogPath $SetupLogPath
    exit 1
}

function Ensure-SystemPythonRequirements {
    param(
        [Parameter(Mandatory = $true)][string]$Variant,
        [Parameter(Mandatory = $true)][string]$InstallDir
    )

    if ($Variant -ne "system") {
        return
    }

    $previousPythonHome = $env:PYTHONHOME
    $previousPythonPath = $env:PYTHONPATH

    try {
        Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue

        $pythonCandidates = @()
        $programFiles = [Environment]::GetEnvironmentVariable("ProgramFiles")
        $programW6432 = [Environment]::GetEnvironmentVariable("ProgramW6432")
        $localAppData = [Environment]::GetEnvironmentVariable("LOCALAPPDATA")
        $systemDrive = [Environment]::GetEnvironmentVariable("SystemDrive")

        foreach ($candidate in @(
            $(if ($programFiles) { Join-Path $programFiles "Python314\python.exe" }),
            $(if ($programW6432) { Join-Path $programW6432 "Python314\python.exe" }),
            $(if ($localAppData) { Join-Path $localAppData "Programs\Python\Python314\python.exe" }),
            $(if ($systemDrive) { Join-Path $systemDrive "opt\Python314\python.exe" })
        )) {
            if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
                $pythonCandidates += $candidate
            }
        }

        if (-not $pythonCandidates) {
            $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
            if ($pyLauncher) {
                $resolvedPython = & $pyLauncher.Source -3.14 -c "import sys; print(sys.executable)" 2>$null
                $resolvedPython = ($resolvedPython | Select-Object -First 1)
                if (-not [string]::IsNullOrWhiteSpace([string]$resolvedPython) -and (Test-Path -LiteralPath ([string]$resolvedPython).Trim())) {
                    $pythonCandidates += ([string]$resolvedPython).Trim()
                }
            }
        }

        if (-not $pythonCandidates) {
            throw "System Python was selected, but Python 3.14 was not found. Install Python 3.14 and try again."
        }

        $pythonCommand = $pythonCandidates | Select-Object -First 1
        $pythonHome = Split-Path -Parent $pythonCommand
        $env:PYTHONHOME = $pythonHome
        $env:PYTHONPATH = (($pythonHome, (Join-Path $pythonHome "Lib"), (Join-Path $pythonHome "DLLs")) -join ";")

        $pythonBindingDir = Join-Path $InstallDir "python"
        $pythonBindingModule = Join-Path $pythonBindingDir "pyovms.pyd"
        if (-not (Test-Path -LiteralPath $pythonBindingModule)) {
            throw "OVMS Python binding was not found at '$pythonBindingModule'. Reinstall OVMS with Python support enabled."
        }

        $versionOutput = & $pythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
        $versionText = ($versionOutput | Select-Object -First 1).ToString().Trim()
        if (-not $versionText) {
            throw "Unable to detect the system Python version."
        }

        if ($versionText -ne "3.14") {
            throw "System Python version $versionText is not supported. OVMS requires Python 3.14 when using the system-Python option."
        }

        $pipArgs = @(
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "setuptools==84.2.1",
            "Jinja2==3.1.6",
            "MarkupSafe==3.0.4"
        )

        & $pythonCommand @pipArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install the required Python dependencies for the system-Python configuration."
        }

        $escapedInstallDir = $InstallDir.Replace('\\', '\\\\').Replace("'", "\\'")
        $validateBindingCommand = "import os,sys; install_dir='${escapedInstallDir}'; binding_dir=os.path.join(install_dir,'python'); os.add_dll_directory(install_dir) if hasattr(os,'add_dll_directory') else None; sys.path.insert(0,binding_dir) if binding_dir not in sys.path else None; import pyovms"
        & $pythonCommand -c $validateBindingCommand

        if ($LASTEXITCODE -ne 0) {
            throw "System Python cannot import OVMS bindings (pyovms). Ensure '$pythonBindingDir' is available and DLL loading from '$InstallDir' is allowed."
        }
    } catch {
        throw "System Python validation failed: $($_.Exception.Message)"
    } finally {
        if ($null -eq $previousPythonHome) {
            Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONHOME = $previousPythonHome
        }

        if ($null -eq $previousPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $previousPythonPath
        }
    }
}

$settingsPath = Join-Path $DataDir "settings.json"
$installMarkerPath = Join-Path $DataDir "install.json"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

Ensure-SystemPythonRequirements -Variant $PythonVariant -InstallDir $InstallDir

$existingSettings = $null
if (Test-Path -LiteralPath $settingsPath) {
    try {
        $existingSettings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json -ErrorAction Stop
    } catch {
        Write-SetupLog -Message "[WARN] Existing settings file is invalid and will be regenerated: $settingsPath" -LogPath $SetupLogPath
    }
}

$hasModelRepositoryOverride = $PSBoundParameters.ContainsKey("ModelRepositoryPath") -and -not [string]::IsNullOrWhiteSpace($ModelRepositoryPath)
if ($hasModelRepositoryOverride) {
    $modelsDir = $ModelRepositoryPath.Trim()
} elseif ($existingSettings -and -not [string]::IsNullOrWhiteSpace([string]$existingSettings.modelRepositoryPath)) {
    $modelsDir = [string]$existingSettings.modelRepositoryPath
} else {
    $modelsDir = Join-Path $DataDir "models"
}

if ($hasModelRepositoryOverride) {
    $configPath = Join-Path $modelsDir "config.json"
} elseif ($existingSettings -and -not [string]::IsNullOrWhiteSpace([string]$existingSettings.configPath)) {
    $configPath = [string]$existingSettings.configPath
} else {
    $configPath = Join-Path $modelsDir "config.json"
}

$logsDir = Join-Path $DataDir "logs"
if ($existingSettings -and -not [string]::IsNullOrWhiteSpace([string]$existingSettings.logPath)) {
    $logPath = [string]$existingSettings.logPath
} else {
    $logPath = Join-Path $logsDir "ovms_server.log"
}

if ($PSBoundParameters.ContainsKey("RestPort")) {
    $effectiveRestPort = $RestPort
} elseif ($existingSettings -and $null -ne $existingSettings.restPort) {
    $effectiveRestPort = [int]$existingSettings.restPort
} else {
    $effectiveRestPort = 8000
}

if ($PSBoundParameters.ContainsKey("GrpcPort")) {
    $effectiveGrpcPort = $GrpcPort
} elseif ($existingSettings -and $null -ne $existingSettings.grpcPort) {
    $effectiveGrpcPort = [int]$existingSettings.grpcPort
} else {
    $effectiveGrpcPort = 0
}

if ($PSBoundParameters.ContainsKey("BindAddress")) {
    $effectiveBindAddress = $BindAddress
} elseif ($existingSettings -and -not [string]::IsNullOrWhiteSpace([string]$existingSettings.bindAddress)) {
    $effectiveBindAddress = [string]$existingSettings.bindAddress
} else {
    $effectiveBindAddress = "127.0.0.1"
}

if ($PSBoundParameters.ContainsKey("LogLevel")) {
    $effectiveLogLevel = $LogLevel
} elseif ($existingSettings -and -not [string]::IsNullOrWhiteSpace([string]$existingSettings.logLevel)) {
    $effectiveLogLevel = [string]$existingSettings.logLevel
} else {
    $effectiveLogLevel = "INFO"
}

if ($PSBoundParameters.ContainsKey("RunMode")) {
    $effectiveRunMode = $RunMode
} elseif ($existingSettings -and -not [string]::IsNullOrWhiteSpace([string]$existingSettings.runMode)) {
    $effectiveRunMode = [string]$existingSettings.runMode
} else {
    $effectiveRunMode = "user-login"
}

$configDir = Split-Path -Parent $configPath
$logDir = Split-Path -Parent $logPath

$pathsToEnsure = @(
    $DataDir,
    $modelsDir,
    $logsDir,
    (Join-Path $DataDir "packages"),
    (Join-Path $DataDir "downloads"),
    (Join-Path $DataDir "diagnostics"),
    $configDir,
    $logDir
) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

New-Item -ItemType Directory -Force -Path $pathsToEnsure | Out-Null

if (-not (Test-Path $configPath)) {
    [System.IO.File]::WriteAllText($configPath, '{"model_config_list":[]}', $utf8NoBom)
}

$settings = [ordered]@{
    installDir = $InstallDir
    dataDir = $DataDir
    modelRepositoryPath = $modelsDir
    configPath = $configPath
    logPath = $logPath
    restPort = $effectiveRestPort
    grpcPort = $effectiveGrpcPort
    bindAddress = $effectiveBindAddress
    logLevel = $effectiveLogLevel
    runMode = $effectiveRunMode
}

$tmpSettings = "$settingsPath.tmp"
[System.IO.File]::WriteAllText($tmpSettings, ($settings | ConvertTo-Json -Depth 10), $utf8NoBom)
Move-Item -Force -Path $tmpSettings -Destination $settingsPath

$version = "unknown"
$ovmsExe = Join-Path $InstallDir "ovms.exe"
if (Test-Path $ovmsExe) {
    try {
        $versionOutput = & $ovmsExe --version 2>$null
        if ($versionOutput) {
            $version = ($versionOutput | Select-Object -First 1).ToString()
        }
    } catch {
        $version = "unknown"
    }
}

$installMarker = [ordered]@{
    productName = "OpenVINO Model Server"
    installDir = $InstallDir
    dataDir = $DataDir
    version = $version
    installedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    installerVersion = "1.0.0"
}

$tmpInstall = "$installMarkerPath.tmp"
[System.IO.File]::WriteAllText($tmpInstall, ($installMarker | ConvertTo-Json -Depth 10), $utf8NoBom)
Move-Item -Force -Path $tmpInstall -Destination $installMarkerPath

Write-SetupLog -Message "[INFO] OVMS configured at $DataDir" -LogPath $SetupLogPath
