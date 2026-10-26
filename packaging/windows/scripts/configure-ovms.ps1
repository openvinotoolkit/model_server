param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$PackageVariant = "python_on",
    [string]$RunMode = "user-login",
    [int]$RestPort = 8000,
    [int]$GrpcPort = 0,
    [string]$BindAddress = "127.0.0.1",
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"
trap {
    $message = $_.Exception.Message
    if ($message) {
        [System.Console]::Error.WriteLine("OVMS configuration failed: $message")
    }
    exit 1
}

function Ensure-SystemPythonRequirements {
    param(
        [Parameter(Mandatory = $true)][string]$Variant
    )

    if ($Variant -ne "system") {
        return
    }

    $pythonCommand = $null
    $pythonName = "python"

    if (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCommand = (Get-Command python).Source
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCommand = "py"
        $pythonName = "py"
    } else {
        throw "System Python was selected, but no Python executable was found on PATH. Install Python 3.12 and try again."
    }

    try {
        if ($pythonName -eq "py") {
            $versionOutput = & $pythonCommand -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        } else {
            $versionOutput = & $pythonCommand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        }

        $versionText = ($versionOutput | Select-Object -First 1).ToString().Trim()
        if (-not $versionText) {
            throw "Unable to detect the system Python version."
        }

        if ($versionText -ne "3.12") {
            throw "System Python version $versionText is not supported. OVMS requires Python 3.12 when using the system-python option."
        }

        $pipArgs = @(
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "setuptools==80.9.0",
            "Jinja2==3.1.6",
            "MarkupSafe==3.0.2"
        )

        if ($pythonName -eq "py") {
            & $pythonCommand -3.12 @pipArgs
        } else {
            & $pythonCommand @pipArgs
        }

        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install the required Python dependencies for the system-Python configuration."
        }
    } catch {
        throw "System Python validation failed: $($_.Exception.Message)"
    }
}

$modelsDir = Join-Path $DataDir "models"
$logsDir = Join-Path $DataDir "logs"
$configPath = Join-Path $modelsDir "config.json"
$settingsPath = Join-Path $DataDir "settings.json"
$installMarkerPath = Join-Path $DataDir "install.json"
$logPath = Join-Path $logsDir "ovms_server.log"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

Ensure-SystemPythonRequirements -Variant $PackageVariant

New-Item -ItemType Directory -Force -Path $DataDir, $modelsDir, $logsDir, (Join-Path $DataDir "packages"), (Join-Path $DataDir "downloads"), (Join-Path $DataDir "diagnostics") | Out-Null

if (-not (Test-Path $configPath)) {
    [System.IO.File]::WriteAllText($configPath, '{"model_config_list":[]}', $utf8NoBom)
}

$settings = [ordered]@{
    installDir = $InstallDir
    dataDir = $DataDir
    modelRepositoryPath = $modelsDir
    configPath = $configPath
    logPath = $logPath
    restPort = $RestPort
    grpcPort = $GrpcPort
    bindAddress = $BindAddress
    logLevel = $LogLevel
    runMode = $RunMode
    startAtLogin = ($RunMode -ne "service")
    showTrayIcon = ($RunMode -ne "service")
    serviceAutoStart = ($RunMode -eq "service")
    packageVariant = $PackageVariant
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
    packageVariant = $PackageVariant
    installedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    installerVersion = "1.0.0"
}

$tmpInstall = "$installMarkerPath.tmp"
[System.IO.File]::WriteAllText($tmpInstall, ($installMarker | ConvertTo-Json -Depth 10), $utf8NoBom)
Move-Item -Force -Path $tmpInstall -Destination $installMarkerPath

Write-Host "[INFO] OVMS configured at $DataDir"
