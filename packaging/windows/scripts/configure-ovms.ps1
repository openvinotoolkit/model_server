param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$PackageVariant = "python_on",
    [int]$RestPort = 8000,
    [int]$GrpcPort = 0,
    [string]$BindAddress = "127.0.0.1",
    [string]$LogLevel = "INFO"
)

$ErrorActionPreference = "Stop"

$modelsDir = Join-Path $DataDir "models"
$logsDir = Join-Path $DataDir "logs"
$configPath = Join-Path $modelsDir "config.json"
$settingsPath = Join-Path $DataDir "settings.json"
$installMarkerPath = Join-Path $DataDir "install.json"
$logPath = Join-Path $logsDir "ovms_server.log"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

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
    runMode = "user-login"
    startAtLogin = $true
    showTrayIcon = $true
    serviceAutoStart = $false
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
