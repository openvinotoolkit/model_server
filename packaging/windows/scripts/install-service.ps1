param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir
)

$ErrorActionPreference = "Stop"

$settingsPath = Join-Path $DataDir "settings.json"
if (-not (Test-Path $settingsPath)) {
    throw "Settings file not found: $settingsPath"
}

$settings = Get-Content $settingsPath -Raw | ConvertFrom-Json
$ovmsExe = Join-Path $InstallDir "ovms.exe"
if (-not (Test-Path $ovmsExe)) {
    throw "ovms.exe not found: $ovmsExe"
}

. (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir $InstallDir -PersistMachine

$binPath = "`"$ovmsExe`" --rest_port $($settings.restPort) --rest_bind_address `"$($settings.bindAddress)`" --config_path `"$($settings.configPath)`" --log_level $($settings.logLevel) --log_path `"$($settings.logPath)`""
if ($settings.grpcPort -and [int]$settings.grpcPort -gt 0) {
    $binPath += " --port $($settings.grpcPort)"
}

$existing = Get-Service ovms -ErrorAction SilentlyContinue
if ($existing) {
    & sc.exe config ovms binPath= $binPath DisplayName= "OpenVINO Model Server" | Out-Host
} else {
    & sc.exe create ovms binPath= $binPath DisplayName= "OpenVINO Model Server" start= demand | Out-Host
}
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create or update ovms service."
}

try {
    & $ovmsExe install | Out-Host
} catch {
    Write-Warning "ovms.exe install returned an error: $_"
}

$settings.runMode = "service"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($settingsPath, ($settings | ConvertTo-Json -Depth 10), $utf8NoBom)

Write-Host "[INFO] OVMS service installed."
