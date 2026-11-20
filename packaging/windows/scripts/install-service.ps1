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

    . (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir $InstallDir -PersistMachine

    $binPath = "`"$ovmsExe`" --rest_port $($settings.restPort) --rest_bind_address `"$($settings.bindAddress)`" --config_path `"$($settings.configPath)`" --log_level $($settings.logLevel) --log_path `"$($settings.logPath)`""
    if ($settings.grpcPort -and [int]$settings.grpcPort -gt 0) {
        $binPath += " --port $($settings.grpcPort)"
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
    & sc.exe create ovms binPath= $binPath DisplayName= "OpenVINO Model Server" start= demand 2>&1 | Tee-Object -FilePath $SetupLogPath -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create ovms service."
    }

    try {
        Write-SetupLog -Message "[INFO] Running ovms.exe install step." -LogPath $SetupLogPath
        & $ovmsExe install 2>&1 | Tee-Object -FilePath $SetupLogPath -Append -Encoding utf8
    } catch {
        Write-SetupLog -Message "[WARN] ovms.exe install returned an error: $($_.Exception.Message)" -LogPath $SetupLogPath
    }

    $settings.runMode = "service"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($settingsPath, ($settings | ConvertTo-Json -Depth 10), $utf8NoBom)

    $service = Get-Service ovms -ErrorAction SilentlyContinue
    if ($service -and $service.Status -ne "Running") {
        Write-SetupLog -Message "[INFO] Starting OVMS service." -LogPath $SetupLogPath
        Start-Service ovms 2>&1 | Tee-Object -FilePath $SetupLogPath -Append -Encoding utf8
    }

    Write-SetupLog -Message "[INFO] OVMS service installed and started." -LogPath $SetupLogPath
} catch {
    Write-SetupLog -Message "[ERROR] OVMS service installation failed: $($_.Exception.Message)" -LogPath $SetupLogPath
    throw
}
