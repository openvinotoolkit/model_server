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

