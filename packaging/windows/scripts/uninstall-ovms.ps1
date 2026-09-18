param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [ValidateSet("PreserveAll", "RemoveSettingsKeepModels", "RemoveAll")][string]$DataMode = "PreserveAll"
)

$ErrorActionPreference = "Stop"

Get-Process -Name "OVMS.Manager" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

$runtimePath = Join-Path $DataDir "runtime.json"
if (Test-Path $runtimePath) {
    try {
        $runtime = Get-Content $runtimePath -Raw | ConvertFrom-Json
        if ($runtime.pid) {
            $process = Get-Process -Id ([int]$runtime.pid) -ErrorAction SilentlyContinue
            if ($process -and $process.ProcessName -ieq "ovms") {
                Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
            }
        }
    } catch {
        Write-Warning "Could not read runtime state: $_"
    }
    Remove-Item -Force -Path $runtimePath -ErrorAction SilentlyContinue
}

& (Join-Path $PSScriptRoot "uninstall-service.ps1")
& (Join-Path $PSScriptRoot "set-path.ps1") -InstallDir $InstallDir -Action Remove
[Environment]::SetEnvironmentVariable("OVMS_DIR", $null, "User")
[Environment]::SetEnvironmentVariable("PYTHONHOME", $null, "User")
[Environment]::SetEnvironmentVariable("ESPEAK_DATA_PATH", $null, "User")

Remove-Item -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run" -Name "OVMS Manager" -ErrorAction SilentlyContinue

switch ($DataMode) {
    "PreserveAll" {
        # Keep everything under $DataDir.
    }
    "RemoveSettingsKeepModels" {
        Remove-Item -Force -Path (Join-Path $DataDir "settings.json") -ErrorAction SilentlyContinue
        Remove-Item -Force -Path (Join-Path $DataDir "install.json") -ErrorAction SilentlyContinue
        Remove-Item -Force -Path (Join-Path $DataDir "runtime.json") -ErrorAction SilentlyContinue

        $logsDir = Join-Path $DataDir "logs"
        if (Test-Path $logsDir) {
            Remove-Item -LiteralPath $logsDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        $packagesDir = Join-Path $DataDir "packages"
        if (Test-Path $packagesDir) {
            Remove-Item -LiteralPath $packagesDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        $downloadsDir = Join-Path $DataDir "downloads"
        if (Test-Path $downloadsDir) {
            Remove-Item -LiteralPath $downloadsDir -Recurse -Force -ErrorAction SilentlyContinue
        }

        $diagnosticsDir = Join-Path $DataDir "diagnostics"
        if (Test-Path $diagnosticsDir) {
            Remove-Item -LiteralPath $diagnosticsDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    "RemoveAll" {
        if (Test-Path $DataDir) {
            Remove-Item -LiteralPath $DataDir -Recurse -Force
        }
    }
}

Write-Host "[INFO] OVMS uninstall cleanup complete."
