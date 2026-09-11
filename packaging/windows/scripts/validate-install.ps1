param(
    [string]$InstallDir
)

$ErrorActionPreference = "Stop"

if (-not $InstallDir) {
    $sourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
    $InstallDir = Join-Path $sourceRoot "dist\windows\ovms"
}

$required = @(
    "ovms.exe",
    "setupvars.bat",
    "setupvars.ps1",
    "install_ovms_service.bat"
)

$missing = @()
foreach ($file in $required) {
    $path = Join-Path $InstallDir $file
    if (-not (Test-Path $path)) {
        $missing += $file
    }
}

if ($missing.Count -gt 0) {
    throw "Missing required files in ${InstallDir}: $($missing -join ', ')"
}

& (Join-Path $InstallDir "ovms.exe") --version
if ($LASTEXITCODE -ne 0) {
    throw "ovms.exe --version failed."
}

Write-Host "[INFO] Install folder validation passed: $InstallDir"
