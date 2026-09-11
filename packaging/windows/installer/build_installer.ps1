param(
    [string]$Configuration = "Release",
    [string]$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path,
    [string]$OvmsSourceDir = "",
    [string]$OutputDir = "",
    [string]$InnoSetupPath = ""
)

$ErrorActionPreference = "Stop"

$ovmsDir = if ($OvmsSourceDir) { (Resolve-Path -LiteralPath $OvmsSourceDir).Path } else { Join-Path $SourceRoot "dist\windows\ovms" }
$managerPublishDir = Join-Path $SourceRoot "packaging\windows\manager\artifacts\publish"
$installerScript = Join-Path $PSScriptRoot "OVMSInstaller.iss"
$outputDir = if ($OutputDir) { $OutputDir } else { Join-Path $SourceRoot "dist\windows" }

if (-not (Test-Path (Join-Path $ovmsDir "ovms.exe"))) {
    throw "Missing OVMS package at $ovmsDir. Run windows_create_package.bat first."
}

if (-not (Test-Path (Join-Path $managerPublishDir "OVMS.Manager.exe"))) {
    & (Join-Path $SourceRoot "packaging\windows\manager\build_manager.ps1") -Configuration $Configuration
}

if (-not $InnoSetupPath) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $InnoSetupPath = $cmd.Source
    } else {
        $candidates = @(
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )
        foreach ($candidate in $candidates) {
            if (Test-Path $candidate) {
                $InnoSetupPath = $candidate
                break
            }
        }
    }
}

if (-not $InnoSetupPath -or -not (Test-Path $InnoSetupPath)) {
    throw "Inno Setup compiler ISCC.exe was not found. Install Inno Setup 6 or pass -InnoSetupPath."
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

& $InnoSetupPath `
    "/DSourceRoot=$SourceRoot" `
    "/DOvmsSourceDir=$ovmsDir" `
    "/DManagerPublishDir=$managerPublishDir" `
    "/DOutputDir=$outputDir" `
    $installerScript

if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE."
}

Write-Host "[INFO] Installer created at $(Join-Path $outputDir 'OVMS-Setup.exe')"
