param(
    [string]$Configuration = "Release",
    [string]$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path,
    [string]$OvmsSourceDir = "",
    [string]$OutputDir = "",
    [string]$InnoSetupPath = "",
    [Alias('h')]
    [switch]$Help,
    [switch]$Silent
)

function Show-InstallerBuildHelp {
    $lines = @(
        "OVMS installer build helper",
        "",
        "Usage:",
        "  .\build_installer.ps1 [-Configuration Release] [-SourceRoot <path>] [-OvmsSourceDir <path>] [-OutputDir <path>] [-InnoSetupPath <path>] [-Silent] [-Help]",
        "",
        "Options:",
        "  -Silent       Suppress normal progress output.",
        "  -Help / -h    Show this help and exit.",
        "",
        "Generated installer supports standard Inno Setup CLI modes:",
        "  OVMS-Setup.exe /SILENT",
        "  OVMS-Setup.exe /VERYSILENT",
        "  OVMS-Setup.exe /SUPPRESSMSGBOXES /NORESTART"
    )

    $lines | ForEach-Object { Write-Output $_ }
}

if ($Help -or $PSBoundParameters.ContainsKey('Help')) {
    Show-InstallerBuildHelp
    exit 0
}

$ErrorActionPreference = "Stop"
trap {
    $message = $_.Exception.Message
    if ($message) {
        [System.Console]::Error.WriteLine("OVMS installer build failed: $message")
    }
    exit 1
}

$buildLogPath = Join-Path (Get-Location) "build_installer.log"
$ovmsDir = if ($OvmsSourceDir) { (Resolve-Path -LiteralPath $OvmsSourceDir).Path } else { Join-Path $SourceRoot "dist\windows\ovms" }
$installerScript = Join-Path $PSScriptRoot "OVMSInstaller.iss"
$outputDir = if ($OutputDir) { $OutputDir } else { Join-Path $SourceRoot "dist\windows" }

if (-not (Test-Path (Join-Path $ovmsDir "ovms.exe"))) {
    throw "Missing OVMS package at $ovmsDir. Run windows_create_package.bat first."
}

if ($Silent) {
    $ProgressPreference = 'SilentlyContinue'
}

if (-not $InnoSetupPath) {
    $cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($cmd) {
        $InnoSetupPath = $cmd.Source
    } else {
        $candidates = @(
            (Join-Path $env:SystemDrive "opt\Inno Setup 7\ISCC.exe"),
            "${env:ProgramFiles(x86)}\Inno Setup 7\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 7\ISCC.exe"
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

$buildLogPath = [System.IO.Path]::GetFullPath($buildLogPath)
$buildLogParent = Split-Path -Parent $buildLogPath
if ($buildLogParent -and -not (Test-Path $buildLogParent)) {
    New-Item -ItemType Directory -Force -Path $buildLogParent | Out-Null
}

$compilerArgs = @(
    "/DSourceRoot=$SourceRoot",
    "/DOvmsSourceDir=$ovmsDir",
    "/DOutputDir=$outputDir",
    $installerScript
)

if (-not $Silent) {
    Write-Host "[INFO] Running Inno Setup compiler: $InnoSetupPath"
    Write-Host "[INFO] Log file: $buildLogPath"
}

& $InnoSetupPath $compilerArgs 2>&1 | Tee-Object -FilePath $buildLogPath
$compilerExitCode = $LASTEXITCODE

if ($compilerExitCode -ne 0) {
    throw "Inno Setup failed with exit code $compilerExitCode. See $buildLogPath for details."
}

if (-not $Silent) {
    Write-Host "[INFO] Installer created at $(Join-Path $outputDir 'OVMS-Setup.exe')"
    Write-Host "[INFO] Build log path: $buildLogPath"
}
