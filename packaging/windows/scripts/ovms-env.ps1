param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [switch]$PersistMachine
)

$ErrorActionPreference = "Stop"

$installDirFull = (Resolve-Path -LiteralPath $InstallDir).Path.TrimEnd("\")
$pythonHome = Join-Path $installDirFull "python"
$pythonScripts = Join-Path $pythonHome "Scripts"
$espeakData = Join-Path $installDirFull "espeak-ng-data"

$env:OVMS_DIR = $installDirFull
if (Test-Path -LiteralPath $pythonHome) {
    $env:PYTHONHOME = $pythonHome
    $env:PATH = (($installDirFull, $pythonHome, $pythonScripts, $env:PATH) -join ";")
} else {
    $env:PATH = (($installDirFull, $env:PATH) -join ";")
}

if (Test-Path -LiteralPath $espeakData) {
    $env:ESPEAK_DATA_PATH = $espeakData
}

if ($PersistMachine) {
    [Environment]::SetEnvironmentVariable("OVMS_DIR", $installDirFull, "Machine")
    if (Test-Path -LiteralPath $pythonHome) {
        [Environment]::SetEnvironmentVariable("PYTHONHOME", $pythonHome, "Machine")
    }
    if (Test-Path -LiteralPath $espeakData) {
        [Environment]::SetEnvironmentVariable("ESPEAK_DATA_PATH", $espeakData, "Machine")
    }
}
