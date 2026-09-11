param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$PackageSource = ""
)

$ErrorActionPreference = "Stop"

$packageCacheDir = Join-Path $DataDir "packages\source"

$resolvedSource = ""
if ($PackageSource -and (Test-Path (Join-Path $PackageSource "ovms.exe"))) {
    $resolvedSource = $PackageSource
} elseif (Test-Path (Join-Path $packageCacheDir "ovms.exe")) {
    $resolvedSource = $packageCacheDir
}

if (-not $resolvedSource) {
    Write-Warning "No package source available to restore missing files from. Falling back to self-verify only."
    Write-Warning "Repair cannot restore missing files without a valid package source (checked: '$PackageSource', '$packageCacheDir')."
}

$required = @(
    "ovms.exe",
    "setupvars.bat",
    "setupvars.ps1",
    "install_ovms_service.bat"
)

$repairedFiles = @()
foreach ($file in $required) {
    $destPath = Join-Path $InstallDir $file
    if (-not (Test-Path $destPath)) {
        if ($resolvedSource) {
            $srcPath = Join-Path $resolvedSource $file
            if (Test-Path $srcPath) {
                $destDir = Split-Path -Parent $destPath
                if (-not (Test-Path $destDir)) {
                    New-Item -ItemType Directory -Force -Path $destDir | Out-Null
                }
                Copy-Item -Path $srcPath -Destination $destPath -Force
                $repairedFiles += $file
                Write-Host "[INFO] Restored missing file: $file"
            } else {
                Write-Warning "Missing file '$file' could not be restored: not found in package source."
            }
        } else {
            Write-Warning "Missing file '$file' could not be restored: no package source available."
        }
    }
}

$genAiInstalled = (Test-Path (Join-Path $InstallDir "openvino_genai.dll")) -and (Test-Path (Join-Path $InstallDir "openvino_tokenizers.dll"))
$pythonInstalled = Test-Path (Join-Path $InstallDir "python\python.exe")

Write-Host "[INFO] GenAI support: $(if ($genAiInstalled) { 'Installed' } else { 'Missing' })"
Write-Host "[INFO] Python support: $(if ($pythonInstalled) { 'Installed' } else { 'Missing' })"

$ovmsExe = Join-Path $InstallDir "ovms.exe"
if (-not (Test-Path $ovmsExe)) {
    throw "ovms.exe is missing from $InstallDir and could not be restored. Repair cannot continue."
}

. (Join-Path $PSScriptRoot "ovms-env.ps1") -InstallDir $InstallDir

try {
    $versionOutput = & $ovmsExe --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "ovms.exe --version exited with code $LASTEXITCODE."
    }
} catch {
    throw "ovms.exe --version failed during repair: $_"
}

$settingsPath = Join-Path $DataDir "settings.json"
$configPath = Join-Path $DataDir "models\config.json"

if ((-not (Test-Path $settingsPath)) -or (-not (Test-Path $configPath))) {
    Write-Host "[INFO] Regenerating missing configuration via configure-ovms.ps1."
    & (Join-Path $PSScriptRoot "configure-ovms.ps1") -InstallDir $InstallDir -DataDir $DataDir
}

Write-Host ""
if ($repairedFiles.Count -gt 0) {
    Write-Host "[INFO] Repair restored $($repairedFiles.Count) file(s): $($repairedFiles -join ', ')"
} else {
    Write-Host "[INFO] No required files were missing; no file repair was needed."
}
Write-Host "[INFO] GenAI support: $(if ($genAiInstalled) { 'Installed' } else { 'Missing' })"
Write-Host "[INFO] Python support: $(if ($pythonInstalled) { 'Installed' } else { 'Missing' })"
Write-Host "[INFO] Repair complete."
