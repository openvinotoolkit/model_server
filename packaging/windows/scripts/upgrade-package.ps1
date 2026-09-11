param(
    [Parameter(Mandatory = $true)][string]$InstallDir,
    [Parameter(Mandatory = $true)][string]$DataDir,
    [Parameter(Mandatory = $true)][string]$PackageUrl,
    [string]$Sha256Url = "",
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$PackageVariant = "python_on"
)

$ErrorActionPreference = "Stop"

$downloadsDir = Join-Path $DataDir "downloads"
$packagesDir = Join-Path $DataDir "packages"
$stagingRoot = Join-Path $packagesDir "staging"
$backupRoot = Join-Path $packagesDir "backup"
$safeVersion = ($Version -replace '[^A-Za-z0-9._-]', '_')
$stagingDir = Join-Path $stagingRoot $safeVersion
$backupDir = Join-Path $backupRoot ("{0}-{1}" -f $safeVersion, (Get-Date -Format "yyyyMMdd-HHmmss"))
$zipPath = Join-Path $downloadsDir ([IO.Path]::GetFileName(([Uri]$PackageUrl).AbsolutePath))
$shaPath = if ($Sha256Url) { Join-Path $downloadsDir ([IO.Path]::GetFileName(([Uri]$Sha256Url).AbsolutePath)) } else { "" }

New-Item -ItemType Directory -Force -Path $downloadsDir, $packagesDir, $stagingRoot, $backupRoot | Out-Null

Write-Host "[INFO] Downloading package: $PackageUrl"
Invoke-WebRequest -Uri $PackageUrl -OutFile $zipPath -UseBasicParsing

if ($Sha256Url) {
    Write-Host "[INFO] Downloading checksum: $Sha256Url"
    Invoke-WebRequest -Uri $Sha256Url -OutFile $shaPath -UseBasicParsing
    $expected = ((Get-Content $shaPath -Raw) -split '\s+')[0].Trim().ToLowerInvariant()
    $actual = (Get-FileHash -Algorithm SHA256 -Path $zipPath).Hash.ToLowerInvariant()
    if ($expected -and $actual -ne $expected) {
        throw "SHA256 mismatch. Expected $expected but got $actual."
    }
    Write-Host "[INFO] SHA256 verified."
}

if (Test-Path $stagingDir) {
    Remove-Item -Recurse -Force -Path $stagingDir
}
New-Item -ItemType Directory -Force -Path $stagingDir | Out-Null

Write-Host "[INFO] Extracting package to staging: $stagingDir"
Expand-Archive -Path $zipPath -DestinationPath $stagingDir -Force

$stagedOvms = Get-ChildItem -Path $stagingDir -Recurse -Filter "ovms.exe" | Select-Object -First 1
if (-not $stagedOvms) {
    throw "Staged package does not contain ovms.exe."
}
$sourceDir = $stagedOvms.Directory.FullName

Write-Host "[INFO] Validating staged ovms.exe."
$versionOutput = & $stagedOvms.FullName --version 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Staged ovms.exe --version failed: $versionOutput"
}

$runtimePath = Join-Path $DataDir "runtime.json"
$wasRunning = $false
if (Test-Path $runtimePath) {
    try {
        $runtime = Get-Content $runtimePath -Raw | ConvertFrom-Json
        if ($runtime.pid) {
            $process = Get-Process -Id ([int]$runtime.pid) -ErrorAction SilentlyContinue
            $wasRunning = $process -and $process.ProcessName -ieq "ovms"
        }
    } catch {
        $wasRunning = $false
    }
}

& (Join-Path $PSScriptRoot "stop-ovms.ps1") -DataDir $DataDir

Write-Host "[INFO] Backing up current install to: $backupDir"
New-Item -ItemType Directory -Force -Path $backupDir | Out-Null
Get-ChildItem -LiteralPath $InstallDir -Force |
    Where-Object { $_.Name -notlike "unins*" -and $_.Name -notin @("OVMS.Manager.exe", "OVMS.Manager.pdb", "scripts", "templates") } |
    ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $backupDir $_.Name) -Recurse -Force
    }

Write-Host "[INFO] Applying staged package."
Copy-Item -Path (Join-Path $sourceDir "*") -Destination $InstallDir -Recurse -Force

$installedOvms = Join-Path $InstallDir "ovms.exe"
$installedVersionOutput = & $installedOvms --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Installed ovms.exe validation failed. Restoring backup."
    Copy-Item -Path (Join-Path $backupDir "*") -Destination $InstallDir -Recurse -Force
    throw "Installed ovms.exe --version failed after upgrade: $installedVersionOutput"
}

& (Join-Path $PSScriptRoot "configure-ovms.ps1") -InstallDir $InstallDir -DataDir $DataDir -PackageVariant $PackageVariant | Out-Host

if ($wasRunning) {
    & (Join-Path $PSScriptRoot "start-ovms.ps1") -DataDir $DataDir
}

Write-Host "[INFO] Upgrade complete."
Write-Host $installedVersionOutput
