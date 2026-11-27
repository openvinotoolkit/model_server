#
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

function Get-RelativePathNormalized {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$FullPath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $fullFull = [System.IO.Path]::GetFullPath($FullPath)

    if ([System.IO.Path].GetMethod('GetRelativePath') -ne $null) {
        $relative = [System.IO.Path]::GetRelativePath($baseFull, $fullFull)
        return $relative.Replace('\\', '/').Replace('\', '/')
    }

    $baseUri = New-Object System.Uri(([System.IO.Path]::GetFullPath($baseFull).TrimEnd('\','/') + [System.IO.Path]::DirectorySeparatorChar))
    $fullUri = New-Object System.Uri($fullFull)
    $relative = [System.Uri]::UnescapeDataString($baseUri.MakeRelativeUri($fullUri).ToString())
    return $relative.Replace('\\', '/').Replace('\', '/')
}

function Get-ManagedInstallFiles {
    param(
        [Parameter(Mandatory = $true)][string]$InstallDir,
        [Parameter(Mandatory = $true)][string]$PackageRoot
    )

    $expected = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)

    if (Test-Path $PackageRoot) {
        Get-ChildItem -LiteralPath $PackageRoot -File -Recurse | ForEach-Object {
            $relative = Get-RelativePathNormalized -BasePath $PackageRoot -FullPath $_.FullName
            $expected.Add($relative) | Out-Null
        }
    }

    foreach ($subdir in @('scripts', 'templates')) {
        $dir = Join-Path $InstallDir $subdir
        if (Test-Path $dir) {
            Get-ChildItem -LiteralPath $dir -File -Recurse | ForEach-Object {
                $relative = Get-RelativePathNormalized -BasePath $InstallDir -FullPath $_.FullName
                $expected.Add($relative) | Out-Null
            }
        }
    }

    Write-Output -NoEnumerate $expected
}

function Remove-DisabledComponentFiles {
    param(
        [Parameter(Mandatory = $true)][string]$InstallDir,
        [Parameter(Mandatory = $true)][hashtable]$ComponentSelection
    )

    $componentToFiles = @{
        opencv_mp = @(
            'opencv_world4140.dll',
            'ovms_opencv_calculators.dll',
            'ovms_opencv_image_decoder.dll'
        )
        espeak = @('espeak-ng.dll', 'espeak-ng-data')
        gpu = @('openvino_intel_gpu_plugin.dll')
        npu = @(
            'openvino_intel_npu_compiler.dll',
            'openvino_intel_npu_compiler_loader.dll',
            'openvino_intel_npu_plugin.dll',
            'openvino_intel_npu_vm_runtime.dll'
        )
        onnx_frontend = @('openvino_onnx_frontend.dll')
        paddle_frontend = @('openvino_paddle_frontend.dll')
        tensorflow_selected = @(
            'openvino_tensorflow_frontend.dll',
            'openvino_tensorflow_lite_frontend.dll'
        )
    }

    $failedRemovals = @()

    foreach ($component in $ComponentSelection.Keys) {
        if (-not $ComponentSelection[$component]) {
            if ($componentToFiles.ContainsKey($component)) {
                foreach ($relativePath in $componentToFiles[$component]) {
                    $targetPath = Join-Path $InstallDir $relativePath
                    if (Test-Path $targetPath) {
                        try {
                            Write-Host "[INFO] Removing deselected component '$component': $relativePath"
                            if ((Get-Item -LiteralPath $targetPath).PSIsContainer) {
                                Remove-Item -LiteralPath $targetPath -Recurse -Force -ErrorAction Stop
                            } else {
                                Remove-Item -LiteralPath $targetPath -Force -ErrorAction Stop
                            }
                            Write-Host "[INFO] Removed deselected component artifact: $relativePath"
                        } catch {
                            $message = "Failed to remove deselected component '$component' for path '$targetPath': $($_.Exception.Message)"
                            Write-Error $message
                            $failedRemovals += $message
                        }
                    }
                }
            }
        }
    }

    if ($failedRemovals.Count -gt 0) {
        throw "Repair cannot continue because one or more deselected component files could not be removed. See the error messages above for the exact paths."
    }
}

function Get-JsonBoolValue {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string[]]$Paths
    )

    foreach ($path in $Paths) {
        $value = $Object
        foreach ($part in ($path -split '\.')) {
            if ($null -eq $value) { $value = $null; break }
            $value = $value.$part
        }
        if ($null -ne $value) {
            return [bool]$value
        }
    }

    return $false
}

function Test-DeselectedComponentPath {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][hashtable]$ComponentSelection,
        [Parameter(Mandatory = $true)][string]$PythonMode
    )

    $relative = $RelativePath.Replace('\', '/').TrimStart('/')

    if (-not $ComponentSelection.opencv_mp) {
        foreach ($path in @('opencv_world4140.dll', 'ovms_opencv_calculators.dll', 'ovms_opencv_image_decoder.dll')) {
            if ($relative.Equals($path, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
    }
    if (-not $ComponentSelection.espeak) {
        if ($relative.Equals('espeak-ng.dll', [System.StringComparison]::OrdinalIgnoreCase) -or
            $relative.StartsWith('espeak-ng-data/', [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    if (-not $ComponentSelection.gpu) {
        if ($relative.Equals('openvino_intel_gpu_plugin.dll', [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    if (-not $ComponentSelection.npu) {
        foreach ($path in @('openvino_intel_npu_compiler.dll', 'openvino_intel_npu_compiler_loader.dll', 'openvino_intel_npu_plugin.dll', 'openvino_intel_npu_vm_runtime.dll')) {
            if ($relative.Equals($path, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
    }
    if (-not $ComponentSelection.onnx_frontend) {
        if ($relative.Equals('openvino_onnx_frontend.dll', [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    if (-not $ComponentSelection.paddle_frontend) {
        if ($relative.Equals('openvino_paddle_frontend.dll', [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    if (-not $ComponentSelection.tensorflow_selected) {
        foreach ($path in @('openvino_tensorflow_frontend.dll', 'openvino_tensorflow_lite_frontend.dll')) {
            if ($relative.Equals($path, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
    }
    if (-not $ComponentSelection.python_embedded) {
        if ($relative.Equals('python', [System.StringComparison]::OrdinalIgnoreCase) -or
            $relative.StartsWith('python/', [System.StringComparison]::OrdinalIgnoreCase)) {
            if ($PythonMode -ieq 'none') {
                return $true
            }
            if ($PythonMode -ieq 'system' -and
                -not $relative.Equals('python/pyovms.pyd', [System.StringComparison]::OrdinalIgnoreCase)) {
                return $true
            }
        }
    }

    return $false
}

function Remove-DisabledPythonFiles {
    param(
        [Parameter(Mandatory = $true)][string]$InstallDir,
        [Parameter(Mandatory = $true)][string]$PythonMode
    )

    $pythonDir = Join-Path $InstallDir 'python'
    if (-not (Test-Path -LiteralPath $pythonDir)) {
        return
    }

    if ($PythonMode -ieq 'none') {
        Remove-Item -LiteralPath $pythonDir -Recurse -Force -ErrorAction Stop
        Write-Host "[INFO] Removed deselected component 'python': $pythonDir"
        return
    }

    if ($PythonMode -ieq 'system') {
        Get-ChildItem -LiteralPath $pythonDir -File -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object {
                $relative = Get-RelativePathNormalized -BasePath $pythonDir -FullPath $_.FullName
                -not $relative.Replace('\', '/').Equals('pyovms.pyd', [System.StringComparison]::OrdinalIgnoreCase)
            } |
            ForEach-Object {
                Remove-Item -LiteralPath $_.FullName -Force -ErrorAction Stop
            }

        Get-ChildItem -LiteralPath $pythonDir -Directory -Recurse -Force -ErrorAction SilentlyContinue |
            Sort-Object -Property FullName -Descending |
            ForEach-Object {
                $children = Get-ChildItem -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
                if (-not $children) {
                    Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
                }
            }
        Write-Host "[INFO] Removed embedded Python runtime; retained system-Python binding."
    }
}

$managedInstallFiles = @{}
if ($resolvedSource) {
    $managedInstallFiles = Get-ManagedInstallFiles -InstallDir $InstallDir -PackageRoot $resolvedSource
}

$installOptionsPath = Join-Path $DataDir 'install-options.json'
$componentSelection = @{}
if (-not (Test-Path $installOptionsPath)) {
    throw "Repair requires install-options.json at '$installOptionsPath', but it was not found. Selective component cleanup cannot run without the saved install state."
}

try {
    $optionsJson = Get-Content -LiteralPath $installOptionsPath -Raw | ConvertFrom-Json -ErrorAction Stop
    $pythonMode = [string]($optionsJson.pythonMode)
    $componentSelection = @{
        opencv_mp = (Get-JsonBoolValue -Object $optionsJson -Paths @('opencv_mp'))
        espeak = (Get-JsonBoolValue -Object $optionsJson -Paths @('espeak'))
        gpu = (Get-JsonBoolValue -Object $optionsJson -Paths @('gpu'))
        npu = (Get-JsonBoolValue -Object $optionsJson -Paths @('npu'))
        onnx_frontend = (Get-JsonBoolValue -Object $optionsJson -Paths @('openvino_frontends.onnx'))
        paddle_frontend = (Get-JsonBoolValue -Object $optionsJson -Paths @('openvino_frontends.paddle'))
        tensorflow_selected = (Get-JsonBoolValue -Object $optionsJson -Paths @('openvino_frontends.tensorflow'))
        python_embedded = ($pythonMode -eq 'embedded')
    }
} catch {
    $message = "install-options.json is invalid or unreadable at '$installOptionsPath': $($_.Exception.Message)"
    Write-Error $message
    throw $message
}

if ($componentSelection.Count -gt 0) {
    Remove-DisabledComponentFiles -InstallDir $InstallDir -ComponentSelection $componentSelection
}
Remove-DisabledPythonFiles -InstallDir $InstallDir -PythonMode $pythonMode

$staleFiles = @()
if ($managedInstallFiles.Count -gt 0) {
    Get-ChildItem -LiteralPath $InstallDir -File -Recurse | ForEach-Object {
        $relative = Get-RelativePathNormalized -BasePath $InstallDir -FullPath $_.FullName
        if (-not $managedInstallFiles.Contains($relative) -and $_.Name -notlike 'unins*' -and $_.Name -notlike 'OVMS.Manager*') {
            $staleFiles += $_.FullName
        }
    }

    foreach ($file in $staleFiles) {
        Remove-Item -LiteralPath $file -Force
        Write-Host "[INFO] Removed stale file no longer in current package: $file"
    }

    $dirs = Get-ChildItem -LiteralPath $InstallDir -Directory -Recurse | Sort-Object -Descending FullName
    foreach ($dir in $dirs) {
        $children = Get-ChildItem -LiteralPath $dir.FullName -Force -ErrorAction SilentlyContinue
        if (-not $children) {
            Remove-Item -LiteralPath $dir.FullName -Force
        }
    }
}

$repairedFiles = @()
if ($resolvedSource) {
    $sourceFiles = Get-ChildItem -LiteralPath $resolvedSource -File -Recurse -ErrorAction Stop
    foreach ($sourceFile in $sourceFiles) {
        $relativePath = Get-RelativePathNormalized -BasePath $resolvedSource -FullPath $sourceFile.FullName
        if (Test-DeselectedComponentPath -RelativePath $relativePath -ComponentSelection $componentSelection -PythonMode $pythonMode) {
            continue
        }

        $destPath = Join-Path $InstallDir $relativePath
        if (Test-Path -LiteralPath $destPath) {
            # Skip files that are already in the destination
            continue
        }

        $destDir = Split-Path -Parent $destPath
        if (-not (Test-Path -LiteralPath $destDir)) {
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
        }
        Copy-Item -LiteralPath $sourceFile.FullName -Destination $destPath -Force -ErrorAction Stop
        $repairedFiles += $relativePath
        Write-Host "[INFO] Restored missing file: $relativePath"
    }
} elseif (-not $resolvedSource) {
    Write-Warning "Missing files could not be restored: no package source available."
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
